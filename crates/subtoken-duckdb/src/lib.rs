//! The `subtoken` DuckDB extension.
//!
//! The whole registered surface is seven functions:
//!
//! | function | returns | why it exists |
//! |---|---|---|
//! | `subtoken_embed(text VARCHAR)` | `FLOAT[]` | the product: one row in, one vector out |
//! | `subtoken_is_truncated(text VARCHAR)` | `BOOLEAN` | whether `subtoken_embed(text)` had to drop content to fit |
//! | `subtoken_version()` | `VARCHAR` | which build, which model, which width |
//! | `subtoken_model_id()` | `VARCHAR` | the model key, joinable: store it beside a vector column |
//! | `subtoken_models()` | `STRUCT(model, backend, revision, width, input_limit, licence, tier, "key")` | the catalogue row for the model this build serves |
//! | `subtoken_cache_stats()` | `STRUCT(hits, misses, encoded, uncached, entries, capacity)` | makes "did it re-embed?" answerable in SQL |
//! | `subtoken_cache_clear()` | `BIGINT` | vectors dropped; lets a session start from a known state |
//!
//! # Comparing vectors across a model change
//!
//! Vectors from two models are not comparable, and nothing about a stored
//! column says which model wrote it: the width is the same, the norm is the
//! same, and the numbers are plausible. `subtoken_model_id()` is the value that
//! makes the question answerable in SQL — stored once beside the vectors, it
//! turns "are these comparable?" into an equality a later session can put in a
//! `WHERE` clause or raise an error on, **before** any vector is read.
//!
//! ```sql
//! CREATE TABLE v AS
//!     SELECT id, subtoken_embed(descr) AS vec, subtoken_model_id() AS model_id
//!     FROM corpus;
//!
//! -- In a later session, after an upgrade:
//! SELECT count(*) FROM v WHERE model_id <> subtoken_model_id();
//! ```
//!
//! `subtoken_embed` is a **scalar**, and that is the product argument rather than an
//! implementation detail: a scalar composes with `WHERE` and `LIMIT`, so a
//! filtered subset of a table can be embedded without materialising the rest,
//! and it behaves the same over a local Parquet file as over a remote table.
//!
//! There is deliberately no similarity or nearest-neighbour function here. The
//! measured position of this model is that a map built from its vectors keeps
//! the cluster structure and does not keep the neighbourhoods, so shipping a
//! "rows most like this one" surface would promise something the model does not
//! deliver. README.md and description.yml both say so to the person reading
//! the page, under "What it is good at, and what it is not", with the figures.
//!
//! # NULL
//!
//! `subtoken_embed(NULL)` is `NULL`. Text with no in-vocabulary tokens — `''`,
//! whitespace, a string of symbols outside the vocabulary — is a zero vector of
//! full width, because that is what the mean over zero tokens is. The two cases
//! are different on purpose: absence of a value is not the same as a value that
//! carries no signal. A caller that wants the pipeline's single behaviour for
//! both writes `subtoken_embed(coalesce(t, ''))`.
//!
//! # Truncation
//!
//! `subtoken_embed` builds its vector from at most 512 tokens of `text`, and from a
//! bounded number of its characters before that; anything past either is
//! discarded before the mean, and the vector that comes back is full width and
//! unit norm either way, so a truncated result looks exactly like a complete
//! one. Which limit bites first is a property of the text: for URLs,
//! run-together identifiers and compound words the character cut lands well
//! before 512 tokens are reached, and for Korean the token cap arrives after a
//! few hundred characters. `subtoken_is_truncated(text)` answers "did this
//! happen" directly, without asking the caller to know either limit — and it
//! answers no for a text that is past a cut but lost nothing to it, which is
//! the question an analyst is actually asking. `subtoken_is_truncated(NULL)` is
//! `NULL`, matching `subtoken_embed(NULL)`.

//!
//! # Registration on DuckDB 2.0
//!
//! The extension is built on DuckDB 2.0's v2 C API, which DuckDB 2.0.0 ships
//! as unstable. The artifact's footer is `C_STRUCT_UNSTABLE` with DuckDB
//! version `v2.0.0`, so it loads on exactly that DuckDB release; each new
//! release means vendoring its headers into `subtoken-duckdb-sys`,
//! regenerating the bindings if they changed, and moving
//! [`MIN_DUCKDB_VERSION`] and the Makefile's `TARGET_DUCKDB_VERSION`, then
//! rebuilding and changing the registry entry.
//!
//! The crate carries its own v2 bindings instead of depending on duckdb-rs:
//! `subtoken-duckdb-sys` vendors DuckDB v2.0.0's headers, and
//! `scripts/regenerate_v2_bindings.sh` generates the bindings from them; CI
//! fails if the checked-in bindings drift from the headers.
//!
//! What `duckdb_functions()` shows for each function (parameter names,
//! description, example, category) comes from the `DOCS` table in this file.
//! `scripts/check_documented_surface.py` and SQL test 13 hold it to the
//! registered catalog.
//!
//! DuckDB calls [`subtoken_init_c_api_v2`] at `LOAD`. It fetches the API table
//! (`subtoken_duckdb_sys::init`), takes a factory from the load's context, and
//! [`register_all`] registers the seven functions through the builder in
//! `capi.rs`: each with its parameters, its return type parsed from SQL type
//! text, its stability, and one exec callback shared by all seven that hands
//! the function a [`capi::Call`] to read its argument from and write its
//! result to. A failure at any step is reported through the load's error slot,
//! and a panic is caught and reported the same way, so nothing unwinds into
//! DuckDB.

mod capi;

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use capi::{guard, Call, Docs, Registrar, ScalarFunction};
use subtoken_duckdb_sys::duckdb_v2_extension_input;

/// The exact DuckDB release this artifact is built for and declares in its
/// footer. The v2 C API is unstable on DuckDB 2.0.0, so DuckDB loads the
/// artifact on this release only.
const MIN_DUCKDB_VERSION: &str = "v2.0.0";

/// The footer's DuckDB version and the API table `init` requests name the same
/// release, or the build fails.
const _: () = assert!(
    bytes_equal(
        MIN_DUCKDB_VERSION.as_bytes(),
        subtoken_duckdb_sys::API_VERSION.to_bytes()
    ),
    "MIN_DUCKDB_VERSION and subtoken_duckdb_sys::API_VERSION disagree"
);

const fn bytes_equal(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// `subtoken_embed(text VARCHAR) → FLOAT[]`
///
/// The vector for one string, from the model bundled in this binary. No network
/// call, no API key, no configuration: the model is parsed out of the binary on
/// the first call and reused for the rest of the session.
fn embed(call: &mut Call) -> Result<(), String> {
    let cells = call.varchar_arg(0)?;
    let mut vectors: Vec<Option<Arc<[f32]>>> = Vec::with_capacity(cells.len());
    for cell in &cells {
        match cell {
            None => vectors.push(None),
            Some(text) => vectors.push(Some(subtoken_core::embed(text)?)),
        }
    }
    call.write_float_lists(&vectors)
}

/// `subtoken_is_truncated(text VARCHAR) → BOOLEAN`
///
/// True if `subtoken_embed(text)` dropped content, so the vector it returns does not
/// reflect all of `text` — even though it is full width and unit norm like any
/// other. False for a text that ran past a limit without losing an id to it.
/// See the module docs' *Truncation* section for the limits and what a caller
/// does with this.
fn is_truncated(call: &mut Call) -> Result<(), String> {
    let cells = call.varchar_arg(0)?;
    let mut rows: Vec<Option<bool>> = Vec::with_capacity(cells.len());
    for cell in &cells {
        match cell {
            None => rows.push(None),
            Some(text) => rows.push(Some(subtoken_core::is_truncated(text)?)),
        }
    }
    call.write_bools(&rows)
}

/// `subtoken_version() → VARCHAR`
fn version(call: &mut Call) -> Result<(), String> {
    call.write_varchar_constant(&subtoken_core::describe())
}

/// `subtoken_model_id() → VARCHAR`
///
/// The full 64-hex model key, on its own. `subtoken_version()` carries the same
/// key truncated to twelve characters inside a sentence, which is readable and
/// not joinable; this is the value an analyst stores beside a vector column so
/// that a later session can refuse to compare across a model change **before**
/// comparing anything.
fn model_id(call: &mut Call) -> Result<(), String> {
    call.write_varchar_constant(&subtoken_core::model_id()?)
}

/// Field order of the STRUCT `subtoken_models()` returns, with the DuckDB type
/// of each. The order is part of the signature, so it is written once and used
/// for both the declared type and the write.
const MODEL_FIELDS: [(&str, &str); 8] = [
    ("model", "VARCHAR"),
    ("backend", "VARCHAR"),
    ("revision", "VARCHAR"),
    ("width", "BIGINT"),
    ("input_limit", "BIGINT"),
    ("licence", "VARCHAR"),
    ("tier", "VARCHAR"),
    ("key", "VARCHAR"),
];

/// `subtoken_models() → STRUCT(model, backend, revision, width, input_limit, licence, tier, "key")`
///
/// The catalogue row for the model this build serves: what wrote a vector, at
/// what revision, how wide, how much text it reads, under what licence, at what
/// support tier, and under what key. One row today, because one model is
/// bundled.
///
/// It is a **scalar returning a STRUCT rather than a table function**, and the
/// reason is the checks rather than the ergonomics. `duckdb_functions()` reports
/// `return_type` as NULL for a table function, and the derived-surface check
/// that holds README.md, description.yml and this module doc to the loaded
/// catalog compares exactly that column — a table function is invisible to it,
/// and so is a documented return type that has drifted from the real one. The
/// row is the same either way; this shape is the one a gate can see.
fn models(call: &mut Call) -> Result<(), String> {
    let row = subtoken_core::catalogue()?;
    // Same order as MODEL_FIELDS, and the compiler is no help if it is not:
    // both lists are written out here so the pairing is one line apart.
    let text: [(usize, &str); 6] = [
        (0, row.model),
        (1, row.backend),
        (2, row.revision),
        (5, row.licence),
        (6, row.tier),
        (7, &row.key),
    ];
    let numeric: [(usize, i64); 2] = [(3, row.width as i64), (4, row.input_limit as i64)];
    for (index, value) in text {
        call.struct_child(index)?.write_varchar_constant(value)?;
    }
    for (index, value) in numeric {
        call.struct_child(index)?.write_bigint_constant(value)?;
    }
    Ok(())
}

/// Field order of the STRUCT `subtoken_cache_stats()` returns, every field a
/// BIGINT. The order is part of the signature, so it is written once and used
/// for both the type and the write.
const STATS_FIELDS: [&str; 6] = [
    "hits", "misses", "encoded", "uncached", "entries", "capacity",
];

/// `subtoken_cache_stats() → STRUCT(hits, misses, encoded, uncached, entries, capacity)`
///
/// `encoded` is the number of times the encoder has actually run since the last
/// `subtoken_cache_clear()`. It is the observable behind "a repeated query
/// does not re-embed": run a query twice and `encoded` does not move.
///
/// `uncached` is how many lookups the cache was too full to store. Non-zero
/// means the column has more distinct values than `capacity`, so a repeated
/// query will re-embed the excess — which is the one condition under which
/// `encoded` legitimately moves on a repeat.
///
/// Registered VOLATILE: otherwise DuckDB folds a zero-argument scalar to a
/// constant and the counters would be read once and reused for the rest of
/// the session.
fn cache_stats(call: &mut Call) -> Result<(), String> {
    let stats = subtoken_core::stats();
    let values = [
        stats.hits,
        stats.misses,
        stats.encoded,
        stats.uncached,
        stats.entries,
        stats.capacity,
    ];
    for (index, value) in values.iter().enumerate() {
        call.struct_child(index)?
            .write_bigint_constant(*value as i64)?;
    }
    Ok(())
}

/// `subtoken_cache_clear() → BIGINT` — vectors dropped. Registered VOLATILE,
/// like `subtoken_cache_stats()`, so every call clears.
fn cache_clear(call: &mut Call) -> Result<(), String> {
    let dropped = subtoken_core::clear_cache() as i64;
    let rows = vec![dropped; call.rows()];
    call.write_bigints(&rows)
}

/// SQL type text for a STRUCT with these fields, in this order. Field names
/// are quoted, so `key` reads as a name rather than a keyword.
fn struct_type<'a>(fields: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let fields: Vec<String> = fields
        .into_iter()
        .map(|(name, ty)| format!("\"{name}\" {ty}"))
        .collect();
    format!("STRUCT({})", fields.join(", "))
}

/// What `duckdb_functions()` reports for each function: its description, one
/// example and one category. This table is the one place the text is written;
/// registration reads it, and `scripts/check_documented_surface.py` compares the
/// loaded catalog with it and runs every example.
/// Each entry is plain literals, so that script can parse it.
pub(crate) const DOCS: [(&str, Docs); 7] = [
    (
        "subtoken_embed",
        Docs {
            description: "Returns a unit-length FLOAT[] vector for text from the model bundled in the extension, with no network call; text with no in-vocabulary tokens (including '') gives the all-zero vector, and NULL gives NULL.",
            example: "subtoken_embed('a quiet street near the river')",
            category: "embedding",
        },
    ),
    (
        "subtoken_is_truncated",
        Docs {
            description: "Returns true when subtoken_embed(text) had to drop part of text to fit the model's input limits, so its vector does not reflect all of it.",
            example: "subtoken_is_truncated(repeat('word ', 2000))",
            category: "embedding",
        },
    ),
    (
        "subtoken_version",
        Docs {
            description: "Returns one line naming this build, the bundled model, its revision and the vector width.",
            example: "subtoken_version()",
            category: "metadata",
        },
    ),
    (
        "subtoken_model_id",
        Docs {
            description: "Returns the bundled model's full key; store it beside a vector column so a later session can refuse to compare vectors from a different model.",
            example: "subtoken_model_id()",
            category: "metadata",
        },
    ),
    (
        "subtoken_models",
        Docs {
            description: "Returns the catalogue row for the bundled model: name, backend, revision, width, input limit, licence, support tier and key.",
            example: "subtoken_models().width",
            category: "metadata",
        },
    ),
    (
        "subtoken_cache_stats",
        Docs {
            description: "Returns the embedding cache's counters since the last subtoken_cache_clear(): hits, misses, encoder runs, lookups too full to store, entries and capacity.",
            example: "subtoken_cache_stats()",
            category: "cache",
        },
    ),
    (
        "subtoken_cache_clear",
        Docs {
            description: "Drops every cached vector, resets the cache counters and returns how many vectors were dropped.",
            example: "subtoken_cache_clear()",
            category: "cache",
        },
    ),
];

/// Registers the whole surface.
fn register_all(r: &Registrar) -> Result<(), String> {
    ScalarFunction::new("subtoken_embed")
        .parameter("text", "VARCHAR")
        .returns("FLOAT[]")
        .docs(&DOCS[0].1)
        .exec(embed)
        .register(r)?;
    ScalarFunction::new("subtoken_is_truncated")
        .parameter("text", "VARCHAR")
        .returns("BOOLEAN")
        .docs(&DOCS[1].1)
        .exec(is_truncated)
        .register(r)?;
    ScalarFunction::new("subtoken_version")
        .returns("VARCHAR")
        .docs(&DOCS[2].1)
        .exec(version)
        .register(r)?;
    ScalarFunction::new("subtoken_model_id")
        .returns("VARCHAR")
        .docs(&DOCS[3].1)
        .exec(model_id)
        .register(r)?;
    ScalarFunction::new("subtoken_models")
        .returns(struct_type(MODEL_FIELDS))
        .docs(&DOCS[4].1)
        .exec(models)
        .register(r)?;
    ScalarFunction::new("subtoken_cache_stats")
        .returns(struct_type(STATS_FIELDS.map(|name| (name, "BIGINT"))))
        .volatile()
        .docs(&DOCS[5].1)
        .exec(cache_stats)
        .register(r)?;
    ScalarFunction::new("subtoken_cache_clear")
        .returns("BIGINT")
        .volatile()
        .docs(&DOCS[6].1)
        .exec(cache_clear)
        .register(r)?;
    Ok(())
}

/// The symbol DuckDB calls when loading the extension.
///
/// If `get_api` returns NULL the load has already failed and DuckDB knows why,
/// so this returns without touching the error slot. Any other failure, or a
/// panic, is written into the slot, which DuckDB owns.
///
/// # Safety
///
/// Called by DuckDB at load with a live `duckdb_v2_extension_input`.
#[no_mangle]
pub unsafe extern "C" fn subtoken_init_c_api_v2(input: *mut duckdb_v2_extension_input) {
    let result = guard(AssertUnwindSafe(|| {
        // SAFETY: DuckDB passes a live input to the entrypoint.
        if !unsafe { subtoken_duckdb_sys::init(input) } {
            return Ok(());
        }
        // SAFETY: `input` is live, `init` succeeded on it, and the registrar
        // is dropped at the end of this closure, before the entrypoint returns.
        let registrar = unsafe { Registrar::from_input(input) }?;
        register_all(&registrar)
    }));
    if let Err(message) = result {
        // SAFETY: `input` is live and its `err` is the slot DuckDB gave this
        // load. `report` calls `api()`, which panics if `init` never stored a
        // table, so it is guarded; then there is nothing left to report with.
        let _ = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            capi::report((*input).err, &message)
        }));
    }
}
