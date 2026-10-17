# subtoken's functions describe themselves in duckdb_functions() on DuckDB 2.0

Written 2026-10-17 against DuckDB's `v2.0-cyanoptera` branch at `f8f3f4dcd95` (2026-10-16), ahead of DuckDB 2.0.0 (due 2026-10-21). No build was run while writing it. Where this document cites DuckDB source, the path is on that commit.

## Intent

An agent connected to a DuckDB 2.0 database with subtoken loaded should be able to learn the whole extension from `duckdb_functions()` alone. For each of the seven functions that means the real parameter names, a one-sentence description, a runnable example and a category. The community registry's DuckDB 2.0 build should carry the same. subtoken promised this publicly on its GitHub issue 11 on 2026-10-12. finetype made the same promise on its issue 141.

Today none of that is possible. DuckDB 1.5's C API has no setter for a description, an example or a category. `COMMENT ON` is refused for functions in the system catalog. On DuckDB 2.0, a function registered through the v1 C API gets the synthetic parameter names `col0`, `col1` and so on (`src/include/duckdb/main/capi/capi_function_signature.hpp:28-33`). DuckDB 2.0's v2 C API adds all four fields: `duckdb_v2_function_signature_add_parameter` (`api_spec/v2/function/signature.yaml:48-89`) and the setters in `function/docs.yaml:22-98`. All of them are marked stable for v2.0.0.

subtoken goes first because it registers only scalar functions, which makes it the smallest real test of the v2 registration path. finetype follows on the same pattern. The section *What finetype takes from this* below says which parts transfer to finetype and which parts it has to prove itself.

## Goals

1. On DuckDB 2.0, each of the seven functions registers through the v2 C API with:
   - its parameter names (`text` for `subtoken_embed` and `subtoken_is_truncated`; none for the other five);
   - a one-sentence description;
   - one example, written as a bare expression that runs;
   - one category.
   Behaviour does not change, and every existing test in `test/sql/` stays green on a 2.0 CLI.
2. A test fails on a 2.0 build when any of the seven lacks any of the four fields, or still shows a `colN` parameter. A second check runs every example the catalog reports and fails if one of them errors.
3. The registry entry is updated so the registry's DuckDB 2.0 build is the v2 build. A note for subtoken issue 11 is drafted for Hugh to post once that build ships.

## Non-goals

- A build that serves DuckDB 1.5 from `main`. A single binary cannot do both: the v2 extension header refuses to compile alongside the v1 header (`api_spec/v2/extension/duckdb_extension_v2.h.in:18-22`), and DuckDB 1.5.6 refuses any extension whose C API major version is not 1 (v1.5.6 `src/main/extension.cpp:66-71`). DuckDB 1.5 users keep the v1 build the registry already serves (see *Calls made on Hugh's behalf*).
- A C++ registration layer on 1.5.x. The commitment on issue 11 is for 2.0.
- New functions, renamed functions or changed behaviour. The names already carry the extension's name.
- A shared bindings crate published for subtoken and finetype together. Each repo carries its own copy until a third user exists.
- Moving to `duckdb-neo` or a future duckdb-rs v2 extension API. Neither exists in usable form today (see *Bindings route*). When duckdb-rs ships one, moving to it is a later, separate change.
- Translations of the descriptions, example titles (DuckDB stores them and shows nothing), or more than one example per function.

## Anti-goals

These would make the work a failure even if every field showed up in `duckdb_functions()`:

- Descriptions or examples that disagree with what the function does. An example that errors, or a description that promises something the model does not deliver, misleads exactly the agent this work is for. The module doc's position that there is no nearest-neighbour surface holds for the descriptions too.
- A wider network or dependency surface. The no-network property is checked on the binary (`scripts/check_no_network_deps.py`), and this change should shrink the dependency tree.
- Bindings that silently drift from the DuckDB that loads them. Vendored headers that do not match the v2.0.0 release are a bug, and the build-start check has to catch them.

## Constraints

- DuckDB 2.0.0 is not released on 2026-10-17. Hugh ruled on 2026-10-15 that this work follows the release rather than racing it. The build starts once DuckDB v2.0.0 is tagged and its CLI binaries are published. The plan's *Before the build starts* section lists what to check.
- One cargo build at a time on Hugh's Mac, and no parallel implementers.
- The repo is public. Specs, plans, commits and pull requests use plain words and no private planning ids. `scripts/check-public-hygiene.sh` enforces this.
- Every existing gate in `make check` stays in force and is updated rather than switched off. That includes the mutation table in `scripts/mutation_check.py`, whose anchors point into the code being rewritten.

## Bindings route

**subtoken carries its own v2 bindings: DuckDB 2.0.0's two v2 headers vendored into the repo, Rust bindings generated from them once and committed, and a small hand-written safe layer over the function-pointer table.** The cdylib then depends on neither `duckdb` nor `libduckdb-sys`.

The issue framed the choice as waiting for a libduckdb-sys built against the 2.0 header, or generating bindings now. On 2026-10-17 that framing turns out to be incomplete, because a 2.0-header libduckdb-sys would not be enough even when it ships:

- **libduckdb-sys.** On duckdb-rs `main` (`4014db76fd0`, version 1.20000.0, not on crates.io), the `capi-v2` feature generates free `extern "C"` functions from `duckdb_v2.h`, for programs that link libduckdb. It does not generate the `duckdb_ext_api_v2` function-pointer table a loadable extension calls through. Its `loadable-extension` feature still means `capi-v1`, and its generator (`buildtime_bindgen.rs:23-100`) handles only the v1 table. The bindings it pins (`v2.0.0-alpha43678`) also predate the docs setters, which DuckDB added on 2026-10-09 (`8c5a63f56fe`). The newest published libduckdb-sys is 1.10506.1 (2026-10-16), built for 1.5.6.
- **duckdb-neo** (`crates/duckdb-neo` in duckdb-rs, MIT) wraps the same free functions. It is not published. Its README says "⚠ DuckDB's v2 C API is in rapid development. Consider `duckdb-neo` unstable." Its `## Extensions` section reads "TODO". It has no entrypoint or table initialisation, and no docs setters. A cdylib built on it would need DuckDB symbols at link time, and a loading DuckDB process does not export them.
- **Our own bindings** need what plume (glennib/plume, the one Rust v2 extension found in the wild) shows works: vendored headers, bindgen output limited to the types, constants and `duckdb_ext_api_v2`, a `OnceLock` holding the table that `get_api` returns, and a `#[no_mangle]` entrypoint. plume has no licence file, so this repo follows its pattern and copies none of its code.

Waiting has no end date: nothing on duckdb-rs's branches or open pull requests adds a v2 loadable-extension API. The issue-11 commitment is for 2.0. So the route is our own bindings.

How the bindings are kept honest:

- Both headers, `src/include/duckdb_v2.h` and `src/include/duckdb_extension_v2.h`, are vendored byte for byte from DuckDB's `v2.0.0` tag under `crates/subtoken-duckdb-sys/vendor/`, with a `SOURCE.txt` naming the tag and commit.
- `scripts/regenerate_v2_bindings.sh` runs a pinned `bindgen` CLI over them and writes `crates/subtoken-duckdb-sys/src/bindings.rs`. The output is committed, so no build needs libclang. `--check` regenerates into a temporary file and fails on any difference. A CI job runs `--check` on Linux, so a hand edit to the bindings, or a header bump without regeneration, goes red.
- bindgen is a tool run by the script, not a Cargo dependency, so it adds nothing to the dependency tree. bindgen-cli is BSD-3-Clause, and the script installs it with `cargo install --locked` at a pinned version.
- The table is requested with `get_api(extension, "v2.0.0")`. The table layout for that version is fixed by DuckDB, so the binary loads into 2.0.0 and every later 2.x that still serves the v2.0.0 table. That is the same promise the v1.2.0 floor makes today.

## Design

### Crates

- **`crates/subtoken-duckdb-sys`** (new, `publish = false`, MIT): the vendored headers, the committed `bindings.rs`, `api()`, which returns the stored `&'static duckdb_ext_api_v2`, and `init(input)`, which calls `get_api` once and stores the table. Nothing else.
- **`crates/subtoken-duckdb`**: the cdylib, as now, with two parts:
  - `src/capi.rs` (new): the safe layer. It holds the entrypoint glue, error reporting, a scalar-function builder, and the vector helpers listed below. Each `unsafe` block carries a `SAFETY` comment, the way `lib.rs` does today.
  - `src/lib.rs`: the seven functions, rewritten onto `capi.rs`, plus the docs table. Their logic, which is mostly calls into `subtoken-core`, does not change.
- **`subtoken-core`** is untouched.

### Loading

- The entrypoint symbol becomes `subtoken_init_c_api_v2(input: *mut duckdb_v2_extension_input)`, returning nothing (`api_spec/v2/extension/extension.yaml:5-62`). It calls `subtoken_duckdb_sys::init`.
  - If `get_api` returns NULL, the entrypoint returns without touching `err`, because DuckDB already knows the version did not match.
  - Otherwise it registers the seven functions through `input.extension`, using a factory taken from `input.context` (`duckdb_v2_context_get_factory`).
  - Any failure sets `input.err`'s code and text and returns. The error handle is borrowed and is never destroyed.
- The whole body runs inside `std::panic::catch_unwind`, as does every exec callback. A panic is reported as an error, because unwinding across `extern "C"` aborts the process.
- The metadata footer keeps the ABI type `C_STRUCT`. Its DuckDB-version field changes from `v1.2.0` to `v2.0.0`, which is how DuckDB 2.0 tells a v2 extension from a v1 one: major version 2 (`src/main/extension/extension_load.cpp:27-37, 369-382, 830-846`). `C_STRUCT_UNSTABLE` is not used, because it pins a binary to one exact DuckDB version.
  - `MIN_DUCKDB_VERSION` in `lib.rs` and `TARGET_DUCKDB_VERSION` in the Makefile both become `v2.0.0`.
  - `scripts/check_artifact_version.py` keeps checking that the two agree and that the ABI is `C_STRUCT`.

### Registration

`capi.rs` exposes one builder, roughly:

```rust
ScalarFunction::new("subtoken_embed")
    .parameter("text", Type::Varchar)
    .returns(Type::parse("FLOAT[]"))
    .docs(&DOCS.embed)
    .volatile(false)
    .exec(embed_exec)
    .register(&ext, &factory)?;
```

It runs the v2 call sequence:

1. `scalar_function_create`
2. `set_name`
3. `get_signature`
4. `add_parameter`, with kind `STANDARD`, so `subtoken_embed(text := 'x')` also works
5. `set_return_type`
6. `get_docs`, then `set_description`, `add_example` (no title) and `add_category`
7. `set_property(STABILITY, VOLATILE)`, for `subtoken_cache_stats` and `subtoken_cache_clear` only, matching today's `volatile()`
8. `set_exec_callback`
9. `extension_register_scalar_function`
10. `scalar_function_destroy`, which is required after registration too

Return types come from `factory_create_type_from_text`. The text is `VARCHAR`, `BOOLEAN`, `BIGINT`, `FLOAT[]`, or the two STRUCTs spelled with field names and types. That puts the STRUCT field order in one string per function, built from the existing `MODEL_FIELDS` and `STATS_FIELDS` constants so it is still written once. Null handling and fallibility stay at their defaults (`NULL_HANDLING_DEFAULT`, `FALLIBLE`).

### Vectors

On 2.0 the executor skips the callback only for a *constant* NULL argument. A flat vector with some NULL rows still reaches the callback (`src/execution/expression_executor/execute_function.cpp:234-241`), so the existing NULL-in, NULL-out handling stays.

The helpers in `capi.rs` cover what the seven functions use:

- **Read a VARCHAR column with validity.** `vector_get_view` gives data, validity and a selection vector. A row's physical index is `sel[i]`, or `i` when there is no selection vector. The string is the 16-byte `duckdb_v2_bytes`: up to 12 bytes are stored inline, and longer strings sit behind a pointer. The helper handles CONSTANT and DICTIONARY inputs through the selection vector rather than flattening. `vector_get_view` refuses `VECTOR_TYPE_OTHER` (FSST, SEQUENCE and shredded vectors, `api_spec/v2/vector/vector.yaml:238`), so for those the helper calls `vector_flatten` first. An FSST-compressed VARCHAR column read from a persistent table is the likely way to get one, and a SQL test reads `subtoken_embed` over such a column (my call, added after the builder check).
- **Write a flat BOOLEAN or BIGINT column with NULLs**: `vector_get_data_mutable`, then `vector_set_null` per NULL row.
- **Write a LIST of FLOAT.**
  1. Size the child with `vector_set_size(child, total)`. This reserves and sets the size in one call, and fetches the child's data pointer only afterwards, because reserving can move the data.
  2. Write the floats.
  3. Write each row's `{offset, length}` entry, keeping today's rule that a NULL row still gets a well-formed `(offset, 0)` entry.
- **Write STRUCT children**: `vector_get_child(i)`, then the flat writers above.
- **Write strings.** Up to 12 bytes go inline. Longer strings are copied into `vector_get_arena` / `arena_allocate`, and the 4-byte prefix is filled in. `CString` goes away, because v2 strings are length-delimited.
- **Report an error from a callback**: `error_info_set_code(INPUT_INVALID)` plus `error_info_set_text`. The engine raises the error after the callback returns (`capi_v2_func_scalar.cpp:223-231`).

The string layout and the list-offset arithmetic are pure functions with Rust unit tests in `capi.rs`. Everything else is tested by loading the artifact into a real DuckDB, as today.

### The docs

The text lives in one `const` table in `lib.rs`, next to the functions. Registration and the checks both read it from there; nothing else restates it. Starting text, which the builder may tighten without changing its claims:

| function | parameters | description | example | category |
|---|---|---|---|---|
| `subtoken_embed` | `text` | Returns a unit-length FLOAT[] vector for text from the model bundled in the extension, with no network call; NULL gives NULL. | `subtoken_embed('a quiet street near the river')` | `embedding` |
| `subtoken_is_truncated` | `text` | Returns true when subtoken_embed(text) had to drop part of text to fit the model's input limits, so its vector does not reflect all of it. | `subtoken_is_truncated(repeat('word ', 2000))` | `embedding` |
| `subtoken_version` | — | Returns one line naming this build, the bundled model, its revision and the vector width. | `subtoken_version()` | `metadata` |
| `subtoken_model_id` | — | Returns the bundled model's full key; store it beside a vector column so a later session can refuse to compare vectors from a different model. | `subtoken_model_id()` | `metadata` |
| `subtoken_models` | — | Returns the catalogue row for the bundled model: name, backend, revision, width, input limit, licence, support tier and key. | `subtoken_models().width` | `metadata` |
| `subtoken_cache_stats` | — | Returns the embedding cache's counters since the last subtoken_cache_clear(): hits, misses, encoder runs, lookups too full to store, entries and capacity. | `subtoken_cache_stats()` | `cache` |
| `subtoken_cache_clear` | — | Drops every cached vector, resets the cache counters and returns how many vectors were dropped. | `subtoken_cache_clear()` | `cache` |

Examples are bare expressions, which is the form DuckDB's own functions use (for example `count(A)`, and `f(1)` in `test/api/capi/v2/test_capi_v2_scalar_function.cpp:1156`). DuckDB shows the string exactly as it is passed. Categories are free text. DuckDB's built-in categories are short lowercase words (`string`, `list`, `text_similarity`), and these follow that form.

### Tests

- **`test/sql/12_each_function_describes_itself.sql`** (new). It reads `duckdb_functions()` for the seven functions and asserts, per function, that:
  - `description` is non-NULL and non-empty;
  - `len(examples) >= 1`;
  - `len(categories) >= 1`;
  - `parameters` equals `['text']` for the two one-argument functions and `[]` for the other five;
  - no parameter matches `^col[0-9]+$`.
  Each assertion is labelled with the function's name, so a failure says which function and which field.
- **The examples run.** `scripts/check_documented_surface.py` already loads the artifact and reads `duckdb_functions()`. It gains a pass that runs `SELECT <example>` for every example the catalog reports, in one session, and fails naming any example that errors. It also fails when the catalog's description, example or category differs from the docs table in `lib.rs`, so the table stays the single source. Its `--self-test` gains cases for both failures.
- **New mutations** in `scripts/mutation_check.py`:
  - drop the description setter for one function;
  - register `subtoken_embed`'s parameter without its name;
  - make one example invalid;
  - drop one category.
  Each must redden the named check.
- **Existing mutations** that anchor into the rewritten glue are re-anchored. Each keeps its intent, for example `a_null_row_is_written_without_its_validity_bit` and `the_list_offset_is_carried_across_chunks`. `--check-anchors` in CI keeps them from going stale.
- All of `test/sql/01`–`11` must pass unchanged on a 2.0 CLI. A test that has to change in order to pass is a behaviour change, and has to be argued in the pull request.

### Toolchain and docs

- **CI:** `DUCKDB_VERSION` moves to `v2.0.0` in `ci.yml` and `MainDistributionPipeline.yml`. The `extension-ci-tools` submodule moves from `v1.5-variegata` to DuckDB's 2.0 branch: `v2.0.0` if it exists when the build starts, otherwise `v2.0-cyanoptera`. Its C API makefiles and metadata script are identical across those branches today.
  - DuckDB 2.0 loads today's v1 build (it accepts C API major 1 or 2, `src/main/extension.cpp:65-86`). So the toolchain can move first, on unchanged code, as its own commit, and every later commit is tested on 2.0.
- **Agent output mode:** the 2.0 CLI switches to it when an agent's variable is set and no output-mode flag is given. Every CLI call in `scripts/` and the Makefile already passes `-csv`, `-list` or `-box`, so none is affected. The builder confirms this on the 2.0 CLI rather than trusting this reading.
- **`check_no_network_deps.py`** loses its note about libduckdb-sys's build-dependencies, which no longer exist.
- **README.md and `description.yml`** say the extension needs DuckDB 2.0 or later, that DuckDB 1.5 users get the earlier build, and that `duckdb_functions()` now describes each function.
- **The `lib.rs` module doc** gains a short section on the v2 registration.
- **Version:** the workspace version moves from 0.1.0 to 0.2.0, so the artifact's footer, `subtoken_version()` and the registry page can tell the v2 build from the v1 build. This is a version string, not a git tag. Where subtoken's first tag goes is still Hugh's call, and nothing here needs a tag.

### The registry

The registry serves a pinned commit, so the entry changes after the pull request merges:

- `repo.ref` and `repo.ref_next` move to the merge commit. The examples checker requires `ref_next`.
- If the registry's `scripts/build.py` still reads a `variegata` key at that point, the entry adds `variegata: 3bf95680e23ba6ad23951023bd65432f53877485`. That keeps any 1.5.6 rebuild on the last v1 commit.

Since 2026-10-15 the registry's `build.yml` builds `ref` against a 2.0 pre-release with the 2.0 CI tools, and its next-version leg (`build_next.yml`) is disabled. So on that day a change to the entry builds `ref` for 2.0 straight away. The builder rereads `build.py` and the workflows at the time.

The registry pull request goes to duckdb's repository, and the note for issue 11 is public text. Under the repo's rules, both are drafted for Hugh, who opens and posts them. They are not sent by the agent.

## Calls made on Hugh's behalf

1. **Our own committed bindings**, not waiting for duckdb-rs or using duckdb-neo. The reasons are under *Bindings route*.
2. **`main` becomes 2.0-only.** DuckDB 1.5 users keep the registry's existing v1 build, pinned to `3bf9568`, through `variegata` if the registry still honours it. The alternative is a 1.5 maintenance branch. It would cost a second toolchain in CI for a version that reaches end of life on 2027-03-01, and the code has had only comment changes since that commit.
3. **Version 0.2.0**, with no tag.
4. **The docs text and the categories** in the table above.
5. **Example-running lives in `check_documented_surface.py`**, not a new script, because that script already loads the artifact and reads the catalog.
6. **Bindings are regenerated by a script and checked in CI**, not generated in a `build.rs`, so no build needs libclang.

## What finetype takes from this

finetype has to keep the same public promise (its issue 141), and it follows this pattern when it moves to DuckDB 2.0.

**Transfers as it is:**

- **The bindings route.** Vendored v2.0.0 headers, bindings generated by a script and committed, `--check` in CI, and no `duckdb`/`libduckdb-sys` in the extension crate. finetype copies `subtoken-duckdb-sys` and the regeneration script. Its other crates that link DuckDB as a library through duckdb-rs, `finetype-eval` and `finetype-train`, are separate binaries and are not affected.
- **Loading.** The entrypoint, which for finetype is `<extension name>_init_c_api_v2` under its new name; the `get_api(…, "v2.0.0")` handshake; `catch_unwind` around the entrypoint and every callback; and a footer of `C_STRUCT` with `v2.0.0`.
- **Scalar registration.** The builder, the parameter and docs calls, the bare-expression example form, and categories as short lowercase words. This covers `ft_version`, `ft_infer`, `ft_validate_text`, `ft_cast` and `ft_unpack`, under whatever names the rename gives them.
- **The vector helpers** for VARCHAR in, BOOLEAN or BIGINT out, LIST and STRUCT out, and string writes through the arena.
- **The two checks.** A SQL test asserting the four fields and no `colN` for every registered function, and an example-running pass that holds the catalog to the docs table.
- **The 1.5 split.** `main` goes 2.0-only, and the last v1 commit stays in the registry for 1.5 through `variegata`.

**Has to prove itself, because subtoken does not exercise it:**

- **Aggregate functions** (`ft_profile`, in `profile_agg.rs`, which uses the raw v1 aggregate calls). v2 has `function/aggregate.yaml`, `duckdb_v2_aggregate_function_get_docs` and `duckdb_v2_extension_register_aggregate_function`, but subtoken registers none of them. This is also where finetype's ORDER BY limitation is tested on 2.0.
- **Table functions** (`finetype_spike`): `function/table.yaml` and `table_function_get_docs`.
- **Any vector type finetype reads or writes** beyond VARCHAR, BOOLEAN, BIGINT, FLOAT[] and the two STRUCTs.
- **Bind or local state.** subtoken's functions keep none, so `set_user_data` and the bind and init callbacks are untested here.

## Left to the builder

- The exact shape of `capi.rs`'s API. It must be small, and it must not grow helpers that the seven functions do not use.
- The pinned bindgen version and its allowlist flags.
- The wording of the docs text, within the claims above.
- Whether `subtoken-duckdb-sys` needs a `build.rs` at all. The aim is that it does not.
