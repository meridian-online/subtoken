//! Static text embeddings from a model bundled at build time.
//!
//! This crate is the engine: it owns the embedded model, the embedding call and
//! the cache. It knows nothing about DuckDB, which is what lets every behaviour
//! below be tested with plain `cargo test`.
//!
//! # The contract, in one place
//!
//! * [`embed`] returns a vector of [`dim`] `f32` values for any `&str`.
//! * Text with no in-vocabulary tokens — the empty string, whitespace, a string
//!   of symbols the vocabulary does not carry — returns a **zero vector of full
//!   width**, not an absent value and not an error. SQL `NULL` is not text and
//!   never reaches this crate; the DuckDB layer propagates it.
//! * Repeating a call for the same text under the same model does not re-embed,
//!   **for as many distinct values as the cache holds**. [`Stats::encoded`]
//!   counts completed encoder runs and is the observable that says so;
//!   [`Stats::uncached`] is how many lookups the cache was too full to serve.
//!   [`cache`] documents why it fills and stops rather than evicting.
//! * Two threads asking for the same text at the same time produce one encode,
//!   not two: the second waits for the first. Without that, a scan at DuckDB's
//!   default thread count reported 29 encodes for a ten-distinct-value column.
//! * [`embed`] builds its vector from at most [`model::MAX_TOKENS`] tokens of
//!   `text`, and from a bounded number of its characters before that; anything
//!   past either is discarded before the mean, and the vector that comes back
//!   is full width and unit norm either way — nothing about it says content was
//!   dropped. [`is_truncated`] is how a caller finds out.
//! * Nothing here reads a file or opens a socket after compilation.

pub mod cache;
pub mod model;

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

use cache::{CacheKey, CacheStats, EmbeddingCache};

/// Extension version, shared by every crate in the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Completed encoder runs since the last [`clear_cache`].
static ENCODED: AtomicU64 = AtomicU64::new(0);

static CACHE: OnceLock<Mutex<EmbeddingCache>> = OnceLock::new();

/// Keys some thread is embedding right now, and the signal for when it stops.
static IN_FLIGHT: OnceLock<(Mutex<HashSet<CacheKey>>, Condvar)> = OnceLock::new();

fn cache() -> MutexGuard<'static, EmbeddingCache> {
    let cache = CACHE.get_or_init(|| {
        let dim = model::bundled().map(|model| model.dim()).unwrap_or(1);
        Mutex::new(EmbeddingCache::with_capacity(cache::capacity_for_budget(
            cache::DEFAULT_BUDGET_BYTES,
            dim,
        )))
    });
    // A panic inside the extension would otherwise poison the cache for the rest
    // of the session; the data behind the lock is a cache, so recovering it is
    // always safe.
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn in_flight() -> &'static (Mutex<HashSet<CacheKey>>, Condvar) {
    IN_FLIGHT.get_or_init(|| (Mutex::new(HashSet::new()), Condvar::new()))
}

/// Exclusive right to embed one key, released on drop.
///
/// Held across the encode so that eight DuckDB worker threads meeting the same
/// value produce one encode between them. `Drop` rather than an explicit
/// release, so a panic in the encoder wakes the waiters instead of stranding
/// them.
struct Flight {
    key: CacheKey,
}

impl Flight {
    fn begin(key: CacheKey) -> Self {
        let (keys, done) = in_flight();
        let mut keys = keys.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while keys.contains(&key) {
            keys = done
                .wait(keys)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        keys.insert(key);
        Self { key }
    }
}

impl Drop for Flight {
    fn drop(&mut self) {
        let (keys, done) = in_flight();
        keys.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.key);
        done.notify_all();
    }
}

/// Everything SQL can observe about the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Lookups answered without embedding.
    pub hits: u64,
    /// Lookups that were not in the cache.
    pub misses: u64,
    /// Completed encoder runs.
    pub encoded: u64,
    /// Lookups the cache was too full to store.
    pub uncached: u64,
    /// Vectors currently held.
    pub entries: u64,
    /// The most vectors this cache will hold.
    pub capacity: u64,
}

/// Width of every vector this build returns.
pub fn dim() -> Result<usize, String> {
    Ok(model::bundled()?.dim())
}

/// The bundled model's content address, as 64 lowercase hex characters.
///
/// This is the same value [`describe`] truncates to twelve characters for its
/// one-line sentence, at full length and on its own. The truncation is fine for
/// a human reading a version string and useless for a column: twelve characters
/// inside a sentence have to be recovered with a regular expression before
/// anything can join on them, and a session that wants to know whether two
/// columns of vectors are comparable is doing exactly that — joining.
///
/// It is derived from the asset bytes rather than from a version number, so a
/// build whose weights, tokenizer or config differ reports a different id even
/// if nobody remembered to bump anything, and a build that only changes this
/// extension's own code reports the same one.
pub fn model_id() -> Result<String, String> {
    Ok(model::bundled()?.key_hex())
}

/// One row of the model catalogue.
///
/// Everything a session needs in order to decide whether a stored column of
/// vectors can be compared with a fresh one, and what the model that wrote them
/// will accept — in fields, so SQL can read them, rather than in a sentence.
pub struct ModelRow {
    /// The upstream repository the assets came from.
    pub model: &'static str,
    /// The family of model, as [`model::MODEL_BACKEND`] documents it.
    pub backend: &'static str,
    /// The full pinned revision of `model`, not an abbreviation of it.
    pub revision: &'static str,
    /// Floats in every vector this model returns — the width of the column a
    /// caller is about to create.
    pub width: u64,
    /// The most tokens of a text that reach the mean. Past this,
    /// [`is_truncated`] answers true and the vector is built from a prefix.
    pub input_limit: u64,
    /// The licence the upstream release declares.
    pub licence: &'static str,
    /// What a caller may expect of this model in this build.
    pub tier: &'static str,
    /// The same string [`model_id`] returns, so a stored id can be joined
    /// against the catalogue rather than compared to a function call.
    pub key: String,
}

/// The catalogue row for the model this build serves.
///
/// Each field is read from the loaded model or from the constant that governs
/// it, never restated: `width` is the width the encoder actually produces and
/// `key` is the loaded model's own key, so a row that has drifted from the
/// binary is a row that cannot be built.
pub fn catalogue() -> Result<ModelRow, String> {
    let model = model::bundled()?;
    Ok(ModelRow {
        model: model::MODEL_ID,
        backend: model::MODEL_BACKEND,
        revision: model::MODEL_REVISION,
        width: model.dim() as u64,
        input_limit: model::MAX_TOKENS as u64,
        licence: model::MODEL_LICENCE,
        tier: model::MODEL_TIER,
        key: model.key_hex(),
    })
}

/// Whether [`embed`] discarded content of `text`: it pooled fewer ids than the
/// whole of `text` would have given it, so the vector it returns does not
/// reflect all of `text`.
///
/// Not cached, and not always cheaper than an encode. It tokenises without
/// pooling, which is the cheaper half of an encode — but for text past the
/// character cut it has to tokenise the whole string, where [`embed`] stops
/// reading at the cut. That is the price of answering "did anything get lost"
/// rather than "did a cut fire": a text can be far past the cut and have lost
/// nothing, and only reading past it can tell the two apart.
pub fn is_truncated(text: &str) -> Result<bool, String> {
    Ok(model::bundled()?.is_truncated(text))
}

fn encode(model: &model::Model, text: &str) -> Result<Arc<[f32]>, String> {
    let vector: Arc<[f32]> = Arc::from(model.embed(text)?.into_boxed_slice());
    // After the encode, so the count is of runs that finished.
    ENCODED.fetch_add(1, Ordering::Relaxed);
    Ok(vector)
}

/// Embed one string, reusing an earlier result when the same text has been seen
/// under the same model.
///
/// See the crate docs for what comes back for text with no tokens, and
/// [`cache`] for what happens once the cache is full.
pub fn embed(text: &str) -> Result<Arc<[f32]>, String> {
    let model = model::bundled()?;
    let key = cache::cache_key(model.key(), text);

    {
        let mut cache = cache();
        if let Some(vector) = cache.get(&key) {
            return Ok(vector);
        }
        if cache.is_full() {
            // Nothing to coordinate: the result cannot be stored, so every
            // caller would re-encode anyway and making them queue behind each
            // other would serialise the scan for no benefit.
            cache.note_uncached();
            drop(cache);
            return encode(model, text);
        }
    }

    let _flight = Flight::begin(key);

    // Look again, unconditionally. Waiting for the flight is not the only way
    // to arrive here with a stale miss: a thread that missed while another was
    // already encoding the same key, and reached `Flight::begin` after that
    // thread released it, never waits at all. Rechecking only after a wait left
    // 454 encodes for 400 distinct values across eight threads.
    //
    // The two do different jobs and both are needed. This recheck is what makes
    // the count right as DuckDB drives it — ten values over 400,000 rows at
    // eight threads give ten encodes with the recheck and no flight. The flight
    // is what makes it right under contention: the same ten values in a tight
    // eight-thread loop give 49 to 65 encodes without it.
    if let Some(vector) = cache().recheck(&key) {
        return Ok(vector);
    }

    let vector = encode(model, text)?;
    cache().insert(key, Arc::clone(&vector));
    Ok(vector)
}

/// Embed one string without consulting or filling the cache.
pub fn embed_uncached(text: &str) -> Result<Vec<f32>, String> {
    let vector = model::bundled()?.embed(text)?;
    ENCODED.fetch_add(1, Ordering::Relaxed);
    Ok(vector)
}

/// Current cache counters.
pub fn stats() -> Stats {
    let CacheStats {
        hits,
        misses,
        uncached,
        entries,
        capacity,
    } = cache().stats();
    Stats {
        hits,
        misses,
        encoded: ENCODED.load(Ordering::Relaxed),
        uncached,
        entries,
        capacity,
    }
}

/// Drop every cached vector and reset the counters, returning the number of
/// vectors dropped.
pub fn clear_cache() -> u64 {
    ENCODED.store(0, Ordering::Relaxed);
    cache().clear()
}

/// One line naming this build: the extension version, the bundled model, and the
/// width of the vectors it returns.
pub fn describe() -> String {
    match model::bundled() {
        Ok(model) => format!(
            "subtoken {} (model {}@{}, key {}, dim {})",
            VERSION,
            model::MODEL_ID,
            &model::MODEL_REVISION[..12],
            &model.key_hex()[..12],
            model.dim()
        ),
        Err(message) => format!("subtoken {VERSION} (model unavailable: {message})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache and the counters are process-global, so the tests that read
    /// them run one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// AC4, at the engine: asking for the same text again does not re-embed.
    ///
    /// `encoded` counts encoder invocations, not cache bookkeeping, so this
    /// fails if the cache is consulted but not honoured.
    #[test]
    fn repeating_a_value_does_not_re_embed_it() {
        let _guard = serial();
        clear_cache();

        let column = ["manufacturing", "logistics", "manufacturing", "logistics"];
        for value in column {
            embed(value).expect("embed");
        }
        let first_pass = stats();
        assert_eq!(first_pass.encoded, 2, "two distinct values, two encodes");

        for value in column {
            embed(value).expect("embed");
        }
        let second_pass = stats();
        assert_eq!(
            second_pass.encoded, first_pass.encoded,
            "a repeated query must not re-embed"
        );
        assert!(second_pass.hits > first_pass.hits);
    }

    /// A cache HIT returns the vector the encoder produced.
    ///
    /// The second `embed` is the point. The first is a miss and returns the
    /// freshly computed vector whatever was written to the cache, so a test that
    /// only called it once would pass against a cache that stored rubbish.
    #[test]
    fn a_vector_read_back_from_the_cache_equals_the_uncached_one() {
        let _guard = serial();
        clear_cache();
        let text = "a supplier of precision bearings";

        let on_miss = embed(text).expect("embed");
        let encodes_after_miss = stats().encoded;

        let on_hit = embed(text).expect("embed");
        assert_eq!(
            stats().encoded,
            encodes_after_miss,
            "the second call must be a cache hit, or this proves nothing"
        );

        let direct = embed_uncached(text).expect("embed");
        assert_eq!(on_hit.as_ref(), direct.as_slice(), "hit vs encoder");
        assert_eq!(on_miss.as_ref(), on_hit.as_ref(), "miss vs hit");
    }

    #[test]
    fn clearing_the_cache_drops_the_entries_and_the_counters() {
        let _guard = serial();
        clear_cache();
        embed("one").expect("embed");
        embed("two").expect("embed");
        assert_eq!(clear_cache(), 2);
        assert_eq!(
            stats(),
            Stats {
                hits: 0,
                misses: 0,
                encoded: 0,
                uncached: 0,
                entries: 0,
                capacity: stats().capacity,
            }
        );
    }

    /// The tokenizer folds case, surrounding whitespace and Unicode composition
    /// away, so these inputs are one value to the model.
    ///
    /// Pinned because it is a property of the bundled tokenizer rather than of
    /// this code: a caller may rely on `embed(name)` matching `embed(NAME)`, and
    /// a tokenizer swap that changed it would change the SQL contract with
    /// nothing else reddening.
    #[test]
    fn case_and_surrounding_whitespace_do_not_change_the_vector() {
        let _guard = serial();
        let pairs = [
            ("steel", "STEEL"),
            ("steel", "  steel  "),
            ("caf\u{e9}", "cafe\u{301}"),
        ];
        for (left, right) in pairs {
            assert_eq!(
                embed(left).expect("embed").as_ref(),
                embed(right).expect("embed").as_ref(),
                "{left:?} and {right:?} should embed alike",
            );
        }
    }

    /// Word order does not change the vector, and repetition does.
    ///
    /// A Model2Vec vector is the mean of its token vectors, so a phrase and its
    /// shuffle land in the same place. That is the model, not a defect, and it
    /// is pinned here because it bounds what the vectors can be asked to do.
    #[test]
    fn the_pool_is_a_mean_so_order_is_lost_and_repetition_is_not() {
        let _guard = serial();
        assert_eq!(
            embed("valve bodies").expect("embed").as_ref(),
            embed("bodies valve").expect("embed").as_ref(),
        );
        assert_ne!(
            embed("steel steel copper").expect("embed").as_ref(),
            embed("steel copper").expect("embed").as_ref(),
        );
    }

    /// Those pairs are nevertheless separate cache entries, because the key is
    /// the exact bytes. This is the cost of not reproducing the tokenizer's
    /// normalisation in the cache layer, counted rather than left implicit.
    #[test]
    fn inputs_that_embed_alike_still_occupy_separate_cache_entries() {
        let _guard = serial();
        clear_cache();
        embed("steel").expect("embed");
        embed("STEEL").expect("embed");
        assert_eq!(stats().entries, 2);
        assert_eq!(stats().encoded, 2);
    }

    #[test]
    fn every_vector_has_the_declared_width() {
        let _guard = serial();
        let width = dim().expect("dim");
        for text in ["", "one", "a longer sentence about steel stockholders"] {
            assert_eq!(
                embed(text).expect("embed").len(),
                width,
                "width for {text:?}"
            );
        }
    }

    #[test]
    fn describe_names_the_bundled_model_and_the_width() {
        let text = describe();
        assert!(text.contains(model::MODEL_ID), "{text}");
        assert!(text.contains(VERSION), "{text}");
        assert!(
            text.contains(&format!("dim {}", dim().expect("dim"))),
            "{text}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // At the scale the product is sold at, rather than at the scale that is
    // convenient to write.
    //
    // Everything above this line uses between two and eight distinct values.
    // The README sells `embed` on "a large text column", and the previous
    // build's cache fell to a ZERO per-cent hit rate above 33,000 distinct
    // values while every test in this file stayed green.
    // ─────────────────────────────────────────────────────────────────────────

    /// Build `count` distinct short strings.
    fn distinct_values(count: usize) -> Vec<String> {
        (0..count).map(|i| format!("row number {i}")).collect()
    }

    /// **AC4 at a scale the product is offered at.**
    ///
    /// Fifty thousand distinct values, embedded twice. The second pass must
    /// re-embed none of them. The count is absolute rather than derived from
    /// `stats().capacity`, which is what makes it pin the default budget: with
    /// the old `DEFAULT_CAPACITY` set to 4, every other test in this repo
    /// stayed green.
    #[test]
    fn a_repeated_query_over_forty_thousand_distinct_values_re_embeds_none_of_them() {
        let _guard = serial();
        clear_cache();
        let column = distinct_values(40_000);

        for value in &column {
            embed(value).expect("embed");
        }
        let first_pass = stats();
        assert_eq!(first_pass.encoded, 40_000);
        assert_eq!(
            first_pass.uncached, 0,
            "50,000 values must fit the default budget"
        );

        for value in &column {
            embed(value).expect("embed");
        }
        let second_pass = stats();
        assert_eq!(
            second_pass.encoded,
            first_pass.encoded,
            "a repeated query over 50,000 distinct values re-embedded {} of them",
            second_pass.encoded - first_pass.encoded
        );
        assert_eq!(second_pass.hits - first_pass.hits, 40_000);
    }

    /// **Past the cache's capacity there is no cliff.**
    ///
    /// A working set larger than the cache is served for a full cache worth on
    /// a rescan, not for none of it. Measured on the previous build, 33,000
    /// distinct values through a 16,384-per-generation cache gave **0** hits on
    /// the second pass; the numbers here are what a rescan gives when the cache
    /// stops admitting instead of evicting.
    ///
    /// The cache is deliberately shrunk for this test rather than the column
    /// grown past the default budget, which would be 61,000 embeddings of setup
    /// to make the same point.
    #[test]
    fn a_rescan_larger_than_the_cache_is_served_for_a_full_cache_worth() {
        let _guard = serial();
        let capacity = 500;
        let distinct = 2_000;
        let column = distinct_values(distinct);
        let key_of = |text: &str| cache::cache_key(model::bundled().expect("model").key(), text);

        // A cache of the same shape as the global one, driven the way `embed`
        // drives it.
        let mut small = EmbeddingCache::with_capacity(capacity);
        let mut encodes = 0usize;
        for value in &column {
            let key = key_of(value);
            if small.get(&key).is_none() {
                encodes += 1;
                small.insert(key, Arc::from(vec![0.0_f32; 4].into_boxed_slice()));
            }
        }
        assert_eq!(encodes, distinct, "the first pass embeds everything");
        let after_first = small.stats();
        assert_eq!(after_first.entries as usize, capacity);

        for value in &column {
            let key = key_of(value);
            if small.get(&key).is_none() {
                encodes += 1;
            }
        }
        let hits_in_rescan = small.stats().hits - after_first.hits;
        assert_eq!(
            hits_in_rescan as usize, capacity,
            "a rescan must hit for the resident set, not for none of it"
        );
        assert_eq!(encodes, distinct + (distinct - capacity));
    }

    /// A column bigger than the cache reports that it did not fit.
    ///
    /// Without this the degradation is invisible: the query is simply slower
    /// the second time and nothing says why.
    #[test]
    fn a_column_larger_than_the_cache_reports_what_it_could_not_hold() {
        let _guard = serial();
        clear_cache();
        let capacity = stats().capacity as usize;
        let column = distinct_values(capacity + 1_000);
        for value in &column {
            embed(value).expect("embed");
        }
        let after = stats();
        assert_eq!(after.entries as usize, capacity);
        assert_eq!(after.uncached, 1_000);
        clear_cache();
    }

    /// **One encode per distinct value, with eight threads on it.**
    ///
    /// Without single-flight this is a race: every thread that misses before
    /// the first insert lands encodes the same value again. Measured on the
    /// previous build through DuckDB at its default thread count, a
    /// ten-distinct-value column reported 29 encodes.
    #[test]
    fn eight_threads_over_ten_values_encode_ten_times() {
        let _guard = serial();
        clear_cache();
        let column: Vec<String> = distinct_values(10);

        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for value in &column {
                        embed(value).expect("embed");
                    }
                });
            }
        });

        let after = stats();
        assert_eq!(
            after.encoded, 10,
            "eight threads over ten distinct values encoded {} times",
            after.encoded
        );
        assert_eq!(after.entries, 10);
        assert_eq!(after.hits + after.misses, 80, "one lookup per row");
    }

    /// The documented cost of being past the bound: a value the cache turned
    /// away is re-embedded on every row that carries it.
    ///
    /// Asserted rather than left as prose, because it is the one condition
    /// under which `encoded` legitimately grows on a repeat and a reader is
    /// entitled to know it is the stated behaviour rather than a regression.
    #[test]
    fn a_value_the_full_cache_turned_away_is_re_embedded_every_time() {
        let _guard = serial();
        clear_cache();
        let capacity = stats().capacity as usize;
        for value in distinct_values(capacity) {
            embed(&value).expect("embed");
        }
        assert_eq!(stats().uncached, 0);
        let encodes_when_full = stats().encoded;

        for _ in 0..5 {
            embed("a value the cache has no room for").expect("embed");
        }
        assert_eq!(stats().encoded - encodes_when_full, 5);
        assert_eq!(stats().uncached, 5);
        clear_cache();
    }

    /// Threads racing on distinct values do not block each other into
    /// incorrectness, and every one of them gets the right vector.
    #[test]
    fn eight_threads_over_distinct_values_each_get_the_right_vector() {
        let _guard = serial();
        clear_cache();
        let column = distinct_values(400);
        let expected: Vec<Vec<f32>> = column
            .iter()
            .map(|value| embed_uncached(value).expect("embed"))
            .collect();
        clear_cache();

        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for (value, want) in column.iter().zip(&expected) {
                        assert_eq!(embed(value).expect("embed").as_ref(), want.as_slice());
                    }
                });
            }
        });

        assert_eq!(stats().encoded, 400);
    }
}
