//! A bounded, content-addressed cache of embeddings.
//!
//! The key is a SHA-256 over the model's content address and the exact input
//! bytes. Both halves are load bearing:
//!
//! * **Model.** Vectors from two models — or two revisions of one model — are
//!   not comparable, so a cached vector must never survive an asset swap. The
//!   model half of the key is [`crate::model::Model::key`], a digest of the
//!   bundled asset bytes themselves, so the invalidation does not depend on
//!   anyone remembering to bump a version string.
//!
//! * **Text.** The exact UTF-8 bytes, with no case folding, trimming or Unicode
//!   normalisation applied first. The only normalisation that would be safe is
//!   the one the tokenizer already applies, and reproducing a dependency's
//!   normalisation here would mean a tokenizer bump silently changing which
//!   inputs share an entry. Keying on the exact bytes is correct whatever the
//!   tokenizer does; what it costs is a duplicate entry for inputs that
//!   normalise together, which
//!   `inputs_that_embed_alike_still_occupy_separate_cache_entries` in the crate
//!   root counts.
//!
//! # Why the cache fills and then stops, rather than evicting
//!
//! The workload this exists for is a scan: `SELECT embed(description) FROM t`,
//! then the same query again. Under any recency-ordered eviction — LRU, FIFO,
//! or the two generations this used to keep — a repeated scan of a working set
//! larger than the cache evicts every entry exactly before the scan returns to
//! it, and the hit rate is **zero**, not merely reduced. That is Bélády's
//! cyclic-scan pathology and it is a cliff rather than a slope: measured on the
//! previous build at 16,384 entries per generation, 25,000 distinct values gave
//! 15,538 hits over two passes and 33,000 distinct values gave 0.
//!
//! So this cache admits entries until it is full and then declines new ones.
//! A repeated query over a column with more distinct values than
//! [`EmbeddingCache::capacity`] is served for `capacity` of them and re-embeds
//! the rest — a hit rate of `capacity / distinct` instead of zero, and it does
//! not collapse as the column grows. What it costs is adaptivity: the resident
//! set is whatever was seen first in the session, so a session that moves on to
//! a different column keeps the old one until someone calls
//! `subtoken_cache_clear()`. [`CacheStats::uncached`] is how a caller sees
//! that this is happening rather than inferring it from a slow query.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

/// Domain tag mixed into every cache key.
const CACHE_KEY_DOMAIN: &[u8] = b"subtoken/cache-key/v1";

/// Memory the cache may spend, in bytes.
///
/// A ceiling, not an estimate: [`capacity_for_budget`] picks the largest number
/// of vectors whose real cost stays under this, and
/// `a_full_cache_costs_no_more_than_the_budget_promised` fills a cache and asks
/// the allocator whether that held.
///
/// 64 MiB is modest beside DuckDB's own default, which is most of the machine.
/// How many vectors it buys is not fixed and is not written down here: it
/// depends on the model's width and on how the platform's allocator rounds, so
/// `subtoken_cache_stats().capacity` is the only place to read it.
pub const DEFAULT_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// A cache key: a 32-byte digest of the model identity and the input text.
pub type CacheKey = [u8; 32];

/// Bytes one bucket of the map costs: the key and the `Arc` inline, plus the
/// one control byte `hashbrown` pairs with every bucket.
const BUCKET_BYTES: usize = std::mem::size_of::<(CacheKey, Arc<[f32]>)>() + 1;

/// Control bytes `hashbrown` allocates beyond one per bucket, so a SIMD group
/// can be read past the end of the array without a bounds check. Sixteen on
/// every target with SSE2 or NEON, which is every target this extension ships
/// for.
///
/// Sixteen bytes on a three-megabyte array looks negligible and is not: it puts
/// the request one byte over a page multiple, and a large allocation is rounded
/// to whole pages. Leaving it out cost 16,384 bytes — the whole gap between the
/// model and the measurement at 10,000 entries and above.
const CONTROL_GROUP_BYTES: usize = 16;

/// Buckets `HashMap` ends up with after `entries` inserts.
///
/// This mirrors `hashbrown::raw::capacity_to_buckets`: the array is a power of
/// two, sized so that seven-eighths of it covers the entries. It is a mirror of
/// a dependency and could drift, which is exactly what
/// `a_full_cache_costs_no_more_than_the_budget_promised` is for — that test
/// asks the allocator rather than asking this function.
///
/// The power of two is the part the first cost model left out. At 61,230
/// entries it charged 56 bytes an entry for the map; the real array is 131,072
/// buckets, which is 104.9 — measured, not derived from this function.
pub fn buckets_for(entries: usize) -> usize {
    if entries < 8 {
        return if entries < 4 { 4 } else { 8 };
    }
    (entries * 8 / 7).next_power_of_two()
}

/// Bytes the allocator really hands out for a block of `requested` bytes.
///
/// **Probed, not modelled.** Allocators round a request up to a size class, and
/// both blocks this cache holds are affected: on macOS the 1,040-byte block
/// behind one vector comes back with 1,280, and a 3,211,264-byte bucket array
/// comes back with 3,227,648. Modelling either would mean encoding one
/// platform's size classes; asking costs one malloc and one free.
///
/// On a platform with no way to ask, this returns the requested size. That
/// under-counts, so a cache there may exceed the budget by whatever the
/// allocator rounds up — the one direction in which the ceiling is soft, and it
/// is soft because the alternative is claiming to know something unmeasured.
pub fn allocation_bytes(requested: usize) -> usize {
    if requested == 0 {
        return 0;
    }
    let mut memo = PROBED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((_, measured)) = memo.iter().find(|(size, _)| *size == requested) {
        return *measured;
    }
    let measured = measure_allocation(requested);
    memo.push((requested, measured));
    measured
}

/// Answers already measured, so a size is probed once per process.
///
/// [`capacity_for_budget`] is a binary search that calls [`bytes_for`] about
/// seventeen times, and without this each call would allocate and free a block
/// of several megabytes. It also makes `bytes_for` answer identically every
/// time it is asked, which the search needs and which
/// `the_probe_is_taken_once_per_size` pins by counting probes.
///
/// A probe can over-report — the allocator may serve it from a recycled block
/// larger than the size class — and can never under-report, since
/// `malloc_size` is at least the request. So a stale or unlucky answer costs a
/// slightly smaller cache and can never breach the budget. Measured once under
/// a loaded process the search returned 49,873 where a quiet one returned
/// 49,907; that is not reproducible on demand, and no test pins it.
static PROBED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Probes actually taken, so a test can see that the memo is doing its job.
static PROBES_TAKEN: AtomicU64 = AtomicU64::new(0);

/// The same count, kept per size.
///
/// The process-wide total cannot answer "was THIS size probed again", because
/// every other size probed anywhere in the process moves it too — and `cargo
/// test` runs the tests on several threads, so a test watching the total is
/// watching every other test's first `embed` as well. That is not hypothetical:
/// `the_probe_is_taken_once_per_size` failed with "the memo let 1 further
/// probes through" for a size it was the only caller of, because the cache's
/// own capacity search — one probe per candidate size it tries — ran on another
/// thread inside its window.
static PROBES_BY_SIZE: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());

/// How many blocks have been allocated to measure a size class this process.
pub fn probes_taken() -> u64 {
    PROBES_TAKEN.load(Ordering::Relaxed)
}

/// How many of those were probes of `size`.
pub fn probes_taken_for(size: usize) -> u64 {
    PROBES_BY_SIZE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .find(|(probed, _)| *probed == size)
        .map(|(_, count)| *count)
        .unwrap_or(0)
}

/// Probe the allocator for a block of `requested` bytes.
fn measure_allocation(requested: usize) -> usize {
    PROBES_TAKEN.fetch_add(1, Ordering::Relaxed);
    {
        let mut by_size = PROBES_BY_SIZE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match by_size.iter_mut().find(|(size, _)| *size == requested) {
            Some((_, count)) => *count += 1,
            None => by_size.push((requested, 1)),
        }
    }
    // 16 is what `hashbrown` aligns its bucket array to and what the system
    // allocator returns anyway; both blocks go through plain `malloc` at this
    // alignment on the platforms probed.
    let Ok(layout) = std::alloc::Layout::from_size_align(requested, 16) else {
        return requested;
    };

    // SAFETY: `layout` has a non-zero size, the pointer is checked before use,
    // and it is freed with the layout it was allocated with.
    let measured = unsafe {
        let block = std::alloc::alloc(layout);
        if block.is_null() {
            return requested;
        }
        let size = allocated_size(block);
        std::alloc::dealloc(block, layout);
        size
    };
    measured.max(requested)
}

/// Bytes one cached vector's heap block costs: an `Arc`'s two reference counts
/// followed by the floats, as the allocator really sizes it.
pub fn vector_bytes(dim: usize) -> usize {
    allocation_bytes(2 * std::mem::size_of::<usize>() + dim * std::mem::size_of::<f32>())
}

/// Bytes the map's bucket array costs at `entries` entries, as the allocator
/// really sizes it.
pub fn bucket_array_bytes(entries: usize) -> usize {
    allocation_bytes(buckets_for(entries) * BUCKET_BYTES + CONTROL_GROUP_BYTES)
}

#[cfg(target_os = "macos")]
extern "C" {
    fn malloc_size(ptr: *const std::ffi::c_void) -> usize;
}

#[cfg(target_os = "linux")]
extern "C" {
    fn malloc_usable_size(ptr: *mut std::ffi::c_void) -> usize;
}

/// # Safety
/// `block` must be a live allocation from the global allocator.
unsafe fn allocated_size(block: *mut u8) -> usize {
    #[cfg(target_os = "macos")]
    {
        malloc_size(block.cast::<std::ffi::c_void>())
    }
    #[cfg(target_os = "linux")]
    {
        malloc_usable_size(block.cast::<std::ffi::c_void>())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = block;
        0
    }
}

/// Bytes a cache holding `entries` vectors of width `dim` costs.
pub fn bytes_for(entries: usize, dim: usize) -> usize {
    bucket_array_bytes(entries).saturating_add(entries.saturating_mul(vector_bytes(dim)))
}

/// The largest number of vectors of width `dim` whose cost stays inside
/// `budget_bytes`.
///
/// **Zero when not even one vector fits**, which turns the cache off rather
/// than letting it hold one vector in defiance of the budget it was given. The
/// budget is a ceiling at every budget or it is not a ceiling.
///
/// A search rather than a division, because [`bytes_for`] is a step function:
/// the bucket array doubles rather than growing smoothly, so there is no
/// per-entry constant to divide by. The division by a per-entry constant is
/// what let a wrong cost model pass a test that divided and multiplied by the
/// same constant.
pub fn capacity_for_budget(budget_bytes: usize, dim: usize) -> usize {
    if bytes_for(1, dim) > budget_bytes {
        return 0;
    }
    let per_vector = vector_bytes(dim).max(1);
    // An upper bound: ignoring the map entirely cannot fit fewer vectors than
    // including it, so the true answer is at or below this.
    let mut high = (budget_bytes / per_vector).max(1);
    let mut low = 1usize;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if bytes_for(middle, dim) <= budget_bytes {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

/// What the cache has been doing, as reported to SQL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups answered from the cache.
    pub hits: u64,
    /// Lookups that had to be embedded.
    pub misses: u64,
    /// Lookups the cache declined to store because it was full.
    ///
    /// Non-zero means the column has more distinct values than the cache can
    /// hold, so a repeated query will re-embed the excess.
    pub uncached: u64,
    /// Vectors currently held.
    pub entries: u64,
    /// The most vectors this cache will hold.
    pub capacity: u64,
}

/// A bounded map from cache key to vector that fills and then stops.
pub struct EmbeddingCache {
    entries: HashMap<CacheKey, Arc<[f32]>>,
    capacity: usize,
    hits: u64,
    misses: u64,
    uncached: u64,
}

impl EmbeddingCache {
    /// A cache holding at most `capacity` vectors.
    ///
    /// A capacity of zero is a cache that is off: it stores nothing, answers
    /// every lookup as a miss, and counts every one in
    /// [`CacheStats::uncached`]. That is what a budget too small for one vector
    /// buys, and it is better than one vector held against the budget's word.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
            hits: 0,
            misses: 0,
            uncached: 0,
        }
    }

    /// Look a key up, counting the outcome.
    pub fn get(&mut self, key: &CacheKey) -> Option<Arc<[f32]>> {
        match self.entries.get(key) {
            Some(vector) => {
                self.hits += 1;
                Some(Arc::clone(vector))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Look again after another thread may have filled the key in.
    ///
    /// This is the second look one caller takes for one row, so it does not
    /// count as a fresh lookup. If the value is there now, the miss that caller
    /// already recorded was stale and becomes a hit — which is what keeps
    /// `hits + misses` equal to the number of rows, and `hits` equal to the
    /// number of rows the cache actually answered.
    pub fn recheck(&mut self, key: &CacheKey) -> Option<Arc<[f32]>> {
        let vector = self.entries.get(key).map(Arc::clone)?;
        self.misses = self.misses.saturating_sub(1);
        self.hits += 1;
        Some(vector)
    }

    /// True when the cache will not take any further distinct key.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Store a vector against its key, if there is room.
    ///
    /// Returns whether it was stored. A full cache declines rather than
    /// evicting; see the module docs for why.
    ///
    /// The full-cache branch here is an invariant guard on a public method, not
    /// the production path: [`crate::embed`] tests [`Self::is_full`] itself and
    /// never reaches this when the answer is yes, because it also has to decide
    /// whether to take a flight. Mutating this branch leaves the SQL suite
    /// green, which is how that was established.
    pub fn insert(&mut self, key: CacheKey, vector: Arc<[f32]>) -> bool {
        if self.is_full() && !self.entries.contains_key(&key) {
            self.uncached += 1;
            return false;
        }
        self.entries.insert(key, vector);
        true
    }

    /// Record that a lookup had to be embedded because the cache was full.
    ///
    /// The caller short-circuits [`Self::insert`] when [`Self::is_full`] is
    /// already true, so the count has to be made here instead.
    pub fn note_uncached(&mut self) {
        self.uncached += 1;
    }

    /// Drop every cached vector, returning how many were dropped.
    ///
    /// The counters go with the entries: they describe the contents that just
    /// went away.
    pub fn clear(&mut self) -> u64 {
        let dropped = self.entries.len() as u64;
        // Replaced rather than cleared: `HashMap::clear` keeps the bucket array,
        // which at a full cache is megabytes. Someone calling
        // `subtoken_cache_clear()` is asking for the memory back.
        self.entries = HashMap::new();
        self.hits = 0;
        self.misses = 0;
        self.uncached = 0;
        dropped
    }

    /// Current counters.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            uncached: self.uncached,
            entries: self.entries.len() as u64,
            capacity: self.capacity as u64,
        }
    }
}

/// Derive the cache key for `text` under the model identified by `model_key`.
///
/// `text` is written last and every field before it is fixed width, so two
/// different inputs cannot assemble into the same byte stream. A field added
/// after `text` would break that and would need a length prefix in front of it.
pub fn cache_key(model_key: &[u8; 32], text: &str) -> CacheKey {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_KEY_DOMAIN);
    hasher.update(model_key);
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_key(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn vector(value: f32) -> Arc<[f32]> {
        Arc::from(vec![value; 4].into_boxed_slice())
    }

    /// The same text under two different models keys to two different entries.
    ///
    /// This is AC4's "keyed on content **and** model version". Without it a
    /// model swap would serve vectors from the previous model, which no other
    /// assertion in this crate would notice.
    #[test]
    fn the_model_key_is_part_of_the_cache_key() {
        let under_a = cache_key(&model_key(0xAA), "manufacturing");
        let under_b = cache_key(&model_key(0xBB), "manufacturing");
        assert_ne!(under_a, under_b);
    }

    /// A cached vector is not reachable under a different model key.
    #[test]
    fn a_cached_vector_does_not_survive_a_model_change() {
        let mut cache = EmbeddingCache::with_capacity(8);
        cache.insert(cache_key(&model_key(0xAA), "steel"), vector(1.0));
        assert!(cache.get(&cache_key(&model_key(0xBB), "steel")).is_none());
    }

    /// The text half of the key is the exact input bytes.
    ///
    /// Every pair below differs only by case, whitespace or Unicode
    /// composition, and this model's tokenizer folds all three away — so these
    /// are pairs the encoder gives the SAME vector for, and the cache still
    /// gives them two entries. That is the deliberate direction to be wrong in:
    /// a duplicate entry costs memory, while a key that collapsed two inputs
    /// the tokenizer did not would return a vector nobody asked for.
    #[test]
    fn the_cache_key_uses_the_exact_input_bytes() {
        let key = model_key(0x11);
        assert_ne!(cache_key(&key, "steel"), cache_key(&key, "steel "));
        assert_ne!(cache_key(&key, "Steel"), cache_key(&key, "steel"));
        assert_ne!(cache_key(&key, ""), cache_key(&key, " "));
        assert_ne!(cache_key(&key, "cafe\u{301}"), cache_key(&key, "caf\u{e9}"));
    }

    #[test]
    fn a_hit_returns_the_stored_vector_and_counts_as_a_hit() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let key = cache_key(&model_key(0x33), "copper");
        cache.insert(key, vector(7.0));
        assert_eq!(cache.get(&key).as_deref(), Some(&[7.0_f32; 4][..]));
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (1, 0, 1));
    }

    #[test]
    fn a_miss_is_counted_and_stores_nothing() {
        let mut cache = EmbeddingCache::with_capacity(8);
        assert!(cache.get(&cache_key(&model_key(0x44), "absent")).is_none());
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (0, 1, 0));
    }

    /// A successful recheck turns the caller's stale miss into a hit, and adds
    /// no second lookup.
    #[test]
    fn a_successful_recheck_corrects_the_stale_miss_it_follows() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let key = cache_key(&model_key(0x55), "nickel");

        assert!(cache.get(&key).is_none());
        assert_eq!((cache.stats().hits, cache.stats().misses), (0, 1));

        // Another thread fills it in while this caller is waiting.
        cache.insert(key, vector(1.0));
        assert!(cache.recheck(&key).is_some());
        assert_eq!(
            (cache.stats().hits, cache.stats().misses),
            (1, 0),
            "one row, one lookup, and the cache answered it"
        );
    }

    /// A failed recheck moves nothing: the miss stands and the caller encodes.
    #[test]
    fn a_failed_recheck_leaves_the_counters_alone() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let key = cache_key(&model_key(0x56), "absent");
        assert!(cache.get(&key).is_none());
        assert!(cache.recheck(&key).is_none());
        assert_eq!((cache.stats().hits, cache.stats().misses), (0, 1));
    }

    /// The live set stays bounded no matter how many distinct values arrive.
    #[test]
    fn the_cache_never_holds_more_than_its_capacity() {
        let capacity = 16;
        let mut cache = EmbeddingCache::with_capacity(capacity);
        let key = model_key(0x66);
        for i in 0..capacity * 10 {
            cache.insert(cache_key(&key, &format!("value {i}")), vector(i as f32));
            assert!(
                cache.stats().entries as usize <= capacity,
                "entries {} exceeded {capacity} after {} inserts",
                cache.stats().entries,
                i + 1
            );
        }
        assert_eq!(cache.stats().entries as usize, capacity);
        assert_eq!(cache.stats().uncached, (capacity * 9) as u64);
    }

    /// **A repeated scan of a working set larger than the cache does not fall
    /// to a zero hit rate.**
    ///
    /// This is the assertion the previous design had none of. Two sequential
    /// passes over 200 distinct keys through a 64-entry cache: the second pass
    /// must hit for the 64 that are resident. Under the two-generation policy
    /// this replaced, and under LRU or FIFO, it hits for none of them, because
    /// a cyclic scan evicts every entry exactly before it is next needed.
    #[test]
    fn a_repeated_scan_larger_than_the_cache_still_hits_for_a_full_cache_worth() {
        let capacity = 64;
        let distinct = 200;
        let mut cache = EmbeddingCache::with_capacity(capacity);
        let model = model_key(0x77);
        let keys: Vec<CacheKey> = (0..distinct)
            .map(|i| cache_key(&model, &format!("value {i}")))
            .collect();

        // First pass: every key is looked up and stored if there is room.
        for (i, key) in keys.iter().enumerate() {
            if cache.get(key).is_none() {
                cache.insert(*key, vector(i as f32));
            }
        }
        let after_first = cache.stats();
        assert_eq!(after_first.entries as usize, capacity);

        // Second pass, in the same order — the pattern a table rescan produces.
        for key in &keys {
            let _ = cache.get(key);
        }
        let hits_in_second_pass = cache.stats().hits - after_first.hits;
        assert_eq!(
            hits_in_second_pass as usize, capacity,
            "a rescan must be served for a full cache worth, not for none of it"
        );
    }

    /// A value the cache declined is still declined the next time round, and is
    /// counted every time.
    #[test]
    fn a_full_cache_declines_and_says_how_often() {
        let mut cache = EmbeddingCache::with_capacity(2);
        let model = model_key(0x88);
        assert!(cache.insert(cache_key(&model, "one"), vector(1.0)));
        assert!(cache.insert(cache_key(&model, "two"), vector(2.0)));
        assert!(!cache.insert(cache_key(&model, "three"), vector(3.0)));
        assert!(!cache.insert(cache_key(&model, "three"), vector(3.0)));
        assert_eq!(cache.stats().uncached, 2);
        // A key already held is still updatable — a full cache is closed to new
        // keys, not to the ones it has.
        assert!(cache.insert(cache_key(&model, "one"), vector(9.0)));
        assert_eq!(
            cache.get(&cache_key(&model, "one")).as_deref(),
            Some(&[9.0_f32; 4][..])
        );
    }

    #[test]
    fn clear_reports_what_it_dropped_and_resets_the_counters() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let model = model_key(0x99);
        cache.insert(cache_key(&model, "one"), vector(1.0));
        cache.insert(cache_key(&model, "two"), vector(2.0));
        let _ = cache.get(&cache_key(&model, "one"));
        assert_eq!(cache.clear(), 2);
        assert_eq!(
            cache.stats(),
            CacheStats {
                hits: 0,
                misses: 0,
                uncached: 0,
                entries: 0,
                capacity: 8
            }
        );
    }

    /// The capacity search returns the largest entry count the cost model
    /// allows.
    ///
    /// **This is a test of the search, not of the cost model**, and it cannot
    /// be anything else: it asks `bytes_for` the same question
    /// `capacity_for_budget` asked it. The version of this that guarded the old
    /// model asserted `capacity * entry_bytes <= budget < (capacity + 1) *
    /// entry_bytes`, called that a budget check, and held for every possible
    /// cost model including one that costed floats at a byte each. Whether the
    /// model is right is
    /// `crate::tests::a_full_cache_costs_no_more_than_the_budget_promised`,
    /// which asks the allocator instead.
    #[test]
    fn the_capacity_search_returns_the_largest_count_the_cost_model_allows() {
        for dim in [4, 256, 1024] {
            for budget in [64 * 1024, 1024 * 1024, 64 * 1024 * 1024] {
                let capacity = capacity_for_budget(budget, dim);
                assert!(
                    bytes_for(capacity, dim) <= budget,
                    "dim {dim} budget {budget} capacity {capacity}"
                );
                assert!(
                    capacity <= 1 || bytes_for(capacity + 1, dim) > budget,
                    "dim {dim} budget {budget} capacity {capacity} was not the largest"
                );
            }
        }
    }

    /// The bucket array is a power of two, which is the term the old model
    /// left out entirely.
    ///
    /// The expected values are worked out from `hashbrown`'s rule rather than
    /// from this function: seven-eighths of a power-of-two array covers the
    /// entries.
    #[test]
    fn the_bucket_count_is_the_power_of_two_that_holds_the_entries() {
        assert_eq!(buckets_for(0), 4);
        assert_eq!(buckets_for(3), 4);
        assert_eq!(buckets_for(4), 8);
        assert_eq!(buckets_for(7), 8);
        assert_eq!(buckets_for(8), 16);
        assert_eq!(buckets_for(14), 16);
        assert_eq!(buckets_for(15), 32);
        // 61,230 * 8 / 7 is 69,977, and the next power of two is 131,072 — the
        // count a real map was measured to allocate at that size.
        assert_eq!(buckets_for(61_230), 131_072);
    }

    /// A wider model gets fewer entries from the same budget.
    #[test]
    fn a_wider_model_gets_fewer_entries_from_the_same_budget() {
        let budget = 64 * 1024 * 1024;
        let narrow = capacity_for_budget(budget, 256);
        let wide = capacity_for_budget(budget, 1024);
        assert!(narrow > wide, "{narrow} should exceed {wide}");
    }

    #[test]
    fn halving_the_budget_leaves_fewer_than_two_thirds_of_the_capacity() {
        let full = capacity_for_budget(64 * 1024 * 1024, 256);
        let half = capacity_for_budget(32 * 1024 * 1024, 256);
        assert!(half < full * 2 / 3, "{half} against {full}");
        assert!(half > full / 3, "{half} against {full}");
    }

    /// A budget too small for one vector turns the cache off rather than
    /// holding a vector the budget cannot pay for.
    #[test]
    fn a_budget_too_small_for_one_vector_gives_no_cache_at_all() {
        assert_eq!(capacity_for_budget(0, 256), 0);
        assert_eq!(capacity_for_budget(1, 256), 0);
        assert_eq!(capacity_for_budget(bytes_for(1, 256) - 1, 256), 0);
        assert_eq!(capacity_for_budget(bytes_for(1, 256), 256), 1);
    }

    /// A cache with no capacity stores nothing and says so.
    #[test]
    fn a_cache_with_no_capacity_declines_everything() {
        let mut cache = EmbeddingCache::with_capacity(0);
        let key = cache_key(&model_key(0xA0), "anything");
        assert!(cache.is_full());
        assert!(!cache.insert(key, vector(1.0)));
        assert!(cache.get(&key).is_none());
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.capacity, stats.uncached), (0, 0, 1));
    }

    /// The probe reports at least what the block asked for, and more when the
    /// allocator rounds up.
    /// A size is probed once and remembered.
    ///
    /// Not a claim about speed. The capacity search calls `bytes_for` about
    /// seventeen times and each probe allocates a block of several megabytes;
    /// more importantly, `bytes_for` has to answer the same way every time or
    /// the search's monotonicity assumption does not hold. Counting the probes
    /// is what makes that checkable — asserting that two calls agree would pass
    /// whether or not anything was remembered.
    #[test]
    fn the_probe_is_taken_once_per_size() {
        // A size nothing else in this binary asks for.
        let size = 1_234_576;
        let first = allocation_bytes(size);
        for _ in 0..50 {
            assert_eq!(allocation_bytes(size), first);
        }
        // Per size, not the process-wide total: `cargo test` runs on several
        // threads, and the total moves for every other test's first `embed`.
        // Watching it here reported "the memo let 1 further probes through" for
        // a size this test is the only caller of.
        assert_eq!(
            probes_taken_for(size),
            1,
            "51 calls for one size took {} probes",
            probes_taken_for(size)
        );
    }

    /// The probe never reports less than the block needs, for either of the two
    /// blocks the cache holds.
    #[test]
    fn the_allocation_probe_never_reports_less_than_the_block_needs() {
        for dim in [1, 4, 256, 1024] {
            let requested = 2 * std::mem::size_of::<usize>() + dim * 4;
            assert!(
                vector_bytes(dim) >= requested,
                "dim {dim}: probe said {} for a {requested}-byte block",
                vector_bytes(dim)
            );
        }
        for entries in [1, 100, 10_000, 60_000] {
            let requested = buckets_for(entries) * BUCKET_BYTES + CONTROL_GROUP_BYTES;
            assert!(
                bucket_array_bytes(entries) >= requested,
                "{entries} entries: probe said {} for a {requested}-byte array",
                bucket_array_bytes(entries)
            );
        }
        assert_eq!(allocation_bytes(0), 0);
    }

    /// The default budget is still worth having after the cost model was
    /// corrected.
    ///
    /// A floor in the units a user cares about. It is deliberately well under
    /// what any platform measured gives — macOS lands near 49,900 and Linux
    /// higher, because glibc rounds a 1,040-byte request less than macOS does —
    /// so this pins the budget without pinning the allocator.
    #[test]
    fn the_default_budget_holds_at_least_thirty_thousand_vectors() {
        let capacity = capacity_for_budget(DEFAULT_BUDGET_BYTES, 256);
        assert!(
            capacity >= 30_000,
            "the default budget holds only {capacity} vectors of width 256"
        );
    }
}
