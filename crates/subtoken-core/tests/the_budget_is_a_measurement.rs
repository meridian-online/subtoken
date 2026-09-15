//! **The cost model is right, measured against the allocator.**
//!
//! Every other assertion about capacity asks `bytes_for` the same question
//! `capacity_for_budget` asked it, and so holds for any cost model at all: the
//! version this replaces asserted `capacity * entry_bytes <= budget <
//! (capacity + 1) * entry_bytes`, which is the definition of integer division.
//! Costing floats at one byte each left the whole suite green at 4.57 times the
//! declared budget. So this one does not consult the model. It fills a cache
//! and asks the global allocator how many bytes the process is holding.
//!
//! # Why this is an integration test and not a unit test
//!
//! It needs a process that has done nothing else. `malloc_size` reports the
//! block you were given, and a block recycled from a large freed region can be
//! bigger than the size class a fresh request would get. Run inside the unit
//! test binary, after other tests had filled and dropped a 64 MB cache, the
//! same code measured **70,319,872** bytes where a quiet process measured
//! **67,108,672** — a 4.8% error, which is larger than two of the three cost
//! model defects this exists to catch, so no tolerance could separate them.
//!
//! Cargo gives each file under `tests/` its own binary. One measurement per
//! file, so nothing precedes it.
//!
//! # What the budget is and is not
//!
//! It is a ceiling on what the cache **holds**. It is not a ceiling on the peak:
//! a map doubling its bucket array holds the old array alongside the new one,
//! so the high-water mark can exceed the budget by half the final array. Both
//! are asserted below, with different bounds, because they are different
//! claims.

#[path = "support/counting_allocator.rs"]
mod counting_allocator;

use std::sync::Arc;

use subtoken_core::cache::{self, EmbeddingCache, DEFAULT_BUDGET_BYTES};

#[global_allocator]
static COUNTING: counting_allocator::Counting = counting_allocator::Counting;

/// **The tolerance is asymmetric — no slack above, ten per cent below — and the
/// two sides are different claims.**
///
/// Above: none. The budget is a ceiling and the cost model is a measurement of
/// the allocator actually running, not an estimate of allocators in general, so
/// there is nothing for a band to absorb. A band would also be the wrong
/// instrument: the three omissions this catches cost 23%, 4.5% and 0.02% of the
/// budget, so any band wide enough to feel safe lets the third one through and
/// most let the second through too. Measured on macOS/aarch64 a full cache
/// lands 192 bytes under 64 MiB. **If a platform needs slack here, that is a
/// measurement to take and record, not a number to guess in advance.**
///
/// Below: ten per cent, and it cannot be tight. The bucket array doubles rather
/// than growing smoothly, so the capacity search can stop up to one doubling
/// short of the budget; that array is under 5% of the budget at this model's
/// width. This side catches a model so conservative that the cache is uselessly
/// small while still technically fitting.
///
/// **Its resolution is correspondingly coarse and that is worth saying.** A
/// model charging twice for every vector lands at 51% and fails. A model
/// charging four times over for the bucket array lands at 90.4% and passes —
/// measured, not supposed. So the floor catches a cost model that is wrong by a
/// factor, not one that is wrong by a sixth, and nothing here catches the
/// latter. The ceiling has no such limit, which is why it has no band.
#[test]
fn a_full_cache_costs_close_to_the_bytes_the_budget_promised() {
    let dim = 256;
    let budget = DEFAULT_BUDGET_BYTES as isize;
    let capacity = cache::capacity_for_budget(DEFAULT_BUDGET_BYTES, dim);
    let model = [0_u8; 32];

    counting_allocator::reset_peak();
    let before = counting_allocator::live_bytes();
    let mut cache = EmbeddingCache::with_capacity(capacity);
    for i in 0..capacity {
        let key = cache::cache_key(&model, &format!("budget probe {i}"));
        cache.insert(key, Arc::from(vec![0.0_f32; dim].into_boxed_slice()));
    }
    let held = counting_allocator::live_bytes() - before;
    let peak = counting_allocator::peak_bytes() - before;
    assert_eq!(
        cache.stats().entries as usize,
        capacity,
        "the cache did not fill"
    );

    assert!(
        held <= budget,
        "a full cache holds {held} bytes against a budget of {budget} — {} bytes over, which is \
         {:.2} bytes an entry the cost model does not charge for. capacity={capacity} \
         vector_bytes={} buckets_for={} bucket_array_bytes={} bytes_for={}",
        held - budget,
        (held - budget) as f64 / capacity as f64,
        cache::vector_bytes(dim),
        cache::buckets_for(capacity),
        cache::bucket_array_bytes(capacity),
        cache::bytes_for(capacity, dim),
    );
    assert!(
        held >= budget - budget / 10,
        "a full cache holds only {held} bytes of a {budget} budget — the cost model over-charges \
         and the cache is smaller than it was asked to be"
    );

    // The high-water mark, which is a different claim from the resting state and
    // a weaker one. When the map doubles its bucket array it holds the old array
    // and the new one at the same time, so the peak legitimately exceeds the
    // budget by up to the old array — half the final one.
    //
    // Whether that shows up at all depends on where the capacity lands relative
    // to a doubling. On macOS/aarch64 the capacity sits well inside a bucket
    // class and the fill peaks a few hundred bytes above where it settles. On
    // Linux/x86-64 it lands just past a doubling and the fill peaks 2,623,360
    // bytes above it — which is what the two bounds below are for, and why the
    // README says the budget is a resting ceiling rather than a peak one.
    let transient = peak - held;
    let previous_array = cache::bucket_array_bytes(capacity) as isize / 2;
    assert!(
        transient <= previous_array + 64 * 1024,
        "filling the cache peaked {transient} bytes above the {held} it settled at, which is \
         more than the {previous_array} bytes a bucket-array doubling can account for"
    );
    assert!(
        peak <= budget + budget / 10,
        "filling the cache peaked at {peak} bytes, more than a tenth above the {budget} budget"
    );
    drop(cache);
}
