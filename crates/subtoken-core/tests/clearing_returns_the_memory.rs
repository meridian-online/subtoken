//! Clearing the cache gives the memory back, including the map's buckets.
//!
//! `HashMap::clear` keeps the bucket array, which at a full cache is megabytes,
//! so `subtoken_cache_clear()` was holding memory back from someone
//! explicitly asking for it.
//!
//! Its own binary for the same reason as `the_budget_is_a_measurement.rs`: a
//! measurement against the allocator wants a process that has done nothing
//! else.

#[path = "support/counting_allocator.rs"]
mod counting_allocator;

use std::sync::Arc;

use subtoken_core::cache::{self, EmbeddingCache};

#[global_allocator]
static COUNTING: counting_allocator::Counting = counting_allocator::Counting;

#[test]
fn clearing_the_cache_returns_the_memory_to_the_allocator() {
    let dim = 256;
    let entries = 20_000;
    let model = [1_u8; 32];

    let before = counting_allocator::live_bytes();
    let mut cache = EmbeddingCache::with_capacity(entries);
    for i in 0..entries {
        let key = cache::cache_key(&model, &format!("clear probe {i}"));
        cache.insert(key, Arc::from(vec![0.0_f32; dim].into_boxed_slice()));
    }
    let while_full = counting_allocator::live_bytes() - before;
    assert!(
        while_full > (entries * dim * 4) as isize,
        "expected a real cache, got {while_full} bytes"
    );

    assert_eq!(cache.clear(), entries as u64);
    let after = counting_allocator::live_bytes() - before;
    assert!(
        after < while_full / 100,
        "clearing left {after} bytes of the {while_full} the cache had taken"
    );
    drop(cache);
}
