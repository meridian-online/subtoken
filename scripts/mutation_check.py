#!/usr/bin/env python3
"""Break the code on purpose and require the named test to redden.

A test that passes against working code AND against deliberately broken code is
testing nothing, and reading it again will not tell you which kind you have.
This applies each mutation below to a source file, runs the one test that should
notice, and fails unless that test reports a failure. Then it restores the file
from git.

It is NOT in CI: each SQL mutation rebuilds the release cdylib, so a full sweep
is minutes rather than seconds. It is a gate you run when you change a test or
the code under one, and its value is that a reviewer can reproduce the proof
with one command instead of believing a paragraph.

    make mutation-check
    scripts/mutation_check.py --list
    scripts/mutation_check.py --only cache_key_drops_the_model

The tree must be clean: mutations are undone with `git checkout --`, which
would take uncommitted work with them.

Stdlib only.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE = "crates/subtoken-core/src"
GLUE = "crates/subtoken-duckdb/src/lib.rs"
ASSETS = "models/potion-base-8M"

#: The packaged artifact `make extension` writes, and which every SQL and script
#: mutation here rebuilds from broken source. `git checkout --` puts the code
#: back and says nothing about this file, so a mutated build of the extension
#: outlives the mutation that made it and answers to the branch's name. A
#: measurement taken against one read `embed('steel logistics')` as
#: order-sensitive on a tree whose own tests said otherwise. Deleted after every
#: mutation, so the next reader has to build one rather than find one.
ARTIFACT = "build/subtoken.duckdb_extension"

#: The doc checks below need no build and no duckdb: they read two files in the
#: tree. Named here so a mutation of one cannot quietly point at the other.
QUALITY_CLAIMS = ["python3", "scripts/check_quality_claims.py"]
QUALITY_CLAIMS_SELF_TEST = [*QUALITY_CLAIMS, "--self-test"]


@dataclass
class Mutation:
    """One deliberate break, and the test that must notice it."""

    name: str
    file: str
    old: str
    new: str
    #: What must appear as failing. For Rust, the test function name; for SQL,
    #: the substring of the test file name.
    expect_red: str = ""
    #: "rust", "sql", or "script" — a check that is itself a script, where
    #: `expect_red` is unused and `command` is what must come back non-zero.
    kind: str = "rust"
    #: For kind="script": the argv to run, relative to the repo root.
    command: list[str] = field(default_factory=list)
    #: Extra test names that also legitimately redden; not required, but not
    #: treated as noise either.
    also_reddens: list[str] = field(default_factory=list)


MUTATIONS: list[Mutation] = [
    # ── the bundled model ────────────────────────────────────────────────────
    Mutation(
        name="weights_asset_is_swapped_for_another_file",
        file=f"{CORE}/model.rs",
        old='const WEIGHTS: &[u8] = include_bytes!("../../../models/potion-base-8M/model.safetensors");',
        new='const WEIGHTS: &[u8] = include_bytes!("../../../models/potion-base-8M/config.json");',
        expect_red="bundled_asset_digests_match_the_pinned_release",
        also_reddens=["the_bundled_model_loads_from_embedded_bytes"],
    ),
    Mutation(
        name="the_model_key_stops_covering_the_weights",
        file=f"{CORE}/model.rs",
        old="    hasher.update(tokenizer);\n    hasher.update(weights);\n    hasher.update(config);",
        new="    hasher.update(tokenizer);\n    let _ = weights;\n    hasher.update(config);",
        expect_red="the_model_key_covers_every_asset_byte",
    ),
    Mutation(
        name="the_loaded_model_keys_itself_off_the_wrong_bytes",
        file=f"{CORE}/model.rs",
        old="        let key = model_key(TOKENIZER, WEIGHTS, CONFIG);",
        new="        let key = model_key(CONFIG, CONFIG, CONFIG);",
        expect_red="the_loaded_model_reports_the_key_its_assets_derive",
    ),
    # ── the width contract ───────────────────────────────────────────────────
    Mutation(
        name="conform_rewrites_a_full_width_vector",
        file=f"{CORE}/model.rs",
        old="        width if width == dim => Ok(vector),",
        new="        width if width == dim => Ok(vec![0.0_f32; width]),",
        expect_red="conform_leaves_a_full_width_vector_alone",
        also_reddens=["ordinary_text_gets_a_full_width_non_zero_vector"],
    ),
    Mutation(
        name="the_no_token_zero_vector_comes_back_one_float_short",
        file=f"{CORE}/model.rs",
        old="        0 => Ok(vec![0.0_f32; dim]),",
        new="        0 => Ok(vec![0.0_f32; dim - 1]),",
        expect_red="conform_turns_an_empty_vector_into_a_full_width_zero_vector",
        also_reddens=["every_vector_has_the_declared_width"],
    ),
    Mutation(
        name="a_wrong_width_vector_is_padded_instead_of_reported",
        file=f"{CORE}/model.rs",
        old='        width => Err(format!(\n            "the encoder returned {width} floats for a model {dim} wide"\n        )),',
        new="        _ => Ok(vec![0.0_f32; dim]),",
        expect_red="conform_reports_any_other_width_rather_than_padding_it",
    ),
    # ── embedding ────────────────────────────────────────────────────────────
    Mutation(
        name="embed_always_takes_the_no_token_path",
        file=f"{CORE}/model.rs",
        old="        conform(vector, self.dim)",
        new="        let _ = vector;\n        conform(Vec::new(), self.dim)",
        expect_red="ordinary_text_gets_a_full_width_non_zero_vector",
        also_reddens=["different_strings_get_different_vectors"],
    ),
    Mutation(
        name="the_empty_string_is_given_a_one_vector_instead_of_a_zero_vector",
        file=f"{CORE}/model.rs",
        old="        let sentence = [text.to_string()];",
        new="        if text.trim().is_empty() {\n            return Ok(vec![1.0_f32; self.dim]);\n        }\n        let sentence = [text.to_string()];",
        expect_red="text_with_no_tokens_gets_a_zero_vector_of_full_width",
    ),
    Mutation(
        name="embed_becomes_stateful_across_calls",
        file=f"{CORE}/model.rs",
        old="        conform(vector, self.dim)",
        new=(
            "        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);\n"
            "        let mut vector = vector;\n"
            "        if CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2 == 0 {\n"
            "            if let Some(first) = vector.first_mut() {\n"
            "                *first += 1.0;\n"
            "            }\n"
            "        }\n"
            "        conform(vector, self.dim)"
        ),
        expect_red="embedding_is_deterministic",
    ),
    Mutation(
        name="the_pool_becomes_order_sensitive",
        file=f"{CORE}/model.rs",
        old="        let sentence = [text.to_string()];",
        new=(
            "        let sentence = [format!(\n"
            '            "{} {}",\n'
            '            text.split_whitespace().next().unwrap_or(""),\n'
            "            text\n"
            "        )];"
        ),
        expect_red="the_pool_is_a_mean_so_order_is_lost_and_repetition_is_not",
        also_reddens=["case_and_surrounding_whitespace_do_not_change_the_vector"],
    ),
    # ── truncation ───────────────────────────────────────────────────────────
    # None of these existed while the mechanism was built and twice rebuilt, and
    # every one of them survived the suite as it stood: the corpus was ASCII, so
    # nothing in it could tell a character from a byte, and no input carried an
    # unknown token in a position that counted.
    Mutation(
        name="the_character_cut_counts_bytes_instead_of_characters",
        file=f"{CORE}/model.rs",
        old=(
            "        text.char_indices()\n"
            "            .nth(max_tokens.saturating_mul(median_token_length))\n"
            "            .map_or(text, |(byte_idx, _)| &text[..byte_idx])"
        ),
        new=(
            "        let limit = max_tokens.saturating_mul(median_token_length);\n"
            "        if text.len() <= limit {\n"
            "            return text;\n"
            "        }\n"
            "        let mut end = limit;\n"
            "        while !text.is_char_boundary(end) {\n"
            "            end -= 1;\n"
            "        }\n"
            "        &text[..end]"
        ),
        expect_red="the_clipped_verdict_matches_what_the_cap_cost_the_vector",
        also_reddens=[
            "a_generated_mixed_script_corpus_reports_clipped_exactly_when_the_cap_cost_it"
        ],
    ),
    # The same break, measured through DuckDB rather than through cargo, so the
    # SQL corpus is shown to be watching the mechanism and not only the Rust one.
    Mutation(
        name="the_character_cut_counts_bytes_instead_of_characters_seen_from_sql",
        file=f"{CORE}/model.rs",
        old=(
            "        text.char_indices()\n"
            "            .nth(max_tokens.saturating_mul(median_token_length))\n"
            "            .map_or(text, |(byte_idx, _)| &text[..byte_idx])"
        ),
        new=(
            "        let limit = max_tokens.saturating_mul(median_token_length);\n"
            "        if text.len() <= limit {\n"
            "            return text;\n"
            "        }\n"
            "        let mut end = limit;\n"
            "        while !text.is_char_boundary(end) {\n"
            "            end -= 1;\n"
            "        }\n"
            "        &text[..end]"
        ),
        expect_red="10_a_long_text",
        kind="sql",
    ),
    Mutation(
        name="the_unknown_token_id_is_never_recovered",
        file=f"{CORE}/model.rs",
        old=(
            "        let unk_token_id = spec\n"
            '            .get("model")\n'
            '            .and_then(|model| model.get("unk_token"))\n'
            "            .and_then(serde_json::Value::as_str)\n"
            "            .and_then(|token| tokenizer.token_to_id(token))\n"
            "            .map(|id| id as usize);"
        ),
        new="        let _ = &spec;\n        let unk_token_id: Option<usize> = None;",
        expect_red="the_clipped_verdict_matches_what_the_cap_cost_the_vector",
        also_reddens=[
            "a_generated_mixed_script_corpus_reports_clipped_exactly_when_the_cap_cost_it"
        ],
    ),
    Mutation(
        name="the_token_cap_moves_by_one",
        file=f"{CORE}/model.rs",
        old="pub const MAX_TOKENS: usize = 512;",
        new="pub const MAX_TOKENS: usize = 511;",
        expect_red="embed_is_byte_identical_to_the_upstream_wrapper_it_replaced",
        also_reddens=[
            "the_clipped_verdict_matches_what_the_cap_cost_the_vector",
            "a_long_text_is_reported_clipped_exactly_where_the_marker_stops_reaching_the_mean",
        ],
    ),
    # The defect the length comparison this replaced actually had: any text over
    # the character cut answered yes, whether or not an id went with it.
    Mutation(
        name="a_text_past_the_character_cut_is_called_clipped_whatever_it_lost",
        file=f"{CORE}/model.rs",
        old="        pooled.truncate(max_tokens);\n\n        pooled != whole",
        new="        pooled.truncate(max_tokens);\n\n        char_cut.len() < text.len() || pooled != whole",
        expect_red="text_with_no_tokens_is_not_reported_clipped",
        also_reddens=["the_clipped_verdict_matches_what_the_cap_cost_the_vector"],
    ),
    # And the defect the version before that had: the predicate never looks past
    # the character cut, so it sees the token cut alone.
    Mutation(
        name="the_predicate_stops_looking_past_the_character_cut",
        file=f"{CORE}/model.rs",
        old="        let whole = self.surviving_ids(text);",
        new="        let whole =\n            self.surviving_ids(Self::char_truncate(text, max_tokens, self.median_token_length));",
        expect_red="text_is_reported_clipped_by_the_character_cut_alone",
        also_reddens=["the_clipped_verdict_matches_what_the_cap_cost_the_vector"],
    ),
    # ── the cache ────────────────────────────────────────────────────────────
    Mutation(
        name="cache_key_drops_the_model_half",
        file=f"{CORE}/cache.rs",
        old="    hasher.update(CACHE_KEY_DOMAIN);\n    hasher.update(model_key);\n    hasher.update(text.as_bytes());",
        new="    hasher.update(CACHE_KEY_DOMAIN);\n    let _ = model_key;\n    hasher.update(text.as_bytes());",
        expect_red="the_model_key_is_part_of_the_cache_key",
        also_reddens=["a_cached_vector_does_not_survive_a_model_change"],
    ),
    Mutation(
        name="cache_key_normalises_the_text_the_way_a_tokenizer_might",
        file=f"{CORE}/cache.rs",
        old="    hasher.update(text.as_bytes());",
        new="    hasher.update(text.trim().to_lowercase().as_bytes());",
        expect_red="the_cache_key_uses_the_exact_input_bytes",
        also_reddens=["inputs_that_embed_alike_still_occupy_separate_cache_entries"],
    ),
    Mutation(
        name="the_cache_never_stops_growing",
        file=f"{CORE}/cache.rs",
        old="        if self.is_full() && !self.entries.contains_key(&key) {\n            self.uncached += 1;\n            return false;\n        }",
        new="",
        expect_red="the_cache_never_holds_more_than_its_capacity",
        also_reddens=["a_full_cache_declines_and_says_how_often"],
    ),
    # The policy this replaced. A full cache that throws everything away and
    # starts again is the two-generation roll in one line, and it is what took
    # a repeated scan of 33,000 distinct values to a zero per-cent hit rate.
    Mutation(
        name="a_full_cache_evicts_instead_of_declining",
        file=f"{CORE}/cache.rs",
        old="        if self.is_full() && !self.entries.contains_key(&key) {\n            self.uncached += 1;\n            return false;\n        }",
        new="        if self.is_full() && !self.entries.contains_key(&key) {\n            self.entries.clear();\n        }",
        expect_red="a_repeated_scan_larger_than_the_cache_still_hits_for_a_full_cache_worth",
        also_reddens=["a_rescan_larger_than_the_cache_is_served_for_a_full_cache_worth"],
    ),
    # The branch `embed` actually takes when the cache is full. The guard inside
    # `insert` never runs in production, because `embed` short-circuits before
    # reaching it — so mutating `insert` leaves SQL green and says nothing.
    Mutation(
        name="a_full_cache_evicts_instead_of_declining_seen_from_sql",
        file=f"{CORE}/lib.rs",
        old="            cache.note_uncached();\n            drop(cache);\n            return encode(model, text);",
        new="            cache.clear();",
        expect_red="07_at_the_scale",
        kind="sql",
    ),
    Mutation(
        name="a_declined_value_is_not_counted",
        file=f"{CORE}/cache.rs",
        old="            self.uncached += 1;\n            return false;",
        new="            return false;",
        expect_red="a_full_cache_declines_and_says_how_often",
    ),
    Mutation(
        name="a_hit_is_not_counted",
        file=f"{CORE}/cache.rs",
        old="            Some(vector) => {\n                self.hits += 1;",
        new="            Some(vector) => {",
        expect_red="a_hit_returns_the_stored_vector_and_counts_as_a_hit",
    ),
    Mutation(
        name="a_miss_is_not_counted",
        file=f"{CORE}/cache.rs",
        old="            None => {\n                self.misses += 1;\n                None\n            }",
        new="            None => None,",
        expect_red="a_miss_is_counted_and_stores_nothing",
    ),
    Mutation(
        name="a_recheck_leaves_the_stale_miss_standing",
        file=f"{CORE}/cache.rs",
        old="        self.misses = self.misses.saturating_sub(1);\n        self.hits += 1;",
        new="",
        expect_red="a_successful_recheck_corrects_the_stale_miss_it_follows",
    ),
    Mutation(
        name="a_recheck_moves_the_counters_before_it_knows_the_answer",
        file=f"{CORE}/cache.rs",
        old="        let vector = self.entries.get(key).map(Arc::clone)?;\n        self.misses = self.misses.saturating_sub(1);\n        self.hits += 1;",
        new="        self.misses = self.misses.saturating_sub(1);\n        self.hits += 1;\n        let vector = self.entries.get(key).map(Arc::clone)?;",
        expect_red="a_failed_recheck_leaves_the_counters_alone",
    ),
    Mutation(
        name="clearing_leaves_the_counters_where_they_were",
        file=f"{CORE}/cache.rs",
        old="        self.hits = 0;\n        self.misses = 0;\n        self.uncached = 0;",
        new="",
        expect_red="clear_reports_what_it_dropped_and_resets_the_counters",
    ),
    # ── the cost model, which nothing measured before ────────────────────────
    # Round 1 left a bare capacity constant settable to 4 with the suite green.
    # Round 2 replaced it with a budget and pinned the budget and the division,
    # neither of which is the cost model the division used. These are the
    # mutations that reach it, and they reach it by measurement: every one is
    # caught by filling a cache and asking the allocator.
    Mutation(
        name="the_cache_budget_is_cut_to_a_few_kilobytes",
        file=f"{CORE}/cache.rs",
        old="pub const DEFAULT_BUDGET_BYTES: usize = 64 * 1024 * 1024;",
        new="pub const DEFAULT_BUDGET_BYTES: usize = 4 * 1024;",
        expect_red="the_default_budget_holds_at_least_thirty_thousand_vectors",
        also_reddens=[
            "a_repeated_query_over_forty_thousand_distinct_values_re_embeds_none_of_them"
        ],
    ),
    Mutation(
        name="the_cache_budget_is_cut_to_a_few_kilobytes_seen_from_sql",
        file=f"{CORE}/cache.rs",
        old="pub const DEFAULT_BUDGET_BYTES: usize = 64 * 1024 * 1024;",
        new="pub const DEFAULT_BUDGET_BYTES: usize = 4 * 1024;",
        expect_red="07_at_the_scale",
        kind="sql",
    ),
    # The reviewer's own probe: cost a float at one byte. Under the previous
    # guard the whole suite stayed green at 4.57 times the declared budget.
    Mutation(
        name="a_float_is_costed_at_one_byte",
        file=f"{CORE}/cache.rs",
        old="    allocation_bytes(2 * std::mem::size_of::<usize>() + dim * std::mem::size_of::<f32>())",
        new="    allocation_bytes(2 * std::mem::size_of::<usize>() + dim)",
        expect_red="a_full_cache_costs_close_to_the_bytes_the_budget_promised",
    ),
    # The size-class rounding: ask for the block and then do not ask how big it
    # really is. Worth 23% of the budget on macOS.
    Mutation(
        name="the_allocator_is_not_asked_how_big_the_block_really_is",
        file=f"{CORE}/cache.rs",
        old="        let size = allocated_size(block);",
        new="        let size = 0_usize;\n        let _ = allocated_size(block);",
        expect_red="a_full_cache_costs_close_to_the_bytes_the_budget_promised",
    ),
    # hashbrown's sixteen extra control bytes. Worth 0.02% of the budget, which
    # is why the ceiling has no tolerance band: any band wide enough to feel
    # comfortable lets this through.
    Mutation(
        name="the_maps_extra_control_group_is_not_charged_for",
        file=f"{CORE}/cache.rs",
        old="const CONTROL_GROUP_BYTES: usize = 16;",
        new="const CONTROL_GROUP_BYTES: usize = 0;",
        expect_red="a_full_cache_costs_close_to_the_bytes_the_budget_promised",
    ),
    # The bucket array is a power of two. Worth 4.5% of the budget.
    Mutation(
        name="the_bucket_array_is_not_rounded_to_a_power_of_two",
        file=f"{CORE}/cache.rs",
        old="    (entries * 8 / 7).next_power_of_two()",
        new="    entries * 8 / 7",
        expect_red="the_bucket_count_is_the_power_of_two_that_holds_the_entries",
        also_reddens=["a_full_cache_costs_close_to_the_bytes_the_budget_promised"],
    ),
    Mutation(
        name="the_bucket_array_is_not_charged_for_at_all",
        file=f"{CORE}/cache.rs",
        old="    bucket_array_bytes(entries).saturating_add(entries.saturating_mul(vector_bytes(dim)))",
        new="    entries.saturating_mul(vector_bytes(dim))",
        expect_red="a_full_cache_costs_close_to_the_bytes_the_budget_promised",
    ),
    # Perturbing `low` inside the loop instead of the bound would be the obvious
    # mutation here and it does not terminate: the search sits at the same pair
    # of bounds forever. It is the upper bound that is safe to break.
    Mutation(
        name="the_capacity_search_starts_from_half_the_room_it_has",
        file=f"{CORE}/cache.rs",
        old="    let mut high = (budget_bytes / per_vector).max(1);",
        new="    let mut high = (budget_bytes / per_vector / 2).max(1);",
        expect_red="the_capacity_search_returns_the_largest_count_the_cost_model_allows",
    ),
    Mutation(
        name="a_budget_too_small_for_one_vector_holds_one_anyway",
        file=f"{CORE}/cache.rs",
        old="    if bytes_for(1, dim) > budget_bytes {\n        return 0;\n    }",
        new="    if bytes_for(1, dim) > budget_bytes {\n        return 1;\n    }",
        expect_red="a_budget_too_small_for_one_vector_gives_no_cache_at_all",
    ),
    Mutation(
        name="a_zero_capacity_cache_quietly_becomes_a_one_entry_cache",
        file=f"{CORE}/cache.rs",
        old="            capacity,\n            hits: 0,",
        new="            capacity: capacity.max(1),\n            hits: 0,",
        expect_red="a_cache_with_no_capacity_declines_everything",
    ),
    # A floor rather than a cap: "never give less than N" stops the capacity
    # responding to the budget, which is what the halving test is for.
    Mutation(
        name="the_capacity_has_a_silent_floor",
        file=f"{CORE}/cache.rs",
        old="    low\n}",
        new="    low.max(40_000)\n}",
        expect_red="halving_the_budget_leaves_fewer_than_two_thirds_of_the_capacity",
    ),
    Mutation(
        name="the_capacity_is_silently_capped",
        file=f"{CORE}/cache.rs",
        old="    low\n}",
        new="    low.min(40_000)\n}",
        expect_red="a_full_cache_costs_close_to_the_bytes_the_budget_promised",
        also_reddens=["the_default_budget_holds_at_least_thirty_thousand_vectors"],
    ),
    Mutation(
        name="an_entry_costs_the_same_whatever_the_model_width",
        file=f"{CORE}/cache.rs",
        old="pub fn vector_bytes(dim: usize) -> usize {",
        new="pub fn vector_bytes(_ignored: usize) -> usize {\n    let dim = 256;",
        expect_red="a_wider_model_gets_fewer_entries_from_the_same_budget",
    ),
    Mutation(
        name="the_allocation_probe_reports_less_than_the_block_needs",
        file=f"{CORE}/cache.rs",
        old="    measured.max(requested)",
        new="    measured.min(requested / 2)",
        expect_red="the_allocation_probe_never_reports_less_than_the_block_needs",
    ),
    # The memo. Counting probes is what makes this checkable: asserting that two
    # calls agree would pass whether or not anything was remembered, which is the
    # shape this whole round is about.
    Mutation(
        name="the_probe_is_repeated_for_every_call",
        file=f"{CORE}/cache.rs",
        old="    if let Some((_, measured)) = memo.iter().find(|(size, _)| *size == requested) {\n        return *measured;\n    }",
        new="",
        expect_red="the_probe_is_taken_once_per_size",
    ),
    # The floor of the measurement: a model that over-charges buys a cache
    # smaller than the budget paid for, and the cache is then quietly worse than
    # it was asked to be rather than broken.
    Mutation(
        name="every_vector_is_charged_for_twice",
        file=f"{CORE}/cache.rs",
        old="pub fn vector_bytes(dim: usize) -> usize {\n    allocation_bytes(",
        new="pub fn vector_bytes(dim: usize) -> usize {\n    2 * allocation_bytes(",
        expect_red="a_full_cache_costs_close_to_the_bytes_the_budget_promised",
    ),
    # Clearing gave the vectors back and kept the bucket array — megabytes
    # withheld from someone who asked for the memory.
    Mutation(
        name="clearing_keeps_the_bucket_array",
        file=f"{CORE}/cache.rs",
        old="        self.entries = HashMap::new();",
        new="        self.entries.clear();",
        expect_red="clearing_the_cache_returns_the_memory_to_the_allocator",
    ),
    # ── the width, which was only ever compared to itself ────────────────────
    Mutation(
        name="the_encoder_width_drifts_from_the_models_own_config",
        file=f"{CORE}/model.rs",
        old='        let dim = inner.encode_single("dimension probe").len();',
        new='        let dim = inner.encode_single("dimension probe").len().saturating_sub(1);',
        expect_red="the_encoder_width_matches_the_width_the_config_declares",
    ),
    Mutation(
        name="the_encoder_width_drifts_seen_from_sql",
        file=f"{CORE}/model.rs",
        old='        let dim = inner.encode_single("dimension probe").len();',
        new='        let dim = inner.encode_single("dimension probe").len().saturating_sub(1);',
        expect_red="01_scalar_composes",
        kind="sql",
    ),
    # ── the checks that are scripts ──────────────────────────────────────────
    # The symbol parser going blind. The self-test this replaces normalised its
    # own planted symbol, so deleting this left it green while the macOS check
    # cleared /usr/bin/nc.
    Mutation(
        name="the_symbol_parser_stops_stripping_the_macho_underscore",
        file="scripts/check_no_network_deps.py",
        old='        names.append(fields[-1].lstrip("_").split("@", 1)[0])',
        new='        names.append(fields[-1].split("@", 1)[0])',
        kind="script",
        command=["python3", "scripts/check_no_network_deps.py", "--self-test"],
    ),
    Mutation(
        name="the_socket_symbol_list_loses_connect",
        file="scripts/check_no_network_deps.py",
        old='    "connect",\n    "connectx",',
        new='    "connectx",',
        kind="script",
        command=["python3", "scripts/check_no_network_deps.py", "--self-test"],
    ),
    # The documented surface drifting from the catalog. This is the defect the
    # check was written for, reproduced.
    Mutation(
        name="the_readme_loses_a_field_from_the_stats_struct",
        file="README.md",
        old="`STRUCT(hits, misses, encoded, uncached, entries, capacity)`",
        new="`STRUCT(hits, misses, encoded, entries, capacity)`",
        kind="script",
        command=[
            "python3",
            "scripts/check_documented_surface.py",
            "--extension",
            "build/subtoken.duckdb_extension",
            "--duckdb",
            "$DUCKDB",
        ],
    ),
    Mutation(
        name="the_readme_reorders_the_stats_struct",
        file="README.md",
        old="`STRUCT(hits, misses, encoded, uncached, entries, capacity)`",
        new="`STRUCT(misses, hits, encoded, uncached, entries, capacity)`",
        kind="script",
        command=[
            "python3",
            "scripts/check_documented_surface.py",
            "--extension",
            "build/subtoken.duckdb_extension",
            "--duckdb",
            "$DUCKDB",
        ],
    ),
    Mutation(
        name="the_surface_check_stops_comparing_field_order",
        file="scripts/check_documented_surface.py",
        old="        signatures.append([field.split()[0] for field in fields])",
        new="        signatures.append(sorted(field.split()[0] for field in fields))",
        kind="script",
        command=["python3", "scripts/check_documented_surface.py", "--self-test"],
    ),
    # ── the published quality position ───────────────────────────────────────
    # README.md and description.yml both tell a stranger how good these vectors
    # are, and description.yml is the copy that becomes the registry page.
    # Neither can be settled by loading the extension, so the mutations below
    # break the pages and the checker in turn. None of them touches Rust, so
    # none rebuilds the cdylib.
    Mutation(
        name="the_readme_hedges_the_neighbourhood_figure",
        file="README.md",
        old="13% are the same rows on long-form prose",
        new="only a minority are the same rows on long-form prose",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    Mutation(
        name="the_registry_entry_drops_the_shape_dependence",
        file="description.yml",
        old="It is worst on long prose and\n    mildest on very short strings",
        new="It is noticeable across the board and\n    much the same whatever the corpus",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    Mutation(
        name="the_registry_entry_and_the_readme_disagree_on_a_figure",
        file="description.yml",
        old="| region structure kept | 71% | 67% | 88% |",
        new="| region structure kept | 71% | 67% | 95% |",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    Mutation(
        name="the_published_figures_drift_from_the_measurement",
        file="scripts/check_quality_claims.py",
        old='        0.13185,\n        "kNN overlap with MiniLM\'s map, long-form prose",',
        new='        0.20185,\n        "kNN overlap with MiniLM\'s map, long-form prose",',
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    # And the checker going blind, measured through its own self-test.
    Mutation(
        name="the_quality_check_stops_comparing_the_two_pages",
        file="scripts/check_quality_claims.py",
        old="    for value in sorted((readme_counts - descriptor_counts).elements()):",
        new="    for value in sorted([]):",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_rounding_at_the_written_precision",
        file="scripts/check_quality_claims.py",
        old="    return abs(figure.value - value) <= 0.5 * 10.0**-places + 1e-12",
        new="    return abs(figure.value - value) <= 0.5",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_pinning_a_figure_to_its_corpus",
        file="scripts/check_quality_claims.py",
        old="    for claim in CLAIMS:",
        new="    for claim in []:",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    # The summary table restates the prose figures, and a literal string pin
    # cannot tie a cell to the corpus it sits under: two cells swapped leave the
    # set of quantities on the page unchanged. It is read cell by cell against
    # FIGURES, by the row's series and the column's corpus, and these are the
    # two edits that used to be invisible.
    Mutation(
        name="the_readme_swaps_two_summary_table_cells",
        file="README.md",
        old="| nearest neighbours that survive | 13% | 28% | 40% |",
        new="| nearest neighbours that survive | 28% | 13% | 40% |",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    Mutation(
        name="the_readme_reverses_the_summary_table_header",
        file="README.md",
        old="| | long-form prose | short text | very short strings |",
        new="| | very short strings | short text | long-form prose |",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    # Region structure is 71%, 67%, 88% — not a gradient. A sentence quoting
    # only its endpoints told a reader with one-line descriptions they were at
    # the good end while the cited metric put them at the worst.
    Mutation(
        name="the_readme_states_a_direction_from_the_endpoints_only",
        file="README.md",
        old="mildest on very short strings — 13%, then 28%, then 40% as the text gets shorter.",
        new="mildest on short strings — 13% against 40% on neighbours, 71% against 88% on regions.",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    # The region sentence says 71% on long-form prose and 67% on short text, and
    # the table two lines below says the same. Swapping which corpus each figure
    # belongs to leaves the quantities, their count and their order untouched in
    # both files, so assertions 2, 3, 5 and 6 all pass: what is left is a
    # sentence contradicting its own table. Only the CLAIMS pin sees it, and
    # until that pin existed this edit passed the whole check.
    Mutation(
        name="the_readme_swaps_the_corpora_in_the_region_sentence",
        file="README.md",
        old="71% on long-form prose, 67% on short text, 88% on very short strings",
        new="71% on short text, 67% on long-form prose, 88% on very short strings",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    # A blanket universal over a section that deliberately mixes our measurement
    # with a third party's, which cancels the per-item sourcing around it.
    Mutation(
        name="the_readme_asserts_a_universal_over_the_mixed_sourcing",
        file="README.md",
        old="and where they compare, the comparison is the bundled",
        new="and every figure below compares the bundled",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    # Speed is ruled out of the published claim. A section-scoped ban let it in
    # three headings down, where it reaches the same reader.
    Mutation(
        name="a_speed_figure_lands_under_the_sql_surface",
        file="README.md",
        old="There is no similarity or nearest-neighbour function, deliberately.",
        new=(
            "There is no similarity or nearest-neighbour function, deliberately. "
            "It embeds 50,000 rows per second."
        ),
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    # And the new assertions going blind, measured through the self-test.
    Mutation(
        name="the_quality_check_stops_reading_the_summary_table",
        file="scripts/check_quality_claims.py",
        old="    for label, series in TABLE_ROWS.items():",
        new="    for label, series in []:",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_requiring_a_whole_series",
        file="scripts/check_quality_claims.py",
        old="            if not written or len(written) == len(members):",
        new="            if True:",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_scanning_for_universals",
        file="scripts/check_quality_claims.py",
        old="    for match in UNIVERSAL.finditer(collapsed):",
        new="    for match in []:",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_expiring_a_permitted_universal",
        file="scripts/check_quality_claims.py",
        old="        if collapse(allowance.phrase) not in haystack:",
        new="        if False:",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_comparing_the_order_of_the_figures",
        file="scripts/check_quality_claims.py",
        old="    for index, (left, right) in enumerate(zip(in_readme, in_descriptor)):",
        new="    for index, (left, right) in enumerate([]):",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_banning_speed_outside_the_section",
        file="scripts/check_quality_claims.py",
        old="    for phrase, reason in BANNED_ON_PAGE:",
        new="    for phrase, reason in []:",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    # The speed ban is the one scan that reads code, because the SQL examples
    # are where a speed figure hides and `description.yml` publishes all of
    # them in fenced blocks. Reading it after `strip_noise` instead makes it
    # page-minus-code-wide, which is what it silently was.
    Mutation(
        name="the_speed_ban_stops_looking_inside_the_sql_examples",
        file="scripts/check_quality_claims.py",
        old="    collapsed = collapse(strip_addresses(page_text))",
        new="    collapsed = collapse(strip_noise(page_text))",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    # These two are the assertions a sweep found could be deleted with every
    # self-test case still green, because in each the case was satisfied by a
    # neighbouring assertion: the hedge case planted `only a minority survive`
    # and the universal scan reported the `only`, and the empty-section case
    # asked only that something was reported, which the missing CLAIMS pins
    # do on their own. Both cases now name the message their own assertion
    # produces, and these two mutations are what says so.
    Mutation(
        name="the_quality_check_stops_banning_the_hedges_the_figures_replaced",
        file="scripts/check_quality_claims.py",
        old="    for phrase, reason in BANNED_IN_SECTION:",
        new="    for phrase, reason in []:",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_reporting_an_empty_section_as_empty",
        file="scripts/check_quality_claims.py",
        old="    if not collapsed:",
        new="    if False:",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    # ── the gate's own wiring ────────────────────────────────────────────────
    # `run` is where every assertion in that file is joined to the two pages it
    # judges, and a self-test that calls the helpers directly cannot see it:
    # each line below was deletable on its own with `--self-test` and the live
    # check both at exit 0, and the last two reduced the whole gate to something
    # that printed `ok:` and read nothing. The staged-tree cases at the foot of
    # `self_test` are what kills them — they run `run`, and then this script as
    # a process, over a temporary tree with one defect planted at a time.
    Mutation(
        name="the_quality_check_stops_judging_the_readme",
        file="scripts/check_quality_claims.py",
        old="    problems += region_problems(README, readme_section)",
        new="    pass",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_judging_the_registry_entry",
        file="scripts/check_quality_claims.py",
        old='    problems += region_problems(f"{DESCRIPTOR} (extended_description)", descriptor_section)',
        new="    pass",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_never_asks_whether_the_two_pages_agree",
        file="scripts/check_quality_claims.py",
        old="    problems += disagreements(readme_section, descriptor_section)",
        new="    pass",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_never_asks_which_figures_are_unused",
        file="scripts/check_quality_claims.py",
        old="    problems += unused_figures(readme_section, descriptor_section)",
        new="    pass",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_never_asks_which_allowances_are_unused",
        file="scripts/check_quality_claims.py",
        old="    problems += unused_allowances(readme_section, descriptor_section)",
        new="    pass",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_speed_ban_never_reaches_the_readme",
        file="scripts/check_quality_claims.py",
        old="    problems += page_problems(README, readme_text)",
        new="    pass",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_speed_ban_never_reaches_the_registry_page",
        file="scripts/check_quality_claims.py",
        old='    problems += page_problems(f"{DESCRIPTOR} (the rendered page)", descriptor_page(descriptor))',
        new="    pass",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_stops_reading_the_model_revision",
        file="scripts/check_quality_claims.py",
        old="    problems += revision_problems(source_path.read_text())",
        new="    pass",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    # The one that made every other assertion decorative: collect the problems,
    # then exit 0 and print `ok:` over them.
    Mutation(
        name="the_quality_check_reports_no_problem_it_found",
        file="scripts/check_quality_claims.py",
        old='    if problems:\n        print("FAIL: the published quality position does not hold up:", file=sys.stderr)',
        new='    if False:\n        print("FAIL: the published quality position does not hold up:", file=sys.stderr)',
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    Mutation(
        name="the_quality_check_parses_its_arguments_and_stops",
        file="scripts/check_quality_claims.py",
        old="    return run(REPO_ROOT)",
        new="    return 0",
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    # The registry renders the worked example and the one-line blurb above the
    # body. Reading the body alone is what the ban did while printing that it
    # had read the whole page.
    Mutation(
        name="the_registry_page_narrows_to_the_body_again",
        file="scripts/check_quality_claims.py",
        old="    for table, field_name in PAGE_FIELDS:",
        new='    for table, field_name in [("docs", "extended_description")]:',
        kind="script",
        command=QUALITY_CLAIMS_SELF_TEST,
    ),
    # And the defect that narrowing admits, on the real entry: a rows-per-second
    # claim in the SQL a stranger reads first, and in the blurb above it.
    Mutation(
        name="a_speed_figure_lands_in_the_published_sql_example",
        file="description.yml",
        old="    -- 256 floats per row, for this model.",
        new="    -- 256 floats per row, for this model. 50,000 rows per second.",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    Mutation(
        name="a_speed_figure_lands_in_the_registry_blurb",
        file="description.yml",
        old="with no API key and no network call",
        new="with no API key, no network call and 397x lower latency",
        kind="script",
        command=QUALITY_CLAIMS,
    ),
    # ── the engine's public behaviour ────────────────────────────────────────
    # Not "the first lookup is skipped": that is a fast path, and `recheck`
    # answers the same question a few lines later, so removing it changes no
    # behaviour and reddens nothing. What is a defect is never storing.
    Mutation(
        name="the_cache_never_stores_anything",
        file=f"{CORE}/cache.rs",
        old="        self.entries.insert(key, vector);\n        true",
        new="        let _ = vector;\n        true",
        expect_red="repeating_a_value_does_not_re_embed_it",
        also_reddens=["a_hit_returns_the_stored_vector_and_counts_as_a_hit"],
    ),
    Mutation(
        name="the_cache_stores_a_zero_vector_instead_of_the_real_one",
        file=f"{CORE}/lib.rs",
        old="    cache().insert(key, Arc::clone(&vector));",
        new="    cache().insert(key, Arc::from(vec![0.0_f32; vector.len()].into_boxed_slice()));",
        expect_red="a_vector_read_back_from_the_cache_equals_the_uncached_one",
    ),
    Mutation(
        name="clearing_leaves_the_encode_counter_running",
        file=f"{CORE}/lib.rs",
        old="    ENCODED.store(0, Ordering::Relaxed);\n    cache().clear()",
        new="    cache().clear()",
        expect_red="clearing_the_cache_drops_the_entries_and_the_counters",
    ),
    Mutation(
        name="the_version_line_stops_naming_the_model",
        file=f"{CORE}/lib.rs",
        old='            "subtoken {} (model {}@{}, key {}, dim {})",\n            VERSION,\n            model::MODEL_ID,',
        new='            "subtoken {} (model {}@{}, key {}, dim {})",\n            VERSION,\n            "a model",',
        expect_red="describe_names_the_bundled_model_and_the_width",
    ),
    # ── one encode per distinct value, under threads ─────────────────────────
    # Pinned in Rust rather than through DuckDB, and the measurement says why.
    # Eight threads in a tight loop over ten values gave 49-65 encodes instead
    # of 10 in ten runs out of ten with the flight removed. The same shape
    # through DuckDB at eight threads gave 12 instead of 10, and only sometimes:
    # DuckDB hands the first chunk to one thread, which warms the cache before
    # the others engage. `recheck` alone is enough to make the DuckDB-visible
    # count right; the flight is what makes it right under contention, and only
    # a test that creates contention can pin it.
    Mutation(
        name="two_threads_on_one_value_both_encode_it",
        file=f"{CORE}/lib.rs",
        old="    let _flight = Flight::begin(key);",
        new="",
        expect_red="eight_threads_over_ten_values_encode_ten_times",
        also_reddens=["eight_threads_over_distinct_values_each_get_the_right_vector"],
    ),
    Mutation(
        name="a_thread_that_took_the_flight_does_not_look_again",
        file=f"{CORE}/lib.rs",
        old="    if let Some(vector) = cache().recheck(&key) {\n        return Ok(vector);\n    }",
        new="",
        expect_red="09_the_encode_count_is_exact",
        kind="sql",
    ),
    Mutation(
        name="a_full_cache_still_queues_every_caller_behind_one_flight",
        file=f"{CORE}/lib.rs",
        old="            cache.note_uncached();",
        new="",
        expect_red="a_column_larger_than_the_cache_reports_what_it_could_not_hold",
        also_reddens=["a_value_the_full_cache_turned_away_is_re_embedded_every_time"],
    ),
    Mutation(
        name="the_uncached_count_is_not_reported_to_sql",
        file=GLUE,
        old="            stats.uncached,\n            stats.entries,",
        new="            stats.entries,\n            stats.uncached,",
        expect_red="07_at_the_scale",
        kind="sql",
    ),
    # ── the DuckDB surface ───────────────────────────────────────────────────
    Mutation(
        name="sql_null_stops_propagating_and_becomes_a_zero_vector",
        file=GLUE,
        old="                Cell::Null => vectors.push(None),",
        new='                Cell::Null => vectors.push(Some(subtoken_core::embed("")?)),',
        expect_red="04_null_and_empty",
        kind="sql",
    ),
    Mutation(
        name="a_null_row_is_written_without_its_validity_bit",
        file=GLUE,
        old="                list.set_entry(row, offset, 0);\n                list.set_null(row);",
        new="                list.set_entry(row, offset, 0);",
        expect_red="04_null_and_empty",
        kind="sql",
    ),
    Mutation(
        name="every_row_reads_the_child_vector_from_offset_zero",
        file=GLUE,
        old="                list.set_entry(row, offset, vector.len());\n                offset += vector.len();",
        new="                list.set_entry(row, 0, vector.len());\n                offset += vector.len();",
        expect_red="04_null_and_empty",
        kind="sql",
    ),
    Mutation(
        name="the_scalar_stops_using_the_cache",
        file=GLUE,
        old="Cell::Text(text) => vectors.push(Some(subtoken_core::embed(text)?)),",
        new="Cell::Text(text) => vectors.push(Some(std::sync::Arc::from(\n                    subtoken_core::embed_uncached(text)?.into_boxed_slice(),\n                ))),",
        expect_red="03_a_repeated_query",
        kind="sql",
    ),
    Mutation(
        name="cache_stats_is_folded_to_a_constant_because_it_is_no_longer_volatile",
        file=GLUE,
        old="    /// Without this DuckDB folds a zero-argument scalar to a constant and the\n    /// counters would be read once and reused for the rest of the session.\n    fn volatile() -> bool {\n        true\n    }",
        new="",
        expect_red="03_a_repeated_query",
        kind="sql",
    ),
    Mutation(
        name="the_version_line_truncates_the_model_revision",
        file=f"{CORE}/lib.rs",
        old="            &model::MODEL_REVISION[..12],",
        new="            &model::MODEL_REVISION[..8],",
        expect_red="02_bundled_model",
        kind="sql",
    ),
    Mutation(
        name="the_pool_becomes_order_sensitive_seen_from_sql",
        file=f"{CORE}/model.rs",
        old="        let sentence = [text.to_string()];",
        new=(
            "        let sentence = [format!(\n"
            '            "{} {}",\n'
            '            text.split_whitespace().next().unwrap_or(""),\n'
            "            text\n"
            "        )];"
        ),
        expect_red="06_text_the_tokenizer",
        kind="sql",
    ),
    # A running offset that is not reset between chunks. Invisible over the five
    # rows every other output-shape assertion uses, because five rows are one
    # chunk; fatal across three.
    Mutation(
        name="the_list_offset_is_carried_across_chunks",
        file=GLUE,
        old="    let mut offset = 0usize;\n    for (row, entry) in rows.iter().enumerate() {",
        new=(
            "    static CARRIED: std::sync::atomic::AtomicUsize =\n"
            "        std::sync::atomic::AtomicUsize::new(0);\n"
            "    let mut offset = CARRIED.load(std::sync::atomic::Ordering::Relaxed);\n"
            "    CARRIED.store(offset + total, std::sync::atomic::Ordering::Relaxed);\n"
            "    for (row, entry) in rows.iter().enumerate() {"
        ),
        expect_red="08_a_scan_wider_than_one_chunk",
        kind="sql",
    ),
    # A neutral name on purpose. It used to register `embed_nearest_neighbour`,
    # which trips test/sql/05's similarity guard as well as its count — so a
    # kill proved that SOMETHING in that file noticed and not which assertion
    # did. `subtoken_eighth` is invisible to the guard, so only the count and
    # the name list can see it; `a_similarity_function_is_registered` below
    # drives the guard, and neither mutation can stand in for the other.
    Mutation(
        name="an_eighth_function_is_registered",
        file=GLUE,
        old='    con.register_scalar_function::<Version>("subtoken_version")?;',
        new='    con.register_scalar_function::<Version>("subtoken_version")?;\n    con.register_scalar_function::<Version>("subtoken_eighth")?;',
        expect_red="05_the_registered_surface",
        kind="sql",
    ),
    # The similarity guard, driven on its own. Measured: with this mutation
    # applied, test/sql/05 reports BOTH "no similarity or nearest-neighbour
    # function is registered" and "the extension registers seven functions" —
    # the CLI runs the file in batch and does not stop at the first raised
    # statement, so the guard is evaluated whatever precedes it.
    Mutation(
        name="a_similarity_function_is_registered",
        file=GLUE,
        old='    con.register_scalar_function::<ModelId>("subtoken_model_id")?;',
        new='    con.register_scalar_function::<ModelId>("subtoken_model_id")?;\n    con.register_scalar_function::<ModelId>("subtoken_similar")?;',
        expect_red="05_the_registered_surface",
        kind="sql",
    ),

    # ── the model identity SQL stores beside a vector ────────────────────────
    #
    # AC1's claim is that changing any one of the three bundled assets moves
    # `subtoken_model_id()`, and that the move is visible from SQL. For the two
    # text assets the mutation edits the FILE, which is the altitude the claim
    # lives at: neither edit changes what the model computes — `model2vec_rs`
    # reads only `normalize` out of the config, and a whitespace change to the
    # tokenizer leaves its vocabulary alone — so the id moving is the only thing
    # either one can be observed to do.
    #
    # `model.safetensors` is binary and this harness reads its target with
    # `read_text()`, so the weights are reached through the hash input instead:
    # one byte short of the whole file, which is what a weights file that
    # differed by a byte would hash to. The model itself still loads from the
    # untouched constant, so that mutation too is observable only as the id
    # moving.
    Mutation(
        name="a_changed_config_asset_leaves_the_id_where_it_was",
        file=f"{ASSETS}/config.json",
        old='"seq_length": 1000000',
        new='"seq_length": 999999',
        expect_red="11_a_stored_vector",
        kind="sql",
    ),
    Mutation(
        name="a_changed_tokenizer_asset_leaves_the_id_where_it_was",
        file=f"{ASSETS}/tokenizer.json",
        old='  "truncation": null,',
        new='  "truncation":null,',
        expect_red="11_a_stored_vector",
        kind="sql",
    ),
    Mutation(
        name="a_changed_weights_asset_leaves_the_id_where_it_was",
        file=f"{CORE}/model.rs",
        old="        let key = model_key(TOKENIZER, WEIGHTS, CONFIG);",
        new="        let key = model_key(TOKENIZER, &WEIGHTS[..WEIGHTS.len() - 1], CONFIG);",
        expect_red="11_a_stored_vector",
        kind="sql",
    ),
    # The scalar stops reporting the key and reports the sentence that carries
    # twelve characters of it — the exact state the card says an analyst is
    # stuck in today, so a version of this function that "worked" that way has
    # to be caught.
    Mutation(
        name="the_model_id_scalar_reports_the_version_sentence",
        file=GLUE,
        old="        let text = CString::new(subtoken_core::model_id()?)?;",
        new="        let text = CString::new(subtoken_core::describe())?;",
        expect_red="11_a_stored_vector",
        kind="sql",
    ),
    Mutation(
        name="the_model_id_is_truncated_the_way_the_sentence_truncates_it",
        file=f"{CORE}/lib.rs",
        old="    Ok(model::bundled()?.key_hex())",
        new="    Ok(model::bundled()?.key_hex()[..12].to_string())",
        expect_red="the_model_id_is_the_whole_key_the_version_sentence_abbreviates",
    ),
    # The catalogue row's `key` is what a stored id joins against. Wired to the
    # revision it is still 40 characters of hex and still stable across a
    # session, which is exactly what a row that had quietly stopped being an
    # identity would look like.
    Mutation(
        name="the_catalogue_key_field_is_wired_to_the_revision",
        file=GLUE,
        old="            (7, &row.key),",
        new="            (7, row.revision),",
        expect_red="11_a_stored_vector",
        kind="sql",
    ),
    Mutation(
        name="the_catalogue_width_is_restated_rather_than_read",
        file=f"{CORE}/lib.rs",
        old="        width: model.dim() as u64,",
        new="        width: 255,",
        expect_red="the_catalogue_row_describes_the_model_that_is_loaded",
    ),
    Mutation(
        name="the_catalogue_input_limit_drifts_from_the_cap_that_bites",
        file=f"{CORE}/lib.rs",
        old="        input_limit: model::MAX_TOKENS as u64,",
        new="        input_limit: model::MAX_TOKENS as u64 + 1,",
        expect_red="11_a_stored_vector",
        kind="sql",
        also_reddens=["the_catalogue_row_describes_the_model_that_is_loaded"],
    ),
    Mutation(
        name="embed_returns_a_list_of_doubles_instead_of_floats",
        file=GLUE,
        old="            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),",
        new="            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Double)),",
        expect_red="01_scalar_composes",
        kind="sql",
    ),
]


#: A mutated build gets this long to produce a verdict. Ten minutes is far more
#: than any check here takes; it exists because a mutation can hang rather than
#: fail, and one did — perturbing the capacity binary search left it looping
#: forever, and the sweep sat on it for thirty-three minutes.
TIMEOUT_SECONDS = 600


class TookTooLong(Exception):
    """A mutated run produced no verdict inside the timeout.

    Neither a kill nor a survival: the check might have failed given longer, and
    counting a hang as either is how a sweep comes to report on work it never
    finished. Same reasoning as `RanNothing`, different cause.
    """


def run(command: list[str], timeout: float | None = TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as expired:
        raise TookTooLong(
            f"`{' '.join(command)}` produced no verdict in {timeout:.0f}s"
        ) from expired


def tree_is_clean() -> bool:
    return run(["git", "status", "--porcelain"]).stdout.strip() == ""


def check_anchors(mutations: list[Mutation]) -> list[str]:
    """Every anchor must be present exactly once, checked before anything runs.

    A sweep is minutes; finding out at mutation 25 that mutation 26's anchor
    moved wastes all of them. Reported together so one edit fixes the lot.
    """
    problems = []
    for mutation in mutations:
        text = (REPO_ROOT / mutation.file).read_text()
        found = text.count(mutation.old)
        if found != 1:
            problems.append(
                f"{mutation.name}: anchor appears {found} times in {mutation.file} "
                f"(it must appear exactly once)"
            )
    return problems


def apply(mutation: Mutation) -> None:
    path = REPO_ROOT / mutation.file
    text = path.read_text()
    if mutation.old not in text:
        raise SystemExit(
            f"mutation {mutation.name!r} no longer applies: its anchor is not in {mutation.file}.\n"
            "The code moved; update the mutation rather than deleting it."
        )
    if text.count(mutation.old) != 1:
        raise SystemExit(
            f"mutation {mutation.name!r} anchor appears {text.count(mutation.old)} times in "
            f"{mutation.file}; it must be unique."
        )
    path.write_text(text.replace(mutation.old, mutation.new))


def restore(mutation: Mutation) -> None:
    run(["git", "checkout", "--", mutation.file])
    (REPO_ROOT / ARTIFACT).unlink(missing_ok=True)


#: `test result: ok. 3 passed; 0 failed; ...`
TEST_RESULT = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed;")


class RanNothing(Exception):
    """The command under a mutation executed no test at all.

    This is neither a kill nor a survival, and counting it as either is how a
    mutation harness comes to report a clean sweep having checked nothing. It
    happened here on the first run: `cargo test NAME -- --exact` matches no test,
    because the names are module-qualified, so every Rust mutation "survived"
    against 0 tests run.
    """


def rust_test_failed(test_name: str, mutated_file: str) -> tuple[bool, str]:
    # No `--exact`: the tests live in a `tests` module, so the bare function
    # name is a substring of the full path rather than equal to it.
    #
    # Scoped to the crate the mutation touched. `--workspace` would relink the
    # 36 MB cdylib for every Rust mutation, and every Rust mutation is in
    # subtoken-core; the DuckDB layer's mutations are measured through SQL.
    scope = (
        ["-p", "subtoken-core"]
        if mutated_file.startswith("crates/subtoken-core")
        else ["--workspace"]
    )
    completed = run(["cargo", "test", *scope, test_name, "--", "--nocapture"])
    output = completed.stdout + completed.stderr

    executed = sum(int(passed) + int(failed) for passed, failed in TEST_RESULT.findall(output))
    if executed == 0 and "error" not in output:
        raise RanNothing(f"`cargo test {test_name}` matched no test")

    if completed.returncode == 0:
        return False, output
    return f"{test_name} ... FAILED" in output or "test result: FAILED" in output, output


def script_check_failed(command: list[str], duckdb: str) -> tuple[bool, str]:
    """A check written as a script: it must come back non-zero under the mutation.

    Some gates in this repo are scripts rather than tests — the socket-symbol
    check and the documented-surface check — and a mutation of one of those has
    to be measured by running it. `--extension` builds are done first when the
    command needs one.
    """
    if any(ARTIFACT in argument for argument in command):
        built = run(["make", "extension"])
        if built.returncode != 0:
            raise RanNothing(f"the mutated tree did not build:\n{built.stdout}{built.stderr}")
    resolved = [sys.executable if part == "python3" else part for part in command]
    resolved = [duckdb if part == "$DUCKDB" else part for part in resolved]
    completed = run(resolved)
    output = completed.stdout + completed.stderr
    if not output.strip():
        raise RanNothing(f"{' '.join(command)} produced no output at all")
    return completed.returncode != 0, output


def sql_test_failed(name_fragment: str, duckdb: str) -> tuple[bool, str]:
    built = run(["make", "extension"])
    if built.returncode != 0:
        # A mutation that does not compile is not evidence about a test.
        raise RanNothing(f"the mutated tree did not build:\n{built.stdout}{built.stderr}")
    completed = run(
        [
            sys.executable,
            "scripts/run_sql_tests.py",
            "--extension",
            "build/subtoken.duckdb_extension",
            "--duckdb",
            duckdb,
            "--only",
            name_fragment,
        ]
    )
    output = completed.stdout + completed.stderr
    if completed.returncode == 2:
        raise RanNothing(f"--only {name_fragment!r} matched no SQL test file")
    if "PASS " not in output and "FAIL " not in output:
        raise RanNothing(f"the SQL runner reported no result for {name_fragment!r}")
    return completed.returncode == 1 and "FAIL " in output, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run only mutations whose name contains this substring")
    parser.add_argument("--list", action="store_true", help="print the mutation table and stop")
    parser.add_argument(
        "--check-anchors",
        action="store_true",
        help="verify every mutation still applies, without running any of them",
    )
    parser.add_argument("--duckdb", default="duckdb")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.check_anchors:
        problems = check_anchors(MUTATIONS)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        if problems:
            return 2
        print(f"ok: all {len(MUTATIONS)} mutation anchors still apply")
        return 0

    if args.list:
        for mutation in MUTATIONS:
            target = mutation.expect_red or " ".join(mutation.command)
            print(f"{mutation.kind:6}  {mutation.name}\n        -> {target}")
        return 0

    if not tree_is_clean():
        print(
            "the working tree is dirty; mutations are undone with `git checkout --` "
            "and would take uncommitted work with them",
            file=sys.stderr,
        )
        return 2

    selected = [m for m in MUTATIONS if not args.only or args.only in m.name]
    if not selected:
        print(f"no mutation matches {args.only!r}", file=sys.stderr)
        return 2

    problems = check_anchors(selected)
    if problems:
        print(
            "the code moved out from under these mutations; update them rather than "
            "deleting them:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    survivors = []
    for mutation in selected:
        apply(mutation)
        try:
            if mutation.kind == "rust":
                reddened, output = rust_test_failed(mutation.expect_red, mutation.file)
            elif mutation.kind == "script":
                reddened, output = script_check_failed(mutation.command, args.duckdb)
            else:
                reddened, output = sql_test_failed(mutation.expect_red, args.duckdb)
        except (RanNothing, TookTooLong) as broken:
            restore(mutation)
            print(f"BROKEN   {mutation.name}: {broken}", file=sys.stderr)
            return 2
        finally:
            restore(mutation)

        verdict = "KILLED " if reddened else "SURVIVED"
        target = mutation.expect_red or " ".join(mutation.command)
        print(f"{verdict}  {mutation.name}  ->  {target}")
        if args.verbose or not reddened:
            print(output)
        if not reddened:
            survivors.append(mutation.name)

    print(f"\n{len(selected) - len(survivors)}/{len(selected)} mutations killed")
    if survivors:
        print("surviving mutations — the named test does not detect them:", file=sys.stderr)
        for name in survivors:
            print(f"  {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
