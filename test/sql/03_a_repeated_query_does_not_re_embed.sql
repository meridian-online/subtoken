-- AC4: repeating a query over unchanged input does not re-embed, keyed on
-- content and model version.
--
-- The content half is here. The model-version half cannot be shown from SQL —
-- one binary carries one model — and is tested in Rust by
-- `the_model_key_is_part_of_the_cache_key` and
-- `a_cached_vector_does_not_survive_a_model_change` in
-- crates/subtoken-core/src/cache.rs.

CREATE TABLE t AS SELECT * FROM (VALUES
    ('alpha'), ('beta'), ('alpha'), ('gamma'), ('beta')
) AS v(s);

SELECT subtoken_cache_clear();

CREATE TABLE run1 AS SELECT s, subtoken_embed(s) AS v FROM t;
SELECT must('five rows of three distinct values encode three times',
    (SELECT subtoken_cache_stats().encoded) = 3);
SELECT must('the repeats within the first pass were served from the cache',
    (SELECT subtoken_cache_stats().hits) = 2);

CREATE TABLE run2 AS SELECT s, subtoken_embed(s) AS v FROM t;
SELECT must('re-running the query re-embeds nothing',
    (SELECT subtoken_cache_stats().encoded) = 3);
SELECT must('the second pass was served entirely from the cache',
    (SELECT subtoken_cache_stats().hits) = 7);

SELECT must('the cached vectors equal the ones first computed',
    (SELECT count(*) FROM run1 a, run2 b WHERE a.s = b.s AND a.v IS DISTINCT FROM b.v) = 0);

-- A cache that never stored anything would also report zero re-embeds if
-- `encoded` were wired to the wrong thing, so pin the entry count too.
SELECT must('the cache holds one entry per distinct value',
    (SELECT subtoken_cache_stats().entries) = 3);

-- Clearing puts the session back to a known state, and the next query pays
-- again — which is what says the numbers above were the cache and not a
-- stuck counter.
SELECT must('clearing reports the entries it dropped',
    subtoken_cache_clear() = 3);
CREATE TABLE run3 AS SELECT s, subtoken_embed(s) AS v FROM t;
SELECT must('after a clear the same query embeds again',
    (SELECT subtoken_cache_stats().encoded) = 3);
SELECT must('the vectors after a clear are the same vectors',
    (SELECT count(*) FROM run1 a, run3 b WHERE a.s = b.s AND a.v IS DISTINCT FROM b.v) = 0);

-- The counters must be re-read within a single statement, not folded to one
-- constant. DuckDB evaluates a zero-argument scalar once and reuses the answer
-- unless the function declares itself volatile, which would make every reading
-- above the first in any statement a stale copy of the first.
CREATE TABLE within_one_statement AS
    SELECT subtoken_cache_stats().entries AS first_reading,
           subtoken_cache_clear()         AS dropped,
           subtoken_cache_stats().entries AS second_reading;

SELECT must('the clear inside that statement dropped the entries',
    (SELECT dropped FROM within_one_statement) = 3);
SELECT must('two readings of the counter in one statement are not one folded constant',
    (SELECT first_reading FROM within_one_statement)
    <> (SELECT second_reading FROM within_one_statement));
