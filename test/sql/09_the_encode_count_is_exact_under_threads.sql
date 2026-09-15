-- One encode per distinct value, with eight threads on it.
--
-- The exact-count assertions elsewhere in this suite run over tables small
-- enough that DuckDB uses one chunk on one thread, so they held without the
-- extension being safe to run in parallel at all. Measured on the first build,
-- a ten-distinct-value column over 400,000 rows reported 29 encodes at DuckDB's
-- default thread count: every worker that missed before the first insert landed
-- embedded the same value again.
--
-- `threads` is set rather than left at the default so this is the same test on
-- a single-core runner as on a sixteen-core one.

SET threads TO 8;

SELECT must('this test really is running with eight threads',
    CAST(current_setting('threads') AS BIGINT) = 8);

CREATE TABLE hot AS SELECT 'value ' || (i % 10) AS s FROM range(400000) r(i);

SELECT subtoken_cache_clear();
CREATE TABLE hot_embedded AS SELECT subtoken_embed(s) AS v FROM hot;
CREATE TABLE threaded AS SELECT subtoken_cache_stats() AS s;

SELECT must('ten distinct values over 400,000 rows encode exactly ten times',
    (SELECT s.encoded FROM threaded) = 10);
SELECT must('every row was accounted for exactly once',
    (SELECT s.hits FROM threaded) + (SELECT s.misses FROM threaded) = 400000);
SELECT must('the cache holds one entry per distinct value',
    (SELECT s.entries FROM threaded) = 10);

-- And the vectors are right, not merely counted: a race that handed one thread
-- another's buffer would keep the counts and corrupt the values.
CREATE TABLE hot_reference AS
    SELECT s, subtoken_embed(s) AS v FROM (SELECT DISTINCT s FROM hot);
SELECT must('every threaded row carries its own text''s vector',
    (SELECT count(*) FROM (SELECT DISTINCT s, v FROM (
        SELECT s, subtoken_embed(s) AS v FROM hot
     )) t JOIN hot_reference r ON t.s = r.s
     WHERE t.v IS DISTINCT FROM r.v) = 0);
