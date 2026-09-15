-- Every other assertion about the shape of the output vector runs over five
-- rows, which DuckDB hands the scalar in a single chunk. `write_float_lists`
-- computes a running offset into one child vector per chunk and resets it for
-- the next, and a defect in that reset is invisible until a scan spans more
-- than one chunk.
--
-- DuckDB's standard vector size is 2048 rows, so 5,000 rows spans at least
-- three chunks. A build configured with a larger vector size would make this
-- test weaker, not wrong.

CREATE TABLE wide AS
    SELECT i,
           CASE WHEN i % 97 = 0 THEN NULL
                WHEN i % 89 = 0 THEN ''
                ELSE 'value ' || (i % 300)
           END AS s
    FROM range(5000) r(i);

CREATE TABLE wide_embedded AS SELECT i, s, subtoken_embed(s) AS v FROM wide;

-- The reference is built from the distinct values only — 300 of them plus the
-- empty string, one chunk — so it cannot carry a cross-chunk offset defect of
-- its own. Comparing the wide scan against it is what makes this a test rather
-- than two copies of the same mistake agreeing.
CREATE TABLE reference AS
    SELECT s, subtoken_embed(s) AS v FROM (SELECT DISTINCT s FROM wide WHERE s IS NOT NULL);

SELECT must('the reference fits in one chunk',
    (SELECT count(*) FROM reference) <= 2048);

SELECT must('every row of a 5,000-row scan carries its own text''s vector',
    (SELECT count(*) FROM wide_embedded w JOIN reference r ON w.s = r.s
     WHERE w.v IS DISTINCT FROM r.v) = 0);

SELECT must('every NULL row across every chunk is NULL',
    (SELECT count(*) FROM wide_embedded WHERE s IS NULL AND v IS NOT NULL) = 0);
SELECT must('and there were NULL rows in more than one chunk',
    (SELECT count(*) FROM wide_embedded WHERE s IS NULL) > 2);

SELECT must('no non-NULL row came back NULL',
    (SELECT count(*) FROM wide_embedded WHERE s IS NOT NULL AND v IS NULL) = 0);

SELECT must('every non-NULL row has the full width',
    (SELECT count(*) FROM wide_embedded
     WHERE v IS NOT NULL
       AND len(v) <> CAST(regexp_extract(subtoken_version(), 'dim (\d+)', 1) AS BIGINT)) = 0);

SELECT must('the empty rows are zero vectors wherever they fall',
    (SELECT count(*) FROM wide_embedded
     WHERE s = '' AND (list_min(v) <> 0 OR list_max(v) <> 0)) = 0);
