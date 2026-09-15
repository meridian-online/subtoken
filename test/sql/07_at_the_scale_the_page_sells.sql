-- AC4 at the scale the README offers, rather than at the scale that is
-- convenient to write.
--
-- Every other assertion about the cache in this suite uses between two and
-- eight distinct values. The page sells `subtoken_embed` on "a large text column" and on
-- `read_parquet('corpus.parquet')`, and the first build of this extension fell
-- to a ZERO per-cent hit rate above 33,000 distinct values with the whole suite
-- green. The counts below are absolute, not derived from
-- `subtoken_cache_stats().capacity`, which is what makes them pin the
-- cache's memory budget: a budget small enough to be useless would leave every
-- other test in this repo passing.

SET threads TO 1;

CREATE TABLE big AS SELECT 'row number ' || i AS s FROM range(40000) r(i);

SELECT subtoken_cache_clear();

CREATE TABLE big_pass_1 AS SELECT subtoken_embed(s) AS v FROM big;
CREATE TABLE first_pass AS SELECT subtoken_cache_stats() AS s;

SELECT must('forty thousand distinct values subtoken_embed forty thousand times',
    (SELECT s.encoded FROM first_pass) = 40000);
SELECT must('forty thousand distinct values all fit the cache',
    (SELECT s.uncached FROM first_pass) = 0
    AND (SELECT s.entries FROM first_pass) = 40000);

CREATE TABLE big_pass_2 AS SELECT subtoken_embed(s) AS v FROM big;
CREATE TABLE second_pass AS SELECT subtoken_cache_stats() AS s;

SELECT must('a repeated query over forty thousand distinct values re-embeds none of them',
    (SELECT s.encoded FROM second_pass) = (SELECT s.encoded FROM first_pass));
SELECT must('the second pass was served entirely from the cache',
    (SELECT s.hits FROM second_pass) - (SELECT s.hits FROM first_pass) = 40000);

-- Past the capacity there must be no cliff: a rescan is served for a full cache
-- worth, not for none of it. The previous build gave 0 hits here; an LRU or a
-- FIFO would too, because a cyclic scan evicts every entry exactly before it is
-- next wanted.
SELECT subtoken_cache_clear();

CREATE TABLE capacity AS SELECT subtoken_cache_stats().capacity AS n;
CREATE TABLE oversized AS
    SELECT 'oversized row ' || i AS s FROM range(0, (SELECT n FROM capacity) + 5000) r(i);

CREATE TABLE over_pass_1 AS SELECT subtoken_embed(s) AS v FROM oversized;
CREATE TABLE over_first AS SELECT subtoken_cache_stats() AS s;

SELECT must('a column larger than the cache fills it exactly',
    (SELECT s.entries FROM over_first) = (SELECT n FROM capacity));
SELECT must('and says how many values it could not hold',
    (SELECT s.uncached FROM over_first) = 5000);

CREATE TABLE over_pass_2 AS SELECT subtoken_embed(s) AS v FROM oversized;
CREATE TABLE over_second AS SELECT subtoken_cache_stats() AS s;

SELECT must('a rescan past the capacity is served for a full cache worth, not for none of it',
    (SELECT s.hits FROM over_second) - (SELECT s.hits FROM over_first)
    = (SELECT n FROM capacity));
SELECT must('and re-embeds only the excess',
    (SELECT s.encoded FROM over_second) - (SELECT s.encoded FROM over_first) = 5000);
