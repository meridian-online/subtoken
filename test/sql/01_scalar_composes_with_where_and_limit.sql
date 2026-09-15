-- AC1: a scalar function returns a vector for a string value, and composes with
-- WHERE and LIMIT on a filtered subset.
--
-- "Composes" is measured, not assumed. `subtoken_cache_stats().encoded`
-- counts encoder invocations, so a filtered query that embedded the whole table
-- would report the whole table's count and fail here. That is the difference
-- between the query running and the filter doing anything.

CREATE TABLE corpus AS SELECT * FROM (VALUES
    (1, 'manufacturing', 'a maker of industrial fasteners'),
    (2, 'logistics',     'a regional freight forwarder'),
    (3, 'manufacturing', 'a foundry casting valve bodies'),
    (4, 'retail',        'a chain of hardware stores'),
    (5, 'manufacturing', 'an injection moulder of housings'),
    (6, 'logistics',     'a bonded warehouse operator'),
    (7, 'retail',        'an online seller of workwear'),
    (8, 'finance',       'a regional commercial lender')
) AS t(id, industry, description);

-- The declared width, taken from the extension rather than written down here,
-- so this file cannot disagree with the model it is testing.
CREATE TABLE width AS
    SELECT CAST(regexp_extract(subtoken_version(), 'dim (\d+)', 1) AS BIGINT) AS n;

SELECT must('subtoken_embed is registered as a scalar returning FLOAT[]',
    (SELECT count(*) FROM duckdb_functions()
     WHERE function_name = 'subtoken_embed'
       AND function_type = 'scalar'
       AND return_type = 'FLOAT[]') = 1);

SELECT must('subtoken_embed returns a vector for a string value',
    typeof(subtoken_embed('a maker of industrial fasteners')) = 'FLOAT[]');

SELECT must('the vector is the width the extension reports',
    len(subtoken_embed('a maker of industrial fasteners')) = (SELECT n FROM width));

-- And the width itself, absolutely. Every other width assertion in this suite
-- compares the vector against `subtoken_version()`, which reads the same
-- `Model::dim` the vector came from — so a build whose encoder returned five
-- floats for everything would report `dim 5` and satisfy all of them. 256 is
-- the width the bundled potion-base-8M publishes, and the Rust test
-- `the_encoder_width_matches_the_width_the_config_declares` holds the encoder
-- to the model's own config.json.
SELECT must('the bundled model returns 256 floats',
    len(subtoken_embed('a maker of industrial fasteners')) = 256);

SELECT must('the vector carries signal rather than zeros',
    list_max(subtoken_embed('a maker of industrial fasteners')) > 0);

-- WHERE: three of the eight rows are manufacturing.
SELECT subtoken_cache_clear();
CREATE TABLE by_where AS
    SELECT id, subtoken_embed(description) AS v FROM corpus WHERE industry = 'manufacturing';

SELECT must('WHERE selected three rows',
    (SELECT count(*) FROM by_where) = 3);
SELECT must('WHERE embedded three rows and not the whole table',
    (SELECT subtoken_cache_stats().encoded) = 3);
SELECT must('every row WHERE selected got a full-width vector',
    (SELECT count(*) FROM by_where WHERE len(v) = (SELECT n FROM width)) = 3);
SELECT must('no row outside the filter was embedded',
    (SELECT count(*) FROM by_where b JOIN corpus c USING (id)
     WHERE c.industry <> 'manufacturing') = 0);

-- WHERE + LIMIT: fewer still.
SELECT subtoken_cache_clear();
CREATE TABLE by_where_limit AS
    SELECT id, subtoken_embed(description) AS v FROM corpus WHERE industry = 'manufacturing' LIMIT 2;

SELECT must('WHERE with LIMIT selected two rows',
    (SELECT count(*) FROM by_where_limit) = 2);
SELECT must('LIMIT further reduced the rows embedded',
    (SELECT subtoken_cache_stats().encoded) = 2);

-- The unfiltered baseline, so the two numbers above mean something.
SELECT subtoken_cache_clear();
CREATE TABLE whole_table AS SELECT id, subtoken_embed(description) AS v FROM corpus;
SELECT must('the whole table embeds all eight rows',
    (SELECT subtoken_cache_stats().encoded) = 8);

-- The same vector whether it arrived through a filter or not.
SELECT must('filtering does not change the vector a row gets',
    (SELECT count(*) FROM by_where b JOIN whole_table w USING (id)
     WHERE b.v IS DISTINCT FROM w.v) = 0);
