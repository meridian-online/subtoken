-- The empty-tokenisation contract, decided rather than inherited.
--
-- Text with no in-vocabulary tokens is a zero vector of full width. SQL NULL is
-- a different case and propagates. Absence of a value is not a value that
-- carries no signal, and a caller wanting one behaviour for both asks for it
-- with coalesce.

CREATE TABLE width AS
    SELECT CAST(regexp_extract(subtoken_version(), 'dim (\d+)', 1) AS BIGINT) AS n;

SELECT must('subtoken_embed(NULL) is NULL', subtoken_embed(NULL) IS NULL);

SELECT must('the empty string is a full-width vector',
    len(subtoken_embed('')) = (SELECT n FROM width));
SELECT must('the empty string is the zero vector',
    list_min(subtoken_embed('')) = 0 AND list_max(subtoken_embed('')) = 0);

SELECT must('whitespace only is the zero vector',
    list_min(subtoken_embed('   ')) = 0 AND list_max(subtoken_embed('   ')) = 0
    AND len(subtoken_embed('   ')) = (SELECT n FROM width));

-- Runic and alchemical characters: outside this vocabulary, so the tokenizer
-- produces only unknown tokens and the mean is over nothing.
SELECT must('text of tokens outside the vocabulary is the zero vector',
    list_min(subtoken_embed(chr(5792) || chr(5794) || chr(5798))) = 0
    AND list_max(subtoken_embed(chr(5792) || chr(5794) || chr(5798))) = 0
    AND len(subtoken_embed(chr(5792) || chr(5794) || chr(5798))) = (SELECT n FROM width));

SELECT must('ordinary text is not the zero vector',
    list_max(subtoken_embed('a foundry casting valve bodies')) > 0);

SELECT must('coalesce is how a caller asks for the zero vector on a missing value',
    list_min(subtoken_embed(coalesce(CAST(NULL AS VARCHAR), ''))) = 0
    AND list_max(subtoken_embed(coalesce(CAST(NULL AS VARCHAR), ''))) = 0
    AND len(subtoken_embed(coalesce(CAST(NULL AS VARCHAR), ''))) = (SELECT n FROM width));

-- A NULL in the middle of a chunk must not shift the vectors either side of it.
CREATE TABLE mixed AS SELECT * FROM (VALUES
    (1, 'a maker of industrial fasteners'),
    (2, CAST(NULL AS VARCHAR)),
    (3, 'a bonded warehouse operator'),
    (4, ''),
    (5, 'a regional commercial lender')
) AS v(id, s);

CREATE TABLE embedded AS SELECT id, subtoken_embed(s) AS vec FROM mixed;

SELECT must('the NULL row is NULL and its neighbours are untouched',
    (SELECT count(*) FROM embedded WHERE
         (id = 1 AND vec IS DISTINCT FROM subtoken_embed('a maker of industrial fasteners'))
      OR (id = 2 AND vec IS NOT NULL)
      OR (id = 3 AND vec IS DISTINCT FROM subtoken_embed('a bonded warehouse operator'))
      OR (id = 4 AND (list_min(vec) <> 0 OR list_max(vec) <> 0))
      OR (id = 5 AND vec IS DISTINCT FROM subtoken_embed('a regional commercial lender'))
    ) = 0);

SELECT must('every non-NULL row has the full width',
    (SELECT count(*) FROM embedded WHERE vec IS NOT NULL AND len(vec) <> (SELECT n FROM width)) = 0);
