-- A stored column of vectors is only comparable with a fresh one if the same
-- model wrote both, and nothing about the column says which model did: the
-- width is the same, the norm is the same, and the numbers are plausible. This
-- file is the SQL-level answer — `subtoken_model_id()` for the identity,
-- `subtoken_models()` for the rest of the row, and a predicate that detects a
-- mismatch without reading a single vector.
--
-- The Protocol side of the estate already refuses a model mismatch and names
-- both sides. Until these two functions the SQL session could not detect one,
-- because the only route to the key was twelve hex characters inside a
-- free-text version sentence, and nothing can join on a sentence.

-- ── the identity ─────────────────────────────────────────────────────────────

SELECT must('subtoken_model_id is registered as a scalar returning VARCHAR',
    (SELECT count(*) FROM duckdb_functions()
     WHERE function_name = 'subtoken_model_id'
       AND function_type = 'scalar'
       AND return_type = 'VARCHAR') = 1);

SELECT must('subtoken_model_id() is 64 characters of lowercase hex',
    regexp_matches(subtoken_model_id(), '^[0-9a-f]{64}$'));

-- The same value the version sentence abbreviates. Recovered here the way
-- arcform recovers it, so this asserts the tie an external caller depends on
-- rather than asserting it about two calls made side by side.
SELECT must('the first twelve characters are the key subtoken_version() reports',
    regexp_extract(subtoken_version(), 'key ([0-9a-f]{12})', 1)
    = substr(subtoken_model_id(), 1, 12));

SELECT must('and the sentence carries only twelve, which is why this function exists',
    NOT regexp_matches(subtoken_version(), 'key [0-9a-f]{13}'));

-- The ledger. This is the key the bundled tokenizer, weights and config derive
-- under the domain `subtoken/model-key/v1`, written down so that a build whose
-- assets differ cannot report the same identity and pass.
--
-- It is the ONE line in this repository that a model change must move by hand,
-- and that is the point: `scripts/mutation_check.py` alters each of the three
-- bundled assets in turn, rebuilds, and requires this file to redden. Without a
-- value pinned here, changing an asset moves `subtoken_model_id()`,
-- `subtoken_models().key` and the version sentence together, every comparison
-- between them still holds, and the whole file stays green over a model nobody
-- intended to ship.
SELECT must('subtoken_model_id() is the key the three bundled assets derive',
    subtoken_model_id()
    = '547fcefc8d6e0e782976680aeef78c8e68f2f29221a0e7876c7e00b4bc2aaad6');

-- ── the catalogue row ────────────────────────────────────────────────────────

SELECT must('subtoken_models() returns exactly one row',
    (SELECT count(*) FROM (SELECT subtoken_models() AS m)) = 1);

-- In order, not sorted: the field order is part of the STRUCT's signature, so a
-- row whose fields were reshuffled would still satisfy a sorted comparison while
-- breaking every positional consumer of it.
SELECT must('and the row has the eight documented fields, in the documented order',
    json_keys(to_json(subtoken_models()))
    = ['model', 'backend', 'revision', 'width', 'input_limit', 'licence', 'tier', 'key']);

SELECT must('the row names the bundled model and its backend',
    (subtoken_models()).model = 'minishlab/potion-base-8M'
    AND (subtoken_models()).backend = 'model2vec');

-- The full SHA, not the twelve characters the version sentence shows. A
-- revision an analyst cannot paste into a URL is a revision they have to
-- reconstruct.
SELECT must('the revision is the whole 40-character SHA',
    regexp_matches((subtoken_models()).revision, '^[0-9a-f]{40}$')
    AND starts_with((subtoken_models()).revision,
                    regexp_extract(subtoken_version(), '@([0-9a-f]{12})', 1)));

SELECT must('the licence is the one SOURCE.md records, and the tier is supported',
    (subtoken_models()).licence = 'MIT' AND (subtoken_models()).tier = 'supported');

-- AC2's three equalities, each against what the extension DOES rather than
-- against another field of the same row.

SELECT must('the row key is the id, so a stored id joins against the catalogue',
    (subtoken_models()).key = subtoken_model_id());

SELECT must('the row width is the width subtoken_embed actually returns',
    (subtoken_models()).width = len(subtoken_embed('a foundry in Rotherham')));

-- `ok ` is one token in this vocabulary and three characters, so a probe built
-- from it stays far under the character cut and pins the TOKEN boundary alone —
-- the same construction, and the same reason, as test/sql/10. `marker` is one
-- distinct trailing token, so the probe at the limit and the probe past it
-- differ in content rather than only in length.
CREATE TABLE limit_probe AS
    SELECT
        (subtoken_models()).input_limit AS input_limit,
        repeat('ok ', ((subtoken_models()).input_limit - 1)::BIGINT) || 'marker' AS at_limit,
        repeat('ok ', ((subtoken_models()).input_limit)::BIGINT) || 'marker' AS past_limit;

SELECT must('at input_limit tokens nothing has been dropped',
    (SELECT subtoken_is_truncated(at_limit) FROM limit_probe) = false);

SELECT must('one token past input_limit, subtoken_is_truncated says so',
    (SELECT subtoken_is_truncated(past_limit) FROM limit_probe) = true);

-- ── the refusal, before anything is compared ─────────────────────────────────

CREATE TABLE corpus AS SELECT * FROM (VALUES
    (1, 'a manufacturer of industrial fasteners in Sheffield'),
    (2, 'a foundry in Rotherham'),
    (3, 'a supplier of hydraulic seals')
) AS t(id, descr);

-- What an analyst writes. The id is stored once, beside the vectors.
CREATE TABLE stored AS
    SELECT id, subtoken_embed(descr) AS v, subtoken_model_id() AS model_id
    FROM corpus;

SELECT must('a column written by this build matches this build',
    (SELECT count(*) FROM stored WHERE model_id <> subtoken_model_id()) = 0);

-- The upgrade, simulated the only way one session can: the stored id is
-- deliberately altered, as it would be had another model written the rows.
UPDATE stored SET model_id = repeat('0', 64) WHERE id IN (2, 3);

SELECT must('the mismatch is a plain equality, and it counts the affected rows',
    (SELECT count(*) FROM stored WHERE model_id <> subtoken_model_id()) = 2);

SELECT must('the rows that do match are still identified',
    (SELECT list(id) FROM stored WHERE model_id = subtoken_model_id()) = [1]);

-- "Before any vector is compared" is the load-bearing word, so it is asserted
-- rather than described: the same predicate over a table whose vector column is
-- entirely NULL still finds the mismatch. Nothing it reads is a vector.
CREATE TABLE vectors_unreadable AS
    SELECT id, NULL::FLOAT[] AS v, model_id FROM stored;

SELECT must('the guard holds with no vector to read at all',
    (SELECT count(*) FROM vectors_unreadable WHERE model_id <> subtoken_model_id()) = 2);

-- And the refusal a session raises on, defined once: NULL when the stored
-- vectors are comparable with this build, and the message to refuse with when
-- they are not. A session writes
--
--     CASE WHEN refusal(model_id) IS NULL THEN v ELSE error(refusal(model_id)) END
--
-- and this file reads the same macro rather than a copy of its condition.
-- `try(error(...))` is not available — DuckDB refuses TRY over a volatile
-- function, and `error` is one — so a macro that raised could not be asserted
-- about here at all: the first mismatched row would end the file.
CREATE OR REPLACE MACRO refusal(stored_id) AS
    CASE WHEN stored_id = subtoken_model_id() THEN NULL
         ELSE 'these vectors were written by model ' || stored_id
              || ', this build serves ' || subtoken_model_id() END;

SELECT must('a matching id does not refuse',
    (SELECT refusal(model_id) FROM stored WHERE id = 1) IS NULL);

SELECT must('a mismatched id refuses, and the message names both models',
    (SELECT refusal(model_id) FROM stored WHERE id = 2)
    = 'these vectors were written by model ' || repeat('0', 64)
      || ', this build serves ' || subtoken_model_id());

SELECT must('the refusal reads the id column alone: it holds with no vector present',
    (SELECT count(*) FROM vectors_unreadable WHERE refusal(model_id) IS NOT NULL) = 2);
