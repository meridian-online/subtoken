-- AC3, from this side of the boundary: subtoken is its own extension with
-- its own surface. finetype is a different repo and a different artifact, and
-- nothing here uses its prefix or its type contract.
--
-- The surface is derived, not asserted from memory: the prelude snapshots
-- duckdb_functions() before LOAD, so the delta below is exactly what this
-- extension registered. Adding an eighth function reddens this file.

CREATE TABLE registered AS
    SELECT DISTINCT function_name FROM duckdb_functions()
    WHERE function_name NOT IN (SELECT function_name FROM subtoken_baseline_functions)
      AND function_name <> 'must';

-- No nearest-neighbour lookup, deliberately. The measured position of this
-- model is that a map built from its vectors keeps the cluster structure and
-- loses the neighbourhoods, which README's "What it is good at, and what it is
-- not" states with the figures behind it. A
-- function that invited "show me the rows most like this one" would promise
-- what the model does not deliver, so the absence is asserted here rather than
-- left to whoever adds the next function to remember.
--
-- It runs BEFORE the count and the name list, and that ordering is the
-- assertion rather than a tidiness: `must` raises, which ends the file, so
-- whichever assertion comes first is the only one a break is ever shown to.
-- With the count first, registering `subtoken_similar` reddened this file on
-- "the extension registers seven functions" and the guard below was never
-- evaluated — a guard that has never been seen to fire reads exactly like
-- one that cannot.
SELECT must('no similarity or nearest-neighbour function is registered',
    (SELECT count(*) FROM registered
     WHERE regexp_matches(function_name, '(?i)similar|neighbou?r|nearest|knn|distance|match|rank')) = 0);

SELECT must('the extension registers seven functions',
    (SELECT count(*) FROM registered) = 7);

SELECT must('and they are exactly the documented seven',
    (SELECT list_sort(list(function_name)) FROM registered)
    = ['subtoken_cache_clear', 'subtoken_cache_stats', 'subtoken_embed', 'subtoken_is_truncated',
       'subtoken_model_id', 'subtoken_models', 'subtoken_version']);

SELECT must('nothing here takes the ft_ prefix that belongs to another extension',
    (SELECT count(*) FROM registered WHERE starts_with(function_name, 'ft_')) = 0);

SELECT must('every registered function is a scalar',
    (SELECT count(*) FROM duckdb_functions()
     WHERE function_name IN (SELECT function_name FROM registered)
       AND function_type <> 'scalar') = 0);
