-- duckdb_functions() is where a user meets a function before calling it, so
-- each of the seven has to say what it does there: a description, an example
-- and a category, and its real parameter names rather than DuckDB's col0 and
-- col1 placeholders.
--
-- scripts/check_documented_surface.py checks that these texts equal the table
-- in lib.rs and that every example runs. This file checks that they are there.

CREATE TABLE surface AS
    SELECT function_name, description, examples, categories, parameters
    FROM duckdb_functions()
    WHERE function_name LIKE 'subtoken_%';

SELECT must('subtoken_embed has a description',
    (SELECT length(description) > 0 FROM surface WHERE function_name = 'subtoken_embed'));
SELECT must('subtoken_embed has an example',
    (SELECT len(examples) >= 1 FROM surface WHERE function_name = 'subtoken_embed'));
SELECT must('subtoken_embed has a category',
    (SELECT len(list_filter(categories, c -> length(c) > 0)) >= 1 FROM surface WHERE function_name = 'subtoken_embed'));
SELECT must('subtoken_embed shows its real parameters',
    (SELECT parameters = ['text'] FROM surface WHERE function_name = 'subtoken_embed'));

SELECT must('subtoken_is_truncated has a description',
    (SELECT length(description) > 0 FROM surface WHERE function_name = 'subtoken_is_truncated'));
SELECT must('subtoken_is_truncated has an example',
    (SELECT len(examples) >= 1 FROM surface WHERE function_name = 'subtoken_is_truncated'));
SELECT must('subtoken_is_truncated has a category',
    (SELECT len(list_filter(categories, c -> length(c) > 0)) >= 1 FROM surface WHERE function_name = 'subtoken_is_truncated'));
SELECT must('subtoken_is_truncated shows its real parameters',
    (SELECT parameters = ['text'] FROM surface WHERE function_name = 'subtoken_is_truncated'));

SELECT must('subtoken_version has a description',
    (SELECT length(description) > 0 FROM surface WHERE function_name = 'subtoken_version'));
SELECT must('subtoken_version has an example',
    (SELECT len(examples) >= 1 FROM surface WHERE function_name = 'subtoken_version'));
SELECT must('subtoken_version has a category',
    (SELECT len(list_filter(categories, c -> length(c) > 0)) >= 1 FROM surface WHERE function_name = 'subtoken_version'));
SELECT must('subtoken_version shows its real parameters',
    (SELECT len(parameters) = 0 FROM surface WHERE function_name = 'subtoken_version'));

SELECT must('subtoken_model_id has a description',
    (SELECT length(description) > 0 FROM surface WHERE function_name = 'subtoken_model_id'));
SELECT must('subtoken_model_id has an example',
    (SELECT len(examples) >= 1 FROM surface WHERE function_name = 'subtoken_model_id'));
SELECT must('subtoken_model_id has a category',
    (SELECT len(list_filter(categories, c -> length(c) > 0)) >= 1 FROM surface WHERE function_name = 'subtoken_model_id'));
SELECT must('subtoken_model_id shows its real parameters',
    (SELECT len(parameters) = 0 FROM surface WHERE function_name = 'subtoken_model_id'));

SELECT must('subtoken_models has a description',
    (SELECT length(description) > 0 FROM surface WHERE function_name = 'subtoken_models'));
SELECT must('subtoken_models has an example',
    (SELECT len(examples) >= 1 FROM surface WHERE function_name = 'subtoken_models'));
SELECT must('subtoken_models has a category',
    (SELECT len(list_filter(categories, c -> length(c) > 0)) >= 1 FROM surface WHERE function_name = 'subtoken_models'));
SELECT must('subtoken_models shows its real parameters',
    (SELECT len(parameters) = 0 FROM surface WHERE function_name = 'subtoken_models'));

SELECT must('subtoken_cache_stats has a description',
    (SELECT length(description) > 0 FROM surface WHERE function_name = 'subtoken_cache_stats'));
SELECT must('subtoken_cache_stats has an example',
    (SELECT len(examples) >= 1 FROM surface WHERE function_name = 'subtoken_cache_stats'));
SELECT must('subtoken_cache_stats has a category',
    (SELECT len(list_filter(categories, c -> length(c) > 0)) >= 1 FROM surface WHERE function_name = 'subtoken_cache_stats'));
SELECT must('subtoken_cache_stats shows its real parameters',
    (SELECT len(parameters) = 0 FROM surface WHERE function_name = 'subtoken_cache_stats'));

SELECT must('subtoken_cache_clear has a description',
    (SELECT length(description) > 0 FROM surface WHERE function_name = 'subtoken_cache_clear'));
SELECT must('subtoken_cache_clear has an example',
    (SELECT len(examples) >= 1 FROM surface WHERE function_name = 'subtoken_cache_clear'));
SELECT must('subtoken_cache_clear has a category',
    (SELECT len(list_filter(categories, c -> length(c) > 0)) >= 1 FROM surface WHERE function_name = 'subtoken_cache_clear'));
SELECT must('subtoken_cache_clear shows its real parameters',
    (SELECT len(parameters) = 0 FROM surface WHERE function_name = 'subtoken_cache_clear'));

SELECT must('there are seven functions to describe',
    (SELECT count(*) FROM surface) = 7);

SELECT must('no function shows a colN parameter',
    (SELECT count(*) FROM duckdb_functions(), unnest(parameters) p(name)
     WHERE function_name LIKE 'subtoken_%' AND regexp_matches(name, '^col[0-9]+$')) = 0);
