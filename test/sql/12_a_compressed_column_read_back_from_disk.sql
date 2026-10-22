-- Every other test feeds subtoken_embed from an in-memory table, which DuckDB
-- hands the function as a flat vector. A column read back from a database file
-- can arrive in other shapes, and this test reads two of them.
--
-- FSST. A VARCHAR column stored with FSST compression, scanned with
-- enable_fsst_vectors on, reaches the function still compressed: DuckDB's v2
-- C API reports such a vector as VECTOR_TYPE_OTHER and refuses to give a view
-- of it until it is flattened. DuckDB 2.0's default storage format folds plain
-- FSST into DICT_FSST, so the file is attached at storage version v1.2.0, the
-- newest one that still writes plain FSST segments. DuckDB 2.0.0 decompresses
-- an FSST column that holds NULLs into a flat vector, so the column with NULL
-- rows (t) covers NULL handling over a compressed segment, and a column with
-- none (u) is the one that arrives as an FSST vector.
--
-- CONSTANT. An INTEGER column of one repeated value is stored as a constant
-- segment, so its scan, and the cast of it to VARCHAR, is a CONSTANT vector:
-- one physical value that every row reads through a selection vector.

ATTACH 'fsst.duckdb' AS disk (STORAGE_VERSION 'v1.2.0');
SET force_compression='fsst';
SET enable_fsst_vectors=true;

CREATE TABLE disk.t AS
    SELECT CASE WHEN i % 97 = 0 THEN NULL ELSE 'value ' || (i % 300) END AS s
    FROM range(5000) r(i);
CHECKPOINT disk;

SELECT must('the column is stored FSST',
    (SELECT count(*) FROM pragma_storage_info('disk.t') WHERE compression = 'FSST') > 0);

-- The reference is embedded from values built in memory, so it never passes
-- through the compressed scan it is compared against.
CREATE TABLE memory_vectors AS
    SELECT s, subtoken_embed(s) AS memory_vector
    FROM (SELECT DISTINCT 'value ' || (i % 300) AS s FROM range(300) r(i));

CREATE TABLE disk_vectors AS
    SELECT s, subtoken_embed(s) AS disk_vector FROM disk.t;

SELECT must('vectors from disk match vectors from memory',
    (SELECT count(*) FROM disk_vectors d JOIN memory_vectors m ON d.s = m.s
     WHERE d.disk_vector IS DISTINCT FROM m.memory_vector) = 0
    AND (SELECT count(*) FROM disk_vectors d JOIN memory_vectors m ON d.s = m.s) > 0);

-- The join above drops the NULL rows, so they are checked on their own.
SELECT must('the stored column has NULL rows',
    (SELECT count(*) FROM disk.t WHERE s IS NULL) > 0);
SELECT must('NULL rows read from disk stay NULL',
    (SELECT count(*) FROM disk.t WHERE s IS NULL AND subtoken_embed(s) IS NOT NULL) = 0);

CREATE TABLE disk.u AS
    SELECT 'value ' || (i % 300) AS s FROM range(5000) r(i);
CHECKPOINT disk;

SELECT must('the NULL-free column is stored FSST',
    (SELECT count(*) FROM pragma_storage_info('disk.u') WHERE compression = 'FSST') > 0);
SELECT must('the NULL-free column reaches a function as an FSST vector',
    (SELECT count(*) FROM disk.u WHERE vector_type(s) = 'FSST_VECTOR') > 0);

-- Embedded straight off the scan, before any join, so the function is handed
-- the scan's vectors rather than a join's flat output.
CREATE TABLE fsst_vectors AS
    SELECT s, subtoken_embed(s) AS disk_vector FROM disk.u;

SELECT must('vectors from an FSST vector match vectors from memory',
    (SELECT count(*) FROM fsst_vectors d JOIN memory_vectors m ON d.s = m.s
     WHERE d.disk_vector IS DISTINCT FROM m.memory_vector) = 0
    AND (SELECT count(*) FROM fsst_vectors d JOIN memory_vectors m ON d.s = m.s) = 5000);

RESET force_compression;

CREATE TABLE disk.c AS SELECT CAST(42 AS INTEGER) AS c FROM range(3000);
CHECKPOINT disk;

SELECT must('the column is stored as a constant segment',
    (SELECT count(*) FROM pragma_storage_info('disk.c')
     WHERE segment_type = 'INTEGER' AND compression = 'Constant') > 0);

SELECT must('a CONSTANT input gives every row the same vector',
    (SELECT count(DISTINCT subtoken_embed(CAST(c AS VARCHAR))) FROM disk.c) = 1
    AND (SELECT count(*) FROM disk.c
         WHERE subtoken_embed(CAST(c AS VARCHAR)) IS DISTINCT FROM subtoken_embed('42')) = 0);
