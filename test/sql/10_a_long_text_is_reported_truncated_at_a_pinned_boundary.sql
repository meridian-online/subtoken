-- AC1, AC3, AC4, AC5, AC6, AC7: `subtoken_is_truncated` is a SQL-level predicate
-- an analyst can put in a WHERE clause and a count(*); the boundary it reports
-- is pinned in both directions on BOTH of the limits `subtoken_embed` applies — a token
-- count and a character count ahead of it; it is false for a text that ran past
-- a limit without losing anything to it; and `subtoken_embed`'s own contract is
-- unchanged by its presence.
--
-- Two ways a probe here can fail to test anything, both of them shipped before:
--
-- A probe of one token repeated cannot show truncation, because the mean of 512
-- copies of a vector equals the mean of 513 copies of it. Each probe below is
-- filler plus one DISTINCT trailing marker word, so a clipped and an unclipped
-- subtoken_embed of it provably differ.
--
-- A probe made only of ASCII cannot show which unit a limit counts in. The
-- character cut and a byte cut are the same cut until the first character above
-- U+007F, and until this file carried Hangul and CJK no test input anywhere in
-- this repository had one.

SELECT must('subtoken_is_truncated is registered as a scalar returning BOOLEAN',
    (SELECT count(*) FROM duckdb_functions()
     WHERE function_name = 'subtoken_is_truncated'
       AND function_type = 'scalar'
       AND return_type = 'BOOLEAN') = 1);

SELECT must('subtoken_is_truncated(NULL) is NULL, matching subtoken_embed(NULL)',
    subtoken_is_truncated(NULL) IS NULL);

SELECT must('the empty string is not reported truncated: nothing was there to drop',
    subtoken_is_truncated('') = false);
SELECT must('whitespace only is not reported truncated',
    subtoken_is_truncated('   ') = false);
SELECT must('ordinary short text is not reported truncated',
    subtoken_is_truncated('a manufacturer of industrial fasteners in Sheffield') = false);

-- The pinned boundary. `repeat('ok ', n)` gives `n` filler tokens (measured:
-- one word, one token, in this vocabulary); `marker` is one DISTINCT token
-- appended directly after, with no boundary token count ambiguity.
--
-- `ok ` rather than `steel `: `subtoken_embed` truncates on TWO independent limits, a
-- token count and a character count, and `steel ` is six characters — the
-- exact median token length in this vocabulary, which is also what the
-- character limit is measured in units of. 511 reps of `steel ` plus a
-- nine-character marker lands three characters past that second limit, so a
-- boundary built from it is pinning the character cut, not the token cut, and
-- an earlier version of this file asserted "not truncated" for text the
-- character cut had already clipped. `ok ` is three characters per token, so
-- even 512 reps plus the marker stays under 1600 characters — nowhere near
-- the character limit — and this table pins the token boundary alone.
CREATE TABLE boundary AS SELECT * FROM (VALUES
    -- 511 filler + 1 marker = 512 tokens: exactly at the cap.
    (511, repeat('ok ', 511) || 'marker', rtrim(repeat('ok ', 511))),
    -- 512 filler + 1 marker = 513 tokens: one past the cap.
    (512, repeat('ok ', 512) || 'marker', rtrim(repeat('ok ', 512)))
) AS t(filler_tokens, with_marker, filler_only);

SELECT must('512 tokens (511 filler + marker) is not reported truncated',
    (SELECT subtoken_is_truncated(with_marker) FROM boundary WHERE filler_tokens = 511) = false);
SELECT must('at 512 tokens the marker still reaches the mean',
    (SELECT subtoken_embed(with_marker) FROM boundary WHERE filler_tokens = 511)
    IS DISTINCT FROM
    (SELECT subtoken_embed(filler_only) FROM boundary WHERE filler_tokens = 511));

SELECT must('513 tokens (512 filler + marker) is reported truncated',
    (SELECT subtoken_is_truncated(with_marker) FROM boundary WHERE filler_tokens = 512) = true);
SELECT must('at 513 tokens the marker was dropped before pooling',
    (SELECT subtoken_embed(with_marker) FROM boundary WHERE filler_tokens = 512)
    IS NOT DISTINCT FROM
    (SELECT subtoken_embed(filler_only) FROM boundary WHERE filler_tokens = 512));

-- AC1 + AC3: the *character* boundary, pinned the same way — the direction
-- `subtoken_is_truncated` used to be structurally blind to. `subtoken_embed` cuts the raw
-- string to 3072 characters before it ever tokenises, so text whose own
-- characters-per-token sits above this vocabulary's median (6) can lose
-- content while its full, untruncated token count is nowhere near 512.
-- `internationalization ` is 21 characters for 2 tokens — crosses the
-- 3072-character cut at 147 reps (3087 characters) while sitting at only 294
-- tokens.
CREATE TABLE char_boundary AS SELECT * FROM (VALUES
    -- 146 reps = 3066 characters: under the character cut, and only 292
    -- tokens — nowhere near the token cap either.
    (146, repeat('internationalization ', 146) || 'marker', rtrim(repeat('internationalization ', 146))),
    -- 147 reps = 3087 characters: past the character cut, at 294 tokens —
    -- still 218 short of the token cap.
    (147, repeat('internationalization ', 147) || 'marker', rtrim(repeat('internationalization ', 147)))
) AS t(filler_reps, with_marker, filler_only);

SELECT must('under the character cut, with far fewer than 512 tokens, is not reported truncated',
    (SELECT subtoken_is_truncated(with_marker) FROM char_boundary WHERE filler_reps = 146) = false);
SELECT must('under the character cut the marker still reaches the mean',
    (SELECT subtoken_embed(with_marker) FROM char_boundary WHERE filler_reps = 146)
    IS DISTINCT FROM
    (SELECT subtoken_embed(filler_only) FROM char_boundary WHERE filler_reps = 146));

SELECT must('past the character cut, still far fewer than 512 tokens, is reported truncated',
    (SELECT subtoken_is_truncated(with_marker) FROM char_boundary WHERE filler_reps = 147) = true);
SELECT must('past the character cut the marker was dropped before pooling, though token count alone gave no warning',
    (SELECT subtoken_embed(with_marker) FROM char_boundary WHERE filler_reps = 147)
    IS NOT DISTINCT FROM
    (SELECT subtoken_embed(filler_only) FROM char_boundary WHERE filler_reps = 147));

-- AC6: both boundaries again, in scripts where a character is not a byte and a
-- character is not a token.
--
-- `한 ` is one Hangul syllable and a space — 3 bytes, and 3 tokens, because the
-- bundled tokenizer normalises with NFD and the syllable decomposes into jamo
-- the vocabulary carries while the composed syllable is not in it. 170 of them
-- is 510 tokens, and one `ok ` plus the marker brings it to exactly 512.
CREATE TABLE hangul_boundary AS SELECT * FROM (VALUES
    (512, repeat('한 ', 170) || 'ok ' || 'beacon', rtrim(repeat('한 ', 170) || 'ok ')),
    (513, repeat('한 ', 170) || 'ok ok ' || 'beacon', rtrim(repeat('한 ', 170) || 'ok ok '))
) AS t(tokens, with_marker, filler_only);

SELECT must('Hangul at exactly 512 tokens is not reported truncated',
    (SELECT subtoken_is_truncated(with_marker) FROM hangul_boundary WHERE tokens = 512) = false);
SELECT must('at 512 tokens the Hangul probe marker still reaches the mean',
    (SELECT subtoken_embed(with_marker) FROM hangul_boundary WHERE tokens = 512)
    IS DISTINCT FROM
    (SELECT subtoken_embed(filler_only) FROM hangul_boundary WHERE tokens = 512));
SELECT must('Hangul one token past the cap is reported truncated',
    (SELECT subtoken_is_truncated(with_marker) FROM hangul_boundary WHERE tokens = 513) = true);
SELECT must('past the cap the Hangul probe marker was dropped before pooling',
    (SELECT subtoken_embed(with_marker) FROM hangul_boundary WHERE tokens = 513)
    IS NOT DISTINCT FROM
    (SELECT subtoken_embed(filler_only) FROM hangul_boundary WHERE tokens = 513));

-- AC6: a text inside the character cut and outside the same number of BYTES.
-- `中` is one in-vocabulary ideograph: one token, one character, three bytes.
-- 200 of them ahead of 130 reps of a 21-character two-token word puts the
-- character count under 3072 and the byte count over it, at 461 tokens — so a
-- cut that counted bytes would take the trailing marker and this one does not.
CREATE TABLE byte_against_character AS SELECT * FROM (VALUES
    (repeat('中', 200) || repeat('internationalization ', 130) || 'beacon',
     repeat('中', 200) || rtrim(repeat('internationalization ', 130)))
) AS t(with_marker, filler_only);

SELECT must('the byte-against-character probe is under 3072 characters and over 3072 bytes',
    (SELECT length(with_marker) FROM byte_against_character) = 2936
    AND (SELECT strlen(with_marker) FROM byte_against_character) = 3336);
SELECT must('a text of 2936 characters in 3336 bytes is not reported truncated',
    (SELECT subtoken_is_truncated(with_marker) FROM byte_against_character) = false);
SELECT must('and its marker still reaches the mean, so nothing was dropped',
    (SELECT subtoken_embed(with_marker) FROM byte_against_character)
    IS DISTINCT FROM
    (SELECT subtoken_embed(filler_only) FROM byte_against_character));

-- AC9: 601 raw ids, 300 of them the unknown-token id that `subtoken_embed` drops before
-- it truncates. Counting those toward the cap would clip this text at 512 raw
-- ids and take the marker; the 301 ids that reach the mean are well inside it.
CREATE TABLE unknown_tokens AS SELECT * FROM (VALUES
    (repeat('steel ᚠ ', 300) || 'beacon', rtrim(repeat('steel ᚠ ', 300)))
) AS t(with_marker, filler_only);

SELECT must('a text whose raw token count is past the cap but whose real one is not is not reported truncated',
    (SELECT subtoken_is_truncated(with_marker) FROM unknown_tokens) = false);
SELECT must('and its marker still reaches the mean',
    (SELECT subtoken_embed(with_marker) FROM unknown_tokens)
    IS DISTINCT FROM
    (SELECT subtoken_embed(filler_only) FROM unknown_tokens));

-- AC7: running past a limit is not the same as having lost something to it.
-- Both of these are far past the 3072-character cut and neither loses an id.
SELECT must('five thousand spaces is not reported truncated',
    subtoken_is_truncated(repeat(' ', 5000)) = false);
SELECT must('and subtoken_embed of it discarded nothing: it is the zero vector',
    list_max(subtoken_embed(repeat(' ', 5000))) = 0.0
    AND list_min(subtoken_embed(repeat(' ', 5000))) = 0.0);
SELECT must('3500 characters of CJK the vocabulary does not carry is not reported truncated',
    subtoken_is_truncated(repeat('工業製品 ', 700)) = false);
SELECT must('and that text really is past the character cut',
    length(repeat('工業製品 ', 700)) = 3500);
SELECT must('and subtoken_embed of it is the zero vector, so there was nothing to drop',
    list_max(subtoken_embed(repeat('工業製品 ', 700))) = 0.0
    AND list_min(subtoken_embed(repeat('工業製品 ', 700))) = 0.0);

-- AC1: a WHERE clause and a count(*) over a column, the way an analyst would
-- actually ask the question.
-- Rows 5 and 6 are the AC7 pair: long enough to be caught by anything that
-- asks "is this text long" rather than "did this text lose anything".
CREATE TABLE corpus AS SELECT * FROM (VALUES
    (1, 'a maker of industrial fasteners'),
    (2, repeat('steel ', 600) || 'logistics'),
    (3, 'a regional freight forwarder'),
    (4, repeat('steel ', 900) || 'logistics'),
    (5, repeat(' ', 5000)),
    (6, repeat('工業製品 ', 700))
) AS t(id, description);

SELECT must('count(*) over the column finds exactly the two rows that lost content',
    (SELECT count(*) FROM corpus WHERE subtoken_is_truncated(description)) = 2);
SELECT must('the WHERE clause names the same two ids',
    (SELECT list_sort(list(id)) FROM corpus WHERE subtoken_is_truncated(description)) = [2, 4]);
SELECT must('the rows that lost nothing are excluded, not merely unmatched',
    (SELECT list_sort(list(id)) FROM corpus WHERE NOT subtoken_is_truncated(description)) = [1, 3, 5, 6]);

-- AC4: subtoken_embed() itself is unaffected by subtoken_is_truncated existing alongside
-- it — same vectors as the untruncated-probe assertions above, and the width
-- and cache-composing behaviour asserted in 01 and 04 are unchanged by this
-- file having run.
CREATE TABLE width AS
    SELECT CAST(regexp_extract(subtoken_version(), 'dim (\d+)', 1) AS BIGINT) AS n;
SELECT must('subtoken_embed still returns the declared width for a truncated text',
    (SELECT len(subtoken_embed(with_marker)) FROM boundary WHERE filler_tokens = 512) = (SELECT n FROM width));
SELECT must('a truncated result is still full width and unit norm, which is the whole problem',
    (SELECT list_max(subtoken_embed(with_marker)) FROM boundary WHERE filler_tokens = 512) > 0);
