# subtoken

Text becomes tokens, tokens become vectors. Inside DuckDB, with no network, no key and no bill.

The model is a static embedding model compiled into the extension binary, so a query embeds a column with no stops along the way — nothing to configure, nothing to fetch, nothing to meter.

```sql
-- Not published to the community registry yet; *Status* below says what to run
-- in the meantime.
INSTALL subtoken FROM community;
LOAD subtoken;

SELECT description, subtoken_embed(description) AS v
FROM read_parquet('corpus.parquet')
WHERE industry = 'manufacturing'
LIMIT 100;
```

## Why this exists

Every other route to an embedding inside DuckDB is either a transformer forward pass or a call to a hosted provider. Over a large text column the first is slow enough that people leave SQL to do it, and the second is an invoice you find out about afterwards. A static embedding is neither: it is a token lookup and a mean, so it is bounded, local, and free at the margin.

Being a scalar is the point. It composes with `WHERE` and `LIMIT`, so you can embed a filtered subset rather than a whole table, and it behaves the same over a local Parquet file as over a remote one.

## What it is good at, and what it is not

**It is not a drop-in replacement for a hosted transformer, and the difference is measured rather than hedged.** A static embedding is the mean of its token vectors with no contextual attention, and what that costs is specific rather than general. The figures in this section come from the harness, corpora and committed `results.json` in `eval/static-embedding-map-fidelity/` in [meridian-online/finetype](https://github.com/meridian-online/finetype), read at the commit `scripts/check_quality_claims.py` pins, and where a figure compares, it compares the bundled `potion-base-8M` against `all-MiniLM-L6-v2`.

**There is a free floor under this, and it was measured beside the model.** DuckDB's `fts` extension scores BM25 over the same corpora with no model loaded, which is the result a reader already has; a bundled model that does not beat it has not earned its download. The ranked-lift figures below are normalised so that a random-vector control scores zero and `all-MiniLM-L6-v2` scores one on the same corpus.

**Two different questions live under "similarity", and this embedder answers them differently.** *Pairwise judgement* asks how alike two given strings are — is A a duplicate of B. *Ranked retrieval* asks a whole corpus for the rows most like A. The neighbourhood figures under *What it is not for* are ranked retrieval, and they are the weak result on this page; the duplicate-detection figures under *What it is good at* are pairwise, and they point the other way. Word-order blindness is a failure of both — a phrase and its shuffle are false neighbours *and* a false duplicate.

### What it is good at

**Pairwise duplicate scoring, which is the strongest result in the run.** Over the 216 column names in 12 semantic classes, pooled average precision at telling a near-duplicate of a string from an unrelated string is 0.9113 for the bundled model, against 0.7262 for BM25 on the same 216 rows, on a scale whose floor is 0.5 because the pool holds one positive and one negative pair per anchor. Telling apart two strings that merely share a class is the harder question, and there the same corpus gives 0.6862 for the bundled model and 0.6815 for `all-MiniLM-L6-v2`.

**Coarse classification and tagging.** Over those same 216 column names in 12 semantic classes, clustering the raw static vectors recovers more of the label structure than clustering MiniLM's — 0.3924 against 0.3510 by adjusted mutual information, which is how much knowing the clusters tells you about the labels. Read that one as indicative rather than settled, because 216 rows is a small sample; it is also the shape of text a database column usually holds.

**Reading a corpus as regions.** Project these vectors down to two dimensions and the groups you see still line up with the corpus's own labels: 71% of what MiniLM's map recovers on long-form prose, 67% on short text, 88% on very short strings. That is measured against a random-vector control rather than against nothing, so it is a share of the structure a real embedder finds and a fake one does not.

### What it is not for

**Ranked nearest-neighbour lookup — "show me the rows most like this one".** Take a point's 20 nearest neighbours in a map built from these vectors, and the same point's in a map built from MiniLM's: 13% are the same rows on long-form prose, 28% on short text, 40% on very short strings. The regions agree and the neighbourhoods do not. There is deliberately no similarity or nearest-neighbour function in this extension, and this is the reason.

**And a BM25 index you already have scores about as well on long prose.** Over 3,000 posts from 20 Newsgroups, ranked lift is 0.763 for the bundled model and 0.746 for BM25 on the same rows. Over their 3,000 subject lines it is 0.853 for the model against 0.694 for BM25. Over the 216 column names it is 0.883 against 0.588. The shorter and more name-like the text, the more the model is worth; on long-form prose the gap is narrow enough that bundling weights is hard to argue for over the `fts` extension DuckDB already ships.

**And the neighbourhood penalty depends on the shape of your text, in the opposite direction to the usual guess.** It is worst on long prose and mildest on very short strings — 13%, then 28%, then 40% as the text gets shorter. The more context a text carries, the more is lost by not attending to it. So a column of names, codes or identifiers is at the good end for neighbourhoods, and a column of paragraphs is at the bad end.

**Region structure does not run down the same line, and a one-line description is the case to watch.** 71% on long-form prose, 67% on short text, 88% on very short strings — short text is the weakest of the three for regions, not the middle of them. A column of titles, subject lines or one-line descriptions is therefore in the better half for neighbourhoods and at the bottom for regions, so read the two rows of the table below as two separate measurements rather than as one gradient.

| | long-form prose | short text | very short strings |
|---|---|---|---|
| corpus | 20 Newsgroups posts | their subject lines | column names |
| rows | 3,000 | 3,000 | 216 |
| ranked lift, this model | 0.763 | 0.853 | 0.883 |
| ranked lift, BM25 with no model | 0.746 | 0.694 | 0.588 |
| nearest neighbours that survive | 13% | 28% | 40% |
| region structure kept | 71% | 67% | 88% |

**It also does not read word order.** A Model2Vec vector is the mean of its token vectors, so `subtoken_embed('valve bodies')` and `subtoken_embed('bodies valve')` are the same vector — a false duplicate, which is a pairwise failure and not a ranked-retrieval one. Repetition does count — a word twice pulls the mean toward it — but any phrase and its shuffle land in the same place.

Our figures were measured with the harness in `eval/static-embedding-map-fidelity/` against `potion-base-8M` as Hugging Face served it at the time; that harness names the model rather than recording a revision, so the tie between these figures and the weights in this binary is asserted here rather than recorded there. What is enforced is narrower, and worth having: vectors from two model versions are not comparable, so `scripts/check_quality_claims.py` registers the figures above against finetype's committed `results.json` at `196d102a`, and reddens when the bundled revision moves off the one in [`models/potion-base-8M/SOURCE.md`](models/potion-base-8M/SOURCE.md) that these figures were published against, which forces a re-run or a deliberate re-blessing instead of a quiet drift. It holds this section and `description.yml`'s copy of it to the same figures at the same time.

## The SQL surface

Seven functions, and `subtoken_embed` is the one you came for.

| function | returns | what it is for |
|---|---|---|
| `subtoken_embed(text VARCHAR)` | `FLOAT[]` | the vector for one string |
| `subtoken_is_truncated(text VARCHAR)` | `BOOLEAN` | whether `subtoken_embed(text)` had to drop content to fit |
| `subtoken_version()` | `VARCHAR` | which build, which model, which vector width |
| `subtoken_model_id()` | `VARCHAR` | the model key on its own, to store beside a vector column |
| `subtoken_models()` | `STRUCT(model, backend, revision, width, input_limit, licence, tier, "key")` | the catalogue row for the model this build serves |
| `subtoken_cache_stats()` | `STRUCT(hits, misses, encoded, uncached, entries, capacity)` | what the cache has been doing |
| `subtoken_cache_clear()` | `BIGINT` | drop the cached vectors; returns how many |

There is no similarity or nearest-neighbour function, deliberately. See *What it is good at, and what it is not* above.

### Which model wrote a stored vector

Vectors from two models are not comparable, and a stored column will not tell you it has the wrong ones: the width is the same, the norm is the same, and the numbers are plausible. Upgrade the extension over a table you built last quarter and the only signal is that the answers drift.

`subtoken_model_id()` is the value that makes it answerable. It is the full 64-character key this build derives from its own tokenizer, weights and config — the same key `subtoken_version()` abbreviates to twelve characters inside its sentence, at full length and on its own, so it can be stored in a column and compared. Store it once, beside the vectors:

```sql
CREATE TABLE corpus_vectors AS
    SELECT id, subtoken_embed(description) AS v, subtoken_model_id() AS model_id
    FROM corpus;
```

One datum per column: the vector in one, the id of the model that wrote it in the next. Any later session can then refuse **before** it compares anything, in a predicate that reads no vector at all:

```sql
-- Nothing to do: every row here was written by the model now loaded.
SELECT count(*) AS written_by_another_model
FROM corpus_vectors
WHERE model_id <> subtoken_model_id();
```

Take a row from a build carrying different weights, and the refusal is what you get instead of a plausible answer:

```sql
INSERT INTO corpus_vectors
VALUES (5, subtoken_embed('a supplier of hydraulic seals'), 'written-by-another-build');

-- Row 5 comes back NULL: the comparison is not attempted, because the id beside
-- its vector is not this build's.
SELECT id,
       CASE WHEN model_id = subtoken_model_id()
            THEN v = subtoken_embed('a foundry casting valve bodies')
       END AS is_the_same_text
FROM corpus_vectors
ORDER BY id;
```

Because the key is derived from the asset bytes, it moves when the weights, the tokenizer or the config move, and it does not move when only this extension's own code changes — so a release that leaves the model alone leaves your stored ids valid.

`subtoken_models()` is the rest of the row: which model, which backend, at which revision, how wide its vectors are, how much text it reads, under what licence, and at what support tier.

```sql
SELECT (subtoken_models()).model, (subtoken_models()).width, (subtoken_models()).input_limit;
```

`key` is quoted in the type above because DuckDB quotes keywords when it prints a `STRUCT`; a field reference does not need the quotes, and `(subtoken_models()).key` returns the same string as `subtoken_model_id()`.

### NULL, and text with nothing in it

`subtoken_embed(NULL)` is `NULL`. Text that tokenises to nothing — the empty string, whitespace, a string of characters the vocabulary does not carry — is a **zero vector of full width**, because the mean over zero tokens is zero. The two are different on purpose: a missing value is not the same as a value that carries no signal, and only one of them should disappear from a `WHERE ... IS NOT NULL`.

If you want the single behaviour a text pipeline usually gives you, ask for it:

```sql
SELECT subtoken_embed(coalesce(description, '')) FROM corpus;
```

### A long text is truncated before the mean, and nothing about the vector says so

`subtoken_embed` builds its vector from at most **512 tokens** of `text` — roughly the first few hundred words of ordinary English for most prose. Anything past that is dropped before the mean is taken, not down-weighted, and the vector that comes back is full width and unit norm either way. Two rows whose descriptions agree up to the cut and then diverge completely embed to the same place, and nothing about the result tells you that happened.

That 512-token figure is not the whole story for every kind of text. Before it tokenises anything, `subtoken_embed` cuts the raw string to a character count derived from the vocabulary — just over three thousand characters for this model. For text made of unusually long, dense tokens — URLs, camelCase or snake_case identifiers, run-together compound words, anything with few word breaks — that is the cut that bites, and it bites while the token count is still nowhere near 512. Non-Latin scripts move the balance the other way: a line of Korean is two or three tokens per character, so it reaches the token cap in a few hundred. You do not need to reason about which case you are in: `subtoken_is_truncated` answers the question either way, which is the point of asking it instead of counting.

`subtoken_is_truncated(text)` is how you find out before you trust a result — a plain question, not a token count, so you never have to know the limit is 512, or where else it might bite, to ask it:

```sql
SELECT count(*) FROM corpus WHERE subtoken_is_truncated(description);
```

`subtoken_is_truncated(NULL)` is `NULL`, the same as `subtoken_embed(NULL)`, so it drops out of a `WHERE` clause the same way.

What it reports is whether `subtoken_embed` pooled less of the text than the whole of it would have given, and that is not the same question as whether the text was long. Five thousand spaces is `false`. So is a column of characters this vocabulary does not carry, however far past the character cut it runs: the cut took nothing that would have reached the mean. It also costs more than `subtoken_embed` does on a very long value — `subtoken_embed` stops reading at the character cut and this has to look past it to know whether anything was there.

### What counts as the same string

The bundled tokenizer lowercases, strips accents and ignores surrounding whitespace, so `subtoken_embed('Steel')`, `subtoken_embed('steel')` and `subtoken_embed('  steel  ')` are the same vector, and `subtoken_embed('café')` matches `subtoken_embed('cafe' || chr(769))`. You do not have to normalise a column before embedding it. Word order still matters, and different words still give different vectors.

The cache keys on the exact input bytes rather than on the tokenizer's folded form, so those variants do occupy separate cache entries. That is deliberate: reproducing a dependency's normalisation in the cache would mean a tokenizer bump quietly changing which inputs share an entry, and the failure would be a vector returned for text nobody embedded.

### Repeating a query does not re-embed, up to a stated number of distinct values

Vectors are cached against the exact input bytes and a digest of the bundled model's own files, so re-running a query re-embeds nothing — **for as many distinct values as the cache holds**, and a future build with different weights cannot serve you the old ones. `subtoken_cache_stats()` is how you see which case you are in rather than guessing from how long the query took:

```sql
SELECT subtoken_cache_clear();
CREATE TABLE v AS SELECT subtoken_embed(description) FROM corpus;
SELECT subtoken_cache_stats();   -- uncached 0 means the column fitted, and
                                    -- then encoded is one per distinct value
CREATE TABLE w AS SELECT subtoken_embed(description) FROM corpus;
SELECT subtoken_cache_stats();   -- encoded unchanged, if uncached was 0
```

**The bound.** The cache spends a fixed memory budget — 64 MiB — so it holds `capacity` vectors and no more. Once it is full it **stops admitting new values rather than evicting old ones**, and `uncached` counts every lookup it turned away.

`subtoken_cache_stats().capacity` is the only place to read the figure, and there is a reason it is not written down here: it depends on the model's vector width and on how your platform's allocator rounds. The extension measures the second of those at startup by allocating one block and asking, rather than assuming, so the same build lands on different capacities on macOS and Linux. A test fills a cache and asks the allocator how many bytes the process is holding, so the budget is a measurement rather than a promise about arithmetic.

The budget is a ceiling on what the cache **holds**, not on its peak. When the internal map doubles its bucket array it briefly holds the old array alongside the new one, so the high-water mark during a fill can sit above the budget by up to half the final array — under a tenth of the budget, and asserted as such.

That choice is the whole behaviour above the bound, so it is worth being plain about. A column with more distinct values than `capacity` is served for `capacity` of them on a re-run and re-embeds the rest — a hit rate of `capacity / distinct`, holding steady as the column grows. Under any recency-ordered policy instead — LRU, FIFO, or the two generations this extension shipped with first — a repeated scan evicts every value exactly before it is next wanted and the hit rate is **zero**, which is a cliff rather than a slope and is much worse than it sounds.

Above the bound the guarantee is weaker in a second way as well: a value the cache turned away is re-embedded for every row that carries it, and two threads meeting the same turned-away value will each embed it. Below the bound neither happens — one encode per distinct value, whatever the thread count.

What it costs is adaptivity: the values kept are the ones seen first in the session, so if you move on to a different column the cache stays full of the old one. `SELECT subtoken_cache_clear();` empties it, and a non-zero `uncached` is the sign that it is time.

## Building it

Needs a Rust toolchain, Python 3 for the packaging and test scripts, and the `duckdb` CLI for the SQL tests. `make check` also needs PyYAML, because it reads `description.yml` with a parser rather than a regex.

```
make check            # formatting, clippy, Rust tests, the artifact, the SQL tests, the registry entry
make extension        # just the loadable artifact, in build/
make community-check  # the recipe duckdb/community-extensions runs, and the checks on what it built
make mutation-check   # break the code on purpose and require the tests to notice
```

`make check` and `make community-check` are what CI runs. `make mutation-check` is not, because each of its SQL mutations rebuilds the release binary; it applies a table of deliberate defects one at a time and fails unless the test named against each one reddens, which is the difference between a suite that is green and a suite that is watching. CI does run `scripts/mutation_check.py --check-anchors`, which applies no mutation and asserts only that each one still names a line that exists.

`make community-check` is the local form of the registry's own build. It needs `git submodule update --init --recursive` for `extension-ci-tools`, and it creates `configure/`, which is gitignored. `configure/extension_version.txt` must never be tracked: the upstream recipe writes that file only when it is absent, so a tracked copy would never be refreshed and would stamp a stale version onto every published artifact.

The model is compiled into the binary, so the artifact is large — most of it is weights. Loading it needs `duckdb -unsigned` until a signed build exists in the community registry:

```sql
LOAD 'build/subtoken.duckdb_extension';
```


## Status

Early. The extension builds, loads and answers queries; nothing is published to the community registry yet, so the `INSTALL ... FROM community` line at the top of this page does not work today. Build it yourself with `make extension` in the meantime.

`description.yml` at the root of this repository is the registry entry, ready to be copied to `extensions/subtoken/description.yml` in [duckdb/community-extensions](https://github.com/duckdb/community-extensions), and `.github/workflows/MainDistributionPipeline.yml` runs the same build that registry would run. Submitting it is a separate decision and has not been taken.

## The model

`minishlab/potion-base-8M`, a Model2Vec static embedding model, taken from its published release at a pinned revision and compiled into the binary. Its files, their checksums and the revision they came from are in [`models/potion-base-8M/SOURCE.md`](models/potion-base-8M/SOURCE.md), and a test recomputes those checksums so a swapped asset reddens rather than shipping. The quality position above was measured on this model; a different one would need the measurement redone.

That file also records why this model and not a larger one from the same family. The same run measured `potion-base-32M` and `potion-retrieval-32M`, which score better than this one on some of the figures above and worse on others; what decided the bundle is that their weights file is over four times the size of this one, on an artifact that is already among the largest the community registry serves.

## Licence

This repository is MIT; the `LICENSE` file at its root is that licence and covers the code here.

The bundled embedding model is a third-party Model2Vec release. Its publisher **declares** it MIT, in the model card at [`models/potion-base-8M/MODEL_CARD.md`](models/potion-base-8M/MODEL_CARD.md) — the frontmatter and the citation both say so. The upstream repository carries no `LICENSE` file at the pinned revision, so **no MIT text or copyright line for the model is reproduced here**, because there is none to copy. If you need the licence in hand rather than declared, take it up with the publisher before redistributing the weights.

## Credits

Part of the [Meridian](https://meridian.online) project.
