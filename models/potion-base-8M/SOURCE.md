# Bundled model

`minishlab/potion-base-8M`, a Model2Vec static embedding model, taken from the published Hugging Face release and embedded in the extension binary at build time.

**Licence: declared MIT, not reproduced.** `MODEL_CARD.md` is the upstream README at the pinned revision; its frontmatter says `license: mit` and its citation repeats it. The upstream repository has no `LICENSE` file at that revision — the files it publishes are `.gitattributes`, `README.md`, `config.json`, `model.safetensors`, `modules.json`, `onnx/model.onnx`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json` and `vocab.txt` — so there is no MIT text or copyright line to carry alongside the weights, and none is claimed here.

Source: `https://huggingface.co/minishlab/potion-base-8M`
Revision: `bf8b056651a2c21b8d2565580b8569da283cab23`

Three files are bundled — the weights, the tokenizer, and the config. The config is here because `model2vec_rs::model::StaticModel::from_bytes` reads the model's `normalize` flag from it; taking it from the release keeps that flag the model's own value rather than one this repo asserts.

| file | sha256 |
|---|---|
| `model.safetensors` | `f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2` |
| `tokenizer.json` | `e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50` |
| `config.json` | `2a6ac0e9aaa356a68a5688070db78fc3a464fefe85d2f06a1905ce3718687553` |

`cargo test -p subtoken-core bundled_asset_digests_match_the_pinned_release` recomputes all three and compares them against the constants in `crates/subtoken-core/src/model.rs`, so a swapped or truncated asset reddens rather than being silently embedded.

**Those constants were checked against the publisher's own hashes, not only against the files as downloaded.** Hugging Face serves the SHA-256 of an LFS object in the `x-linked-etag` header and the git blob SHA-1 of a small file in `etag`, so all three can be confirmed without trusting the download:

| file | upstream says | how to reproduce |
|---|---|---|
| `model.safetensors` | `x-linked-etag: f65d0f32…23de2` (SHA-256) | `curl -sI .../model.safetensors` against `shasum -a 256` |
| `tokenizer.json` | `etag: 3e511f68ccf95c33b9ffd214a94c6d25bdb3034f` (git blob) | `git hash-object tokenizer.json` |
| `config.json` | `etag: 7df26884a1aaaefbd7a30b37c32a25477cfb4c0e` (git blob) | `git hash-object config.json` |

All three matched on 2026-08-24. This is not in the test suite because it needs the network, and the point of the bundle is that nothing at build or run time does.

## Why this model, and not one of the two 32M candidates

The map-fidelity run in [meridian-online/finetype](https://github.com/meridian-online/finetype) — `eval/static-embedding-map-fidelity/`, at commit `196d102a` — measured four Model2Vec arms against `all-MiniLM-L6-v2` and against two floors, a random-vector control and DuckDB's `fts` extension scoring BM25 with no model loaded. Two of those arms are candidates that could have been bundled here instead: `minishlab/potion-base-32M`, and `minishlab/potion-retrieval-32M`, which is the same base fine-tuned on a retrieval objective.

**The measurement did not hand either of them the decision, because they split.** On the 216-row column-name corpus — the shape of text closest to what a database column holds — the two candidates disagree three ways, and the disagreement runs straight down the task the reader cares about:

| figure, column-name corpus, 216 rows | `potion-base-32M` | `potion-retrieval-32M` | which it picks |
|---|---|---|---|
| ranked lift over the random control | 0.9033 | 0.9202 | retrieval-32M |
| region structure kept, against MiniLM's map | 0.8865 | 0.8296 | base-32M |
| pairwise near-duplicate average precision | 0.9200 | 0.9024 | base-32M |

The bundled `potion-base-8M` is behind both candidates on ranked lift (0.8827 vs 0.9033 and 0.9202), sits between them on region structure (0.8750 between 0.8865 and 0.8296), and sits between them on pairwise precision (0.9113 between 0.9200 and 0.9024); the whole ladder from it to `potion-retrieval-32M` moves ranked lift on this corpus by 0.0375. The page publishes the pairwise and region figures as well as the ranked one, so on the task order this repository publishes, `potion-base-32M` is the better of the two candidates and `potion-retrieval-32M` is the better one for ranked retrieval alone, which this extension deliberately does not serve.

**What decided it is the size of the download, and that is not close.** The bundled `model.safetensors` is 30,236,760 bytes. The same file is 129,210,456 bytes for both 32M candidates — 4.27 times larger — so bundling either would take the weights alone from 30.2 MB to 129.2 MB. For comparison, the largest artifact the DuckDB community registry serves for v1.5.5 on `osx_arm64` is `ldbc_data_gen` at 37,608,828 bytes, and the next below it is `pic2vec` at 27,443,851 bytes. This extension packaged and gzipped is 30,410,741 bytes, which already sits between those two; a 32M model would put it at more than three times the largest thing the registry carries, in exchange for 0.0375 of ranked lift on the corpus this extension is most used on.

**How and when those sizes were read.** On 2026-09-15.

- The packaged artifact: `gzip -c build/subtoken.duckdb_extension | wc -c` over a local release build, 30,410,741 bytes (30,399,166 at `gzip -9`).
- The registry artifacts: the `Content-Length` of a `HEAD` request to `https://community-extensions.duckdb.org/v1.5.5/osx_arm64/<name>.duckdb_extension.gz`, for the 339 extension directories in `duckdb/community-extensions`, of which 263 answered 200 for that version and platform.
- The model weights: the `Content-Length` of a `HEAD` request, following redirects, to `https://huggingface.co/minishlab/<model>/resolve/main/model.safetensors`.

None of the three is in the test suite, for the same reason as the checksum table above: they need the network, and the point of the bundle is that neither the build nor a query does.
