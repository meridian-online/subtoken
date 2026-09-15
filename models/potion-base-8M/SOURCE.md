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
