//! The bundled model, and the one place a string becomes a vector.
//!
//! The three asset files are compiled into the binary with `include_bytes!` and
//! handed to `model2vec_rs` as bytes. Nothing here opens a file at runtime and
//! nothing here opens a socket: `StaticModel::from_bytes` is the only loader
//! this crate calls, and the `local-only` feature on `model2vec-rs` compiles the
//! download path out of the dependency entirely.

use std::sync::OnceLock;

use model2vec_rs::model::StaticModel;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

/// The Hugging Face repository the bundled assets came from.
pub const MODEL_ID: &str = "minishlab/potion-base-8M";

/// The exact revision of that repository the bundled assets were taken at.
pub const MODEL_REVISION: &str = "bf8b056651a2c21b8d2565580b8569da283cab23";

/// The family of model that produces the vectors, as the catalogue reports it.
///
/// Not the crate name and not the extension name: it is the answer to "what
/// kind of thing wrote this vector", which is the question a session asks
/// before deciding whether a stored column can be compared with a fresh one.
/// A transformer-backed column and a static-embedding column can carry the same
/// width and the same number of rows and share nothing else.
pub const MODEL_BACKEND: &str = "model2vec";

/// The licence the upstream release declares for the bundled weights.
///
/// `models/potion-base-8M/SOURCE.md` records why no licence text travels with
/// them: the upstream repository publishes none at the pinned revision, so this
/// is the declaration from the model card's frontmatter and nothing more.
pub const MODEL_LICENCE: &str = "MIT";

/// What a caller may expect of this model in this build.
///
/// `supported` is the only value this extension emits today, because it bundles
/// exactly one model and that model is the product. The field exists because
/// the catalogue is the place a second model would arrive, and a row that could
/// not say "this one is experimental" would have to be re-shaped to say it.
pub const MODEL_TIER: &str = "supported";

/// SHA-256 of `models/potion-base-8M/model.safetensors` as published.
pub const WEIGHTS_SHA256: &str = "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2";
/// SHA-256 of `models/potion-base-8M/tokenizer.json` as published.
pub const TOKENIZER_SHA256: &str =
    "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50";
/// SHA-256 of `models/potion-base-8M/config.json` as published.
pub const CONFIG_SHA256: &str = "2a6ac0e9aaa356a68a5688070db78fc3a464fefe85d2f06a1905ce3718687553";

const WEIGHTS: &[u8] = include_bytes!("../../../models/potion-base-8M/model.safetensors");
const TOKENIZER: &[u8] = include_bytes!("../../../models/potion-base-8M/tokenizer.json");
const CONFIG: &[u8] = include_bytes!("../../../models/potion-base-8M/config.json");

/// The token cap [`Model::embed`] hands to `model2vec_rs`: text that tokenises
/// to more in-vocabulary ids than this has the excess dropped before the mean.
/// It is not the only way content is lost — `encode_with_args` also cuts the
/// raw string to `MAX_TOKENS * median_token_length` characters before it
/// tokenises anything, so text made of long, sparse tokens can lose content
/// well before it reaches 512 of them. See [`Truncation`] for both cuts and
/// [`Model::is_truncated`] for how a caller finds out.
///
/// This is `model2vec_rs::StaticModel::encode`'s own default, `Some(512)`,
/// declared here so that moving it moves `embed` as well as the predicate —
/// which `embed_is_byte_identical_to_the_upstream_wrapper_it_replaced` pins in
/// both directions.
pub const MAX_TOKENS: usize = 512;

/// Domain tag mixed into the model key so the digest cannot be confused with a
/// plain SHA-256 of any one asset.
const MODEL_KEY_DOMAIN: &[u8] = b"subtoken/model-key/v1";

/// Derive a model's content address from the three asset files it is made of.
///
/// Taken as arguments rather than read from the embedded constants so that a
/// test can hand it altered bytes and see the key move. A version of this that
/// hashed only some of its arguments would pass any test that recomputed the
/// same fields beside it; it cannot pass one that calls it twice.
pub fn model_key(tokenizer: &[u8], weights: &[u8], config: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(MODEL_KEY_DOMAIN);
    hasher.update(MODEL_ID.as_bytes());
    hasher.update([0u8]);
    hasher.update(MODEL_REVISION.as_bytes());
    hasher.update([0u8]);
    hasher.update(tokenizer);
    hasher.update(weights);
    hasher.update(config);
    hasher.finalize().into()
}

/// A loaded static embedding model.
pub struct Model {
    inner: StaticModel,
    dim: usize,
    key: [u8; 32],
    truncation: Truncation,
}

/// Whether `model2vec_rs` dropped any of a text before pooling it.
///
/// **[`Model::embed`] does not use this type.** It hands the whole text and
/// `Some(MAX_TOKENS)` to `model2vec_rs::StaticModel::encode_with_args` and
/// takes what comes back, which is the call `encode_single` makes — so the
/// vectors are upstream's own and there is no second implementation of the
/// cuts for them to disagree with. Two earlier versions of this file did the
/// cutting here and handed `encode_with_args` a pre-cut string with no limit;
/// the first missed the character cut, and the second reconstructed the token
/// cut as a byte-offset slice of the source text, which is faithful only if
/// token index maps injectively to source span. Under this tokenizer it does
/// not: `BertNormalizer` runs NFD, every Hangul syllable decomposes into two
/// or three jamo tokens reporting one shared span, and a boundary inside such
/// a run kept all of them. `embed` then pooled more tokens than upstream, at a
/// cosine of 0.999994 — a difference that reads as rounding.
///
/// What is left for this type is the question upstream does not answer: given
/// a text, did the pipeline pool fewer ids than the whole text would have
/// given? `encode_with_args` performs two cuts when asked for at most
/// `max_tokens` ids — first a **character** pre-cut of the raw string to
/// `max_tokens * median_token_length` characters, a performance guard so an
/// arbitrarily long text is not tokenised in full only to be thrown away, and
/// then a **token** cut of the tokenised, out-of-vocabulary-filtered result to
/// `max_tokens` ids. Both are private with no accessor, so
/// [`Truncation::truncates`] reproduces them and compares the id list they
/// leave against the id list the whole text gives. It reaches no byte offset
/// and reconstructs no substring: the only thing it ever compares is ids.
///
/// It carries its own [`Tokenizer`], loaded from the identical `TOKENIZER`
/// bytes [`Model`] itself loads from, because `model2vec_rs::StaticModel`
/// keeps its tokenizer private with no accessor.
struct Truncation {
    tokenizer: Tokenizer,
    /// The id `model2vec_rs` drops from a token list before truncating and
    /// pooling it. Recomputed here the same way its own (private) metadata
    /// step does: there is no accessor for it either.
    unk_token_id: Option<usize>,
    /// The median byte length of a vocabulary token, used to reproduce
    /// `model2vec_rs`'s private character-level pre-cut. Computed the same
    /// way its own (private) `compute_metadata` does: byte lengths of every
    /// vocabulary token, sorted, middle element.
    median_token_length: usize,
}

impl Truncation {
    fn from_tokenizer_bytes(bytes: &[u8]) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|e| format!("the truncation tokenizer did not load: {e}"))?;

        // No accessor exists for the unk token either, so recover it the way
        // `model2vec_rs`'s private `compute_metadata` does: round-trip the
        // tokenizer through JSON and read `model.unk_token`.
        let spec: serde_json::Value = serde_json::to_value(&tokenizer)
            .map_err(|e| format!("could not inspect the tokenizer for its unk token: {e}"))?;
        let unk_token_id = spec
            .get("model")
            .and_then(|model| model.get("unk_token"))
            .and_then(serde_json::Value::as_str)
            .and_then(|token| tokenizer.token_to_id(token))
            .map(|id| id as usize);

        let mut lens: Vec<usize> = tokenizer.get_vocab(false).keys().map(String::len).collect();
        lens.sort_unstable();
        let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);

        Ok(Self {
            tokenizer,
            unk_token_id,
            median_token_length,
        })
    }

    /// Reproduce `model2vec_rs`'s private character-level pre-cut: at most
    /// `max_tokens * median_token_length` characters of `text`, on a char
    /// boundary.
    ///
    /// Characters, not bytes, and the two part company on the first character
    /// above U+007F. `median_token_length` is itself measured in bytes — that
    /// mismatch is upstream's, reproduced rather than corrected, because the
    /// question here is what `encode_with_args` did and not what it should
    /// have done.
    fn char_truncate(text: &str, max_tokens: usize, median_token_length: usize) -> &str {
        text.char_indices()
            .nth(max_tokens.saturating_mul(median_token_length))
            .map_or(text, |(byte_idx, _)| &text[..byte_idx])
    }

    /// Every id of `text` that reaches `pool_ids`, before any token cut:
    /// tokenise, then drop the unknown-token id the way `encode_with_args`
    /// does on its way to the mean.
    ///
    /// [`Tokenizer::encode_fast`] is the call upstream makes
    /// (`encode_batch_fast`), and it is safe here for the reason it was not
    /// safe before: it documents that it does not compute offsets and returns
    /// `(0, 0)` for every token, and nothing in this file reads an offset any
    /// more.
    fn surviving_ids(&self, text: &str) -> Vec<u32> {
        let Ok(encoding) = self.tokenizer.encode_fast(text, false) else {
            // Upstream panics here (`.expect("tokenization failed")`), which
            // would take the whole DuckDB session with it. Reporting no ids
            // makes both sides of the comparison in `truncates` empty, so a
            // text the tokenizer cannot handle is reported as losing nothing
            // rather than as losing everything.
            return Vec::new();
        };
        let mut ids = encoding.get_ids().to_vec();
        if let Some(unk) = self.unk_token_id {
            ids.retain(|&id| id as usize != unk);
        }
        ids
    }

    /// True if [`Model::embed`] pooled fewer ids for `text` than the whole of
    /// `text` would have given it.
    ///
    /// Not "did a cut fire" — the character cut fires for any text over
    /// `max_tokens * median_token_length` characters whether or not a single
    /// token was lost with it, so 5,000 spaces would answer yes to that
    /// question while `embed` discarded nothing. The two id lists are the
    /// answer to the question a caller is actually asking, and they are equal
    /// exactly when the cuts took nothing that would have reached the mean:
    /// trailing whitespace, a run of symbols outside the vocabulary, a text
    /// with no in-vocabulary token in it at all.
    fn truncates(&self, text: &str, max_tokens: usize) -> bool {
        // The whole text, neither cut applied.
        let whole = self.surviving_ids(text);

        let char_cut = Self::char_truncate(text, max_tokens, self.median_token_length);
        // When the character cut took nothing, `char_cut` *is* `text` and
        // tokenising it again would return `whole` a second time, at the cost
        // of a second pass over the string. That is the only thing this branch
        // avoids: both arms produce the same ids.
        let mut pooled = if char_cut.len() == text.len() {
            whole.clone()
        } else {
            self.surviving_ids(char_cut)
        };
        pooled.truncate(max_tokens);

        pooled != whole
    }
}

static BUNDLED: OnceLock<Result<Model, String>> = OnceLock::new();

/// The model compiled into this binary, loaded on first call and reused after.
///
/// Loading is a parse of the embedded bytes — no filesystem lookup, no network,
/// no environment variable, and no configuration a caller has to supply.
pub fn bundled() -> Result<&'static Model, &'static str> {
    match BUNDLED.get_or_init(Model::from_embedded_bytes) {
        Ok(model) => Ok(model),
        Err(message) => Err(message.as_str()),
    }
}

impl Model {
    /// Load the model from the bytes embedded in this binary.
    fn from_embedded_bytes() -> Result<Self, String> {
        let inner = StaticModel::from_bytes(TOKENIZER, WEIGHTS, CONFIG, None)
            .map_err(|e| format!("the bundled {MODEL_ID} assets did not load: {e}"))?;

        // The dimension is read from the model rather than declared here, so a
        // future asset swap cannot leave a stale constant behind.
        let dim = inner.encode_single("dimension probe").len();
        if dim == 0 {
            return Err(format!(
                "the bundled {MODEL_ID} assets produced a zero-width embedding"
            ));
        }

        let key = model_key(TOKENIZER, WEIGHTS, CONFIG);
        let truncation = Truncation::from_tokenizer_bytes(TOKENIZER)?;

        Ok(Model {
            inner,
            dim,
            key,
            truncation,
        })
    }

    /// Number of floats in every vector this model returns.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The model's content address: a SHA-256 over its identity, its pinned
    /// revision and the three bundled asset files.
    ///
    /// This is the "model version" half of the cache key. It is derived from the
    /// bytes rather than from a version string, so replacing the weights
    /// invalidates every cached vector even if nobody remembers to bump a
    /// number.
    pub fn key(&self) -> &[u8; 32] {
        &self.key
    }

    /// The model key as lowercase hex.
    pub fn key_hex(&self) -> String {
        hex(&self.key)
    }

    /// Embed one string.
    ///
    /// # What comes back for text with no tokens
    ///
    /// A zero vector of [`Model::dim`] floats — not `None`, not an error, and
    /// not a short vector. That is the case for the empty string, for
    /// whitespace, and for text made only of tokens the vocabulary does not
    /// contain: the tokenizer yields no in-vocabulary ids, so the mean over
    /// zero ids is the zero vector.
    ///
    /// This function does not decide what a SQL `NULL` means. `NULL` never
    /// reaches here — the DuckDB layer propagates it, so `embed(NULL)` is
    /// `NULL`. A caller that wants the zero vector for a missing value asks for
    /// it: `embed(coalesce(t, ''))`.
    ///
    /// A vector of any width other than [`Model::dim`] or zero is a bug in the
    /// encoder, and comes back as an error rather than as silent zeros: a query
    /// that fails is recoverable, and a column of zero vectors that looks like
    /// data is not.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        // The whole text and the cap, handed to upstream together. Both cuts
        // are then upstream's to apply, to a string this crate has not touched
        // — which is what makes this `encode_single`'s own call.
        // `encode_with_args` takes two arguments besides the text: the cap,
        // which `encode_single` hard-codes at 512, and the batch size, which
        // chunks a one-sentence slice into one chunk whatever it is. So this
        // and `encode_single` can differ only in `MAX_TOKENS`, and
        // `embed_is_byte_identical_to_the_upstream_wrapper_it_replaced` pins
        // that.
        let sentence = [text.to_string()];
        let vector = self
            .inner
            .encode_with_args(&sentence, Some(MAX_TOKENS), 1)
            .into_iter()
            .next()
            .unwrap_or_default();
        conform(vector, self.dim)
    }

    /// Whether embedding `text` discards content: [`Model::embed`] pooled
    /// fewer ids than the whole of `text` would have given it, so the excess
    /// never reached the mean.
    ///
    /// The vector `embed` returns is full width and unit norm whether or not
    /// this is true — nothing about it says content was dropped, which is the
    /// whole reason to ask first.
    pub fn is_truncated(&self, text: &str) -> bool {
        self.truncation.truncates(text, MAX_TOKENS)
    }
}

/// Apply the width contract to whatever the encoder returned.
///
/// A pure function rather than a `match` inline in [`Model::embed`], because the
/// error arm is unreachable from outside the crate and a test that could only
/// call `embed` could never reach it.
fn conform(vector: Vec<f32>, dim: usize) -> Result<Vec<f32>, String> {
    match vector.len() {
        width if width == dim => Ok(vector),
        // The no-token contract: nothing to average is the zero vector, at the
        // model's full width.
        0 => Ok(vec![0.0_f32; dim]),
        width => Err(format!(
            "the encoder returned {width} floats for a model {dim} wide"
        )),
    }
}

/// Lowercase hex of a byte string.
pub fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    /// The embedded assets are the published release, byte for byte.
    ///
    /// AC5's quality figures were measured against `potion-base-8M`. A swapped,
    /// truncated or re-serialised asset would keep every other test green while
    /// invalidating every published claim, so the bytes are pinned here.
    #[test]
    fn bundled_asset_digests_match_the_pinned_release() {
        assert_eq!(digest(WEIGHTS), WEIGHTS_SHA256, "model.safetensors");
        assert_eq!(digest(TOKENIZER), TOKENIZER_SHA256, "tokenizer.json");
        assert_eq!(digest(CONFIG), CONFIG_SHA256, "config.json");
    }

    /// The width the encoder produces is the width the model's own config
    /// declares.
    ///
    /// Everything else that mentions the width takes it from `Model::dim`,
    /// which is itself measured by encoding a probe string — so an encoder that
    /// returned five floats for everything would set `dim` to five and satisfy
    /// every one of those assertions. `config.json` is a second, independent
    /// statement of the same fact, published with the weights, and this is the
    /// only place the two are made to agree.
    #[test]
    fn the_encoder_width_matches_the_width_the_config_declares() {
        let text = std::str::from_utf8(CONFIG).expect("config.json is UTF-8");
        let key = "\"hidden_dim\":";
        let start = text.find(key).expect("config.json declares hidden_dim") + key.len();
        let declared: usize = text[start..]
            .trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .and_then(|digits| digits.parse().ok())
            .expect("hidden_dim is a number");

        let model = bundled().expect("the bundled model loads");
        assert_eq!(model.dim(), declared, "encoder width against config.json");
    }

    #[test]
    fn the_bundled_model_loads_from_embedded_bytes() {
        let model = bundled().expect("the bundled model loads");
        assert!(model.dim() > 0);
    }

    /// A real sentence gets a real vector: full width, and not the zero vector.
    ///
    /// The second half is what matters. Every "does it return something"
    /// assertion stays green if the encoder silently degrades to zeros, which is
    /// exactly what the no-token fallback in `Model::embed` would look like if
    /// it fired for ordinary text.
    #[test]
    fn ordinary_text_gets_a_full_width_non_zero_vector() {
        let model = bundled().expect("the bundled model loads");
        let vector = model
            .embed("a manufacturer of industrial fasteners in Sheffield")
            .expect("embed");
        assert_eq!(vector.len(), model.dim());
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.5, "expected a normalised vector, got norm {norm}");
    }

    /// Same string in, same vector out — the property the cache depends on.
    #[test]
    fn embedding_is_deterministic() {
        let model = bundled().expect("the bundled model loads");
        assert_eq!(
            model.embed("repeatable").expect("embed"),
            model.embed("repeatable").expect("embed")
        );
    }

    #[test]
    fn different_strings_get_different_vectors() {
        let model = bundled().expect("the bundled model loads");
        assert_ne!(
            model.embed("bicycle").expect("embed"),
            model.embed("sovereign debt").expect("embed")
        );
    }

    /// The empty-tokenisation contract, pinned.
    ///
    /// Empty and whitespace-only text yield a zero vector of the model's width.
    /// This is upstream behaviour, which is precisely why it is asserted here:
    /// a dependency bump that changed it would otherwise change the extension's
    /// SQL contract without anything reddening.
    #[test]
    fn text_with_no_tokens_gets_a_zero_vector_of_full_width() {
        let model = bundled().expect("the bundled model loads");
        // The last two are runic and alchemical characters: the vocabulary this
        // model was built on does not carry them, so the tokenizer produces only
        // unknown tokens and there is nothing to average.
        for text in [
            "",
            " ",
            "\t\n  ",
            "\u{16A0}\u{16A2}\u{16A6}",
            "\u{1F701}\u{1F702}\u{1F703}",
        ] {
            let vector = model.embed(text).expect("embed");
            assert_eq!(vector.len(), model.dim(), "width for {text:?}");
            assert!(
                vector.iter().all(|v| *v == 0.0),
                "expected a zero vector for {text:?}"
            );
        }
    }

    /// AC3 + AC5: the *token* boundary between "clipped" and "not", pinned in
    /// both directions with a probe built to actually show truncation.
    ///
    /// A text of one repeated token cannot show this: the mean of 512 copies of
    /// a vector equals the mean of 513 copies of the same vector, so a probe
    /// like that reports "not clipped" whichever token the encoder happens to
    /// drop and would pass against a limit of any size, correct or broken. This
    /// probe is `N` copies of one filler word plus one DISTINCT trailing marker
    /// word, so whether the marker reached the mean is a fact a byte-equality
    /// check on the two vectors can actually observe.
    ///
    /// The filler is `"ok "` — three characters per token — rather than a
    /// six-character word, on purpose: `MAX_TOKENS * median_token_length`
    /// characters (512 * 6 = 3072 here) is a *second*, independent boundary,
    /// and a filler word whose own characters-per-token happens to sit at the
    /// vocabulary median crosses it at almost the same repetition count as the
    /// token cap. An earlier version of this test used `"steel "` — six
    /// characters including the space, exactly the median — and 511 reps plus
    /// a nine-character marker landed at 3075 characters, three past the
    /// 3072-character cut. That test asserted "not clipped" for text that the
    /// character cut had already clipped, and passed, because nothing before
    /// this file's `Truncation` type checked the character cut at all. `"ok "`
    /// keeps this probe's total character count under 1600 at these
    /// repetition counts — nowhere near 3072 — so it is pinning the token
    /// boundary alone, uncontaminated by the other one.
    /// [`text_is_reported_clipped_by_the_character_cut_alone`] pins that other
    /// boundary the same way.
    #[test]
    fn a_long_text_is_reported_clipped_exactly_where_the_marker_stops_reaching_the_mean() {
        let model = bundled().expect("the bundled model loads");
        let filler = |tokens: usize| "ok ".repeat(tokens).trim_end().to_string();
        let filler_plus_marker = |tokens: usize| {
            let mut text = "ok ".repeat(tokens);
            text.push_str("marker");
            text
        };

        // 511 filler tokens + 1 marker = 512 tokens: exactly at the cap, so the
        // marker is the 512nd token and is still pooled. 1539 characters total
        // — far short of the 3072-character cut, so only the token cap is in
        // play here.
        let at_cap = filler_plus_marker(MAX_TOKENS - 1);
        assert!(
            !model.is_truncated(&at_cap),
            "{} tokens must not report clipped",
            MAX_TOKENS
        );
        assert_ne!(
            model.embed(&at_cap).expect("embed"),
            model.embed(&filler(MAX_TOKENS - 1)).expect("embed"),
            "the marker at position {MAX_TOKENS} must still move the mean"
        );

        // 512 filler tokens + 1 marker = 513 tokens: one past the cap, so the
        // marker is truncated away before pooling. 1542 characters — still far
        // short of 3072.
        let past_cap = filler_plus_marker(MAX_TOKENS);
        assert!(
            model.is_truncated(&past_cap),
            "{} tokens must report clipped",
            MAX_TOKENS + 1
        );
        assert_eq!(
            model.embed(&past_cap).expect("embed"),
            model.embed(&filler(MAX_TOKENS)).expect("embed"),
            "the marker at position {} must have been dropped before pooling",
            MAX_TOKENS + 1
        );
    }

    /// AC1 + AC3 + AC5: the *character* boundary between "clipped" and "not" —
    /// the one `subtoken_is_truncated` used to be structurally blind to.
    ///
    /// `model2vec_rs` cuts the raw string to `MAX_TOKENS * median_token_length`
    /// characters (3072, here) *before* tokenising, so text whose own
    /// characters-per-token sits above that median can lose content while
    /// still tokenising to far fewer than [`MAX_TOKENS`] ids — a token-count
    /// check alone, run on the untruncated text, would say "plenty of room"
    /// right up to the point content is already gone. `"internationalization "`
    /// is 21 characters for 2 tokens — 10.5 characters per token, well above
    /// the median of 6 — so it crosses the 3072-character cut at 147
    /// repetitions (3087 characters) while sitting at only 294 tokens, 218
    /// short of the token cap.
    #[test]
    fn text_is_reported_clipped_by_the_character_cut_alone() {
        let model = bundled().expect("the bundled model loads");
        let word = "internationalization ";
        let filler = |reps: usize| word.repeat(reps).trim_end().to_string();
        let filler_plus_marker = |reps: usize| {
            let mut text = word.repeat(reps);
            text.push_str("marker");
            text
        };

        // 145 reps of filler plus the marker is 3051 characters, well under
        // the 3072-character cut, and 291 tokens, well under MAX_TOKENS —
        // neither cap is in play, so the marker is pooled.
        let under_the_cut = filler_plus_marker(145);
        assert!(
            !model.is_truncated(&under_the_cut),
            "3051 characters, 291 tokens: must not report clipped"
        );
        assert_ne!(
            model.embed(&under_the_cut).expect("embed"),
            model.embed(&filler(145)).expect("embed"),
            "the marker must still move the mean"
        );

        // 147 reps of filler alone is already 3087 characters — past the
        // 3072-character cut — at only 294 tokens, 218 short of MAX_TOKENS.
        // The trailing marker sits entirely past the character cut, so it is
        // discarded before it is ever tokenised, and embedding it changes
        // nothing: the vector is identical to the filler alone.
        let over_the_cut = filler_plus_marker(147);
        assert!(
            model.is_truncated(&over_the_cut),
            "3093 characters, 295 tokens: must report clipped even though \
             token count alone is nowhere near {MAX_TOKENS}"
        );
        assert!(
            model.is_truncated(&filler(147)),
            "the filler alone, past the character cut, must also report clipped"
        );
        assert_eq!(
            model.embed(&over_the_cut).expect("embed"),
            model.embed(&filler(147)).expect("embed"),
            "the marker past the character cut must have been dropped before pooling"
        );
    }

    /// The one token that appears in no probe's filler, appended to every probe
    /// a clip could be visible in.
    const MARKER: &str = "beacon";

    /// What `mixed_script_texts(200)` actually contains, measured. Exact rather
    /// than floors: the seed is fixed and every step from it is deterministic,
    /// so if a tokenizer bump moves one of these, the corpus the biconditional
    /// was validated against is not the corpus being run.
    const COVERAGE_CLIPPED: usize = 99;
    const COVERAGE_INTACT: usize = 101;
    const COVERAGE_TOKEN_CUT_ALONE: usize = 17;
    const COVERAGE_BYTES_OVER_CHARACTERS_UNDER: usize = 50;

    /// A probe text and what this crate must say about it.
    struct Probe {
        name: &'static str,
        text: String,
        /// What [`Model::is_truncated`] must report for it.
        clipped: bool,
        /// Whether a clip of this text would show in the vector it embeds to.
        ///
        /// True for every probe ending in [`MARKER`]: a cut of either kind
        /// removes a *suffix*, so a clip of any size takes the marker with it,
        /// and the marker is a token the rest of the probe does not contain —
        /// so the mean moves. False only where there is no in-vocabulary token
        /// to lose in the first place, and nothing a vector could show.
        observable: bool,
    }

    /// The corpus every truncation assertion below reads.
    ///
    /// One list rather than a set of inputs per test, because three defects
    /// have now been shipped past this file's suite and they were the same
    /// defect: a mechanism checked against inputs chosen to suit it. A probe of
    /// one repeated token cannot show truncation at all — the mean of 512
    /// copies of a vector is the mean of 513 copies. `"steel "` is six
    /// characters per token against a vocabulary median of exactly six, so the
    /// character cut and the token cut land on the same repetition and no count
    /// exists where one fires without the other. And until this list existed,
    /// no test input in this repository — in `src`, in `tests/` or in
    /// `test/sql/` — carried a byte above U+007F, which is why an
    /// implementation that sliced the source text at a token offset passed the
    /// suite while disagreeing with upstream on Korean.
    fn probes() -> Vec<Probe> {
        // One Hangul syllable. `BertNormalizer` runs NFD with accent stripping,
        // so it decomposes into its jamo — three tokens for this syllable, two
        // for one with no final consonant. The jamo are in the vocabulary and
        // the composed syllable is not: 70 jamo appear in it, and no syllable
        // from the U+AC00 block does.
        let hangul = "한 ";
        // One in-vocabulary CJK ideograph: one token, three bytes, and no
        // decomposition, which is why Chinese and Japanese passed the offset
        // slice that Korean broke.
        let cjk = "中";
        // 21 characters for 2 tokens — 10.5 per token against the median 6, so
        // the character cut reaches it long before the token cut does.
        let dense = "internationalization ";

        let probe = |name, text: String, clipped, observable| Probe {
            name,
            text,
            clipped,
            observable,
        };

        vec![
            probe("the empty string", String::new(), false, false),
            probe(
                "ordinary english",
                "a manufacturer of industrial fasteners in Sheffield".into(),
                false,
                false,
            ),
            // 511 filler tokens plus the marker is exactly MAX_TOKENS, and
            // 1539 characters — half the character cut, so this pair moves the
            // token cut alone.
            probe(
                "ascii exactly at the token cap",
                format!("{}{MARKER}", "ok ".repeat(MAX_TOKENS - 1)),
                false,
                true,
            ),
            probe(
                "ascii one token past the cap",
                format!("{}{MARKER}", "ok ".repeat(MAX_TOKENS)),
                true,
                true,
            ),
            // 3051 characters at 291 tokens, then 3093 characters at 295: the
            // character cut fires while the token count is 200 short of the
            // cap, which a token count alone cannot see.
            probe(
                "dense ascii under the character cut",
                format!("{}{MARKER}", dense.repeat(145)),
                false,
                true,
            ),
            probe(
                "dense ascii past the character cut",
                format!("{}{MARKER}", dense.repeat(147)),
                true,
                true,
            ),
            // 170 syllables is 510 jamo tokens; one ascii token and the marker
            // bring it to exactly MAX_TOKENS, and one more past it.
            probe(
                "hangul exactly at the token cap",
                format!("{}ok {MARKER}", hangul.repeat(170)),
                false,
                true,
            ),
            probe(
                "hangul one token past the cap",
                format!("{}ok ok {MARKER}", hangul.repeat(170)),
                true,
                true,
            ),
            // 171 syllables is 513 jamo, so the cut falls between the second
            // and third jamo of the last syllable — the case a byte-offset
            // slice of the source text cannot express, because all three jamo
            // report the one span the composed syllable occupied.
            probe(
                "hangul clipped inside one syllable's jamo",
                format!("{}{MARKER}", hangul.repeat(171)),
                true,
                true,
            ),
            probe(
                "hangul far past the cap",
                format!("{}{MARKER}", hangul.repeat(400)),
                true,
                true,
            ),
            probe(
                "cjk past the token cap",
                format!("{}{MARKER}", format!("{cjk} ").repeat(600)),
                true,
                true,
            ),
            // 3500 characters, none of them a token this vocabulary carries,
            // so every raw id is the unknown-token id and none reaches the
            // mean. The character cut fires and takes 428 characters with it,
            // and not one id the mean would have seen.
            probe(
                "cjk outside the vocabulary, past the character cut",
                "工業製品 ".repeat(700),
                false,
                false,
            ),
            // 2936 characters in 3336 bytes at 461 tokens: under both of the
            // cuts upstream applies, and over a cut that counted bytes. A
            // character cut and a byte cut are the same thing until the first
            // character above U+007F, and this repository had none.
            probe(
                "more bytes than the character cut but fewer characters",
                format!("{}{}{MARKER}", cjk.repeat(200), dense.repeat(130)),
                false,
                true,
            ),
            // 601 raw ids, 300 of them the unknown-token id that
            // `encode_with_args` drops before it truncates. Counting them
            // toward the cap would clip this at 512 raw ids; upstream pools
            // all 301 real ones and clips nothing.
            probe(
                "more raw tokens than the cap, half of them unknown",
                format!("{}{MARKER}", "steel \u{16A0} ".repeat(300)),
                false,
                true,
            ),
            // AC7, twice. Both are far past the character cut and neither
            // loses an id: `embed` of either is the zero vector with or
            // without the cap.
            probe("five thousand spaces", " ".repeat(5000), false, false),
            probe(
                "four thousand runic characters",
                "\u{16A0}".repeat(4000),
                false,
                false,
            ),
        ]
    }

    /// AC4: `embed` returns what it returned at `b1c1e03`, for every probe.
    ///
    /// `conform(inner.encode_single(text), dim)` *is* the body `embed` had at
    /// `b1c1e03`, so this is that implementation and this one answering the
    /// same inputs in the same process — a stronger comparison than two builds,
    /// because there is no second build to differ for another reason.
    ///
    /// It can now fail in exactly one way. `embed` hands the untouched text and
    /// `Some(MAX_TOKENS)` to `encode_with_args`, which is the call
    /// `encode_single` makes with its own hard-coded 512, so the two differ
    /// only if `MAX_TOKENS` stops being 512 — the batch size is the other
    /// argument and a one-sentence slice is one chunk at any of them. That is
    /// the point of the shape: two earlier versions of this file cut the text
    /// themselves and passed upstream a pre-cut string, which is what gave them
    /// room to disagree with it, and what this pins is that there is no longer
    /// any such room. The Hangul probes are the regression guard for the
    /// version that did — measured against the merged build, ASCII, Japanese,
    /// Chinese, accented French and emoji agreed with it byte for byte and
    /// Korean did not.
    #[test]
    fn embed_is_byte_identical_to_the_upstream_wrapper_it_replaced() {
        let model = bundled().expect("the bundled model loads");
        for probe in probes() {
            assert_eq!(
                model.embed(&probe.text).expect("embed"),
                conform(model.inner.encode_single(&probe.text), model.dim()).expect("conform"),
                "{}: {} chars",
                probe.name,
                probe.text.chars().count()
            );
        }
    }

    /// Everything `embed` would have pooled with no cap at all — upstream's own
    /// call with `None`, which skips both cuts and keeps filtering the unknown
    /// token.
    ///
    /// This is the oracle for what `is_truncated` claims. It is upstream
    /// answering, not this crate: a predicate that reproduces the cuts wrongly
    /// disagrees with it, whichever direction the mistake runs in.
    fn embed_with_no_cap(model: &Model, text: &str) -> Vec<f32> {
        conform(
            model
                .inner
                .encode_with_args(&[text.to_string()], None, 1)
                .into_iter()
                .next()
                .unwrap_or_default(),
            model.dim(),
        )
        .expect("conform")
    }

    /// AC1, AC3, AC6, AC7, AC9: `is_truncated` is true exactly when the cap
    /// changed the vector, on a corpus that can tell the two apart.
    ///
    /// Two assertions per probe, and the second is the one that does not
    /// depend on anybody having written down the right answer:
    ///
    /// * the reported verdict equals the pinned one, which is what makes a cap
    ///   that moves by one redden;
    /// * the reported verdict equals `embed(text) != embed_with_no_cap(text)`,
    ///   which is upstream's own answer to "did the cap cost you anything".
    ///   For probes with no marker only the safe half of that holds — "not
    ///   clipped" must mean the vectors agree — because a text with no
    ///   in-vocabulary token has no way to show a clip in a vector either way.
    #[test]
    fn the_clipped_verdict_matches_what_the_cap_cost_the_vector() {
        let model = bundled().expect("the bundled model loads");
        for probe in probes() {
            let reported = model.is_truncated(&probe.text);
            assert_eq!(
                reported,
                probe.clipped,
                "{}: {} chars, {} bytes",
                probe.name,
                probe.text.chars().count(),
                probe.text.len()
            );

            let capped = model.embed(&probe.text).expect("embed");
            let whole = embed_with_no_cap(model, &probe.text);
            if probe.observable {
                assert_eq!(
                    reported,
                    capped != whole,
                    "{}: reported clipped = {reported}, but the cap {} the vector",
                    probe.name,
                    if capped == whole { "left" } else { "moved" }
                );
            } else if !reported {
                assert_eq!(
                    capped, whole,
                    "{}: reported unclipped, yet the cap moved the vector",
                    probe.name
                );
            }
        }
    }

    /// The same property, on inputs nobody chose.
    ///
    /// The table above is a list of cases somebody thought of, and each defect
    /// this file has carried was the case nobody thought of: a filler that
    /// could not show truncation, a corpus with no byte above U+007F in any
    /// input, an offset slice that Hangul broke. Adding the case that was
    /// missed buys the next one. This generates its inputs instead — 200 texts
    /// drawn
    /// from a mixed alphabet of ASCII words, Hangul, in- and out-of-vocabulary
    /// CJK, kana, accented Latin, runic, emoji, a URL, an identifier and runs
    /// of whitespace, at lengths that straddle both cuts — and asserts the same
    /// biconditional against the same upstream oracle.
    ///
    /// Every generated text ends with [`MARKER`], which none of the pieces
    /// contain, so a clip of any size takes it and the mean moves: the
    /// biconditional is exact for all 200 rather than one-directional. The
    /// generator is a fixed-seed LCG, so a failure names a text that can be
    /// regenerated rather than one that has already gone.
    #[test]
    fn a_generated_mixed_script_corpus_reports_clipped_exactly_when_the_cap_cost_it() {
        let model = bundled().expect("the bundled model loads");
        let mut clipped = 0usize;
        let mut clipped_under_the_character_cut = 0usize;
        let mut intact = 0usize;
        let mut inside_the_cut_but_over_it_in_bytes = 0usize;
        for (index, text) in mixed_script_texts(200).into_iter().enumerate() {
            let reported = model.is_truncated(&text);
            let capped = model.embed(&text).expect("embed");
            let whole = embed_with_no_cap(model, &text);
            if reported {
                clipped += 1;
                // 6 is this vocabulary's median token length, so this counts
                // the texts the token cut clipped on its own, with the
                // character cut nowhere near them.
                if text.chars().count() <= MAX_TOKENS * 6 {
                    clipped_under_the_character_cut += 1;
                }
            } else {
                intact += 1;
                // The region a byte cut and a character cut disagree over:
                // short enough in characters to survive the real cut, long
                // enough in bytes to be taken by a cut that counted those, and
                // far enough short of the token cap that neither cap hides the
                // difference.
                if text.chars().count() <= MAX_TOKENS * 6 && text.len() > MAX_TOKENS * 6 {
                    inside_the_cut_but_over_it_in_bytes += 1;
                }
            }

            assert_eq!(
                model.embed(&text).expect("embed"),
                conform(model.inner.encode_single(&text), model.dim()).expect("conform"),
                "text {index}: {} chars, {} bytes — embed is no longer upstream's call",
                text.chars().count(),
                text.len()
            );
            assert_eq!(
                reported,
                capped != whole,
                "text {index}: {} chars, {} bytes, reported clipped = {reported}, cap {} the vector",
                text.chars().count(),
                text.len(),
                if capped == whole { "left" } else { "moved" }
            );
        }

        // What the corpus actually contains, asserted rather than assumed. A
        // biconditional over 200 texts that all landed on one side of it would
        // pass while checking half of what it says it checks, and a generator
        // is one edit away from that at any time. The seed is fixed and every
        // step from it is deterministic, so these are exact rather than floors:
        // if a tokenizer bump moves them, the corpus this was validated against
        // is not the corpus being run, and that is worth reddening for.
        assert_eq!(
            (clipped, intact, clipped_under_the_character_cut),
            (COVERAGE_CLIPPED, COVERAGE_INTACT, COVERAGE_TOKEN_CUT_ALONE),
            "the generated corpus no longer straddles the boundary it was built to straddle"
        );
        assert_eq!(
            inside_the_cut_but_over_it_in_bytes, COVERAGE_BYTES_OVER_CHARACTERS_UNDER,
            "the generated corpus no longer reaches the region a byte cut and a \
             character cut disagree over, which is the region it was extended for"
        );
    }

    /// 200 mixed-script texts from a fixed seed, each ending in [`MARKER`].
    ///
    /// Two families, because a generator that draws its pieces evenly cannot
    /// reach every region that matters and this one could not reach the region
    /// the character cut lives in. Measured while writing it: the first family
    /// alone left the byte-cut mutation alive across all 200 texts, because a
    /// text long enough in bytes to cross 3072 was already long enough in
    /// tokens to cross 512, so both a right and a wrong cut called it clipped.
    ///
    /// * a free mixture from the whole alphabet, at budgets straddling both
    ///   cuts;
    /// * texts that are heavy in bytes and light in tokens — a run of
    ///   multi-byte characters ahead of words of ten or more characters each —
    ///   which is the only shape that sits inside the character cut, outside
    ///   the same number of bytes, and far short of the token cap at once.
    ///
    /// The count of texts landing in that second region is asserted by the
    /// caller. Without that the generator is one edit away from the hole it was
    /// written to close.
    fn mixed_script_texts(count: usize) -> Vec<String> {
        // Deliberately uneven in characters per token and in bytes per
        // character, because those two ratios are what decide which cut bites
        // first and every earlier corpus held them fixed.
        const PIECES: [&str; 17] = [
            "steel ",
            "ok ",
            "a foundry casting valve bodies ",
            "internationalization ",
            "snake_case_identifier_name ",
            "https://example.invalid/a/deeply/nested/path ",
            "한 ",
            "한국어 ",
            "中 ",
            "中",
            "日本語 ",
            "こんにちは ",
            "工業製品 ",
            "café ",
            "\u{16A0} ",
            "\u{1F701} ",
            "     ",
        ];
        // Fifteen or more characters to the token, so a few thousand of them
        // are still only a couple of hundred ids. Every one of these five is a
        // single vocabulary entry except `internationalization`, which is two.
        //
        // A long word is not the same as a word with a high
        // characters-per-token ratio, and this is where that distinction bites:
        // the first version of this family used the URL and the identifier from
        // `PIECES` and reached none of the region it was written for.
        // `snake_case_identifier_name` is 27 characters and splits on every
        // underscore, and the URL splits on every punctuation mark, so both are
        // around three characters to the token — denser than ordinary prose,
        // not sparser.
        const SPARSE: [&str; 5] = [
            "telecommunications ",
            "responsibilities ",
            "characteristics ",
            "entrepreneurship ",
            "internationalization ",
        ];
        // Three bytes each: one in the vocabulary, one that decomposes into
        // three jamo, and two the vocabulary does not carry.
        const WIDE: [&str; 4] = ["中", "한", "工", "\u{16A0}"];
        // Character budgets straddling the 3072-character cut and the lengths
        // at which the token cut bites for each script.
        const BUDGETS: [usize; 9] = [8, 300, 1200, 2900, 3050, 3071, 3080, 3400, 5200];

        let mut state = 0x5EED_1234_9ABC_DEF0_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize
        };

        let mut texts = Vec::with_capacity(count);
        for index in 0..count {
            let mut text = String::new();
            if index % 4 == 3 {
                // Byte-heavy and token-light: 120 to 220 wide characters, then
                // sparse words up to a budget that stays inside the character
                // cut while the byte count passes it. 120 is the floor at which
                // the two extra bytes each wide character costs carry the
                // smallest budget here past 3072.
                let wide = 120 + next() % 101;
                for _ in 0..wide {
                    text.push_str(WIDE[next() % WIDE.len()]);
                }
                let budget = 2850 + next() % 131;
                while text.chars().count() < budget {
                    text.push_str(SPARSE[next() % SPARSE.len()]);
                }
            } else {
                let budget = BUDGETS[index % BUDGETS.len()] + next() % 64;
                while text.chars().count() < budget {
                    text.push_str(PIECES[next() % PIECES.len()]);
                }
            }
            text.push_str(MARKER);
            texts.push(text);
        }
        texts
    }

    /// AC7: text that loses nothing is not reported clipped, however long it
    /// runs.
    ///
    /// The character cut fires for any text over 3072 characters whether or not
    /// a single id went with it, and a predicate that asked "did a cut fire"
    /// answered yes for all four of these. `embed` of every one is the zero
    /// vector, with the cap and without it.
    #[test]
    fn text_with_no_tokens_is_not_reported_clipped() {
        let model = bundled().expect("the bundled model loads");
        for text in [
            String::new(),
            "   ".to_string(),
            "\u{16A0}\u{16A2}\u{16A6}".to_string(),
            " ".repeat(5000),
            "\u{16A0}".repeat(4000),
            "工業製品 ".repeat(700),
        ] {
            assert!(
                !model.is_truncated(&text),
                "{} characters, {} bytes",
                text.chars().count(),
                text.len()
            );
        }
    }

    /// Ordinary short text is not clipped, and a text built to be many times
    /// past the cap is — the coarse sanity check either side of the pinned
    /// boundary above.
    #[test]
    fn ordinary_text_is_not_clipped_and_a_much_longer_text_is() {
        let model = bundled().expect("the bundled model loads");
        assert!(!model.is_truncated("a manufacturer of industrial fasteners in Sheffield"));
        assert!(model.is_truncated(&"steel ".repeat(MAX_TOKENS * 3)));
    }

    /// The model key is a content address: flipping one byte of any of the three
    /// assets moves it.
    ///
    /// This calls `model_key` itself rather than recomputing the same fields
    /// beside it. A version that recomputed them would still pass with an asset
    /// dropped from the production hash, because the test's own copy would keep
    /// including it — which is what this test used to do.
    #[test]
    fn the_model_key_covers_every_asset_byte() {
        let live = model_key(TOKENIZER, WEIGHTS, CONFIG);
        assert_eq!(hex(&live).len(), 64);

        let flip_last = |bytes: &[u8]| {
            let mut copy = bytes.to_vec();
            let last = copy.len() - 1;
            copy[last] ^= 0x01;
            copy
        };

        assert_ne!(
            live,
            model_key(&flip_last(TOKENIZER), WEIGHTS, CONFIG),
            "tokenizer"
        );
        assert_ne!(
            live,
            model_key(TOKENIZER, &flip_last(WEIGHTS), CONFIG),
            "weights"
        );
        assert_ne!(
            live,
            model_key(TOKENIZER, WEIGHTS, &flip_last(CONFIG)),
            "config"
        );
    }

    /// The key a loaded model reports is the key its own asset bytes derive.
    #[test]
    fn the_loaded_model_reports_the_key_its_assets_derive() {
        let model = bundled().expect("the bundled model loads");
        assert_eq!(model.key(), &model_key(TOKENIZER, WEIGHTS, CONFIG));
    }

    /// A full-width vector passes through untouched.
    #[test]
    fn conform_leaves_a_full_width_vector_alone() {
        assert_eq!(
            conform(vec![1.0, 2.0, 3.0, 4.0], 4),
            Ok(vec![1.0, 2.0, 3.0, 4.0])
        );
    }

    /// Nothing to average becomes the zero vector at full width.
    #[test]
    fn conform_turns_an_empty_vector_into_a_full_width_zero_vector() {
        assert_eq!(conform(vec![], 4), Ok(vec![0.0, 0.0, 0.0, 0.0]));
    }

    /// Any other width is a bug in the encoder and is reported, not padded.
    ///
    /// Silent padding here would turn a dimension mismatch into a column of
    /// plausible-looking zeros, which nothing downstream could distinguish from
    /// text that genuinely had no tokens.
    #[test]
    fn conform_reports_any_other_width_rather_than_padding_it() {
        let reported = conform(vec![1.0, 2.0, 3.0], 4).expect_err("a short vector is an error");
        assert!(
            reported.contains('3') && reported.contains('4'),
            "{reported}"
        );
    }
}
