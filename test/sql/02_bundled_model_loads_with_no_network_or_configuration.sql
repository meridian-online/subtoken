-- AC2: the model is bundled in the extension binary and loads on first call
-- with no network and no configuration.
--
-- The runner executes this with HOME and every model-cache variable pointed at
-- empty temporary directories and the proxy variables removed, so a model that
-- needed a download, a cache or a config file has nowhere to find one. The
-- other half of the claim — that no HTTP or TLS crate is in the runtime
-- dependency tree at all — is scripts/check_no_network_deps.py.

SELECT must('the extension names the model it bundles',
    subtoken_version() LIKE '%minishlab/potion-base-8M%');

SELECT must('the extension reports the revision the assets were pinned at',
    subtoken_version() LIKE '%bf8b056651a2%');

SELECT must('the extension reports a positive vector width',
    CAST(regexp_extract(subtoken_version(), 'dim (\d+)', 1) AS BIGINT) > 0);

-- The first embedding call in this process. No LOAD-time configuration, no
-- SET, no environment variable: the weights are in the binary.
SELECT must('the first call returns a vector without any configuration',
    len(subtoken_embed('the first call in this process')) =
    CAST(regexp_extract(subtoken_version(), 'dim (\d+)', 1) AS BIGINT));

-- Weights that failed to load would still produce a correctly shaped zero
-- vector, so shape alone is not evidence the model is there.
SELECT must('the vector is not the zero vector, so real weights were read',
    list_max(subtoken_embed('the first call in this process')) > 0
    OR list_min(subtoken_embed('the first call in this process')) < 0);

SELECT must('two different strings get two different vectors',
    subtoken_embed('a foundry casting valve bodies') IS DISTINCT FROM subtoken_embed('a bonded warehouse operator'));
