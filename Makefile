# subtoken — build, package and test the DuckDB extension.
#
# `make check` is the whole gate, and it is what .github/workflows/ci.yml runs.

EXTENSION_NAME := subtoken
# The stable C_STRUCT ABI floor the packaged artifact declares, so one build
# loads on DuckDB 1.2 and later. Mirrors MIN_DUCKDB_VERSION in
# crates/subtoken-duckdb/src/lib.rs.
TARGET_DUCKDB_VERSION := v1.2.0
EXTENSION_VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
	LIB_FILE := libsubtoken.dylib
	UNAME_PLATFORM := $(if $(filter arm64,$(UNAME_M)),osx_arm64,osx_amd64)
else
	LIB_FILE := libsubtoken.so
	UNAME_PLATFORM := $(if $(filter aarch64,$(UNAME_M)),linux_arm64,linux_amd64)
endif

DUCKDB ?= duckdb

# The platform string stamped into the metadata trailer must be the one the
# DuckDB that loads the artifact reports, or LOAD refuses it. Ask DuckDB rather
# than deriving it from uname, so the two cannot disagree; fall back to uname
# when the CLI is absent, and let a cross-compile override it outright.
CLI_PLATFORM := $(shell command -v $(DUCKDB) >/dev/null 2>&1 && $(DUCKDB) -init /dev/null -noheader -csv -c "SELECT platform FROM pragma_platform();" 2>/dev/null)
DUCKDB_PLATFORM ?= $(if $(CLI_PLATFORM),$(CLI_PLATFORM),$(UNAME_PLATFORM))

BUILD_DIR := build
EXTENSION := $(BUILD_DIR)/$(EXTENSION_NAME).duckdb_extension

.PHONY: all check build extension test test-sql description-examples no-network documented-surface quality-claims assertions-can-fail mutation-check fmt clippy clean

all: extension

## Everything CI runs, in the order it runs it.
check: fmt clippy test extension no-network documented-surface test-sql description-examples quality-claims assertions-can-fail

build:
	cargo build --workspace --release

## The packaged artifact: the cdylib plus the DuckDB metadata trailer. Without
## the trailer DuckDB rejects the file rather than loading it, so this is the
## step that makes the build testable at all.
extension: build
	python3 scripts/append_extension_metadata.py \
		--library-file target/release/$(LIB_FILE) \
		--out-file $(EXTENSION) \
		--platform $(DUCKDB_PLATFORM) \
		--duckdb-version $(TARGET_DUCKDB_VERSION) \
		--extension-version $(EXTENSION_VERSION)

## Rust unit tests: the model, the cache and the embedding contract.
test:
	cargo test --workspace

## AC2's other half: nothing that can open a socket is compiled into the
## artifact. The self-test runs first, so a checker that has gone blind is
## caught before its clean report is believed.
no-network: extension
	python3 scripts/check_no_network_deps.py --self-test
	python3 scripts/check_no_network_deps.py --artifact $(EXTENSION)

## The function table and STRUCT signatures written in README.md, the module doc
## and description.yml — the registry page a stranger reads — against
## duckdb_functions() of a loaded build. Each copy of a signature is another
## chance to be wrong, and the page was wrong.
documented-surface: extension
	python3 scripts/check_documented_surface.py --self-test
	python3 scripts/check_documented_surface.py --extension $(EXTENSION) --duckdb $(DUCKDB)

## SQL tests: LOAD the packaged artifact into a real DuckDB and query it.
## Needs the duckdb CLI on PATH.
test-sql: extension
	python3 scripts/run_sql_tests.py --extension $(EXTENSION) --duckdb $(DUCKDB)

## Every SQL example in the registry entry, run in one session against a real
## build. The registry's own doc_test job carries `if: false`, so nothing
## upstream ever executes what description.yml publishes — a broken example
## reaches the extension's page and stays there. Needs PyYAML: the examples are
## extracted with a YAML parser rather than a regex over the file, because a
## regex cannot tell a block scalar from the prose around it.
description-examples: extension
	python3 scripts/check_description_examples.py --self-test
	python3 scripts/check_description_examples.py --extension $(EXTENSION) --duckdb $(DUCKDB)

## The quality position published in README.md and in description.yml's
## extended_description, against each other and against the measurement that
## produced it. A function signature can be settled by loading the extension;
## a quality claim cannot, so this holds the two copies to one set of figures
## and reddens when the bundled model moves out from under them. Needs PyYAML
## for the same reason `description-examples` does. No build, no duckdb.
quality-claims:
	python3 scripts/check_quality_claims.py --self-test
	python3 scripts/check_quality_claims.py

## Neuter one branch of a checked script at a time and require the script to
## notice. This one IS part of `check` and IS in CI: it needs no build and no
## duckdb, and the whole sweep is seconds. Every review of check_quality_claims.py
## so far has found an assertion that could be deleted with every gate green, and
## every one was found by a person reading the file rather than by a check.
assertions-can-fail:
	python3 scripts/check_assertions_can_fail.py --self-test
	python3 scripts/check_assertions_can_fail.py

## Break the code on purpose and require the named test to redden. NOT part of
## `check`: every SQL mutation rebuilds the release cdylib, so a sweep is
## minutes. Run it when you change a test or the code under one. Needs a clean
## tree — mutations are undone with `git checkout --`.
mutation-check:
	python3 scripts/mutation_check.py --duckdb $(DUCKDB)

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

clean:
	cargo clean
	rm -rf $(BUILD_DIR)

# ─── The community-extensions build contract ──────────────────────────────────
#
# `duckdb/community-extensions` does not read a build recipe out of the registry
# entry. It checks this repository out at the `repo.ref` its `description.yml`
# names and runs, at the root, `make configure_ci`, then `make release`, then
# `make test_release`, through
# `duckdb/extension-ci-tools/.github/workflows/_extension_distribution.yml`.
# The targets below ARE that recipe, and `.github/workflows/MainDistributionPipeline.yml`
# calls the same reusable workflow at the same ref the registry pins, so a green
# run there is a rehearsal of the registry's build rather than a lookalike.
#
# `make extension` above is the local convenience path and is a different
# recipe: it packages the whole-workspace `cargo build --release` with this
# repo's own metadata script, while the contract below builds only
# `subtoken-duckdb` and stamps the trailer with extension-ci-tools' script.
# Two recipes that agree today is exactly what an acceptance criterion warned
# against, so `make community-check` builds both and
# `scripts/check_artifact_version.py --compare` fails when the trailers they
# write stop agreeing.
#
# STABLE C API: USE_UNSTABLE_C_API is deliberately unset, so the trailer carries
# the C_STRUCT ABI at the TARGET_DUCKDB_VERSION floor set at the top of this
# file and one binary stays loadable across DuckDB 1.2 and later.

COMMUNITY_RECIPE := extension-ci-tools/makefiles/c_api_extensions/base.Makefile

ifeq ($(wildcard $(COMMUNITY_RECIPE)),)

# A clone without `--recursive` has no extension-ci-tools, and `make check`
# must still work in one. Name the reason rather than failing on a missing
# include halfway through a parse.
.PHONY: configure release debug test_release test_debug community-check
configure release debug test_release test_debug community-check:
	@echo "$(COMMUNITY_RECIPE) is missing." >&2
	@echo "Run: git submodule update --init --recursive" >&2
	@exit 1

else

include $(COMMUNITY_RECIPE)

# base.Makefile writes configure/extension_version.txt ONLY when that file is
# absent — it is a make file-target rule with no prerequisites. A copy of it
# that survives a Cargo.toml bump is therefore never refreshed, and `make
# release` stamps the old number onto every artifact while `make extension`,
# which reads Cargo.toml directly, stays right. That is a measured defect in a
# sibling repo, not a hypothesis. Two things stop it here: `/configure/` is
# gitignored, so the file cannot arrive tracked in a checkout; and this rule
# adds Cargo.toml as a prerequisite, so a version bump in a live tree
# regenerates it. `scripts/check_artifact_version.py` is what reddens if both
# fail.
#
# A prerequisite with no recipe: it ADDS to base.Makefile's rule rather than
# replacing it, so the version still comes from $(EXTENSION_VERSION) above.
configure/extension_version.txt: Cargo.toml

# The community matrix builds osx_amd64 and osx_arm64 from one arm64 runner.
COMMUNITY_CARGO_TARGET :=
COMMUNITY_CARGO_OUT    := target
ifeq ($(DUCKDB_PLATFORM),osx_amd64)
	COMMUNITY_CARGO_TARGET := --target x86_64-apple-darwin
	COMMUNITY_CARGO_OUT    := target/x86_64-apple-darwin
else ifeq ($(DUCKDB_PLATFORM),osx_arm64)
	COMMUNITY_CARGO_TARGET := --target aarch64-apple-darwin
	COMMUNITY_CARGO_OUT    := target/aarch64-apple-darwin
endif

# What the registry uploads and what a stranger's INSTALL downloads.
COMMUNITY_EXTENSION := $(EXTENSION_BUILD_PATH)/release/extension/$(EXTENSION_NAME)/$(EXTENSION_FILENAME)

.PHONY: configure release debug test_release test_debug community-check community-artifact-checks

configure: venv platform extension_version

## `extension_version` first, and this is load-bearing rather than tidy.
## base.Makefile's own `release` prerequisites do not include it: only `make
## configure` ever asks for configure/extension_version.txt, so on a tree where
## that file already exists `make release` reads it without regenerating it and
## stamps whatever it says. Adding it here, with the Cargo.toml prerequisite
## below, means a version bump is picked up by the build that publishes rather
## than only by the one that configures.
release: extension_version build_extension_library_release build_extension_with_metadata_release
debug:   extension_version build_extension_library_debug   build_extension_with_metadata_debug

## Only the cdylib, not the whole workspace: subtoken-core's dev-dependencies
## and test binaries have no business in a distribution build.
build_extension_library_release: check_configure
	DUCKDB_EXTENSION_NAME=$(EXTENSION_NAME) DUCKDB_EXTENSION_MIN_DUCKDB_VERSION=$(TARGET_DUCKDB_VERSION) \
		cargo build -p subtoken-duckdb --release $(COMMUNITY_CARGO_TARGET)
	mkdir -p $(EXTENSION_BUILD_PATH)/release/extension/$(EXTENSION_NAME)
	cp $(COMMUNITY_CARGO_OUT)/release/$(EXTENSION_LIB_FILENAME) $(EXTENSION_BUILD_PATH)/release/$(EXTENSION_LIB_FILENAME)

build_extension_library_debug: check_configure
	DUCKDB_EXTENSION_NAME=$(EXTENSION_NAME) DUCKDB_EXTENSION_MIN_DUCKDB_VERSION=$(TARGET_DUCKDB_VERSION) \
		cargo build -p subtoken-duckdb $(COMMUNITY_CARGO_TARGET)
	mkdir -p $(EXTENSION_BUILD_PATH)/debug/extension/$(EXTENSION_NAME)
	cp $(COMMUNITY_CARGO_OUT)/debug/$(EXTENSION_LIB_FILENAME) $(EXTENSION_BUILD_PATH)/debug/$(EXTENSION_LIB_FILENAME)

## What the registry runs after its build, on the artifact it is about to
## publish. SKIP_TESTS is base.Makefile's own verdict on whether this
## environment can run anything — it is set for linux_amd64, for the musl and
## mingw targets, and inside the Linux build container — so honour it here the
## same way base.Makefile does for its own test targets. The platforms it turns
## off are covered instead by the `no-network` job in
## .github/workflows/MainDistributionPipeline.yml, which downloads the uploaded
## artifact and checks it on a runner of that platform.
ifeq ($(SKIP_TESTS),1)
test_release: tests_skipped
test_debug:   tests_skipped
else
test_release: community-artifact-checks
test_debug:   tests_skipped
endif

community-artifact-checks: check_configure
	$(PYTHON_BIN) scripts/check_artifact_version.py --self-test
	$(PYTHON_BIN) scripts/check_artifact_version.py --artifact $(COMMUNITY_EXTENSION)
	$(PYTHON_BIN) scripts/check_no_network_deps.py --self-test
	$(PYTHON_BIN) scripts/check_no_network_deps.py --artifact $(COMMUNITY_EXTENSION) --require-inspection

## Both recipes, then the assertion that they still agree. This is the local
## equivalent of the `community-recipe` job in .github/workflows/ci.yml.
community-check: extension configure release
	$(PYTHON_BIN) scripts/check_artifact_version.py --self-test
	$(PYTHON_BIN) scripts/check_artifact_version.py --artifact $(COMMUNITY_EXTENSION)
	$(PYTHON_BIN) scripts/check_artifact_version.py --compare $(EXTENSION) $(COMMUNITY_EXTENSION)
	$(PYTHON_BIN) scripts/check_no_network_deps.py --artifact $(COMMUNITY_EXTENSION) --require-inspection
	$(PYTHON_BIN) scripts/check_description_examples.py --self-test
	$(PYTHON_BIN) scripts/check_description_examples.py --extension $(COMMUNITY_EXTENSION) --duckdb $(DUCKDB) --require-complete

endif
