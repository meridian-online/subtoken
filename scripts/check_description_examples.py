#!/usr/bin/env python3
"""Run every SQL example in `description.yml`, and check the entry describes this tree.

`description.yml` is the registry submission. What it says is what a stranger
reads on the extension's page and pastes into a shell, and **nothing upstream
ever runs it**: the `doc_test` job in `duckdb/community-extensions`'
`.github/workflows/build.yml` carries `if: false`, so it is not merely skipped
for a given run, it is switched off. On a real registry PR the observed result
is 7 pass / 6 skip / 0 fail, with `doc_test` among the skips. A published
example that throws on paste therefore stays published until a person complains.

WHAT THIS RUNS. Every example, in ONE DuckDB session, in the order the page
presents them, against a packaged artifact. The session is given nothing the
descriptor does not create for itself: the prelude is an `extension_directory`
pointed at an empty temporary directory and a `LOAD` of the artifact, and that
is all. A later example may use a table an earlier one created, because that is
how someone reads a page; it may not use one nobody created.

The environment is scrubbed the same way `run_sql_tests.py` scrubs it — `HOME`
and the model-cache variables point at empty temporary directories and the proxy
variables are removed — so an example that only works because the developer's
machine has something in it fails here.

HOW THE EXAMPLES ARE FOUND. With a YAML parser, not a regex over the file.
`hello_world` and `extended_description` are block scalars, and their content is
prose that contains lines beginning `extension:`, `repo:` and `## ` — a
line-oriented pattern reading this file as text stops at the first of those and
silently drops the rest of the examples, which is a checker that passes because
it found nothing. Inside `extended_description`, examples are the ```sql fenced
blocks; a ```python or ```text block is not one.

WHAT ELSE IT CHECKS. The entry has to describe THIS tree, and the parts that can
go stale silently are the ones asserted here: the version against `Cargo.toml`,
the repository against the `repository` field in `Cargo.toml`, the platform
exclusions against the distribution matrix that defines those names, and the
function table against `duckdb_functions()` of the loaded artifact — derived
from the build rather than from a list written here, so registering a sixth
function reddens this until the page mentions it.

It also holds the distribution workflow's inspection job to the platforms that
workflow builds: an arch that survives the exclusions and opt-ins has to have an
entry there, and that entry has to load the artifact or say what stops it.
Reading a binary's trailer and its imports is not running it, and an artifact
that builds, stamps and imports correctly can still refuse to `LOAD`.

`--require-complete` turns an absent input from a named skip into a failure. The
distribution matrix arrives through a submodule and `make check` is documented
as working in a non-recursive clone, so CI passes the flag and the Makefile does
not.

Needs PyYAML. The registry's own `scripts/build.py` does too.

    scripts/check_description_examples.py --extension build/subtoken.duckdb_extension
    scripts/check_description_examples.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DESCRIPTION = REPO_ROOT / "description.yml"
MATRIX = REPO_ROOT / "extension-ci-tools" / "config" / "distribution_matrix.json"
REHEARSAL = REPO_ROOT / ".github" / "workflows" / "MainDistributionPipeline.yml"
BUILD_JOB = "duckdb-stable-build"
#: The job that downloads every uploaded artifact and looks at it.
INSPECTION_JOB = "no-network"

MARKER = "SUBTOKEN_EXAMPLE"

PRELUDE = """\
SET extension_directory='{extension_directory}';
LOAD '{extension}';
"""

SCRUBBED_ENV_TO_TEMP = ("HOME", "XDG_CACHE_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
SCRUBBED_ENV_TO_UNSET = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")


def load_yaml(path: pathlib.Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is needed to read description.yml with a parser rather than a regex.\n"
            "    pip install pyyaml"
        )
    return yaml.safe_load(path.read_text())


def fenced_sql(markdown: str) -> list[str]:
    """Every ```sql block, in order.

    A fence is a line whose first non-space characters are three or more
    backticks. The opening fence's info string decides whether the block counts;
    the closing fence is the next fence line of at least the same length. Blocks
    tagged anything other than `sql` are skipped rather than run, which is what
    keeps a ```text sample of output from being executed as a query.
    """
    blocks: list[str] = []
    inside = False
    is_sql = False
    collected: list[str] = []
    fence_length = 0
    for line in markdown.splitlines():
        match = re.match(r"(`{3,})\s*([A-Za-z0-9_+-]*)\s*$", line.lstrip())
        if not inside:
            if match:
                inside = True
                fence_length = len(match.group(1))
                is_sql = match.group(2).lower() == "sql"
                collected = []
            continue
        if match and len(match.group(1)) >= fence_length:
            if is_sql:
                blocks.append("\n".join(collected))
            inside = False
            continue
        collected.append(line)
    return blocks


def examples(descriptor: dict) -> list[tuple[str, str]]:
    """(label, sql) for every example the entry publishes."""
    docs = descriptor.get("docs") or {}
    found: list[tuple[str, str]] = []
    hello = docs.get("hello_world")
    if hello:
        found.append(("docs.hello_world", hello))
    extended = docs.get("extended_description")
    if extended:
        for index, block in enumerate(fenced_sql(extended), start=1):
            found.append((f"docs.extended_description sql block {index}", block))
    return found


def scrubbed_environment(sandbox: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in SCRUBBED_ENV_TO_TEMP:
        env[name] = str(sandbox)
    for name in SCRUBBED_ENV_TO_UNSET:
        env.pop(name, None)
    return env


def run_session(
    duckdb: str,
    extension: pathlib.Path | None,
    blocks: list[tuple[str, str]],
) -> tuple[bool, str, int]:
    """Run every block in one session. Returns (ok, output, blocks_reached)."""
    script = []
    for index, (_label, sql) in enumerate(blocks, start=1):
        script.append(f"SELECT '{MARKER}_{index}' AS example;")
        script.append(sql)
    body = "\n".join(script)

    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        extension_directory = work / "extensions"
        extension_directory.mkdir()
        sandbox = work / "home"
        sandbox.mkdir()

        prelude = (
            PRELUDE.format(extension_directory=extension_directory, extension=extension)
            if extension is not None
            else f"SET extension_directory='{extension_directory}';\n"
        )
        completed = subprocess.run(
            [duckdb, "-unsigned", "-init", os.devnull, "-batch", "-box"],
            input=prelude + body,
            text=True,
            capture_output=True,
            cwd=work,
            env=scrubbed_environment(sandbox),
        )

    output = completed.stdout + completed.stderr
    reached = len(re.findall(rf"{MARKER}_(\d+)", output))
    ok = completed.returncode == 0 and "Error:" not in output
    return ok, output, reached


def registered_functions(duckdb: str, extension: pathlib.Path) -> list[str]:
    """The function names the artifact adds to the catalog, derived by loading it."""
    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        extension_directory = work / "extensions"
        extension_directory.mkdir()
        sandbox = work / "home"
        sandbox.mkdir()
        script = (
            f"SET extension_directory='{extension_directory}';\n"
            "CREATE TABLE before AS SELECT DISTINCT function_name FROM duckdb_functions();\n"
            f"LOAD '{extension}';\n"
            "SELECT function_name FROM (SELECT DISTINCT function_name FROM duckdb_functions()) "
            "WHERE function_name NOT IN (SELECT function_name FROM before) ORDER BY 1;\n"
        )
        completed = subprocess.run(
            [duckdb, "-unsigned", "-init", os.devnull, "-batch", "-noheader", "-csv"],
            input=script,
            text=True,
            capture_output=True,
            cwd=work,
            env=scrubbed_environment(sandbox),
        )
    if completed.returncode != 0:
        raise SystemExit(f"could not read the catalog from {extension}:\n{completed.stdout}{completed.stderr}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def repository_in_cargo_toml() -> str:
    """`owner/name` from the `repository` URL in Cargo.toml."""
    text = (REPO_ROOT / "Cargo.toml").read_text()
    found = re.search(r'^\s*repository\s*=\s*"https://github\.com/([^"/]+/[^"/]+?)/?"', text, re.MULTILINE)
    if not found:
        raise SystemExit("Cargo.toml has no github repository URL")
    return found.group(1)


def matrix_entries() -> list[dict] | None:
    """Every entry the distribution matrix defines, or None if it is absent."""
    if not MATRIX.is_file():
        return None
    matrix = json.loads(MATRIX.read_text())
    return [entry for group in matrix.values() for entry in group.get("include", [])]


def known_platforms() -> set[str] | None:
    """Every `duckdb_arch` the distribution matrix defines, or None if absent."""
    entries = matrix_entries()
    return None if entries is None else {entry["duckdb_arch"] for entry in entries}


def built_platforms(excluded: str, opted_in: str) -> set[str] | None:
    """The platforms a given exclusion and opt-in list actually build and upload.

    This mirrors `should_run` in extension-ci-tools'
    `scripts/modify_distribution_matrix.py`: an arch is built when it is not
    excluded, and is either not opt-in or named in the opt-in list.
    `reduced_ci_mode` is left at the reusable workflow's default of `auto`,
    which that script maps to off.
    """
    entries = matrix_entries()
    if entries is None:
        return None
    excluded_set = set(filter(None, excluded.split(";")))
    opted_in_set = set(filter(None, opted_in.split(";")))
    return {
        entry["duckdb_arch"]
        for entry in entries
        if entry["duckdb_arch"] not in excluded_set
        and (not entry.get("opt_in") or entry["duckdb_arch"] in opted_in_set)
    }


def without_comments(text: str) -> str:
    """YAML text with its comments removed, for asking whether a name is used.

    A `#` opens a comment when it starts a line or follows whitespace, which is
    YAML's own rule. This does not track quoting, so a `#` inside a quoted
    string preceded by a space would truncate the line — that can only make a
    used name look unused, which reddens loudly, never the other way round.
    """
    return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in text.splitlines())


def dead_env_problems(workflow: dict, text: str) -> list[str]:
    """A workflow-level `env:` key that nothing in the file reads.

    A declared variable no step and no expression uses still reads like a pin,
    so the next person keeps it in step with something it does not control.
    Comments are stripped first: a name that appears only in the comment
    explaining it is not a use.
    """
    body = without_comments(text)
    problems = []
    for name in (workflow.get("env") or {}):
        # One occurrence is the declaration itself.
        if len(re.findall(rf"\b{re.escape(name)}\b", body)) <= 1:
            problems.append(
                f"{REHEARSAL.name} declares env.{name} and nothing outside its own comment "
                f"reads it. A variable no step uses still reads as a pin — wire it in or "
                f"delete it"
            )
    return problems


def inspection_coverage_problems(workflow: dict, inputs: dict) -> list[str]:
    """Every artifact this configuration uploads has to be looked at, and loaded or excused.

    The platform list is derived from the distribution matrix and the build
    job's own exclusions rather than read from a list written here, so opting a
    platform in — or upstream adding one — cannot quietly produce an artifact
    with no entry in the inspection job. Each entry either carries `load: true`,
    meaning the job runs the binary, or `load_skipped_because:` naming what
    stops it; a platform that is merely read and never run without saying so is
    the gap this closes.
    """
    built = built_platforms(
        str(inputs.get("exclude_archs", "")), str(inputs.get("opt_in_archs", ""))
    )
    if built is None:
        # The matrix is absent. `descriptor_problems` reports that once, rather
        # than every derived check reporting it again or going quiet.
        return []

    job = (workflow.get("jobs") or {}).get(INSPECTION_JOB) or {}
    include = ((job.get("strategy") or {}).get("matrix") or {}).get("include") or []
    by_arch = {str(entry.get("duckdb_arch")): entry for entry in include}

    problems = []
    for arch in sorted(built):
        entry = by_arch.get(arch)
        if entry is None:
            problems.append(
                f"{REHEARSAL.name} builds and uploads {arch}, but its `{INSPECTION_JOB}` job "
                f"has no entry for it — that artifact would be published with nothing having "
                f"looked at it"
            )
        elif entry.get("load") is not True and not str(entry.get("load_skipped_because", "")).strip():
            problems.append(
                f"{REHEARSAL.name}'s `{INSPECTION_JOB}` entry for {arch} neither loads the "
                f"artifact nor says why not — give it `load: true`, or a "
                f"`load_skipped_because:` naming what stops it"
            )
    for arch in sorted(by_arch):
        if arch not in built:
            problems.append(
                f"{REHEARSAL.name}'s `{INSPECTION_JOB}` job has an entry for {arch}, which this "
                f"configuration does not build — the download step would find no artifact"
            )
    return problems


def rehearsal_problems(descriptor: dict, workflow: dict | None) -> list[str]:
    """Where our own distribution workflow and the registry entry disagree.

    The entry declares which platforms the registry builds and which toolchains
    it installs; `MainDistributionPipeline.yml` calls the same reusable workflow
    with its own copy of both. Two copies that happen to agree today is what an
    acceptance criterion warned against, so this is where they are held
    together: a platform excluded in one and not the other means the run that
    goes green is not the run the registry will do. The platforms that survive
    those exclusions are then required to have an entry in the inspection job,
    so an artifact cannot be uploaded with nothing having looked at it.
    """
    if workflow is None:
        return [
            f"{REHEARSAL.name} is not in the tree, so the entry was compared with nothing. "
            f"A rehearsal that cannot be read is not a rehearsal that agrees"
        ]
    problems: list[str] = []
    extension = descriptor.get("extension") or {}
    job = ((workflow.get("jobs") or {}).get(BUILD_JOB) or {})
    inputs = job.get("with") or {}
    if not inputs:
        return [f"{REHEARSAL.name} has no `{BUILD_JOB}` job with inputs to compare"]

    for entry_field, workflow_field in (
        ("name", "extension_name"),
        ("excluded_platforms", "exclude_archs"),
        ("opt_in_platforms", "opt_in_archs"),
        ("requires_toolchains", "extra_toolchains"),
    ):
        entry_value = str(extension.get(entry_field, ""))
        workflow_value = str(inputs.get(workflow_field, ""))
        if entry_value != workflow_value:
            problems.append(
                f"description.yml's {entry_field} is {entry_value!r} but "
                f"{REHEARSAL.name}'s {workflow_field} is {workflow_value!r} — "
                f"the rehearsal would not build what the registry builds"
            )

    # The reusable workflow's ref and the ci_tools_version it is given have to
    # be the same, or the workflow that runs and the makefiles it drives come
    # from two different releases of extension-ci-tools.
    uses = str(job.get("uses", ""))
    pinned = uses.rsplit("@", 1)[-1] if "@" in uses else ""
    if pinned != str(inputs.get("ci_tools_version", "")):
        problems.append(
            f"{REHEARSAL.name} calls the reusable workflow at {pinned!r} but passes "
            f"ci_tools_version {inputs.get('ci_tools_version')!r}"
        )

    declared = str((workflow.get("env") or {}).get("DUCKDB_VERSION", ""))
    if declared != str(inputs.get("duckdb_version", "")):
        problems.append(
            f"{REHEARSAL.name}'s env.DUCKDB_VERSION is {declared!r} but the build job is "
            f"given duckdb_version {inputs.get('duckdb_version')!r}; the artifact names the "
            f"jobs download are built from the first and produced by the second"
        )

    problems += inspection_coverage_problems(workflow, inputs)

    return problems


def descriptor_problems(
    descriptor: dict, functions: list[str] | None, require_complete: bool = False
) -> list[str]:
    """Everything about the entry that disagrees with this tree.

    `require_complete` decides what an absent input means. The distribution
    matrix arrives through the `extension-ci-tools` submodule, and `make check`
    is documented as working in a clone made without `--recursive`, so its
    absence is a named skip locally. In CI, where the checkout is recursive, it
    is a failure — a check that reports clean because it could not look is the
    shape this repository keeps deleting, and `--require-inspection` in
    `check_no_network_deps.py` settles the same question the same way.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_artifact_version import version_in_cargo_toml

    problems: list[str] = []
    extension = descriptor.get("extension") or {}
    repo = descriptor.get("repo") or {}

    expected_version = version_in_cargo_toml()
    if str(extension.get("version")) != expected_version:
        problems.append(
            f"extension.version is {extension.get('version')!r}, but Cargo.toml says {expected_version!r}"
        )

    expected_repo = repository_in_cargo_toml()
    if extension.get("name") != expected_repo.split("/")[-1]:
        problems.append(
            f"extension.name is {extension.get('name')!r}, but the repository is {expected_repo!r} — "
            f"the registry requires the directory, the name and the built library to agree"
        )
    if repo.get("github") != expected_repo:
        problems.append(f"repo.github is {repo.get('github')!r}, but Cargo.toml says {expected_repo!r}")

    ref = str(repo.get("ref", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        problems.append(
            f"repo.ref is {ref!r}; the registry pins a full 40-character commit SHA, and an "
            f"abbreviated one or a branch name is not a pin"
        )

    platforms = known_platforms()
    if platforms is None:
        absent = (
            f"{MATRIX.relative_to(REPO_ROOT)} is not in the tree, so the entry's platform names "
            f"and the inspection job's coverage were not compared — run "
            f"`git submodule update --init --recursive`"
        )
        if require_complete:
            problems.append(absent)
        else:
            print(f"NOT CHECKED: {absent}", file=sys.stderr)
    else:
        for field in ("excluded_platforms", "opt_in_platforms"):
            for name in filter(None, str(extension.get(field, "")).split(";")):
                if name not in platforms:
                    problems.append(
                        f"extension.{field} names {name!r}, which is not a duckdb_arch in the "
                        f"distribution matrix — a misspelt exclusion excludes nothing"
                    )

    workflow = load_yaml(REHEARSAL) if REHEARSAL.is_file() else None
    problems += rehearsal_problems(descriptor, workflow)
    if workflow is not None:
        problems += dead_env_problems(workflow, REHEARSAL.read_text())

    if functions is not None:
        docs = descriptor.get("docs") or {}
        page = f"{docs.get('hello_world', '')}\n{docs.get('extended_description', '')}"
        for name in functions:
            if not re.search(rf"\b{re.escape(name)}\b", page):
                problems.append(
                    f"the artifact registers {name!r} and the entry never names it — "
                    f"the page a stranger reads would not mention a function they have"
                )

    return problems


def self_test() -> int:
    """Prove the extractor extracts and the runner reddens.

    Every case plants a specific defect and requires it to be reported. A
    checker that clears the file it was written against and has never been shown
    a broken one is indistinguishable from a checker that always clears.
    """
    failures: list[str] = []

    def expect(label: str, condition: bool) -> None:
        if not condition:
            failures.append(label)

    # A block scalar whose CONTENT contains lines that look like top-level YAML
    # keys and a markdown heading. This is the case a line-oriented regex over
    # the file gets wrong: it ends the block at `repo:` and finds one example
    # where there are three.
    synthetic = """\
extension:
  name: subtoken
  version: 9.9.9
repo:
  github: example/subtoken
  ref: deadbeef
docs:
  hello_world: |
    SELECT subtoken_embed('hello') AS hello;
    -- repo: this line is prose, not a key
    SELECT 2 AS also_hello;
  extended_description: |
    Prose about the extension.

    ```sql
    SELECT 3 AS first_block;
    ```

    ## extension: a heading that looks like a key

    ```python
    print("not sql, and must not be run")
    ```

    ```text
    extension: neither is this
    ```

    ```sql
    SELECT 4 AS second_block;
    ```
"""
    with tempfile.TemporaryDirectory() as workdir:
        path = pathlib.Path(workdir) / "description.yml"
        path.write_text(synthetic)
        descriptor = load_yaml(path)

        found = examples(descriptor)
        expect("finds the hello_world block and both sql blocks", len(found) == 3)
        if len(found) == 3:
            expect("hello_world survives a line that looks like a key", "also_hello" in found[0][1])
            expect("the first sql block is the first sql block", "first_block" in found[1][1])
            expect("the second sql block is found past a python block", "second_block" in found[2][1])
        expect(
            "the python block is not extracted",
            all("not sql" not in sql for _label, sql in found),
        )
        expect(
            "the text block is not extracted",
            all("neither is this" not in sql for _label, sql in found),
        )

        # Descriptor agreement: each of these must be reported.
        problems = descriptor_problems(descriptor, ["subtoken_embed", "a_function_the_page_never_names"])
        expect("a wrong version is caught", any("extension.version" in p for p in problems))
        expect("a wrong repository is caught", any("repo.github" in p for p in problems))
        expect("an abbreviated ref is caught", any("repo.ref" in p for p in problems))
        expect(
            "a registered function the page never names is caught",
            any("a_function_the_page_never_names" in p for p in problems),
        )
        expect(
            "a function the page does name is not reported",
            not any("'subtoken_embed'" in p for p in problems),
        )

    # A misspelt platform exclusion, checked only when the matrix is available.
    if known_platforms() is not None:
        typo = {
            "extension": {
                "name": "subtoken",
                "version": "0.0.0",
                "excluded_platforms": "wasm_mvp;windows_amd64_mingww",
            },
            "repo": {"github": "x/y", "ref": "0" * 40},
        }
        expect(
            "a misspelt platform exclusion is caught",
            any("windows_amd64_mingww" in p for p in descriptor_problems(typo, None)),
        )
        expect(
            "a real platform name is not reported",
            not any("'wasm_mvp'" in p for p in descriptor_problems(typo, None)),
        )
    else:
        print("self-test: SKIPPED the platform-name check — extension-ci-tools is not checked out")

    # The rehearsal workflow against the entry. Each field is disagreed with in
    # turn, because a comparison nobody has seen report a difference is a
    # comparison of a value with itself.
    # The exclusions the real entry declares, so `built_platforms` gives the
    # five this repository ships and the inspection matrix below is realistic.
    real_exclusions = "wasm_mvp;wasm_eh;wasm_threads;linux_amd64_musl;linux_arm64_musl;windows_amd64_mingw"
    entry = {
        "extension": {
            "name": "subtoken",
            "excluded_platforms": real_exclusions,
            "requires_toolchains": "rust;python3",
        }
    }
    inspection = {
        "strategy": {
            "matrix": {
                "include": [
                    {"duckdb_arch": "linux_amd64", "load": True},
                    {"duckdb_arch": "linux_arm64", "load": True},
                    {"duckdb_arch": "osx_amd64", "load": False, "load_skipped_because": "no Intel runner"},
                    {"duckdb_arch": "osx_arm64", "load": True},
                    {"duckdb_arch": "windows_amd64", "load": True},
                ]
            }
        }
    }
    agreeing = {
        "env": {"DUCKDB_VERSION": "v1.5.5"},
        "jobs": {
            "duckdb-stable-build": {
                "uses": "duckdb/extension-ci-tools/.github/workflows/_extension_distribution.yml@v1.5-variegata",
                "with": {
                    "extension_name": "subtoken",
                    "exclude_archs": real_exclusions,
                    "extra_toolchains": "rust;python3",
                    "ci_tools_version": "v1.5-variegata",
                    "duckdb_version": "v1.5.5",
                },
            },
            "no-network": inspection,
        },
    }
    expect("an agreeing rehearsal reports nothing", rehearsal_problems(entry, agreeing) == [])

    import copy

    for field, value, needle in (
        ("exclude_archs", "wasm_mvp", "excluded_platforms"),
        ("extra_toolchains", "rust", "requires_toolchains"),
        ("extension_name", "subtokend", "name"),
        ("ci_tools_version", "v1.5.5", "ci_tools_version"),
        ("duckdb_version", "v1.5.4", "DUCKDB_VERSION"),
        ("opt_in_archs", "windows_arm64", "opt_in_platforms"),
    ):
        broken = copy.deepcopy(agreeing)
        broken["jobs"]["duckdb-stable-build"]["with"][field] = value
        expect(
            f"a rehearsal that changed {field} is caught",
            any(needle in problem for problem in rehearsal_problems(entry, broken)),
        )

    missing = {"env": {}, "jobs": {"something-else": {}}}
    expect(
        "a rehearsal with no build job is caught",
        any("no `duckdb-stable-build` job" in problem for problem in rehearsal_problems(entry, missing)),
    )

    # An absent workflow file. This returned [] until 2026-08-25, so moving the
    # file aside left the whole comparison reporting "ok".
    expect(
        "a rehearsal that is not there at all is caught",
        any("is not in the tree" in problem for problem in rehearsal_problems(entry, None)),
    )

    # Inspection coverage, checked only when the matrix is available because the
    # platform list is derived from it rather than written here.
    if matrix_entries() is not None:
        expect(
            "the real exclusions build the five platforms the inspection job lists",
            built_platforms(real_exclusions, "")
            == {"linux_amd64", "linux_arm64", "osx_amd64", "osx_arm64", "windows_amd64"},
        )
        expect(
            "an opted-in platform joins the built set",
            "windows_arm64" in (built_platforms(real_exclusions, "windows_arm64") or set()),
        )

        dropped = copy.deepcopy(agreeing)
        dropped["jobs"]["no-network"]["strategy"]["matrix"]["include"] = [
            item
            for item in inspection["strategy"]["matrix"]["include"]
            if item["duckdb_arch"] != "linux_arm64"
        ]
        expect(
            "a built platform with no inspection entry is caught",
            any("no entry for it" in problem for problem in rehearsal_problems(entry, dropped)),
        )

        silent = copy.deepcopy(agreeing)
        for item in silent["jobs"]["no-network"]["strategy"]["matrix"]["include"]:
            if item["duckdb_arch"] == "windows_amd64":
                item["load"] = False
        expect(
            "an entry that stops loading without saying why is caught",
            any("neither loads" in problem for problem in rehearsal_problems(entry, silent)),
        )

        excused = copy.deepcopy(agreeing)
        for item in excused["jobs"]["no-network"]["strategy"]["matrix"]["include"]:
            if item["duckdb_arch"] == "windows_amd64":
                item["load"] = False
                item["load_skipped_because"] = "a stated reason"
        expect("and a stated reason is accepted", rehearsal_problems(entry, excused) == [])

        opted = copy.deepcopy(agreeing)
        opted["jobs"]["duckdb-stable-build"]["with"]["opt_in_archs"] = "windows_arm64"
        opted_entry = {"extension": {**entry["extension"], "opt_in_platforms": "windows_arm64"}}
        expect(
            "opting a platform in without an inspection entry is caught",
            any("windows_arm64" in problem for problem in rehearsal_problems(opted_entry, opted)),
        )

        surplus = copy.deepcopy(agreeing)
        surplus["jobs"]["no-network"]["strategy"]["matrix"]["include"].append(
            {"duckdb_arch": "wasm_mvp", "load": True}
        )
        expect(
            "an inspection entry for a platform nothing builds is caught",
            any("does not build" in problem for problem in rehearsal_problems(entry, surplus)),
        )
    else:
        print("self-test: SKIPPED the inspection-coverage checks — extension-ci-tools is not checked out")

    # A declared env variable nothing reads. env.CI_TOOLS_VERSION was exactly
    # this: named in its own comment as a pin, and read by no step.
    expect(
        "an env variable used by a step is not reported",
        dead_env_problems(
            {"env": {"DUCKDB_VERSION": "v1.5.5"}},
            "env:\n  DUCKDB_VERSION: v1.5.5\njobs:\n  x:\n    steps:\n"
            "      - run: echo ${{ env.DUCKDB_VERSION }}\n",
        )
        == [],
    )
    expect(
        "an env variable nothing reads is caught",
        any(
            "CI_TOOLS_VERSION" in problem
            for problem in dead_env_problems(
                {"env": {"CI_TOOLS_VERSION": "v1.5-variegata"}},
                "env:\n  # Must equal the registry's CI_TOOLS_VERSION.\n"
                "  CI_TOOLS_VERSION: v1.5-variegata\njobs: {}\n",
            )
        ),
    )
    expect(
        "a comment is not a use",
        "#" not in without_comments("  DUCKDB_VERSION: v1.5.5  # and a comment"),
    )
    expect(
        "and stripping comments does not eat the declaration",
        "DUCKDB_VERSION: v1.5.5" in without_comments("  DUCKDB_VERSION: v1.5.5  # a comment"),
    )

    # And the runner itself. No extension needed: this proves that a failing
    # example makes this script fail, which is the whole point of running them.
    duckdb = shutil.which("duckdb")
    if duckdb is None:
        print("self-test: SKIPPED the runner check — the duckdb CLI is not on PATH")
    else:
        ok, _output, reached = run_session(duckdb, None, [("good", "SELECT 1 AS a;")])
        expect("a working example passes", ok and reached == 1)

        ok, output, reached = run_session(
            duckdb,
            None,
            [("first", "SELECT 1 AS a;"), ("second", "SELECT * FROM a_table_nobody_created;")],
        )
        expect("an example referring to a fixture nobody created fails", not ok)
        expect("and the output says how far the session got", reached == 2)
        expect("and names the missing table", "a_table_nobody_created" in output)

        ok, _output, _reached = run_session(duckdb, None, [("syntax", "SELEKT 1;")])
        expect("a syntactically broken example fails", not ok)

    if failures:
        for failure in failures:
            print(f"self-test FAILED: {failure}", file=sys.stderr)
        return 1

    print(
        "self-test ok: the extractor finds a block scalar past lines that look like YAML keys, "
        "takes sql fences and leaves python and text ones, the runner fails on a missing "
        "fixture and on a syntax error, an absent rehearsal workflow is a failure rather than "
        "a clean report, a built platform with no inspection entry and one that stops loading "
        "without saying why are both caught, and a declared env variable nothing reads is "
        "reported"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extension", type=pathlib.Path)
    parser.add_argument("--description", type=pathlib.Path, default=DESCRIPTION)
    parser.add_argument("--duckdb", default="duckdb")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail rather than skip when an input this check reads is absent",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.extension is None:
        parser.error("--extension is required: the examples are run against a packaged artifact")
    if not args.extension.is_file():
        raise SystemExit(f"no artifact at {args.extension}")
    # The session runs with its cwd in a temporary directory, so a relative path
    # would resolve to nothing there.
    args.extension = args.extension.resolve()
    if shutil.which(args.duckdb) is None:
        raise SystemExit(f"{args.duckdb} is not on PATH; the examples are run through the DuckDB CLI")

    descriptor = load_yaml(args.description)
    failures = 0

    functions = registered_functions(args.duckdb, args.extension)
    problems = descriptor_problems(descriptor, functions, require_complete=args.require_complete)
    if problems:
        print(f"FAIL: {args.description} does not describe this tree:", file=sys.stderr)
        for problem in problems:
            print(f"       {problem}", file=sys.stderr)
        failures += 1
    else:
        built = built_platforms(
            str((descriptor.get("extension") or {}).get("excluded_platforms", "")),
            str((descriptor.get("extension") or {}).get("opt_in_platforms", "")),
        )
        # The sentence names what was compared. When the matrix is absent the
        # platform names were not read at all, and saying they agree would be
        # the same defect this check exists to catch.
        coverage = (
            f"the {len(built)} platforms it builds carry matrix names and are each inspected"
            if built is not None
            else "NOTHING about platform names or inspection coverage — see the NOT CHECKED line"
        )
        print(
            f"ok: the entry's version, repository and ref shape agree with the tree, it names "
            f"all {len(functions)} functions the artifact registers, and {coverage}"
        )

    blocks = examples(descriptor)
    if not blocks:
        print(f"FAIL: {args.description} publishes no SQL example", file=sys.stderr)
        return 1

    ok, output, reached = run_session(args.duckdb, args.extension, blocks)
    if not ok:
        stopped = blocks[min(reached, len(blocks)) - 1][0] if reached else "the prelude"
        print(
            f"FAIL: the published examples do not run. The session stopped in {stopped} "
            f"({reached} of {len(blocks)} reached).",
            file=sys.stderr,
        )
        print(output, file=sys.stderr)
        failures += 1
    else:
        print(
            f"ok: all {len(blocks)} published examples ran in one session against "
            f"{args.extension}, with no fixture the entry does not create for itself"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
