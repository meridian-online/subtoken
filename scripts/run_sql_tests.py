#!/usr/bin/env python3
"""Run every `test/sql/*.sql` file against a real DuckDB with the extension loaded.

Each file is executed by the `duckdb` CLI in its own process, preceded by a
prelude that points `extension_directory` at an empty temporary directory,
snapshots the function catalog, loads the packaged artifact, and defines the
`must(label, condition)` macro the assertions use. `must` raises a DuckDB error
when its condition is anything other than TRUE — NULL fails too, so a mistyped
column name cannot pass silently — and a raised error makes the CLI exit
non-zero, which is what this script reads.

The environment each file runs in is scrubbed: `HOME` and every model-cache
variable point at empty temporary directories, and the proxy variables are
removed. A model that needed a download, a cache directory or a configuration
file would fail here rather than quietly finding one on the developer's machine.
That is evidence for "no network and no configuration"; it is not a network
sandbox, and the dependency-tree half of that claim is
`scripts/check_no_network_deps.py`.

Stdlib only.

    scripts/run_sql_tests.py --extension build/subtoken.duckdb_extension
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

PRELUDE = """\
SET extension_directory='{extension_directory}';
CREATE OR REPLACE MACRO must(label, condition) AS
    CASE WHEN condition IS TRUE THEN label
         ELSE error('ASSERTION FAILED: ' || label) END;
CREATE TABLE subtoken_baseline_functions AS
    SELECT DISTINCT function_name FROM duckdb_functions();
LOAD '{extension}';
"""

# Cleared or redirected so nothing under the developer's home directory can
# stand in for the bundled model.
SCRUBBED_ENV_TO_TEMP = ("HOME", "XDG_CACHE_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
SCRUBBED_ENV_TO_UNSET = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")


def scrubbed_environment(sandbox: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in SCRUBBED_ENV_TO_TEMP:
        env[name] = str(sandbox)
    for name in SCRUBBED_ENV_TO_UNSET:
        env.pop(name, None)
    return env


def run_one(duckdb: str, extension: pathlib.Path, sql_file: pathlib.Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        extension_directory = work / "extensions"
        extension_directory.mkdir()
        sandbox = work / "home"
        sandbox.mkdir()

        script = PRELUDE.format(
            extension_directory=extension_directory,
            extension=extension,
        ) + sql_file.read_text()

        completed = subprocess.run(
            [duckdb, "-unsigned", "-init", os.devnull, "-batch", "-box"],
            input=script,
            text=True,
            capture_output=True,
            cwd=work,
            env=scrubbed_environment(sandbox),
        )

    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        return False, output
    # Belt and braces: a future CLI that reported an error without a non-zero
    # exit code would otherwise pass.
    if "ASSERTION FAILED" in output or "Error:" in output:
        return False, output
    return True, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", required=True, type=pathlib.Path)
    parser.add_argument("--duckdb", default=os.environ.get("DUCKDB", "duckdb"))
    parser.add_argument("--sql-dir", type=pathlib.Path, default=REPO_ROOT / "test" / "sql")
    parser.add_argument(
        "--only",
        help="run only SQL files whose name contains this substring",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    duckdb = shutil.which(args.duckdb)
    if duckdb is None:
        print(
            f"the duckdb CLI ({args.duckdb}) is not on PATH; the SQL tests need it",
            file=sys.stderr,
        )
        return 2

    extension = args.extension.resolve()
    if not extension.is_file():
        print(f"no packaged extension at {extension}; run `make extension`", file=sys.stderr)
        return 2

    sql_files = sorted(args.sql_dir.glob("*.sql"))
    if args.only:
        sql_files = [path for path in sql_files if args.only in path.name]
    if not sql_files:
        where = f"{args.sql_dir}" + (f" matching {args.only!r}" if args.only else "")
        print(f"no SQL test files in {where}", file=sys.stderr)
        return 2

    failures = 0
    for sql_file in sql_files:
        passed, output = run_one(duckdb, extension, sql_file)
        print(f"{'PASS' if passed else 'FAIL'}  {sql_file.relative_to(REPO_ROOT)}")
        if args.verbose or not passed:
            print(output)
        failures += 0 if passed else 1

    print(f"\n{len(sql_files) - failures}/{len(sql_files)} SQL test files passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
