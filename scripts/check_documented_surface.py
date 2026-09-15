#!/usr/bin/env python3
"""Fail when the documented SQL surface disagrees with the loaded catalog.

Each file in `SCANNED` writes the extension's function table out in prose: the
function, its return type, and for `subtoken_cache_stats()` the field list of
the STRUCT it returns. Each copy of a signature is another chance to be wrong,
and one of them was: the page listed five fields for a six-field struct,
omitting `uncached` — the counter that tells a reader which side of the cache's
bound they are on.

`description.yml` is in `SCANNED` because it is the copy a stranger reads. It is
the registry submission, it carries the same table, and it is the only one of
the three that nobody arrives at through this repository.

DuckDB settles it mechanically. `duckdb_functions()` reports what a loaded
extension really registered, with the return type and, for a STRUCT, its field
names in order. This loads the local build and compares.

WHAT IS DERIVED (the source of truth)
    `duckdb_functions()` after `LOAD`, taken as the delta against a snapshot
    made before the load, so nothing built into the CLI is mistaken for ours.
    `extension_directory` points at an empty temporary directory, so an
    installed community build cannot be picked up instead.

WHAT IS ASSERTED, for every scanned file
    1. Its function table names every function the catalog registers, and names
       no function the catalog has not.
    2. The return type it gives each function is the one the catalog reports. A
       STRUCT is written without field types, so those two are compared on field
       names in order; everything else is compared literally.
    3. Every `STRUCT(...)` written anywhere else in the file — in prose, in a
       doc comment — also matches a registered signature.
    4. The file is present and does write a function table. A scan that reports
       clean because it found nothing to read is not a scan.

WHAT IS NOT ASSERTED
    Behaviour. This says the docs describe the registered surface; whether that
    surface does anything is `test/sql/`.

Needs the duckdb CLI and a packaged extension. Stdlib only.

    scripts/check_documented_surface.py --extension build/subtoken.duckdb_extension
    scripts/check_documented_surface.py --self-test
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Files that write the surface out in prose and must agree with the catalog.
SCANNED = ("README.md", "crates/subtoken-duckdb/src/lib.rs", "description.yml")

#: A `STRUCT(...)` signature as the docs write it: field names, no types.
DOCUMENTED_STRUCT = re.compile(r"STRUCT\(([^)]*)\)")

#: A `STRUCT(...)` signature as DuckDB writes it: `name TYPE, name TYPE`.
CATALOG_STRUCT = re.compile(r"^STRUCT\((.*)\)$", re.DOTALL)

#: A row of a documented function table — the call, then the return type. The
#: optional `//!` is a Rust module doc's; the leading whitespace is the indent a
#: YAML block scalar puts on every line of `description.yml`'s page.
DOCUMENTED_ROW = re.compile(r"^\s*(?://!\s*)?\|\s*`(\w+)\([^`]*\)`\s*\|\s*`([^`]+)`\s*\|")


def catalog(duckdb: str, extension: pathlib.Path) -> dict[str, str]:
    """Function name to return type, for the functions this extension registers."""
    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        extensions = work / "extensions"
        extensions.mkdir()
        script = f"""
SET extension_directory='{extensions}';
CREATE TABLE before AS SELECT DISTINCT function_name FROM duckdb_functions();
LOAD '{extension}';
SELECT function_name || chr(9) || return_type
FROM (SELECT DISTINCT function_name, return_type FROM duckdb_functions()
      WHERE function_name NOT IN (SELECT function_name FROM before))
ORDER BY function_name;
"""
        completed = subprocess.run(
            [duckdb, "-unsigned", "-init", os.devnull, "-batch", "-noheader", "-list"],
            input=script,
            text=True,
            capture_output=True,
            cwd=work,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"reading the catalog failed:\n{completed.stdout}{completed.stderr}")

    found = {}
    for line in completed.stdout.splitlines():
        if "\t" not in line:
            continue
        name, return_type = line.split("\t", 1)
        found[name.strip()] = return_type.strip()
    if not found:
        raise SystemExit("the catalog delta is empty; the extension registered nothing")
    return found


def catalog_struct_fields(return_type: str) -> list[str] | None:
    """Field names, in order, from a catalog STRUCT return type."""
    match = CATALOG_STRUCT.match(return_type)
    if not match:
        return None
    return [field.strip().split()[0] for field in match.group(1).split(",") if field.strip()]


def documented_struct_fields(text: str) -> list[list[str]]:
    """Every STRUCT field list a document writes out."""
    signatures = []
    for match in DOCUMENTED_STRUCT.finditer(text):
        fields = [field.strip() for field in match.group(1).split(",") if field.strip()]
        # A signature written with types is the catalog's own spelling quoted
        # back; take the names so both forms compare alike.
        signatures.append([field.split()[0] for field in fields])
    return signatures


def documented_rows(text: str) -> list[tuple[str, str]]:
    """(function name, written return type) for every function-table row."""
    rows = []
    for line in text.splitlines():
        match = DOCUMENTED_ROW.match(line)
        if match:
            rows.append((match.group(1), match.group(2)))
    return rows


def return_types_agree(written: str, reported: str) -> bool:
    """Is the return type a document writes the one the catalog reports?

    A STRUCT is written on the page without its field types, so when either side
    is one the comparison is on field names in order. Anything else — `FLOAT[]`,
    `BOOLEAN`, `BIGINT`, `VARCHAR` — is compared literally, so `DOUBLE[]` where
    the catalog says `FLOAT[]` is a difference rather than a near miss.
    """
    written_fields = catalog_struct_fields(written)
    reported_fields = catalog_struct_fields(reported)
    if written_fields is not None or reported_fields is not None:
        return written_fields == reported_fields
    return written.strip() == reported.strip()


def file_problems(relative: str, text: str, registered: dict[str, str]) -> list[str]:
    """Everything a scanned file says about the surface that the catalog denies."""
    struct_fields = {
        name: fields
        for name, return_type in registered.items()
        if (fields := catalog_struct_fields(return_type)) is not None
    }
    problems: list[str] = []

    for written in documented_struct_fields(text):
        if not any(written == fields for fields in struct_fields.values()):
            problems.append(
                f"{relative} writes STRUCT({', '.join(written)}), which matches no registered "
                f"function. The catalog has: "
                + "; ".join(
                    f"{name} STRUCT({', '.join(fields)})" for name, fields in struct_fields.items()
                )
            )

    rows = documented_rows(text)
    if not rows:
        problems.append(
            f"{relative} writes no function-table row. It is scanned because it publishes the "
            f"surface; a file that has stopped publishing it agrees with the catalog only by "
            f"saying nothing"
        )
        return problems

    for name, written in rows:
        if name not in registered:
            problems.append(
                f"{relative}'s function table names {name}(), which the catalog has not. "
                f"Registered: {sorted(registered)}"
            )
        elif not return_types_agree(written, registered[name]):
            problems.append(
                f"{relative} gives {name}() the return type `{written}`, but the loaded "
                f"catalog reports `{registered[name]}`"
            )

    omitted = set(registered) - {name for name, _ in rows}
    if omitted:
        problems.append(
            f"{relative}'s function table omits registered functions: {sorted(omitted)}"
        )
    return problems


def self_test() -> int:
    """Prove each matcher sees what it is looking for."""
    catalog_type = "STRUCT(hits BIGINT, misses BIGINT, encoded BIGINT, uncached BIGINT)"
    fields = catalog_struct_fields(catalog_type)
    if fields != ["hits", "misses", "encoded", "uncached"]:
        print(f"self-test FAILED: catalog parse gave {fields}", file=sys.stderr)
        return 1
    if catalog_struct_fields("BIGINT") is not None:
        print("self-test FAILED: a non-STRUCT was parsed as one", file=sys.stderr)
        return 1

    stale = "| `subtoken_cache_stats()` | `STRUCT(hits, misses, encoded, entries, capacity)` |"
    found = documented_struct_fields(stale)
    if found != [["hits", "misses", "encoded", "entries", "capacity"]]:
        print(f"self-test FAILED: doc parse gave {found}", file=sys.stderr)
        return 1
    if found[0] == fields:
        print("self-test FAILED: the stale signature compared equal", file=sys.stderr)
        return 1

    reordered = documented_struct_fields("STRUCT(misses, hits)")
    if reordered == [["hits", "misses"]]:
        print("self-test FAILED: field order is not being compared", file=sys.stderr)
        return 1

    # The three spellings of one table row: README's plain markdown, the Rust
    # module doc's `//!` prefix, and description.yml's block-scalar indent. A
    # row parser that reads only the first leaves the other two unscanned while
    # reporting clean.
    three_spellings = (
        "| `subtoken_embed(text VARCHAR)` | `FLOAT[]` | the vector for one string |\n"
        "//! | `subtoken_version()` | `VARCHAR` | which build |\n"
        "    | `subtoken_cache_clear()` | `BIGINT` | drop the cached vectors |\n"
    )
    rows = documented_rows(three_spellings)
    if rows != [
        ("subtoken_embed", "FLOAT[]"),
        ("subtoken_version", "VARCHAR"),
        ("subtoken_cache_clear", "BIGINT"),
    ]:
        print(f"self-test FAILED: the row parser gave {rows}", file=sys.stderr)
        return 1
    if documented_rows("subtoken_embed(text VARCHAR) returns FLOAT[]") != []:
        print("self-test FAILED: prose was read as a table row", file=sys.stderr)
        return 1

    # A return type is compared, not merely present. `DOUBLE[]` for `FLOAT[]` is
    # the mutation that went green while description.yml was unscanned.
    if not return_types_agree("FLOAT[]", "FLOAT[]"):
        print("self-test FAILED: a matching scalar return type compared unequal", file=sys.stderr)
        return 1
    if return_types_agree("DOUBLE[]", "FLOAT[]"):
        print("self-test FAILED: DOUBLE[] compared equal to FLOAT[]", file=sys.stderr)
        return 1
    if not return_types_agree("STRUCT(hits, misses)", "STRUCT(hits BIGINT, misses BIGINT)"):
        print("self-test FAILED: an untyped STRUCT did not match the catalog's", file=sys.stderr)
        return 1
    if return_types_agree("STRUCT(hits, evictions)", "STRUCT(hits BIGINT, misses BIGINT)"):
        print("self-test FAILED: a renamed STRUCT field compared equal", file=sys.stderr)
        return 1
    if return_types_agree("STRUCT(hits, misses)", "BIGINT"):
        print("self-test FAILED: a STRUCT compared equal to a scalar", file=sys.stderr)
        return 1

    # And the per-file comparison, each defect planted in turn.
    truth = {"subtoken_embed": "FLOAT[]", "subtoken_cache_stats": "STRUCT(hits BIGINT, misses BIGINT)"}
    good = (
        "| `subtoken_embed(text VARCHAR)` | `FLOAT[]` | the vector |\n"
        "| `subtoken_cache_stats()` | `STRUCT(hits, misses)` | the cache |\n"
    )
    for label, text, needle in (
        ("a clean file reports nothing", good, None),
        (
            "a wrong return type is caught",
            good.replace("`FLOAT[]`", "`DOUBLE[]`"),
            "DOUBLE[]",
        ),
        (
            "a renamed STRUCT field is caught",
            good.replace("STRUCT(hits, misses)", "STRUCT(hits, evictions)"),
            "evictions",
        ),
        (
            "an omitted function is caught",
            "| `subtoken_embed(text VARCHAR)` | `FLOAT[]` | the vector |\n",
            "omits registered functions",
        ),
        (
            "a function the catalog has not is caught",
            good + "| `embed_all(text VARCHAR)` | `FLOAT[]` | invented |\n",
            "embed_all",
        ),
        ("a file with no table row is caught", "prose only\n", "no function-table row"),
    ):
        found = file_problems("a_file", text, truth)
        if needle is None:
            if found:
                print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
                return 1
        elif not any(needle in problem for problem in found):
            print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
            return 1

    print(
        "self-test ok: the catalog parser reads a STRUCT and rejects a scalar, the row parser "
        "reads all three spellings of a table row and not prose, DOUBLE[] does not compare equal "
        "to FLOAT[], and a wrong type, a renamed field, an omitted function, an invented one and "
        "an empty table are each reported"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=pathlib.Path)
    parser.add_argument("--duckdb", default=os.environ.get("DUCKDB", "duckdb"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.extension is None:
        print("--extension is required unless --self-test is given", file=sys.stderr)
        return 2
    duckdb = shutil.which(args.duckdb)
    if duckdb is None:
        print(f"the duckdb CLI ({args.duckdb}) is not on PATH", file=sys.stderr)
        return 2
    if not args.extension.is_file():
        print(f"no packaged extension at {args.extension}; run `make extension`", file=sys.stderr)
        return 2

    registered = catalog(duckdb, args.extension.resolve())

    problems: list[str] = []
    rows_read = 0
    for relative in SCANNED:
        path = REPO_ROOT / relative
        if not path.is_file():
            problems.append(
                f"{relative} is not in the tree. It is one of the files that publishes the "
                f"surface, and a scan that skips a missing file reports clean for having "
                f"nothing to read"
            )
            continue
        text = path.read_text()
        rows_read += len(documented_rows(text))
        problems += file_problems(relative, text, registered)

    if problems:
        print("FAIL: the documented surface disagrees with the loaded catalog:", file=sys.stderr)
        for problem in problems:
            print(f"       {problem}", file=sys.stderr)
        return 1

    print(
        f"ok: {len(registered)} registered functions, and the {rows_read} function-table rows "
        f"written across {len(SCANNED)} files — {', '.join(SCANNED)} — name them all, with the "
        f"return types the loaded catalog reports"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
