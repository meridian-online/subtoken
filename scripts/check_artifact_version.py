#!/usr/bin/env python3
"""Fail if the version a packaged artifact self-reports is not the version in this tree.

An extension carries its own version in the DuckDB metadata trailer, and that
number is what `SELECT * FROM duckdb_extensions()` shows a user and what the
registry records against a published build. Nothing else in the build reads it
back, so a wrong one ships silently: every test still passes, the extension
still loads, and the only symptom is that the version a stranger sees is not the
version they got.

THE DEFECT THIS EXISTS FOR is mechanical and was measured on a sibling repo, not
imagined. `extension-ci-tools/makefiles/c_api_extensions/base.Makefile` writes
the version into `configure/extension_version.txt` with

    configure/extension_version.txt:
    	@ $(VERSION_COMMAND)

which is a make file-target rule with no prerequisites: it runs when the file is
absent and never again. A copy of that file committed to git therefore arrives
in every checkout already present, is never refreshed, and `make release` stamps
whatever it says onto the artifact. In the sibling repo it said `0.6.23` while
`Cargo.toml` said `0.6.56`, and every gate stayed green, because the other build
path takes its version from `Cargo.toml` and is unaffected.

WHAT THIS CHECKS, and why each expectation comes from where it does:

    extension version   `Cargo.toml`, `[workspace.package] version`
        Read here rather than accepted as an argument. Taking it from the same
        `$(EXTENSION_VERSION)` the Makefile stamped with would compare a number
        against itself, which is a check that cannot fail.

    duckdb version      `MIN_DUCKDB_VERSION` in crates/subtoken-duckdb/src/lib.rs
        The trailer's FIELD3 is the stable-C-API floor the artifact declares.
        The Makefile's `TARGET_DUCKDB_VERSION` is a second copy of that constant
        and says in a comment that it mirrors this one; comparing the trailer to
        the Rust source is what makes the two copies unable to drift apart in
        silence.

    abi type            C_STRUCT, always
        `USE_UNSTABLE_C_API` is deliberately unset, and the unstable ABI would
        pin the binary to one exact DuckDB release. A build that acquired it
        would still load — on precisely one version — so it has to be asserted.

Stdlib only.

    scripts/check_artifact_version.py --artifact build/release/extension/subtoken/subtoken.duckdb_extension
    scripts/check_artifact_version.py --compare build/subtoken.duckdb_extension build/release/.../subtoken.duckdb_extension
    scripts/check_artifact_version.py --show build/subtoken.duckdb_extension
    scripts/check_artifact_version.py --self-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

FIELD_WIDTH = 32
SIGNATURE_BYTES = 256
FIELD_BYTES = FIELD_WIDTH * 8
TRAILER_BYTES = 534

# The 22 bytes that open the trailer: a WebAssembly custom section header naming
# `duckdb_signature`. Checked so that a file with no trailer at all is an error
# rather than 256 bytes of whatever happened to be at the end being read as
# fields.
TRAILER_HEADER = b"\x00\x93\x04\x10duckdb_signature\x80\x04"

# Written FIELD8 first, FIELD1 last, so this is file order.
FIELD_NAMES = (
    "field8",
    "field7",
    "field6",
    "abi_type",
    "extension_version",
    "duckdb_version",
    "platform",
    "magic",
)

# FIELD1's fixed value. DuckDB reads it to decide the file is an extension.
MAGIC = "4"

EXPECTED_ABI = "C_STRUCT"

# The fields two recipes building the same tree must agree on. `platform` is
# excluded on purpose: a cross-compiled build legitimately names a platform the
# host does not.
COMPARED_FIELDS = ("extension_version", "duckdb_version", "abi_type", "magic")


class TrailerError(Exception):
    """The file does not carry a readable DuckDB metadata trailer."""


def read_trailer(artifact: pathlib.Path) -> dict[str, str]:
    """The eight metadata fields of a packaged `.duckdb_extension`.

    Raises TrailerError rather than returning a partial answer: every caller
    here treats an unreadable trailer as a failure, and a function that returned
    empty strings for a truncated file would let the comparison below pass by
    matching nothing against nothing.
    """
    if not artifact.is_file():
        raise TrailerError(f"no such artifact: {artifact}")
    data = artifact.read_bytes()
    if len(data) < TRAILER_BYTES:
        raise TrailerError(f"{artifact} is {len(data)} bytes, shorter than the {TRAILER_BYTES}-byte trailer")

    header = data[-TRAILER_BYTES : -TRAILER_BYTES + len(TRAILER_HEADER)]
    if header != TRAILER_HEADER:
        raise TrailerError(
            f"{artifact} does not end in a duckdb_signature section — it is a raw "
            f"library, not a packaged extension"
        )

    block = data[-(SIGNATURE_BYTES + FIELD_BYTES) : -SIGNATURE_BYTES]
    fields = {}
    for index, name in enumerate(FIELD_NAMES):
        raw = block[index * FIELD_WIDTH : (index + 1) * FIELD_WIDTH]
        fields[name] = raw.rstrip(b"\0").decode("ascii", errors="replace")
    if fields["magic"] != MAGIC:
        raise TrailerError(
            f"{artifact}'s trailer has magic {fields['magic']!r}, expected {MAGIC!r} — "
            f"the fields are not where they are supposed to be"
        )
    return fields


def version_in_cargo_toml(path: pathlib.Path | None = None) -> str:
    """The workspace version, from the `[workspace.package]` table.

    Scoped to that table rather than taken from the first `version =` in the
    file: a `[workspace.dependencies]` entry carries versions too, and the first
    match in a differently ordered manifest would be one of those.
    """
    path = path or (REPO_ROOT / "Cargo.toml")
    text = path.read_text()
    section = re.search(r"^\[workspace\.package\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL)
    if not section:
        raise SystemExit(f"{path} has no [workspace.package] table")
    found = re.search(r'^\s*version\s*=\s*"([^"]+)"', section.group(1), re.MULTILINE)
    if not found:
        raise SystemExit(f"{path}'s [workspace.package] table has no version")
    return found.group(1)


def min_duckdb_version(path: pathlib.Path | None = None) -> str:
    """`MIN_DUCKDB_VERSION` from the extension crate's source."""
    path = path or (REPO_ROOT / "crates" / "subtoken-duckdb" / "src" / "lib.rs")
    found = re.search(r'MIN_DUCKDB_VERSION\s*:\s*&str\s*=\s*"([^"]+)"', path.read_text())
    if not found:
        raise SystemExit(f"{path} does not define MIN_DUCKDB_VERSION")
    return found.group(1)


def disagreements(
    fields: dict[str, str],
    expected_version: str,
    expected_duckdb_version: str,
    expected_platform: str | None,
) -> list[str]:
    problems = []
    if fields["extension_version"] != expected_version:
        problems.append(
            f"the artifact self-reports version {fields['extension_version']!r}, "
            f"but Cargo.toml says {expected_version!r}"
        )
    if fields["duckdb_version"] != expected_duckdb_version:
        problems.append(
            f"the artifact declares a DuckDB floor of {fields['duckdb_version']!r}, "
            f"but MIN_DUCKDB_VERSION says {expected_duckdb_version!r}"
        )
    if fields["abi_type"] != EXPECTED_ABI:
        problems.append(
            f"the artifact declares ABI {fields['abi_type']!r}, expected {EXPECTED_ABI!r} — "
            f"the unstable ABI pins the binary to one exact DuckDB release"
        )
    if expected_platform is not None and fields["platform"] != expected_platform:
        problems.append(
            f"the artifact is stamped for platform {fields['platform']!r}, expected {expected_platform!r}"
        )
    return problems


def synthetic_artifact(
    directory: pathlib.Path,
    name: str,
    *,
    extension_version: str = "9.9.9",
    duckdb_version: str = "v1.2.0",
    platform: str = "osx_arm64",
    abi_type: str = EXPECTED_ABI,
    magic: str = MAGIC,
    header: bytes = TRAILER_HEADER,
    body: bytes = b"not really a shared library, and it does not need to be",
) -> pathlib.Path:
    """A file carrying a metadata trailer, for the self-test to read back.

    Composed here from the constants above rather than by calling
    `append_extension_metadata.py`: driving the writer this reader is meant to
    check would make the two agree by construction, and a layout error in both
    would cancel out.
    """
    values = {
        "field8": "",
        "field7": "",
        "field6": "",
        "abi_type": abi_type,
        "extension_version": extension_version,
        "duckdb_version": duckdb_version,
        "platform": platform,
        "magic": magic,
    }
    trailer = bytearray(header)
    for field in FIELD_NAMES:
        trailer += values[field].encode("ascii").ljust(FIELD_WIDTH, b"\0")
    trailer += b"\0" * SIGNATURE_BYTES
    path = directory / name
    path.write_bytes(body + bytes(trailer))
    return path


def self_test() -> int:
    """Prove the reader reads, and that each expectation can actually fail.

    A checker that clears the tree it was written against and has never been
    shown a wrong one is indistinguishable from a checker that always clears.
    Every case below plants a specific defect and requires it to be reported.
    """
    failures = []

    def expect(label: str, condition: bool) -> None:
        if not condition:
            failures.append(label)

    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)

        # The reader reads what a writer wrote, at the right offsets.
        good = synthetic_artifact(work, "good.duckdb_extension", extension_version="1.2.3")
        fields = read_trailer(good)
        expect("reads the extension version", fields["extension_version"] == "1.2.3")
        expect("reads the duckdb version", fields["duckdb_version"] == "v1.2.0")
        expect("reads the platform", fields["platform"] == "osx_arm64")
        expect("reads the abi type", fields["abi_type"] == EXPECTED_ABI)
        expect("the unused fields are empty", fields["field6"] == "" and fields["field8"] == "")

        # And a matching tree clears.
        expect(
            "a matching artifact reports nothing",
            disagreements(fields, "1.2.3", "v1.2.0", "osx_arm64") == [],
        )

        # The measured defect: the artifact says an older version than the tree.
        expect(
            "a stale version is caught",
            any("self-reports version" in problem for problem in disagreements(fields, "1.2.4", "v1.2.0", None)),
        )
        expect(
            "a wrong DuckDB floor is caught",
            any("DuckDB floor" in problem for problem in disagreements(fields, "1.2.3", "v1.5.5", None)),
        )
        expect(
            "a wrong platform is caught",
            any("platform" in problem for problem in disagreements(fields, "1.2.3", "v1.2.0", "linux_amd64")),
        )

        unstable = read_trailer(
            synthetic_artifact(work, "unstable.duckdb_extension", abi_type="C_STRUCT_UNSTABLE")
        )
        expect(
            "the unstable ABI is caught",
            any("ABI" in problem for problem in disagreements(unstable, "9.9.9", "v1.2.0", None)),
        )

        # A raw library is not an extension. Without the header check the last
        # 256 bytes of any file parse as fields and a truncated build would
        # compare empty against empty and pass.
        raw = work / "raw.dylib"
        raw.write_bytes(b"\0" * 4096)
        try:
            read_trailer(raw)
            failures.append("a file with no trailer was accepted")
        except TrailerError as error:
            expect("a file with no trailer names the reason", "duckdb_signature" in str(error))

        # The fields being at the wrong offset must be an error, not a silent
        # misread: that is what a layout change upstream would look like.
        try:
            read_trailer(synthetic_artifact(work, "shifted.duckdb_extension", magic="7"))
            failures.append("a trailer with the wrong magic was accepted")
        except TrailerError as error:
            expect("a wrong magic names the reason", "magic" in str(error))

        short = work / "short.duckdb_extension"
        short.write_bytes(b"\0" * 12)
        try:
            read_trailer(short)
            failures.append("a file shorter than the trailer was accepted")
        except TrailerError:
            pass

        # --compare: two recipes stamping different versions must not pass.
        left = synthetic_artifact(work, "left.duckdb_extension", extension_version="1.0.0")
        right = synthetic_artifact(work, "right.duckdb_extension", extension_version="1.0.1")
        same = synthetic_artifact(work, "same.duckdb_extension", extension_version="1.0.0", platform="linux_amd64")
        expect("compare catches a version that differs", compare_fields(read_trailer(left), read_trailer(right)) != [])
        expect("compare ignores a platform that differs", compare_fields(read_trailer(left), read_trailer(same)) == [])

    # The two expectations really are read out of the tree, not defaulted.
    version = version_in_cargo_toml()
    expect("Cargo.toml yields a version", bool(re.fullmatch(r"\d+\.\d+\.\d+", version)))
    floor = min_duckdb_version()
    expect("lib.rs yields a DuckDB floor", bool(re.fullmatch(r"v\d+\.\d+\.\d+", floor)))

    # And a manifest whose [workspace.package] has no version is an error rather
    # than a fallback onto some other table's version.
    with tempfile.TemporaryDirectory() as workdir:
        manifest = pathlib.Path(workdir) / "Cargo.toml"
        manifest.write_text('[workspace.dependencies]\nsha2 = { version = "0.10" }\n\n[workspace.package]\nedition = "2021"\n')
        try:
            version_in_cargo_toml(manifest)
            failures.append("a manifest with no workspace version was accepted")
        except SystemExit:
            pass

    if failures:
        for failure in failures:
            print(f"self-test FAILED: {failure}", file=sys.stderr)
        return 1

    print(
        f"self-test ok: the trailer reader recovers all {len(FIELD_NAMES)} fields from a "
        f"synthetic artifact, reports a stale version, a wrong DuckDB floor, a wrong "
        f"platform and the unstable ABI, refuses a raw library and a shifted trailer, and "
        f"reads {version} out of Cargo.toml and {floor} out of lib.rs"
    )
    return 0


def compare_fields(left: dict[str, str], right: dict[str, str]) -> list[str]:
    return [
        f"{name}: {left[name]!r} vs {right[name]!r}"
        for name in COMPARED_FIELDS
        if left[name] != right[name]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=pathlib.Path)
    parser.add_argument("--compare", nargs=2, type=pathlib.Path, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--show", type=pathlib.Path)
    parser.add_argument("--expect-platform")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not (args.artifact or args.compare or args.show):
        parser.error("one of --artifact, --compare, --show or --self-test is required")

    if args.show:
        for name, value in read_trailer(args.show).items():
            print(f"{name:>18} = {value!r}")

    failures = 0

    if args.artifact:
        try:
            fields = read_trailer(args.artifact)
        except TrailerError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        expected_version = version_in_cargo_toml()
        expected_floor = min_duckdb_version()
        problems = disagreements(fields, expected_version, expected_floor, args.expect_platform)
        if problems:
            print(f"FAIL: {args.artifact} does not report this tree:", file=sys.stderr)
            for problem in problems:
                print(f"       {problem}", file=sys.stderr)
            print(
                "       If configure/extension_version.txt exists and is stale, that is the "
                "cause: the recipe writes it only when it is absent. `make clean_configure`.",
                file=sys.stderr,
            )
            failures += 1
        else:
            print(
                f"ok: {args.artifact} self-reports version {fields['extension_version']}, "
                f"matching Cargo.toml; DuckDB floor {fields['duckdb_version']}, matching "
                f"MIN_DUCKDB_VERSION; ABI {fields['abi_type']}; platform {fields['platform']}"
            )

    if args.compare:
        left_path, right_path = args.compare
        try:
            left, right = read_trailer(left_path), read_trailer(right_path)
        except TrailerError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        differences = compare_fields(left, right)
        if differences:
            print(
                f"FAIL: the two build recipes no longer stamp the same metadata\n"
                f"       {left_path}\n       {right_path}",
                file=sys.stderr,
            )
            for difference in differences:
                print(f"       {difference}", file=sys.stderr)
            failures += 1
        else:
            print(
                f"ok: both recipes stamp the same {', '.join(COMPARED_FIELDS)} "
                f"(version {left['extension_version']})"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
