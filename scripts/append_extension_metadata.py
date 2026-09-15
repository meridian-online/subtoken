#!/usr/bin/env python3
"""Turn a compiled cdylib into a loadable `.duckdb_extension`.

DuckDB refuses to `LOAD` a shared library that does not carry its metadata
trailer, so the raw `libsubtoken.dylib` cargo produces is not yet an
extension. The trailer is a 534-byte WebAssembly custom section named
`duckdb_signature`: eight 32-byte null-padded ASCII fields written in reverse
order, then 256 zero bytes reserved for a signature the community registry adds
when it signs a build.

Stdlib only, so `make test` needs no virtualenv.

    scripts/append_extension_metadata.py \\
        --library-file target/release/libsubtoken.dylib \\
        --out-file build/subtoken.duckdb_extension \\
        --platform osx_arm64 --duckdb-version v1.2.0 --extension-version 0.1.0
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

FIELD_WIDTH = 32
SIGNATURE_BYTES = 256
TRAILER_BYTES = 534


def padded(value: str) -> bytes:
    encoded = value.encode("ascii")
    if len(encoded) > FIELD_WIDTH:
        raise SystemExit(f"metadata field longer than {FIELD_WIDTH} bytes: {value!r}")
    return encoded.ljust(FIELD_WIDTH, b"\0")


def trailer(platform: str, duckdb_version: str, extension_version: str, abi_type: str) -> bytes:
    out = bytearray()
    out.append(0x00)              # custom section
    out += b"\x93\x04"            # LEB128 payload length: 531
    out.append(0x10)              # name length: 16
    out += b"duckdb_signature"
    out += b"\x80\x04"            # LEB128 content length: 512

    # Fields are written FIELD8 down to FIELD1.
    out += padded("")             # FIELD8 unused
    out += padded("")             # FIELD7 unused
    out += padded("")             # FIELD6 unused
    out += padded(abi_type)       # FIELD5
    out += padded(extension_version)  # FIELD4
    out += padded(duckdb_version)     # FIELD3
    out += padded(platform)           # FIELD2
    out += padded("4")                # FIELD1 magic

    out += b"\0" * SIGNATURE_BYTES
    if len(out) != TRAILER_BYTES:
        raise SystemExit(f"trailer is {len(out)} bytes, expected {TRAILER_BYTES}")
    return bytes(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-file", required=True, type=pathlib.Path)
    parser.add_argument("--out-file", required=True, type=pathlib.Path)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--duckdb-version", required=True)
    parser.add_argument("--extension-version", required=True)
    parser.add_argument("--abi-type", default="C_STRUCT")
    args = parser.parse_args()

    if not args.library_file.is_file():
        raise SystemExit(f"no such library: {args.library_file}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.library_file, args.out_file)
    with args.out_file.open("ab") as handle:
        handle.write(
            trailer(args.platform, args.duckdb_version, args.extension_version, args.abi_type)
        )

    print(
        f"{args.out_file} "
        f"[platform={args.platform} duckdb={args.duckdb_version} "
        f"extension={args.extension_version} abi={args.abi_type}]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
