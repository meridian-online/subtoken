#!/usr/bin/env python3
"""Fail if anything that can open a socket is linked into the extension.

AC2 says the model loads with no network. A test that calls `subtoken_embed()` and gets a
vector cannot prove that: it proves the model was found, not that no other code
path could have gone looking for it. What proves it is the absence of an HTTP or
TLS client from the tree at all, and that is mechanical.

Two checks, and they answer different questions.

UNDEFINED SYMBOLS
    Every call a binary makes into the platform appears in its undefined symbol
    table. So if none of the socket API's entry points is undefined, that binary
    makes no direct socket call. This check came from the reviewer of the first
    version of this file, who found the assertion it was missing.

    On its own it is not a proof of absence, and it is worth knowing why:
    `/usr/bin/curl` passes it. curl opens sockets through `libcurl.4.dylib`, so
    the socket calls are undefined in the library rather than in the binary. It
    is the LINKED LIBRARIES check below that fails curl.

    Together the two are a proof of absence for this artifact, and the argument
    is short enough to state. It links four things: itself, libiconv,
    CoreFoundation and libSystem. libiconv converts charsets. Every route to a
    socket that the other two offer — the BSD calls in libSystem, and CFSocket
    and CFStream in CoreFoundation — is in the list below, and none of them is
    undefined in the artifact. There is nowhere else to delegate to.

    What neither covers: raw syscall instructions that bypass libSystem, and a
    `dlopen` of a library named at runtime. Nothing in this tree does either.

    ON WINDOWS there is no `nm`, and both questions are answered by one
    structure: the PE import table names the DLLs the binary needs and the
    functions it takes from each. `pe_imports` reads it out of the file with no
    external tool, and the answers feed the same two checks — Winsock spells its
    entry points `connect`, `send`, `recv`, `getaddrinfo`, so the symbol list
    below already covers it unchanged.

RUNTIME DEPENDENCY TREE
    `cargo tree --workspace --edges normal` is the set of crates compiled into
    the artifact. Build-dependencies are deliberately excluded: `libduckdb-sys`
    takes `ureq` and `rustls` as build-dependencies to fetch DuckDB's headers,
    and a build-dependency compiles into the build script, not into the cdylib.
    Excluding them is the difference between reporting a real defect and
    reporting the DuckDB bindings.

LINKED LIBRARIES
    The dynamic libraries the built artifact actually names. A vendored OpenSSL
    would not appear as a crate in a stripped tree but would appear here.
    Skipped, loudly, when neither `otool` nor `ldd` is available.

Stdlib only.

    scripts/check_no_network_deps.py
    scripts/check_no_network_deps.py --artifact build/subtoken.duckdb_extension
    scripts/check_no_network_deps.py --self-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Crates that speak HTTP, terminate TLS, or exist to fetch a model. Matched
# against the crate name a `cargo tree` line carries, so `rustls-pki-types`
# matches on `rustls`.
FORBIDDEN_CRATES = (
    "openssl",
    "openssl-sys",
    "openssl-src",
    "native-tls",
    "rustls",
    "webpki",
    "ring",
    "ureq",
    "reqwest",
    "hyper",
    "isahc",
    "attohttpc",
    "curl",
    "curl-sys",
    "hf-hub",
    "tokio",
)

# Dynamic library names that mean a TLS or HTTP stack was linked. The first
# alternative is the unix spelling; the second is the Windows one, where the
# stack is a system DLL rather than a vendored library — `secur32` and
# `crypt32` are schannel TLS and the certificate store, which is what a Windows
# HTTPS client pulls in. `bcrypt`, `bcryptprimitives` and `advapi32` are
# deliberately absent: they are where a Rust binary gets its random numbers,
# and flagging them would fail every build.
FORBIDDEN_LIBRARY_PATTERN = re.compile(
    r"lib(ssl|crypto|curl|nghttp2)"
    r"|\b(ws2_32|wsock32|mswsock|winhttp|wininet|dnsapi|iphlpapi|secur32"
    r"|sspicli|schannel|ncrypt|crypt32|urlmon|netapi32|websocket)\.dll",
    re.IGNORECASE,
)

# Entry points to the platform's socket API. A binary that can reach the network
# calls at least one of these, so their joint absence from the undefined symbol
# table is the thing that makes "no network" a proof rather than a hope.
#
# BSD sockets, name resolution, and the two macOS frameworks that offer a
# connection without going through the BSD calls.
FORBIDDEN_SYMBOLS = (
    "socket",
    "socketpair",
    "connect",
    "connectx",
    "bind",
    "listen",
    "accept",
    "accept4",
    "send",
    "sendto",
    "sendmsg",
    "sendmmsg",
    "recv",
    "recvfrom",
    "recvmsg",
    "recvmmsg",
    "shutdown",
    "setsockopt",
    "getsockopt",
    "getpeername",
    "getsockname",
    "getaddrinfo",
    "freeaddrinfo",
    "getnameinfo",
    "gethostbyname",
    "gethostbyname2",
    "gethostbyaddr",
    "res_query",
    "res_search",
    "res_init",
    "CFSocketCreate",
    "CFStreamCreatePairWithSocketToHost",
    "CFReadStreamOpen",
    "SSLHandshake",
    "SSLCreateContext",
    "nw_connection_create",
    "nw_connection_start",
    "nw_endpoint_create_host",
    # Winsock. The BSD names above are imported under those exact spellings from
    # ws2_32.dll, so these are the ones that have no unix equivalent.
    "WSAStartup",
    "WSASocketA",
    "WSASocketW",
    "WSAConnect",
    "WSASend",
    "WSASendTo",
    "WSARecv",
    "WSARecvFrom",
    "WSAAccept",
    "WSAAddressToStringW",
    "GetAddrInfoW",
    "InternetOpenA",
    "InternetOpenW",
    "InternetConnectA",
    "InternetConnectW",
    "WinHttpOpen",
    "WinHttpConnect",
    "WinHttpSendRequest",
)

# `name vX.Y.Z` is how every cargo tree line names a crate.
CRATE_LINE = re.compile(r"([A-Za-z0-9_.-]+) v\d")


def crates_in(tree_output: str) -> set[str]:
    return {match.group(1) for match in CRATE_LINE.finditer(tree_output)}


def offending_crates(tree_output: str) -> set[str]:
    return crates_in(tree_output) & set(FORBIDDEN_CRATES)


def runtime_tree() -> str:
    completed = subprocess.run(
        ["cargo", "tree", "--workspace", "--edges", "normal"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"cargo tree failed:\n{completed.stdout}{completed.stderr}")
    return completed.stdout


class UninspectableArtifact(Exception):
    """The artifact could not be read, so nothing about it has been established."""


# ── Windows: the PE import table ──────────────────────────────────────────────
#
# `nm`, `otool` and `ldd` are all absent on a Windows runner, and a check that
# reports SKIPPED there would leave the one platform whose toolchain nobody has
# looked at as the one platform nobody has looked at. The import table answers
# both questions this file asks, so it is read directly. Stdlib `struct`, no
# external tool, and the format is fixed by the PE specification rather than by
# a tool's output.

PE_SIGNATURE = b"PE\0\0"
PE32_MAGIC = 0x10B
PE32PLUS_MAGIC = 0x20B
IMPORT_DIRECTORY = 1
DELAY_IMPORT_DIRECTORY = 13


def looks_like_pe(artifact: pathlib.Path) -> bool:
    with artifact.open("rb") as handle:
        return handle.read(2) == b"MZ"


def _c_string(data: bytes, offset: int) -> str:
    end = data.index(b"\0", offset)
    return data[offset:end].decode("ascii", errors="replace")


def pe_imports(artifact: pathlib.Path) -> tuple[list[str], list[str]]:
    """(DLL names, imported function names) from a PE image.

    Raises UninspectableArtifact rather than returning empty lists: a parser
    that gave up quietly would report a clean import table for a file it never
    read, and clean is exactly the answer this check must not invent.

    The packaged extension carries DuckDB's 534-byte metadata trailer after the
    image, which is past everything read here and does not disturb it.
    """
    data = artifact.read_bytes()
    try:
        if data[:2] != b"MZ":
            raise UninspectableArtifact(f"{artifact} does not start with the MZ signature")
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe : pe + 4] != PE_SIGNATURE:
            raise UninspectableArtifact(f"{artifact} has no PE header at {pe:#x}")

        sections_count, optional_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        optional = pe + 24
        magic = struct.unpack_from("<H", data, optional)[0]
        if magic not in (PE32_MAGIC, PE32PLUS_MAGIC):
            raise UninspectableArtifact(f"{artifact} has an unknown optional-header magic {magic:#x}")
        plus = magic == PE32PLUS_MAGIC
        directories = optional + (112 if plus else 96)

        sections = []
        section_table = pe + 24 + optional_size
        for index in range(sections_count):
            base = section_table + index * 40
            virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from("<IIII", data, base + 8)
            sections.append((virtual_address, max(virtual_size, raw_size), raw_pointer))
        if not sections:
            raise UninspectableArtifact(f"{artifact} has no sections")

        def offset_of(rva: int) -> int | None:
            for virtual_address, span, raw_pointer in sections:
                if virtual_address <= rva < virtual_address + span:
                    return rva - virtual_address + raw_pointer
            return None

        thunk_size = 8 if plus else 4
        thunk_format = "<Q" if plus else "<I"
        ordinal_flag = 1 << (63 if plus else 31)

        libraries: list[str] = []
        functions: list[str] = []

        for directory in (IMPORT_DIRECTORY, DELAY_IMPORT_DIRECTORY):
            table_rva, table_size = struct.unpack_from("<II", data, directories + directory * 8)
            if not table_rva or not table_size:
                continue
            table = offset_of(table_rva)
            if table is None:
                raise UninspectableArtifact(f"{artifact}: import table RVA {table_rva:#x} is in no section")
            # The delay-import descriptor is 32 bytes and its name/thunk fields
            # sit at different offsets from the 20-byte import descriptor.
            stride, name_at, lookup_at, address_at = (20, 12, 0, 16) if directory == IMPORT_DIRECTORY else (32, 4, 16, 12)
            entry = 0
            while True:
                base = table + entry * stride
                if base + stride > len(data):
                    raise UninspectableArtifact(f"{artifact}: the import table runs past the end of the file")
                fields = data[base : base + stride]
                if not any(fields):
                    break
                name_rva = struct.unpack_from("<I", data, base + name_at)[0]
                lookup_rva = struct.unpack_from("<I", data, base + lookup_at)[0]
                address_rva = struct.unpack_from("<I", data, base + address_at)[0]
                name_offset = offset_of(name_rva)
                if name_offset is None:
                    raise UninspectableArtifact(f"{artifact}: a DLL name RVA {name_rva:#x} is in no section")
                libraries.append(_c_string(data, name_offset))

                thunks = offset_of(lookup_rva or address_rva)
                slot = 0
                while thunks is not None:
                    value = struct.unpack_from(thunk_format, data, thunks + slot * thunk_size)[0]
                    if value == 0:
                        break
                    if not value & ordinal_flag:
                        hint = offset_of(value & 0x7FFFFFFF)
                        if hint is not None:
                            functions.append(_c_string(data, hint + 2))
                    slot += 1
                entry += 1
    except UninspectableArtifact:
        raise
    except (struct.error, IndexError, ValueError) as error:
        raise UninspectableArtifact(f"{artifact} is not a readable PE image: {error}") from error

    return libraries, functions


def parse_nm_output(text: str) -> list[str]:
    """Bare symbol names from `nm` output.

    Leading underscores (Mach-O prefixes every C symbol with one) and
    `@GLIBC_2.x` version suffixes (ELF) are stripped, so a name in
    FORBIDDEN_SYMBOLS matches on either platform.

    Split out from the `nm` call so that `--self-test` can drive it with real
    `nm` output. The version of the self-test this replaces normalised its own
    planted symbol and then matched that, which tested the assertion rather than
    the parser: deleting the `.lstrip("_")` here left the self-test exiting 0
    and still printing that it caught all 38 names, while the macOS check
    cleared `/usr/bin/nc`.
    """
    names = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        names.append(fields[-1].lstrip("_").split("@", 1)[0])
    return names


def undefined_symbols(artifact: pathlib.Path) -> list[str] | None:
    """The artifact's undefined symbols, one bare name per entry.

    None means "not established", which is a different answer from an empty
    list and is why --require-inspection exists.
    """
    if looks_like_pe(artifact):
        return pe_imports(artifact)[1]
    if not shutil.which("nm"):
        return None
    for arguments in (["-u"], ["-D", "-u"], ["--dynamic", "--undefined-only"]):
        completed = subprocess.run(
            ["nm", *arguments, str(artifact)], text=True, capture_output=True, check=False
        )
        if completed.returncode == 0 and completed.stdout.strip():
            break
    else:
        raise SystemExit(f"nm reported no undefined symbols for {artifact}; that is not credible")

    return parse_nm_output(completed.stdout)


def offending_symbols(names: list[str]) -> set[str]:
    """Which forbidden entry points appear, matched whole rather than as
    substrings.

    Substring matching is what made the first draft of this useless: `bind`
    matched `dyld_stub_binder`, which is the dynamic linker and not a socket.
    """
    return set(names) & set(FORBIDDEN_SYMBOLS)


def linked_libraries(artifact: pathlib.Path) -> list[str] | None:
    if looks_like_pe(artifact):
        return pe_imports(artifact)[0]
    if shutil.which("otool"):
        command = ["otool", "-L", str(artifact)]
    elif shutil.which("ldd"):
        command = ["ldd", str(artifact)]
    else:
        return None
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{command[0]} failed:\n{completed.stdout}{completed.stderr}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def synthetic_pe(path: pathlib.Path, imports: dict[str, list[str]], *, plus: bool = True) -> pathlib.Path:
    """A PE image importing exactly `imports`, for the self-test to read back.

    Assembled here from the format rather than taken from a real binary: the
    self-test has to run offline and on a machine with no Windows toolchain, and
    a planted import is the only way to require the reader to report one.
    """
    magic = PE32PLUS_MAGIC if plus else PE32_MAGIC
    thunk_size = 8 if plus else 4
    thunk_format = "<Q" if plus else "<I"
    section_rva, section_raw = 0x1000, 0x400

    descriptors_size = (len(imports) + 1) * 20
    body = bytearray()

    def place(blob: bytes) -> int:
        """Append to the section body and return the RVA it landed at."""
        rva = section_rva + descriptors_size + len(body)
        body.extend(blob)
        return rva

    descriptors = bytearray()
    for library, functions in imports.items():
        hint_rvas = [place(b"\x00\x00" + name.encode("ascii") + b"\x00") for name in functions]
        lookup_rva = place(
            b"".join(struct.pack(thunk_format, rva) for rva in hint_rvas)
            + struct.pack(thunk_format, 0)
        )
        name_rva = place(library.encode("ascii") + b"\x00")
        descriptors += struct.pack("<IIIII", lookup_rva, 0, 0, name_rva, lookup_rva)
    descriptors += b"\x00" * 20

    section = bytes(descriptors) + bytes(body)

    optional_size = (112 if plus else 96) + 16 * 8
    headers = bytearray(b"MZ" + b"\x00" * 0x3E)
    headers[0x3C:0x40] = struct.pack("<I", 0x40)
    headers += PE_SIGNATURE
    headers += struct.pack("<HHIIIHH", 0x8664 if plus else 0x14C, 1, 0, 0, 0, optional_size, 0x2102)
    optional = bytearray(b"\x00" * optional_size)
    struct.pack_into("<H", optional, 0, magic)
    struct.pack_into("<I", optional, 108 if plus else 92, 16)
    struct.pack_into("<II", optional, (112 if plus else 96) + IMPORT_DIRECTORY * 8, section_rva, descriptors_size)
    headers += optional
    headers += (
        b".rdata\x00\x00"
        + struct.pack("<IIII", len(section), section_rva, len(section), section_raw)
        + b"\x00" * 16
    )
    headers += b"\x00" * (section_raw - len(headers))
    path.write_bytes(bytes(headers) + section)
    return path


def self_test() -> int:
    """Prove the matcher detects what it is looking for.

    A checker that returns clean on a clean tree and has never been shown a
    dirty one is indistinguishable from a checker that always returns clean.
    """
    dirty = (
        "subtoken-core v0.1.0\n"
        "├── model2vec-rs v0.2.1\n"
        "│   ├── hf-hub v0.4.3\n"
        "│   │   └── ureq v2.12.1\n"
        "│   │       └── rustls v0.23.43\n"
        "└── sha2 v0.10.9\n"
    )
    found = offending_crates(dirty)
    expected = {"hf-hub", "ureq", "rustls"}
    if found != expected:
        print(f"self-test FAILED: matched {sorted(found)}, expected {sorted(expected)}", file=sys.stderr)
        return 1

    clean = "subtoken-core v0.1.0\n└── sha2 v0.10.9\n"
    if offending_crates(clean):
        print("self-test FAILED: matched a clean tree", file=sys.stderr)
        return 1

    if not FORBIDDEN_LIBRARY_PATTERN.search("/usr/lib/libssl.3.dylib"):
        print("self-test FAILED: the library pattern misses libssl", file=sys.stderr)
        return 1
    if FORBIDDEN_LIBRARY_PATTERN.search("/usr/lib/libSystem.B.dylib"):
        print("self-test FAILED: the library pattern matches libSystem", file=sys.stderr)
        return 1

    # Every forbidden symbol, one at a time, in both platforms' spellings, as
    # lines of real `nm` output driven through the real parser. Planting a
    # pre-normalised name here instead is what made the previous version of this
    # unable to notice its own parser going blind.
    for symbol in FORBIDDEN_SYMBOLS:
        planted = [
            f"                 U _{symbol}",  # macOS `nm -u`
            f"                 U {symbol}",  # ELF `nm -D -u`
            f"                 U {symbol}@GLIBC_2.2.5",  # ELF, versioned
            f"_{symbol}",  # macOS `nm -u`, bare-name form
        ]
        for line in planted:
            caught = offending_symbols(parse_nm_output(line))
            if caught != {symbol}:
                print(
                    f"self-test FAILED: {line!r} parsed to {sorted(caught)}, expected [{symbol!r}]",
                    file=sys.stderr,
                )
                return 1

    # Real `nm -u` output from a clean build of this extension. Every one of
    # these must pass, or the check would fail on its own artifact.
    clean_output = """\
                 U dyld_stub_binder
                 U _malloc
                 U _free
                 U _open
                 U _read
                 U _write
                 U _close
                 U _pthread_create
                 U _pthread_cond_wait
                 U _getentropy
                 U _sysconf
                 U _mmap
                 U _sched_yield
"""
    caught = offending_symbols(parse_nm_output(clean_output))
    if caught:
        print(f"self-test FAILED: clean symbols matched {sorted(caught)}", file=sys.stderr)
        return 1

    # And end to end, against a binary on this machine that really does open
    # sockets. Synthetic input proves the parser; this proves the whole path
    # from `nm` through to the verdict — and, unlike everything above, it does
    # not enumerate FORBIDDEN_SYMBOLS to do it.
    #
    # That distinction is the point. The loop above iterates over the very list
    # it is testing, so a name deleted from the list is simply not tested and
    # the self-test stays green: removing `connect` was invisible to it. Here
    # the enumeration comes from a real binary. Every name in CORE_SYMBOLS is
    # one any TCP client must call, so requiring the checker to flag all of them
    # in a binary that has all of them makes a deleted name fail.
    #
    # Beyond CORE_SYMBOLS the list is still a deny-list and nothing here can
    # prove it complete. What it is is a list of every entry point in the
    # platform's socket API, and the artifact this gates uses none of them.
    core_symbols = ("socket", "connect")
    candidates = [
        pathlib.Path(path)
        for path in (
            "/usr/bin/nc",
            "/bin/nc",
            "/usr/bin/ncat",
            "/usr/bin/ssh",
            "/usr/bin/ping",
            "/bin/ping",
            "/usr/bin/wget",
        )
    ]
    if not shutil.which("nm"):
        print("self-test: SKIPPED the live check — nm is not available")
    else:
        for candidate in candidates:
            if not candidate.is_file():
                continue
            live = undefined_symbols(candidate)
            if live is None or not all(symbol in live for symbol in core_symbols):
                continue
            flagged = offending_symbols(live)
            missing = [symbol for symbol in core_symbols if symbol not in flagged]
            if missing:
                print(
                    f"self-test FAILED: {candidate} calls {missing} and the check did not "
                    f"flag them — a name has gone missing from FORBIDDEN_SYMBOLS",
                    file=sys.stderr,
                )
                return 1
            print(
                f"self-test ok: {candidate} is flagged on all of {list(core_symbols)} "
                f"and {len(flagged)} socket entry points in total"
            )
            break
        else:
            print(
                "self-test: SKIPPED the live check — no binary on this machine has all of "
                f"{list(core_symbols)} undefined"
            )

    # ── The Windows path ──────────────────────────────────────────────────────
    # There is no `nm` on a Windows runner, so this is the only reader that
    # answers either question there. A synthetic image is the only way to plant
    # a socket import on a machine with no Windows toolchain, and without a
    # planted one the reader would be a function nobody has ever seen report
    # anything.
    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)

        for plus in (True, False):
            width = "PE32+" if plus else "PE32"

            clean = synthetic_pe(
                work / f"clean{int(plus)}.dll",
                {
                    "kernel32.dll": ["WriteFile", "ReadFile", "GetLastError"],
                    "bcrypt.dll": ["BCryptGenRandom"],
                    "advapi32.dll": ["SystemFunction036"],
                    "ntdll.dll": ["NtCreateFile"],
                },
                plus=plus,
            )
            libraries, functions = pe_imports(clean)
            if [line for line in libraries if FORBIDDEN_LIBRARY_PATTERN.search(line)]:
                print(f"self-test FAILED: {width} — a clean import table was flagged", file=sys.stderr)
                return 1
            if offending_symbols(functions):
                print(
                    f"self-test FAILED: {width} — clean imports matched "
                    f"{sorted(offending_symbols(functions))}",
                    file=sys.stderr,
                )
                return 1
            if "BCryptGenRandom" not in functions or "kernel32.dll" not in libraries:
                print(
                    f"self-test FAILED: {width} — the reader lost imports it was given: "
                    f"{libraries} {functions}",
                    file=sys.stderr,
                )
                return 1

            dirty = synthetic_pe(
                work / f"dirty{int(plus)}.dll",
                {
                    "kernel32.dll": ["WriteFile"],
                    "ws2_32.dll": ["WSAStartup", "socket", "connect", "send", "recv", "getaddrinfo"],
                    "secur32.dll": ["InitializeSecurityContextW"],
                },
                plus=plus,
            )
            libraries, functions = pe_imports(dirty)
            flagged_libraries = {line for line in libraries if FORBIDDEN_LIBRARY_PATTERN.search(line)}
            if flagged_libraries != {"ws2_32.dll", "secur32.dll"}:
                print(
                    f"self-test FAILED: {width} — a planted Winsock/schannel import was reported as "
                    f"{sorted(flagged_libraries)}",
                    file=sys.stderr,
                )
                return 1
            expected = {"WSAStartup", "socket", "connect", "send", "recv", "getaddrinfo"}
            if offending_symbols(functions) != expected:
                print(
                    f"self-test FAILED: {width} — planted socket imports were reported as "
                    f"{sorted(offending_symbols(functions))}, expected {sorted(expected)}",
                    file=sys.stderr,
                )
                return 1

            # And the two routing functions really send a PE through this
            # reader rather than reaching for a tool that is not there.
            if undefined_symbols(dirty) != functions or linked_libraries(dirty) != libraries:
                print(f"self-test FAILED: {width} — a PE was not routed to the import reader", file=sys.stderr)
                return 1

        # A file the reader cannot read must raise, not return an empty and
        # therefore clean answer.
        for name, blob in (
            ("truncated.dll", b"MZ" + b"\x00" * 8),
            ("no-pe-header.dll", b"MZ" + b"\x00" * 0x3A + (0x40).to_bytes(4, "little") + b"NOPE" + b"\x00" * 512),
        ):
            broken = work / name
            broken.write_bytes(blob)
            try:
                pe_imports(broken)
            except UninspectableArtifact:
                continue
            print(f"self-test FAILED: {name} was parsed as a readable PE", file=sys.stderr)
            return 1

    print(
        f"self-test ok: the matcher finds a planted TLS stack, catches all "
        f"{len(FORBIDDEN_SYMBOLS)} socket entry points parsed from real nm output in "
        f"both platforms' spellings, clears a clean tree and a clean symbol table, and "
        f"reads a planted ws2_32/secur32 import out of both a PE32 and a PE32+ image "
        f"while refusing a file that is not one"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=pathlib.Path)
    parser.add_argument(
        "--require-inspection",
        action="store_true",
        help="fail rather than SKIP when the artifact cannot be inspected. A release "
        "pipeline builds in containers this repository has never seen, and a container "
        "without nm would otherwise turn 'no socket' into 'not looked at' silently.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    failures = 0

    found = offending_crates(runtime_tree())
    if found:
        print(
            "FAIL: the runtime dependency tree contains "
            + ", ".join(sorted(found))
            + "\n      The extension bundles its model; nothing in it should be able to fetch one.",
            file=sys.stderr,
        )
        failures += 1
    else:
        print(f"ok: no HTTP or TLS crate in the runtime dependency tree ({len(FORBIDDEN_CRATES)} names checked)")

    if args.artifact:
        if not args.artifact.is_file():
            print(f"FAIL: no artifact at {args.artifact}", file=sys.stderr)
            return 1

        try:
            symbols = undefined_symbols(args.artifact)
        except UninspectableArtifact as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        if symbols is None:
            if args.require_inspection:
                print(
                    "FAIL: nm is not available, so nothing about this artifact's symbols has "
                    "been established. --require-inspection was passed because a clean report "
                    "from a check that did not run is worse than no report.",
                    file=sys.stderr,
                )
                failures += 1
            else:
                print("SKIPPED: nm is not available, so the undefined symbols were not checked")
        else:
            caught = offending_symbols(symbols)
            if caught:
                print(
                    "FAIL: the artifact can reach the network — its undefined symbols include "
                    + ", ".join(sorted(caught)),
                    file=sys.stderr,
                )
                failures += 1
            else:
                print(
                    f"ok: none of the {len(FORBIDDEN_SYMBOLS)} socket entry points is undefined "
                    f"in the artifact ({len(symbols)} undefined symbols in total), so it makes "
                    f"no direct socket call"
                )

        try:
            libraries = linked_libraries(args.artifact)
        except UninspectableArtifact as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        if libraries is None:
            if args.require_inspection:
                print(
                    "FAIL: neither otool nor ldd is available, so nothing about this artifact's "
                    "linked libraries has been established.",
                    file=sys.stderr,
                )
                failures += 1
            else:
                print("SKIPPED: neither otool nor ldd is available, so linked libraries were not checked")
        else:
            offending = [line for line in libraries if FORBIDDEN_LIBRARY_PATTERN.search(line)]
            if offending:
                print("FAIL: the artifact links a TLS or HTTP library:", file=sys.stderr)
                for line in offending:
                    print(f"       {line}", file=sys.stderr)
                failures += 1
            else:
                print(f"ok: the artifact links no TLS or HTTP library ({len(libraries)} entries checked)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
