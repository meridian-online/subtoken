#!/usr/bin/env python3
"""Neuter one branch of a checked script at a time, and require the script to notice.

A test that passes against working code and against deliberately broken code is
testing nothing. `scripts/mutation_check.py` says that about the product and
proves it for a hand-written table of breakages. This says it about the checks
themselves, and enumerates the breakages from the source rather than from a
table, because the table is written by the same person as the check and misses
the same things they did.

Every review of `check_quality_claims.py` so far has found an assertion that
could be deleted with every gate green, and every one of those was found by a
person reading the file. Reading it once more would have found the next. Running
it does not depend on anybody's attention, and the first thing it did when
pointed at itself was report two site kinds its own self-test never exercised.
The second thing it did, once it was pointed at its own `main` as well, was
report that `main`'s loop over the targets, its accumulation into `problems`,
its `if problems:` and both of its report loops could each be deleted while
every Python gate stayed at exit 0 — which is the defect this file exists to
find, in this file, hidden behind an exclusion written for one line and applied
to a whole function.

WHAT IT DOES
    For each `Target` below, parse the script and enumerate every site where a
    decision is taken — `if <test>:`, `for <name> in <iterable>:`, and
    `problems += <call>` — outside the functions listed in `not_swept`, which
    are the script's own self-test machinery and would only be measuring itself.
    Neuter one site at a time (`if False:`, `for x in []:`, `problems += []`),
    run the script's own commands, and require at least one of them to come back
    non-zero. A site where every command still exits 0 is a line that can be
    deleted with the gate reporting itself green.

    A site that stays green must be written into `allowed` with the reason it
    cannot fail, and an entry there that no longer names a live green site is
    reported too — a permission that outlives what it permitted quietly widens
    what the sweep tolerates. That is the same shape as `ALLOWED_UNIVERSALS` in
    `check_quality_claims.py`, and for the same reason.

    `not_swept` carries a reason per entry and is held to the second half of
    that rule: an entry naming a function the file does not define is reported.
    It was a bare tuple of names with neither property, which is how this file's
    own entry came to list `main` — one line of it genuinely unmeasurable, five
    of them nobody had looked at. It remains the blunter of the two hatches,
    because it excludes a whole function: adding `"run"` to
    `check_quality_claims.py`'s entry drops that sweep from 69 sites to 56.

WHAT IT DOES NOT DO
    It does not neuter a guard in the other direction. `if <test>:` becomes
    `if False:` and never `if True:`, so a condition that must *not* fire — a
    fast path, an early return — is not measured here.

    It does not touch expressions. A comparison flipped from `!=` to `==`, a
    regex that matches nothing, a constant edited: none of those are branches
    and none are enumerated. `SENTENCE_END` compiling to something that matches
    nothing was one of those, and it is a real hole this cannot see. The
    hand-written table in `scripts/mutation_check.py` is where a breakage of
    that shape goes.

    It does not measure list content. `CLAIMS`, `PAGE_FIELDS`,
    `BANNED_ON_PAGE` and `BANNED_IN_SECTION` in `check_quality_claims.py` are
    lists an assertion reads rather than branches, so an entry deleted from one
    is invisible here. Each of those four has entries its own file's self-test
    does not see either, which is a live hole and not a limitation of this
    sweep: `CLAIMS` builds the fixture its cases run against —
    `good = " ".join(CLAIMS)` — so an entry witnesses itself and nothing else
    does, and dropping one stops it pinning the published sentence it was
    written for; `PAGE_FIELDS` losing `("docs", "extended_description")` goes
    unreported; `throughput` and `rows/s` in `BANNED_ON_PAGE` have no case that
    they alone satisfy, so dropping `throughput` would let a throughput claim
    ship as the opening line of the registry entry; and of `BANNED_IN_SECTION`'s
    four hedges, `most of the cluster` and `on the evidence we have` are each
    deletable with both of that file's commands at exit 0, which lets back in
    the hedge a measured figure replaced. Measured on this tree, and left open.
    The hand-written table in `scripts/mutation_check.py` is where a breakage of
    that shape goes.

    It does not say the assertions are the right ones. A check that asserts
    nothing useful, thoroughly, passes this.

    It sweeps the scripts in `TARGETS` and no others. Adding one is a `Target`
    entry plus however many `allowed` entries its first run turns up, and each
    of those is a sentence somebody has to mean.

Stdlib only.

    scripts/check_assertions_can_fail.py
    scripts/check_assertions_can_fail.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Long enough that a slow machine is not a failure, short enough that a neuter
#: which turns a loop infinite does not hold a CI runner for an hour.
TIMEOUT_SECONDS = 120

#: The timeout the group-kill case in `self_test` runs `reddened` with, instead
#: of TIMEOUT_SECONDS: that case has to reach the timeout path to reach the kill,
#: and waiting two minutes for it once per swept site is not affordable. Two
#: interpreters have to start inside it, so a run slow enough to miss that fails
#: saying the child never started rather than passing on nothing.
KILL_TIMEOUT_SECONDS = 3.0

#: How long that case then watches the heartbeat for. Six writes at the interval
#: `HEARTBEAT` uses.
HEARTBEAT_WINDOW_SECONDS = 0.3

#: And how long the heartbeat runs for at the outside. Bounded so that a tree
#: where the group kill has been broken leaks one process for seconds rather
#: than until the machine is rebooted.
HEARTBEAT_SECONDS = 10

#: Set in the environment of every command the sweep runs, and refused by `run`.
#: Without it, neutering `main`'s dispatch to `self_test` turns the `--self-test`
#: this sweep runs against itself into another live sweep, which sweeps this file
#: again. Measured before the refusal existed: the outer command hit
#: TIMEOUT_SECONDS, `subprocess.run` killed the process it started and not the
#: tree beneath it, and the orphans were still rewriting
#: `scripts/check_assertions_can_fail.py` four seconds after the sweep's `finally`
#: had put it back.
SWEEP_MARKER = "SUBTOKEN_SWEEP_RUNNING"


@dataclass(frozen=True)
class Unswept:
    """A function whose body is not swept, and why measuring it would measure nothing.

    The same two properties as `Allowed`, for the same reason. This started as a
    bare tuple of names, which is an escape hatch with no reason attached and no
    check that it still names anything: adding `"run"` to
    `check_quality_claims.py`'s entry drops that sweep from sixty-nine sites to
    fifty-six, silently, and a name left behind by a rename goes on excluding
    nothing while reading as though it excludes something.
    """

    function: str
    why: str


@dataclass(frozen=True)
class Allowed:
    """A site that stays green when neutered, and why it cannot be otherwise."""

    #: The function the site is in, innermost first.
    function: str
    #: The exact source text of the expression that gets neutered, so that an
    #: entry stops matching when the line it permits is edited.
    source: str
    why: str


@dataclass(frozen=True)
class Target:
    """A script to sweep, the commands that must notice, and its exceptions."""

    script: str
    #: Run in order; the first non-zero exit ends the site. Put the fastest and
    #: most sensitive first — for a checker with a self-test, that is the
    #: self-test.
    commands: tuple[tuple[str, ...], ...]
    #: Functions whose bodies are not swept, because they are the harness
    #: rather than the check: neutering a branch of a self-test measures the
    #: self-test against itself and reports whatever it likes.
    not_swept: tuple[Unswept, ...]
    allowed: tuple[Allowed, ...]

    def unswept(self) -> tuple[str, ...]:
        """Just the names, for `sites`."""
        return tuple(exclusion.function for exclusion in self.not_swept)


TARGETS: tuple[Target, ...] = (
    Target(
        script="scripts/check_quality_claims.py",
        commands=(
            ("python3", "scripts/check_quality_claims.py", "--self-test"),
            ("python3", "scripts/check_quality_claims.py"),
        ),
        not_swept=(
            Unswept("self_test", "the harness: neutering a branch of it measures it against itself"),
            Unswept("stage_tree", "harness — writes the two-page tree a case plants its defect in"),
            Unswept("staged_run", "harness — calls `run` over a staged tree and captures what it printed"),
            Unswept("staged_process", "harness — runs that file as a process over a staged tree"),
        ),
        allowed=(
            Allowed(
                function="table_problems",
                source="expected is None",
                why="unreachable while the self-test's own first block holds. That block "
                "requires FIGURES to register a figure for every (series, corpus) pair in "
                "TABLE_ROWS and DIRECTIONAL_SERIES, so `figure_for` cannot return None here. "
                "It is a guard against a state a stronger assertion already refuses, and "
                "deleting it would turn that state from a message into a traceback",
            ),
            Allowed(
                function="main",
                source="args.self_test",
                why="no self-test can prove it was itself invoked. Neutered, `--self-test` "
                "runs the live check, which passes on a correct tree: both CI commands "
                "become the same command and nothing inside the file can notice. Closing it "
                "needs something outside the file to assert on what the run printed, which "
                "is a check on a string rather than on behaviour",
            ),
        ),
    ),
    # This file, measured by its own self-test. `main` was excluded here by name
    # until the sweep was pointed at itself with only `self_test` excluded and
    # reported five of `main`'s six sites surviving: the loop over the targets,
    # the accumulation into `problems`, `if problems:` and both report loops
    # could each be deleted with every Python gate at exit 0. The reason written
    # against the exclusion — that a neutered dispatch recurses — was true of
    # `args.self_test` and of nothing else it excluded. `run` is the split that
    # let the other five be driven; SWEEP_MARKER is what makes the sixth redden
    # in milliseconds instead of recursing.
    Target(
        script="scripts/check_assertions_can_fail.py",
        commands=(("python3", "scripts/check_assertions_can_fail.py", "--self-test"),),
        not_swept=(
            Unswept("self_test", "the harness: neutering a branch of it measures it against itself"),
        ),
        allowed=(),
    ),
)


@dataclass(frozen=True)
class Site:
    """One decision in a source file, and what neutering it looks like."""

    function: str
    lineno: int
    start: int
    end: int
    source: str
    replacement: str

    def describe(self) -> str:
        return f"{self.function}:{self.lineno} `{self.source}` -> {self.replacement}"


def sites(source: str, not_swept: tuple[str, ...]) -> list[Site]:
    """Every branch, loop and `problems +=` outside `not_swept`, innermost function wins."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        return offsets[lineno - 1] + col

    found: list[Site] = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = function
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = child.name
            elif function and function not in not_swept:
                target = None
                if isinstance(child, ast.If):
                    target, replacement = child.test, "False"
                elif isinstance(child, ast.For):
                    target, replacement = child.iter, "[]"
                elif (
                    isinstance(child, ast.AugAssign)
                    and isinstance(child.target, ast.Name)
                    and child.target.id == "problems"
                ):
                    target, replacement = child.value, "[]"
                if target is not None:
                    start = offset(target.lineno, target.col_offset)
                    end = offset(target.end_lineno, target.end_col_offset)
                    found.append(
                        Site(
                            function=function,
                            lineno=child.lineno,
                            start=start,
                            end=end,
                            source=source[start:end],
                            replacement=replacement,
                        )
                    )
            visit(child, inner)

    visit(ast.parse(source), "")
    return found


def reddened(
    root: pathlib.Path,
    commands: tuple[tuple[str, ...], ...],
    timeout: float = TIMEOUT_SECONDS,
) -> bool:
    """Whether any command comes back non-zero. A command that hangs counts as noticing.

    SWEEP_MARKER goes into every child's environment so that a command which
    reaches the live sweep — which is what a neutered `--self-test` dispatch
    does — refuses instead of sweeping the tree that is already being swept.
    That marker is what bounds the recursion, and the kill below does not help
    with it and cannot. Each child is given a session of its own, so a nested
    sweep's own children land in a group of their own; killing the group named
    by `child.pid` reaches one level and every level under it goes on running.
    Measured with the group kill in place and the marker removed: twenty-three
    orphans in twenty-three distinct process groups outlived the timeout and
    were still multiplying twenty seconds after it, and the swept file, which
    `survivors`' own `finally` had put back, was intact a second later and
    gutted four seconds later.

    What the group kill does bound is a leaf hang — a command that stops
    responding having spawned children that stay in its group.
    `subprocess.run(timeout=...)` kills the process it started and leaves those
    running; killing the group takes them with it. A sweep that can leave the
    tree it swept modified is worse than no sweep.

    Neither the marker nor the kill is a branch, so the sweep does not enumerate
    either and removing one is invisible to it. The last two cases in
    `self_test` drive them, and are what reddens: measured with each removed on
    its own, every other Python gate in this repo stayed at exit 0.

    `timeout` is a parameter for that self-test alone: the group-kill case has
    to reach the timeout path, and nothing else passes it.
    """
    environment = {**os.environ, SWEEP_MARKER: "1"}
    for command in commands:
        argv = [sys.executable if part == "python3" else part for part in command]
        child = subprocess.Popen(  # noqa: S603
            argv,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        try:
            child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.communicate()
            return True
        if child.returncode != 0:
            return True
    return False


def survivors(root: pathlib.Path, target: Target) -> tuple[list[Site], int]:
    """The sites that stay green when neutered, and how many were tried."""
    path = root / target.script
    source = path.read_text()
    swept = sites(source, target.unswept())
    green: list[Site] = []
    try:
        for site in swept:
            path.write_text(source[: site.start] + site.replacement + source[site.end :])
            if not reddened(root, target.commands):
                green.append(site)
    finally:
        path.write_text(source)
    return green, len(swept)


def problems_for(root: pathlib.Path, target: Target) -> tuple[list[str], int, int]:
    """Everything wrong with one target: unallowed survivors, and allowances nothing needs."""
    green, tried = survivors(root, target)
    source = (root / target.script).read_text()
    problems = []
    for site in green:
        if not any(a.function == site.function and a.source == site.source for a in target.allowed):
            problems.append(
                f"{target.script}: {site.describe()} leaves every command at exit 0. That line "
                f"can be deleted with the gate reporting itself green. Give the assertion a "
                f"case that only it can satisfy, or write the site into `allowed` with the "
                f"reason it cannot fail"
            )
    live = {(site.function, site.source) for site in green}
    every = {(site.function, site.source) for site in sites(source, target.unswept())}
    defined = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for exclusion in target.not_swept:
        if exclusion.function not in defined:
            problems.append(
                f"{target.script}: `not_swept` skips `{exclusion.function}` and the file defines "
                f"no such function. An exclusion left behind by a rename excludes nothing while "
                f"reading as though it excludes something: delete the entry or repoint it"
            )
    for allowance in target.allowed:
        key = (allowance.function, allowance.source)
        if key not in every:
            problems.append(
                f"{target.script}: `allowed` permits `{allowance.source}` in {allowance.function} "
                f"and there is no such branch any more. The code moved out from under the "
                f"permission: delete the entry or repoint it"
            )
        elif key not in live:
            problems.append(
                f"{target.script}: `allowed` permits `{allowance.source}` in {allowance.function} "
                f"and neutering it is now noticed. A permission that outlives what it permitted "
                f"widens this sweep without anyone deciding to: delete the entry"
            )
    return problems, len(green), tried


def targets_from(described: pathlib.Path) -> tuple[Target, ...]:
    """The targets a JSON file describes, instead of `TARGETS`.

    `--targets` is passed by `self_test`, which is how a case drives this entry
    point end to end over a tree it staged, without the staged tree having to
    contain the real scripts `TARGETS` names — which is what would turn the
    staged sweep back into a sweep of this file. Neither CI command passes it.
    """
    return tuple(
        Target(
            script=entry["script"],
            commands=tuple(tuple(command) for command in entry["commands"]),
            not_swept=tuple(Unswept(**exclusion) for exclusion in entry["not_swept"]),
            allowed=tuple(Allowed(**allowance) for allowance in entry["allowed"]),
        )
        for entry in json.loads(described.read_text())
    )


def run(root: pathlib.Path, targets: tuple[Target, ...]) -> int:
    """The whole sweep over one tree: judge every target, report, return an exit code.

    Split out of `main` so that `self_test` can drive it over a staged tree.
    That split is the point rather than a tidiness, and it is the same one
    `check_quality_claims.py` made for the same reason: every assertion above is
    wired together here and nowhere else, so a wiring line deleted here disables
    it while the assertion — and every self-test case that calls it directly —
    stays green. While `main` was excluded from the sweep by name, five of the
    six lines now below were deletable one at a time with every Python gate at
    exit 0, `if problems:` among them, which reduced this to something that
    printed `ok:` over a list of survivors.
    """
    if os.environ.get(SWEEP_MARKER):
        print(
            f"refusing to run the live sweep: {SWEEP_MARKER} says this process is already inside "
            f"a sweep. Reaching here from a `--self-test` means the dispatch in `main` is not "
            f"doing its job, and sweeping from here sweeps the file the outer sweep is mutating",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []
    counts = []
    for target in targets:
        found, green, tried = problems_for(root, target)
        problems += found
        counts.append((target.script, tried, green))

    if problems:
        print("FAIL: a check in this tree can be neutered and stay green:", file=sys.stderr)
        for problem in problems:
            print(f"       {problem}", file=sys.stderr)
        return 1

    for script, tried, green in counts:
        print(
            f"ok: {script} — {tried} branches neutered one at a time, {tried - green} noticed by "
            f"the script's own commands and {green} recorded in `allowed` with a reason"
        )
    return 0


#: A script carrying one driven and one undriven site of each kind this sweep
#: enumerates. Both halves are load-bearing. The driven three prove the sweep
#: runs the commands and reads their exit codes; the undriven three prove it
#: enumerates `if`, `for` and `problems +=` at all. An earlier victim had only
#: `if` statements in it, and this sweep — run against itself — reported that
#: `isinstance(child, ast.For)` and the `problems +=` test could both be deleted
#: with its own self-test green, which would have narrowed what it looks for
#: without narrowing anything it says.
VICTIM = '''#!/usr/bin/env python3
import sys

DRIVEN_WORDS = ("driven-loop",)
UNDRIVEN_WORDS = ("undriven-loop",)


def driven_call(text):
    return [word for word in ("driven-call",) if word in text]


def undriven_call(text):
    return [word for word in ("undriven-call",) if word in text]


def problems_in(text):
    problems = []
    if "driven-if" in text:
        problems.append("driven-if")
    if "undriven-if" in text:
        problems.append("undriven-if")
    for word in DRIVEN_WORDS:
        problems.extend([word] if word in text else [])
    for word in UNDRIVEN_WORDS:
        problems.extend([word] if word in text else [])
    problems += driven_call(text)
    problems += undriven_call(text)
    return problems


def self_test():
    found = problems_in("driven-if driven-loop driven-call")
    if found != ["driven-if", "driven-loop", "driven-call"]:
        return 1
    return 0


def main():
    if "--self-test" in sys.argv:
        return self_test()
    return 0


sys.exit(main())
'''

#: In source order, which is the order `sites` returns them.
VICTIM_SITES = (
    '"driven-if" in text',
    '"undriven-if" in text',
    "DRIVEN_WORDS",
    "UNDRIVEN_WORDS",
    "driven_call(text)",
    "undriven_call(text)",
)
VICTIM_UNDRIVEN = ('"undriven-if" in text', "UNDRIVEN_WORDS", "undriven_call(text)")

VICTIM_TARGET = Target(
    script="victim.py",
    commands=(("python3", "victim.py", "--self-test"), ("python3", "victim.py")),
    not_swept=(
        Unswept("self_test", "the victim's harness, which is what `not_swept` is for"),
        Unswept("main", "argv dispatch, which no self-test of the victim's can prove it reached"),
    ),
    allowed=(),
)

#: `VICTIM_TARGET` with its three undriven sites permitted, so a sweep over the
#: victim has nothing to report. Both the process cases that need a clean tree
#: use it.
PERMITTED_TARGET = dataclasses.replace(
    VICTIM_TARGET,
    allowed=tuple(
        Allowed("problems_in", source, "on purpose") for source in VICTIM_UNDRIVEN
    ),
)


#: A command for `reddened` that spawns a child of its own and then stops
#: responding, so the timeout path runs with something still underneath it. The
#: child stays in the parent's process group — no `start_new_session` — because
#: that group is what the kill names. Its output goes to DEVNULL rather than to
#: the pipes `reddened` opens: a child holding those open makes
#: `child.communicate()` wait for it, and a survivor waited for is a survivor
#: that has finished writing by the time anything looks.
SPAWNS_AND_HANGS = """\
import subprocess
import sys
import time

subprocess.Popen(
    [sys.executable, "-c", sys.argv[1], sys.argv[2], sys.argv[3]],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(3600)
"""

#: What that child runs: a byte appended every 50ms, for at most the seconds it
#: is handed. A file that stops growing is a process that stopped.
HEARTBEAT = """\
import sys
import time

path, deadline = sys.argv[1], time.time() + float(sys.argv[2])
while time.time() < deadline:
    with open(path, "a") as beat:
        beat.write(".")
    time.sleep(0.05)
"""


def stage_entry(
    root: pathlib.Path, victim: str, target: Target
) -> tuple[pathlib.Path, pathlib.Path]:
    """Copy this file into `root` as a sweep of `target` alone; return entry and targets.

    The file is copied in under a name no target claims, so its own `REPO_ROOT`
    resolves to the staged tree while the staged sweep still has only the victim
    to sweep. Split out because two cases need the tree: `staged_sweep` runs the
    entry point over it, and the re-entry case hands the same command to
    `reddened` instead, which is where SWEEP_MARKER is put into the environment.
    """
    entry = root / "scripts" / "sweep.py"
    entry.parent.mkdir()
    shutil.copy(pathlib.Path(__file__).resolve(), entry)
    (root / target.script).write_text(victim)
    described = root / "targets.json"
    described.write_text(json.dumps([dataclasses.asdict(target)]))
    return entry, described


def staged_sweep(
    victim: str, target: Target, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """This file, run as a process over a tree staged in a temporary directory.

    Everything `self_test` does above calls `sites` and `problems_for` directly,
    which leaves `run` — and `main`'s dispatch to it, the two lines CI actually
    invokes — with nothing behind them. The file is copied in under a name no
    target claims, so its own `REPO_ROOT` resolves to the staged tree while the
    staged sweep still has only the victim to sweep. SWEEP_MARKER is stripped
    from the child's environment unless a case sets it: when the real sweep runs
    this file's `--self-test` the marker is already set, and inheriting it would
    make every case here refuse.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        entry, described = stage_entry(root, victim, target)
        env = {name: value for name, value in os.environ.items() if name != SWEEP_MARKER}
        env.update(environment or {})
        return subprocess.run(
            [sys.executable, str(entry), "--targets", str(described)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
        )


def self_test() -> int:
    """Sweep a script whose uncovered assertions are known, and require exactly those."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "victim.py").write_text(VICTIM)

        found = tuple(site.source for site in sites(VICTIM, VICTIM_TARGET.unswept()))
        if found != VICTIM_SITES:
            print(
                f"self-test FAILED: the enumerated sites were {list(found)} and the victim "
                f"carries {list(VICTIM_SITES)}. A sweep that enumerates the wrong sites reports "
                f"on the wrong lines, and `not_swept` is what keeps the victim's own self-test "
                f"out of them",
                file=sys.stderr,
            )
            return 1

        problems, green, tried = problems_for(root, VICTIM_TARGET)
        if (tried, green) != (len(VICTIM_SITES), len(VICTIM_UNDRIVEN)):
            print(
                f"self-test FAILED: {tried} sites tried and {green} survived, and the victim "
                f"has {len(VICTIM_SITES)} sites of which {len(VICTIM_UNDRIVEN)} are undriven. "
                f"The driven half is what says the commands are run and their exit codes read",
                file=sys.stderr,
            )
            return 1
        for source in VICTIM_UNDRIVEN:
            if not any(source in problem for problem in problems):
                print(
                    f"self-test FAILED: the undriven site `{source}` was not reported: "
                    f"{problems}",
                    file=sys.stderr,
                )
                return 1
        if len(problems) != len(VICTIM_UNDRIVEN):
            print(
                f"self-test FAILED: {len(problems)} problems for {len(VICTIM_UNDRIVEN)} undriven "
                f"sites, so a driven one was reported as well: {problems}",
                file=sys.stderr,
            )
            return 1
        if (root / "victim.py").read_text() != VICTIM:
            print("self-test FAILED: the swept file was not put back", file=sys.stderr)
            return 1

        # The exception mechanism, and both ways an exception goes stale.
        permitted = PERMITTED_TARGET
        if problems_for(root, permitted)[0] != []:
            print(
                f"self-test FAILED: permitted survivors were still reported: "
                f"{problems_for(root, permitted)[0]}",
                file=sys.stderr,
            )
            return 1

        needless = dataclasses.replace(
            VICTIM_TARGET, allowed=(Allowed("problems_in", '"driven-if" in text', "not needed"),)
        )
        if not any("is now noticed" in problem for problem in problems_for(root, needless)[0]):
            print(
                f"self-test FAILED: an allowance for a branch that IS noticed was not reported: "
                f"{problems_for(root, needless)[0]}",
                file=sys.stderr,
            )
            return 1

        moved = dataclasses.replace(
            VICTIM_TARGET, allowed=(Allowed("problems_in", '"deleted" in text', "gone"),)
        )
        if not any("no such branch" in problem for problem in problems_for(root, moved)[0]):
            print(
                f"self-test FAILED: an allowance naming a branch that does not exist was not "
                f"reported: {problems_for(root, moved)[0]}",
                file=sys.stderr,
            )
            return 1

        # And the other escape hatch going stale the same way. `not_swept` used
        # to be bare names with no reason and no check, which is how `main` sat
        # in this file's own entry excluding five sites nobody had looked at.
        renamed = dataclasses.replace(
            permitted,
            not_swept=permitted.not_swept + (Unswept("problems_in_v2", "renamed away"),),
        )
        if not any("no such function" in problem for problem in problems_for(root, renamed)[0]):
            print(
                f"self-test FAILED: `not_swept` naming a function that does not exist was not "
                f"reported: {problems_for(root, renamed)[0]}",
                file=sys.stderr,
            )
            return 1

    # `run`, and `main`'s dispatch to it, driven end to end as a process. Every
    # case above calls the pieces directly and leaves the lines that wire them
    # together untested, which is exactly what `main` being excluded by name had
    # hidden. These three are what those lines redden.
    broken = staged_sweep(VICTIM, VICTIM_TARGET)
    if broken.returncode != 1:
        print(
            f"self-test FAILED: swept as a process, a victim with undriven sites in it left the "
            f"entry point at {broken.returncode} rather than 1. That is the loop over the "
            f"targets, the accumulation into `problems` and `if problems:` — the wiring, not the "
            f"assertions: {broken.stdout}{broken.stderr}",
            file=sys.stderr,
        )
        return 1
    for source in VICTIM_UNDRIVEN:
        if source not in broken.stderr:
            print(
                f"self-test FAILED: swept as a process, the entry point exited 1 without naming "
                f"`{source}` in what it printed. An exit code with no report behind it does not "
                f"say which line survived: {broken.stdout}{broken.stderr}",
                file=sys.stderr,
            )
            return 1

    clean = staged_sweep(VICTIM, PERMITTED_TARGET)
    if clean.returncode != 0 or f"ok: {VICTIM_TARGET.script} —" not in clean.stdout:
        print(
            f"self-test FAILED: swept as a process, a victim with every survivor permitted left "
            f"the entry point at {clean.returncode} and printed {clean.stdout!r}. It should exit "
            f"0 with one `ok:` line naming each target: {clean.stderr}",
            file=sys.stderr,
        )
        return 1

    nested = staged_sweep(VICTIM, PERMITTED_TARGET, {SWEEP_MARKER: "1"})
    if nested.returncode != 2 or "already inside a sweep" not in nested.stderr:
        print(
            f"self-test FAILED: reached from inside a sweep the entry point returned "
            f"{nested.returncode} and said {nested.stderr!r}, rather than refusing. Without that "
            f"refusal, neutering `args.self_test` turns every `--self-test` this sweep runs "
            f"against itself into another live sweep of this file: measured, that recursed until "
            f"the outer command hit TIMEOUT_SECONDS, and the orphaned processes were still "
            f"rewriting the file seconds after the sweep had restored it",
            file=sys.stderr,
        )
        return 1

    # `reddened`'s end of the marker, which no case above reaches. Each of them
    # either sets SWEEP_MARKER itself or strips it, and none asks whether
    # `reddened` is what puts it there. Measured: replacing that environment
    # with a bare `dict(os.environ)` and leaving everything else alone left
    # every Python gate in this repo at exit 0 — including the live sweep, which
    # printed its usual `ok:` line — while twenty-three orphans in twenty-three
    # distinct process groups sat under it, still multiplying twenty seconds
    # later, and both checked scripts were left gutted on disk.
    #
    # The command below re-enters this entry point over a staged tree. With the
    # marker delivered, `run` refuses in milliseconds and `reddened` reads a
    # non-zero exit. Without it the staged sweep runs for real over a victim
    # that has nothing to report and comes back 0, so this fails rather than
    # recursing — and the `clean` case above is the control for that half,
    # being the same file, victim and target run as a process with the marker
    # stripped, required to exit 0.
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        entry, described = stage_entry(root, VICTIM, PERMITTED_TARGET)
        reentry = (sys.executable, str(entry), "--targets", str(described))
        if not reddened(root, (reentry,)):
            print(
                f"self-test FAILED: a command that re-enters this entry point came back 0 "
                f"through `reddened`, so {SWEEP_MARKER} never reached the child's environment "
                f"and the sweep it re-entered ran instead of refusing. That marker is what "
                f"bounds a neutered `args.self_test` dispatch: without it every `--self-test` "
                f"the sweep runs against this file becomes another live sweep of it, and the "
                f"group kill reaches one level of that",
                file=sys.stderr,
            )
            return 1

    # The group kill, which no case above reaches either: replacing
    # `os.killpg(...)` with `child.kill()` left every Python gate at exit 0 too.
    # The victim spawns a child of its own into the same process group and then
    # stops responding, so the timeout path runs; afterwards the heartbeat that
    # child was writing has to have stopped. `subprocess.run(timeout=...)` kills
    # the process it started and leaves that child running, which is the shape
    # of what went on rewriting a tracked file after the sweep had restored it.
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        beat = root / "heartbeat"
        hangs = ("python3", "-c", SPAWNS_AND_HANGS, HEARTBEAT, str(beat), str(HEARTBEAT_SECONDS))
        if not reddened(root, (hangs,), timeout=KILL_TIMEOUT_SECONDS):
            print(
                "self-test FAILED: a command that never exits came back green. A hang has to "
                "count as noticing, or a neuter that turns a loop infinite reads as a check "
                "that passed",
                file=sys.stderr,
            )
            return 1
        if not beat.exists():
            print(
                f"self-test FAILED: the hanging command's own child never wrote {beat.name}, so "
                f"the kill was measured against nothing running and this case could not have "
                f"failed. It has KILL_TIMEOUT_SECONDS ({KILL_TIMEOUT_SECONDS}s) to start two "
                f"interpreters",
                file=sys.stderr,
            )
            return 1
        written = beat.stat().st_size
        time.sleep(HEARTBEAT_WINDOW_SECONDS)
        if beat.stat().st_size != written:
            print(
                f"self-test FAILED: a process below the timed-out command was still writing "
                f"{HEARTBEAT_WINDOW_SECONDS}s after it was killed. The kill has to name the "
                f"process group: `subprocess.run(timeout=...)` kills the process it started and "
                f"nothing beneath it, and what survives that is what went on rewriting a "
                f"tracked file four seconds after this sweep had put it back",
                file=sys.stderr,
            )
            return 1

    print(
        f"self-test ok: over a staged script carrying a driven and an undriven `if`, `for` and "
        f"`problems +=`, the sweep enumerates all {len(VICTIM_SITES)} in source order, reports "
        f"the {len(VICTIM_UNDRIVEN)} undriven ones and no other, puts the file back, reports "
        f"nothing once they are permitted, and reports a permission both when the branch it "
        f"names is noticed after all and when the branch is gone, and a `not_swept` entry naming "
        f"a function the file does not define; and with this file run as a process over a staged "
        f"tree, `run` reports every undriven site and exits 1, exits 0 with an `ok:` line per "
        f"target once they are permitted, and refuses with 2 when it is reached from inside a "
        f"sweep. Through `reddened`: a command that re-enters that entry point is refused "
        f"rather than sweeping, which is what says {SWEEP_MARKER} is delivered, and a command "
        f"that hangs having spawned a child of its own leaves nothing of it writing "
        f"{HEARTBEAT_WINDOW_SECONDS}s later. {len(TARGETS)} scripts are swept for real"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--targets",
        metavar="JSON",
        help="sweep the targets this JSON file describes instead of TARGETS. Passed by "
        "--self-test, to drive this entry point over a tree it staged",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.targets:
        return run(REPO_ROOT, targets_from(pathlib.Path(args.targets)))
    return run(REPO_ROOT, TARGETS)


if __name__ == "__main__":
    sys.exit(main())
