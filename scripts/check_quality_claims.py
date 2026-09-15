#!/usr/bin/env python3
"""Fail when the two published pages disagree about how good these vectors are.

`README.md` and `description.yml` both tell a stranger what this embedder can
and cannot do, and `description.yml` is the copy that becomes the extension's
page in the DuckDB community registry — the one nobody arrives at through this
repository. Two prose copies of a quality claim are two chances to be wrong, and
this repository has already shipped that defect once on the function table,
which is why `check_documented_surface.py` exists.

A quality claim cannot be settled by loading the extension the way a function
signature can. What it can be settled against is the measurement that produced
it, and against the other copy of itself.

WHAT IS ASSERTED
    1. Both files carry a section headed `SECTION_HEADING`, and neither is
       empty. A scan that reports clean because a heading was renamed out from
       under it is not a scan.
    2. Every quantity written in either section is one of `FIGURES` — a number
       somebody measured, recorded here with what it measures and where it came
       from. An unregistered quantity is a claim with nothing behind it, so the
       set is closed: a new figure reddens until it is entered here with its
       source.
    3. The two sections write the same quantities, the same number of times, in
       the same order. Order matters because a figure moved to another row or
       another corpus leaves the set of quantities untouched: a summary-table
       cell swapped in one file and not the other used to pass a set comparison
       in both directions.
    4. Every string in `CLAIMS` appears in both sections. These pin the wording
       a figure is embedded in, so a rewrite that drops the corpus a figure
       belongs to reddens.
    5. The summary table's columns run `CORPORA` in that order, and every cell
       of a `TABLE_ROWS` row is the figure `FIGURES` registers for that row's
       series and that column's corpus. This is the assertion that survives a
       maintainer editing page and pin together: `13%` under `short text`
       reddens against a registered 0.27758 however the prose is worded.
    6. No sentence writes part of a series in `DIRECTIONAL_SERIES` without
       writing all of it. Both series run over the same three corpora and one of
       them is not monotone — region structure is 71%, 67%, 88%, so short text
       is the worst of the three and not the middle. A sentence quoting the two
       endpoints reads as a gradient and is not one, and this is what stops the
       page claiming a direction the reader cannot check from the same sentence.
    7. No universal quantifier appears in either section unless it is recorded
       in `ALLOWED_UNIVERSALS` with why it holds. See the block below for what
       this covers and what it cannot see; it is the widest of these assertions
       and the one with the largest blind spot.
    8. No phrase in `BANNED_IN_SECTION` appears in either section — the hedges
       the figures replaced. And no phrase in `BANNED_ON_PAGE` appears in
       `README.md`, or in any of the `description.yml` fields listed in
       `PAGE_FIELDS`: speed was ruled out of the product claim on purpose,
       because the embedding-only number is an order of magnitude larger than
       the end-to-end one, and a speed figure planted under *The SQL surface*
       flatters the product to the same reader as one planted here. That ban is
       scoped to the page for that reason rather than to the section, and it is
       the only scan here read over `strip_addresses` rather than `strip_noise`,
       so it sees code as well as prose. It has to: `docs.hello_world` is a
       published SQL example that is not fenced, `description.yml` fences the
       rest and `README.md` fences its own, and a ban that ran after code was
       stripped read none of them.
    9. The bundled model has not moved off `MEASURED_ON_REVISION`.
   10. Every entry in `FIGURES` and every entry in `ALLOWED_UNIVERSALS` is
       written on at least one of the two pages. A permitted value or a
       permitted universal that outlives the sentence it permitted widens
       assertions 2 and 7 without anyone deciding to, which is the same shape as
       a mutation anchor naming a line that has been deleted.

WHAT THE UNIVERSAL-QUANTIFIER SCAN COVERS, AND WHAT IT CANNOT SEE
    It covers exactly the words in `UNIVERSAL_WORDS`, matched whole, inside the
    two quality sections, after code spans and links are removed. It exists
    because both published defects it was written for were the same shape: a
    blanket sentence asserting a uniform property over a list whose items do not
    share it. "Every figure below compares potion-base-8M against MiniLM" stood
    over a section that deliberately mixes our measurement with a third party's,
    and cancelled the per-item sourcing the rest of the section does carefully.
    Pinning sentences in `CLAIMS` could not have found it, because `CLAIMS` pins
    what somebody thought to enumerate.

    It cannot see a universal written without one of those words. A bare plural
    generic — "the figures below compare X against Y" — asserts precisely the
    same thing as "every figure below compares X against Y" and carries no
    quantifier at all. That is the largest hole and it is not closeable by
    widening the word list; this scan is a floor, not a proof.

    It cannot see a spelling with no `UNIVERSAL_WORDS` member in it —
    `invariably`, `without exception`, `universally`. It does see `in each case`
    and `bar none`, which name no listed word and carry `each` and `none` as
    whole words, so the list reaches further than reading it suggests. The
    self-test pins both halves of that, because neither is derivable by eye and
    the second invites a maintainer to widen a list that already covers it.
    Adding a spelling is a one-line change and should be made when one is met,
    rather than assumed absent.

    It does not judge truth. An entry in `ALLOWED_UNIVERSALS` is a person's
    recorded reason, not a derivation, and nothing here re-checks it. What it
    does guarantee is that the reason exists, in this file, next to the phrase
    it permits, where a reviewer reads it — and that it disappears when the
    sentence does, by assertion 10.

    It is scoped to the two quality sections and not to the whole page. The rest
    of `README.md` is full of true universals about a deterministic function
    ("`subtoken_embed(NULL)` is always NULL"), and a scan that cried wolf there would be
    turned off. The speed ban in assertion 8 is the one that runs page-wide and
    into the code examples, because that is where its defects were planted.

WHAT IS NOT ASSERTED
    That the figures are true. `FIGURES` is a transcription of a measurement
    made elsewhere, named in each entry's `source`; nothing here re-runs it.

    That the figures were measured on the bundled revision. The harness in
    finetype downloads `potion-base-8M` by name and records no revision, and
    `results.json` carries none, so `MEASURED_ON_REVISION` is a tripwire and not
    a proof: it is the revision `SOURCE.md` recorded when these figures were
    published. Assertion 9 forces a model bump to be noticed and the measurement
    re-run or deliberately re-blessed. It cannot certify what the harness ran
    against, and the page does not claim it can.

    A quality claim carrying no quantity at all. "Roughly as good as a
    transformer" in one file and not the other passes every assertion above.
    `BANNED_IN_SECTION` catches the specific hedges this page used to carry and
    no others.

Needs PyYAML, because `extended_description` is a block scalar inside a YAML
document and a regex cannot tell one from the prose around it.

    scripts/check_quality_claims.py
    scripts/check_quality_claims.py --self-test
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

README = "README.md"
DESCRIPTOR = "description.yml"
MODEL_SOURCE = "models/potion-base-8M/SOURCE.md"

#: What "the `description.yml` page" means to assertion 8, as (table, field).
#: The registry renders all three: `extension.description` is the one-line
#: blurb, `docs.hello_world` is the worked SQL example, and
#: `docs.extended_description` is the body. Assertion 8 read the body alone
#: until this existed, so a speed figure in the published example — the SQL a
#: stranger reads first — passed it. `check_description_examples.py` searches
#: the same two `docs` fields for the names of the functions the artifact
#: registers, for the same reason: the example is part of what is read.
PAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("extension", "description"),
    ("docs", "hello_world"),
    ("docs", "extended_description"),
)

#: The level-2 heading whose section carries the quality position, in both
#: files, spelled identically. Renaming it in one file is caught by assertion 1.
SECTION_HEADING = "What it is good at, and what it is not"

#: The revision `models/potion-base-8M/SOURCE.md` recorded when the figures in
#: `FIGURES` were published. A tripwire, not a proof — see WHAT IS NOT ASSERTED.
MEASURED_ON_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"

#: The finetype commit `results.json` was generated at. Every figure below is
#: read from that file at that commit, and the self-test holds FIGURES to it:
#: a figure sourced anywhere else is a figure nobody in this estate measured,
#: which is what the SwiftEmbed citations this page used to carry were.
MEASURED_AT_COMMIT = "196d102a"

FINETYPE_EVAL = (
    f"finetype eval/static-embedding-map-fidelity/results.json at {MEASURED_AT_COMMIT}"
)

#: The three corpora, in the order the summary table's columns run and in the
#: order the prose lists them. Assertion 5 reads the table against this, so a
#: reversed header row is a mismatch rather than an invisible edit.
CORPORA = ("long-form prose", "short text", "very short strings")


@dataclass(frozen=True)
class Figure:
    """A quantity the pages are allowed to write, and what it is a figure of."""

    value: float
    what: str
    source: str
    #: The named run of figures over `CORPORA` this belongs to, if any. Used by
    #: assertion 5 to read a summary-table cell and by assertion 6 to require a
    #: whole series in one sentence.
    series: str = ""
    #: Which of `CORPORA` this figure is for. Empty when the figure is not one
    #: of a per-corpus series.
    corpus: str = ""


#: Every quantity either section may contain. Closed on purpose: a figure that
#: is not here reddens, so a new claim cannot reach the registry page without
#: someone writing down what it measures and who measured it.
FIGURES: list[Figure] = [
    # potion-base-8M against all-MiniLM-L6-v2, seed 42, 20 nearest neighbours,
    # UMAP(metric="cosine", n_neighbors=15, random_state=42), k=10 for ranked
    # retrieval. `bm25` is DuckDB's own `fts` extension over the same corpora
    # with no model loaded: the floor a reader has before installing this one.
    Figure(
        0.13185,
        "kNN overlap with MiniLM's map, long-form prose",
        FINETYPE_EVAL,
        series="kNN overlap",
        corpus="long-form prose",
    ),
    Figure(
        0.27758,
        "kNN overlap with MiniLM's map, short text",
        FINETYPE_EVAL,
        series="kNN overlap",
        corpus="short text",
    ),
    Figure(
        0.40301,
        "kNN overlap with MiniLM's map, very short strings",
        FINETYPE_EVAL,
        series="kNN overlap",
        corpus="very short strings",
    ),
    Figure(
        0.71005,
        "cluster-structure retention vs MiniLM, long-form prose",
        FINETYPE_EVAL,
        series="region retention",
        corpus="long-form prose",
    ),
    Figure(
        0.66735,
        "cluster-structure retention vs MiniLM, short text",
        FINETYPE_EVAL,
        series="region retention",
        corpus="short text",
    ),
    Figure(
        0.87503,
        "cluster-structure retention vs MiniLM, very short strings",
        FINETYPE_EVAL,
        series="region retention",
        corpus="very short strings",
    ),
    Figure(
        0.76335,
        "ranked lift over the random control, potion-base-8M, long-form prose",
        FINETYPE_EVAL,
        series="ranked lift",
        corpus="long-form prose",
    ),
    Figure(
        0.85301,
        "ranked lift over the random control, potion-base-8M, short text",
        FINETYPE_EVAL,
        series="ranked lift",
        corpus="short text",
    ),
    Figure(
        0.88271,
        "ranked lift over the random control, potion-base-8M, very short strings",
        FINETYPE_EVAL,
        series="ranked lift",
        corpus="very short strings",
    ),
    Figure(
        0.74618,
        "ranked lift over the random control, BM25 with no model, long-form prose",
        FINETYPE_EVAL,
        series="bm25 lift",
        corpus="long-form prose",
    ),
    Figure(
        0.69389,
        "ranked lift over the random control, BM25 with no model, short text",
        FINETYPE_EVAL,
        series="bm25 lift",
        corpus="short text",
    ),
    Figure(
        0.58767,
        "ranked lift over the random control, BM25 with no model, very short strings",
        FINETYPE_EVAL,
        series="bm25 lift",
        corpus="very short strings",
    ),
    # Pairwise, pooled under one global threshold, on the column-name corpus.
    # Registered for that corpus alone because it is the one the pages publish:
    # near-duplicate AP is above 0.99 on both prose corpora for the model and
    # for BM25 alike, so quoting it there would read as a strength nothing in
    # the run distinguishes.
    Figure(0.91133, "pairwise near-duplicate AP, potion-base-8M, very short strings", FINETYPE_EVAL),
    Figure(0.72620, "pairwise near-duplicate AP, BM25 with no model, very short strings", FINETYPE_EVAL),
    Figure(0.68615, "pairwise same-class AP, potion-base-8M, very short strings", FINETYPE_EVAL),
    Figure(0.68151, "pairwise same-class AP, all-MiniLM-L6-v2, very short strings", FINETYPE_EVAL),
    Figure(0.5, "the floor pooled AP sits at: one positive and one negative pair per anchor", FINETYPE_EVAL),
    Figure(0.39244, "potion-base-8M AMI over raw vectors, very short strings", FINETYPE_EVAL),
    Figure(0.35104, "all-MiniLM-L6-v2 AMI over raw vectors, very short strings", FINETYPE_EVAL),
    # Two entries for one value on purpose: the long-form and short-text corpora
    # are separate samples that happen to be the same size, and assertion 5
    # reads one table cell per corpus.
    Figure(
        3000,
        "rows sampled from the 20 Newsgroups posts",
        FINETYPE_EVAL,
        series="corpus size",
        corpus="long-form prose",
    ),
    Figure(
        3000,
        "rows sampled from their subject lines",
        FINETYPE_EVAL,
        series="corpus size",
        corpus="short text",
    ),
    Figure(
        216,
        "rows in the column-name corpus",
        FINETYPE_EVAL,
        series="corpus size",
        corpus="very short strings",
    ),
    Figure(12, "classes in the column-name corpus", FINETYPE_EVAL),
    Figure(20, "nearest neighbours compared; also the 20 Newsgroups corpus name", FINETYPE_EVAL),
]

#: Assertion 5. Summary-table row label → the series its cells are figures of.
#: The `corpus` row is prose rather than numbers and is pinned in `CLAIMS`.
TABLE_ROWS: dict[str, str] = {
    "rows": "corpus size",
    "ranked lift, this model": "ranked lift",
    "ranked lift, BM25 with no model": "bm25 lift",
    "nearest neighbours that survive": "kNN overlap",
    "region structure kept": "region retention",
}

#: Assertion 6. Series a reader could mistake for a gradient. `region retention`
#: is 71%, 67%, 88% — not monotone — and a sentence quoting 71% and 88% alone
#: told a reader with one-line descriptions they were at the good end when the
#: cited metric puts them at the worst. `corpus size` is left out: nobody claims
#: a direction over sample sizes, and requiring all three would redden the
#: sentence that says 216 rows is a small sample. `ranked lift` and `bm25 lift`
#: are left out for a different reason: both run monotonically with the shape of
#: the text — 0.763, 0.853, 0.883 and 0.746, 0.694, 0.588 — so a sentence
#: quoting one corpus's pair states no direction it has to walk back, and the
#: per-corpus sentences the page carries are each one such pair. Their cells are
#: still read one at a time by assertion 5, through TABLE_ROWS.
DIRECTIONAL_SERIES: tuple[str, ...] = ("kNN overlap", "region retention")

#: Assertion 4. Each pins a figure to what it is a figure of, or pins a
#: distinction the page would otherwise lose in a rewrite. Matched after
#: whitespace is collapsed and case folded, so `description.yml` wrapping one
#: across two lines does not hide it.
CLAIMS: list[str] = [
    # AC4: the distinction the page was missing, in both files, and the one
    # weakness that sits on the pairwise side of it.
    "pairwise judgement",
    "ranked retrieval",
    "a false duplicate, which is a pairwise failure and not a ranked-retrieval one",
    # What it is good at, each with its figure. Pairwise duplicate scoring is
    # the strongest thing this run measured and is our own, which the SwiftEmbed
    # citations these three replaced were not: they were another serving
    # system's numbers on another benchmark, read as this extension's.
    "0.9113 for the bundled model, against 0.7262 for BM25 on the same 216 rows",
    "floor is 0.5",
    "0.6862 for the bundled model and 0.6815",
    "216 column names in 12 semantic classes",
    "0.3924 against 0.3510",
    # AC1: the figures that replaced "most" and "a minority".
    "71% of what MiniLM's map recovers on long-form prose, 67% on short text, "
    "88% on very short strings",
    "13% are the same rows on long-form prose, 28% on short text, 40% on very short strings",
    "20 nearest neighbours",
    # AC2: the direction of the shape dependence for neighbourhoods, and the
    # fact that region structure does not follow it.
    "worst on long prose and mildest on very short strings",
    "13%, then 28%, then 40% as the text gets shorter",
    # The free floor, per corpus. Each of these three sentences carries its
    # corpus and its row count, because a lift figure without them says nothing
    # about which shape of text it was measured on — and the long-prose pair is
    # the one a reader most needs, since it is where the model barely clears a
    # BM25 index the engine already ships.
    "ranked lift is 0.763 for the bundled model and 0.746 for BM25 on the same rows",
    "over their 3,000 subject lines it is 0.853 for the model against 0.694 for BM25",
    "over the 216 column names it is 0.883 against 0.588",
    "a bundled model that does not beat it has not earned its download",
    # The region sentence, tied to its corpora rather than to its conclusion.
    # Its trailing clause survives the first two figures being transposed, and
    # a transposed pair passes assertions 2, 3, 5 and 6 untouched: both values
    # stay registered, both files agree, the table is not edited and the series
    # is still quoted whole. What it leaves is a sentence that contradicts the
    # table two lines below it and tells a reader with one-line descriptions
    # the wrong end to be at, which is the defect this page was rewritten for.
    "71% on long-form prose, 67% on short text, 88% on very short strings",
    "short text is the weakest of the three for regions, not the middle of them",
    # The summary table's one prose row. Its numeric rows are read against
    # FIGURES by assertion 5, which a literal pin cannot do.
    "| corpus | 20 Newsgroups posts | their subject lines | column names |",
]

#: Assertion 8, inside the section. Case-folded substring match.
BANNED_IN_SECTION: list[tuple[str, str]] = [
    ("a minority", "the hedge a measured figure replaced; it is compatible with 45%"),
    ("recovers most", "the hedge a measured figure replaced"),
    ("most of the cluster", "the hedge a measured figure replaced"),
    ("on the evidence we have", "a hedge standing where the corpus and the sample size belong"),
]

#: Assertion 8, over `README.md` and the `PAGE_FIELDS` of `description.yml`.
#: Speed is ruled out of the product claim, and a speed figure three sections
#: down — or in the worked example above the prose — reaches the same reader.
BANNED_ON_PAGE: list[tuple[str, str]] = [
    ("faster", "speed is not part of the published claim, deliberately"),
    ("speedup", "speed is not part of the published claim, deliberately"),
    ("throughput", "speed is not part of the published claim, deliberately"),
    ("latency", "speed is not part of the published claim, deliberately"),
    ("per second", "speed is not part of the published claim, deliberately"),
    ("rows/s", "speed is not part of the published claim, deliberately"),
    ("×", "a multiplier is how a speed figure arrives; speed is ruled out here"),
]

#: Assertion 7. Words that assert a property over a whole set. Matched whole, so
#: `all-MiniLM-L6-v2` and `overall` are not hits — though both are usually
#: inside a code span and removed before this runs anyway.
UNIVERSAL_WORDS: tuple[str, ...] = (
    "every",
    "everything",
    "all",
    "always",
    "any",
    "anything",
    "each",
    "never",
    "only",
    "none",
)


@dataclass(frozen=True)
class Universal:
    """A universal these sections may write, and why it holds over its set."""

    phrase: str
    why: str


#: Assertion 7's exceptions, and the whole exception mechanism: a universal not
#: written out here reddens. Kept short on purpose — every entry is a sentence
#: nobody will re-derive, so the reason has to carry it.
ALLOWED_UNIVERSALS: list[Universal] = [
    Universal(
        "any phrase and its shuffle land in the same place",
        "permutation invariance is a property of the arithmetic mean rather than a sample of "
        "it. `crates/subtoken-core` pools by averaging token vectors; "
        "`the_pool_is_a_mean_so_order_is_lost_and_repetition_is_not` asserts it in Rust and "
        "`test/sql/06_text_the_tokenizer_treats_as_one_value.sql` asserts it against a loaded "
        "extension, so this quantifies over inputs nobody tried",
    ),
]

#: A number in prose. The lookbehind keeps the `6` of `all-MiniLM-L6-v2` out,
#: and the lookahead keeps the `8` of `potion-base-8M` out while still admitting
#: `top-20` and `13%`. Thousands separators are kept and stripped when parsed.
QUANTITY = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%?)(?![\w-])")

#: A whole word from `UNIVERSAL_WORDS`, over already-collapsed (case-folded) text.
UNIVERSAL = re.compile(r"(?<![\w-])(" + "|".join(UNIVERSAL_WORDS) + r")(?![\w-])")

#: A sentence end in collapsed prose. Trailing `*` and `_` are consumed so a
#: bolded lead-in ends its sentence where a reader sees it end.
SENTENCE_END = re.compile(r"(?<=[.;:!?])[*_]*\s+")

FENCED = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
BARE_URL = re.compile(r"https?://\S+")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def strip_addresses(text: str) -> str:
    """Remove links, bare URLs and dates. Code is left where it is.

    An address is not a quantity — `arxiv.org/abs/2510.24793` and `2026-08-24`
    both parse as one — and neither is it a claim about the product. Code is a
    different matter: it is where a speed figure hides. This is what assertion
    8 scans over, so it reads the SQL examples — `docs.hello_world` raw, the
    fenced ones as they are written — as well as the prose around them.
    """
    text = LINK_TARGET.sub("]", text)
    text = BARE_URL.sub(" ", text)
    text = ISO_DATE.sub(" ", text)
    return text


def strip_noise(text: str) -> str:
    """Remove everything in a section that is code or an address, not a claim.

    SQL examples are checked by `check_description_examples.py` against a real
    build, so a number inside one is not a published figure.
    """
    text = FENCED.sub(" ", text)
    text = INLINE_CODE.sub(" ", text)
    return strip_addresses(text)


def collapse(text: str) -> str:
    """Case-fold and squeeze whitespace, so a wrapped line matches an unwrapped one."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def section(text: str, heading: str) -> str | None:
    """The body under the level-2 `heading`, up to the next heading of that level or above.

    `None` when the heading is absent or written more than once — both are
    reported rather than treated as an empty section, because a section this
    cannot find is a section it is not checking.
    """
    lines = text.splitlines()
    starts = []
    for index, line in enumerate(lines):
        match = HEADING.match(line.strip())
        if match and len(match.group(1)) == 2 and match.group(2) == heading:
            starts.append(index)
    if len(starts) != 1:
        return None

    body = []
    for line in lines[starts[0] + 1 :]:
        match = HEADING.match(line.strip())
        if match and len(match.group(1)) <= 2:
            break
        body.append(line)
    return "\n".join(body)


def parse_quantity(digits: str, percent: str) -> tuple[float, int]:
    """A written quantity as (value, decimal places it was written to).

    `13%` is 0.13 written to two places; `0.3924` is itself written to four;
    `3,000` is three thousand written to none. The precision is returned so a
    figure can be compared at the precision the page rounded it to, which is how
    `13%` matches a measured 0.13185 and `14%` does not.

    A percentage carries two more places than its digits show: `13%` is 0.13,
    which is written to two decimal places, not to none.
    """
    places = len(digits.split(".")[1]) if "." in digits else 0
    value = float(digits.replace(",", ""))
    if percent:
        value /= 100.0
        places += 2
    return value, places


def quantities(section_text: str) -> list[tuple[float, int, str]]:
    """(value, decimal places, as written) for every quantity in a section."""
    found = []
    for match in QUANTITY.finditer(strip_noise(section_text)):
        value, places = parse_quantity(match.group(1), match.group(2))
        found.append((value, places, match.group(0).strip()))
    return found


def rounds_onto(figure: Figure, value: float, places: int) -> bool:
    """Whether a quantity written to `places` places is a rounding of `figure`.

    The interval is used rather than `round()` because `round()` has to pick a
    side of a tie and 0.13185 is one: at four places it is as good a rounding to
    0.1318 as to 0.1319, and a check that admits one spelling and reddens on the
    other is arbitrary.
    """
    return abs(figure.value - value) <= 0.5 * 10.0**-places + 1e-12


def measured(value: float, places: int) -> Figure | None:
    """The registered figure a written quantity is a rounding of, if there is one.

    A written quantity matches when the measurement falls inside the interval
    that rounds to it at the precision it was written to — `13%` covers
    everything from 0.125 to 0.135, so it covers a measured 0.13185, and `14%`
    covers none of it.
    """
    for figure in FIGURES:
        if rounds_onto(figure, value, places):
            return figure
    return None


def figure_for(series: str, corpus: str) -> Figure | None:
    """The one registered figure for a series and a corpus. Assertion 5."""
    for figure in FIGURES:
        if figure.series == series and figure.corpus == corpus:
            return figure
    return None


def unused_figures(*sections: str | None) -> list[str]:
    """Registered figures no section writes. Assertion 10."""
    written = [
        (value, places)
        for text in sections
        if text is not None
        for value, places, _ in quantities(text)
    ]
    problems = []
    for figure in FIGURES:
        if not any(rounds_onto(figure, value, places) for value, places in written):
            problems.append(
                f"FIGURES registers {figure.value} ({figure.what}, from {figure.source}) and "
                f"neither page writes it. A permitted value nothing uses widens what the pages "
                f"may say without anyone deciding to: delete it, or put the claim back"
            )
    return problems


def unused_allowances(*sections: str | None) -> list[str]:
    """Permitted universals no section writes. Assertion 10."""
    haystack = " ".join(collapse(strip_noise(text)) for text in sections if text is not None)
    problems = []
    for allowance in ALLOWED_UNIVERSALS:
        if collapse(allowance.phrase) not in haystack:
            problems.append(
                f"ALLOWED_UNIVERSALS permits {allowance.phrase!r} and neither page writes it. A "
                f"live exception to the universal-quantifier scan that no longer covers any "
                f"sentence widens assertion 7 silently: delete it"
            )
    return problems


def table(section_text: str) -> list[list[str]]:
    """Every markdown table row in a section, as stripped cells, separators dropped."""
    rows = []
    for line in section_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if all(set(cell) <= set("-: ") and cell for cell in cells):
            continue
        rows.append(cells)
    return rows


def prose(section_text: str) -> str:
    """The section with its table rows removed, for assertions read per sentence."""
    kept = [
        line
        for line in section_text.splitlines()
        if not (line.strip().startswith("|") and line.strip().endswith("|"))
    ]
    return "\n".join(kept)


def table_problems(name: str, section_text: str) -> list[str]:
    """The summary table's cells are the figures they claim to be. Assertion 5."""
    rows = table(section_text)
    headers = [row for row in rows if row and row[0] == "" and tuple(row[1:]) == CORPORA]
    if len(headers) != 1:
        actual = [row[1:] for row in rows if row and row[0] == ""]
        return [
            f"{name}'s `## {SECTION_HEADING}` section has no single summary table whose columns "
            f"run {list(CORPORA)} in that order — found {actual}. Every figure below the header "
            f"is read by its column, so a reordered or missing header is not a cosmetic edit"
        ]
    width = len(CORPORA)

    problems = []
    for label, series in TABLE_ROWS.items():
        matches = [row for row in rows if row and row[0] == label]
        if len(matches) != 1:
            problems.append(
                f"{name}'s summary table has {len(matches)} rows labelled {label!r} and needs "
                f"exactly one: it is the row carrying the {series!r} figures"
            )
            continue
        cells = matches[0][1:]
        if len(cells) != width:
            problems.append(
                f"{name}'s summary table row {label!r} has {len(cells)} cells under "
                f"{width} corpora"
            )
            continue
        for corpus, cell in zip(CORPORA, cells):
            expected = figure_for(series, corpus)
            if expected is None:
                problems.append(
                    f"FIGURES registers no {series!r} figure for {corpus!r}, so the summary "
                    f"table's {label!r} row cannot be checked against it"
                )
                continue
            written = quantities(cell)
            if len(written) != 1:
                problems.append(
                    f"{name}'s summary table cell for {label!r} under {corpus!r} is {cell!r}, "
                    f"which carries {len(written)} quantities and needs exactly one"
                )
                continue
            value, places, as_written = written[0]
            if not rounds_onto(expected, value, places):
                problems.append(
                    f"{name}'s summary table puts `{as_written}` under {corpus!r} in the "
                    f"{label!r} row, and the measured {series} for {corpus!r} is "
                    f"{expected.value} ({expected.what}). Two cells swapped between corpora "
                    f"leave the set of quantities on the page unchanged, which is why this is "
                    f"read cell by cell"
                )
    return problems


def series_problems(name: str, section_text: str) -> list[str]:
    """No sentence writes part of a directional series. Assertion 6.

    The summary table is excluded: its rows are read cell by cell against
    `FIGURES` by assertion 5, which is a stronger statement than this one.
    """
    problems = []
    for sentence in SENTENCE_END.split(collapse(strip_noise(prose(section_text)))):
        for series in DIRECTIONAL_SERIES:
            members = [figure for figure in FIGURES if figure.series == series]
            written = {
                figure.corpus
                for value, places, _ in quantities(sentence)
                for figure in members
                if rounds_onto(figure, value, places)
            }
            if not written or len(written) == len(members):
                continue
            missing = [figure for figure in members if figure.corpus not in written]
            problems.append(
                f"{name} writes {len(written)} of the {len(members)} {series} figures in one "
                f"sentence and leaves out "
                + ", ".join(f"{figure.value} ({figure.corpus})" for figure in missing)
                + f". The sentence is: {sentence.strip()[:160]!r}. A partial series reads as a "
                f"direction, and {series} over {list(CORPORA)} is "
                + ", ".join(str(figure.value) for figure in members)
                + " — quote all three or the reader cannot tell a gradient from a dip"
            )
    return problems


def universal_problems(name: str, section_text: str) -> list[str]:
    """No unpermitted universal quantifier. Assertion 7."""
    collapsed = collapse(strip_noise(section_text))
    permitted = []
    for allowance in ALLOWED_UNIVERSALS:
        needle = collapse(allowance.phrase)
        start = collapsed.find(needle)
        while start != -1:
            permitted.append((start, start + len(needle)))
            start = collapsed.find(needle, start + 1)

    problems = []
    for match in UNIVERSAL.finditer(collapsed):
        if any(low <= match.start() and match.end() <= high for low, high in permitted):
            continue
        context = collapsed[max(0, match.start() - 60) : match.end() + 60]
        problems.append(
            f"{name}'s section writes the universal {match.group(1)!r} in: …{context}…  A "
            f"universal asserts one property over a whole set, and this section deliberately "
            f"mixes figures measured against MiniLM, figures measured against a no-model BM25 "
            f"floor and figures that compare nothing, so a blanket sentence cancels the "
            f"per-item sourcing the rest of it does. Rewrite it as "
            f"a bounded claim, or add it to ALLOWED_UNIVERSALS with why it holds"
        )
    return problems


def region_problems(name: str, section_text: str | None) -> list[str]:
    """Everything wrong with one file's section, judged without the other file."""
    if section_text is None:
        return [
            f"{name} has no single `## {SECTION_HEADING}` section. It is one of the two files "
            f"that publish the quality position, and a section this cannot find is a section "
            f"it is not checking"
        ]
    collapsed = collapse(strip_noise(section_text))
    if not collapsed:
        return [
            f"{name}'s `## {SECTION_HEADING}` section is empty once code and links are removed; "
            f"an empty section agrees with anything"
        ]

    problems = []
    for value, places, written in quantities(section_text):
        if measured(value, places) is None:
            problems.append(
                f"{name} writes the quantity `{written}`, which is not one of the measured "
                f"figures in FIGURES. Either it is wrong, or it is a new claim and belongs in "
                f"FIGURES with what it measures and who measured it"
            )
    for claim in CLAIMS:
        if collapse(claim) not in collapsed:
            problems.append(f"{name}'s section does not carry the claim: {claim!r}")
    for phrase, reason in BANNED_IN_SECTION:
        if collapse(phrase) in collapsed:
            problems.append(f"{name}'s section contains {phrase!r} — {reason}")
    problems += table_problems(name, section_text)
    problems += series_problems(name, section_text)
    problems += universal_problems(name, section_text)
    return problems


def page_problems(name: str, page_text: str) -> list[str]:
    """Assertion 8, over one whole page: speed vocabulary anywhere in `page_text`.

    Read over `strip_addresses` and not `strip_noise`, so fenced blocks and
    inline code spans are scanned as well as prose. The SQL examples are the
    surface this ban was widened to cover — a speed figure in a comment inside
    one reaches the same reader as a speed figure in a sentence — and they are
    not all fenced: `docs.hello_world` is published raw.

    What `page_text` is for the registry entry is `PAGE_FIELDS` joined, which is
    `descriptor_page`'s business and not this function's. Passing it one field
    is how this scan came to assert a reach it did not have.
    """
    collapsed = collapse(strip_addresses(page_text))
    problems = []
    for phrase, reason in BANNED_ON_PAGE:
        if collapse(phrase) in collapsed:
            position = collapsed.find(collapse(phrase))
            context = collapsed[max(0, position - 70) : position + 70]
            problems.append(
                f"{name} contains {phrase!r} — {reason}. Found in: …{context}…  This ban is "
                f"page-wide rather than section-wide: a speed figure under another heading "
                f"reaches the same reader"
            )
    return problems


def descriptor_page(descriptor: dict) -> str:
    """The registry entry's page, assembled from `PAGE_FIELDS` in the order it renders.

    Assertion 8 is scoped to this and not to `docs.extended_description`, which
    is the one field it used to read. `docs.hello_world` is the worked SQL
    example the registry shows above the body, and a speed figure in a comment
    inside it reaches a stranger before the prose does.

    A field this cannot find contributes nothing rather than raising: a
    descriptor missing `docs.extended_description` is caught by `run` with a
    message about what that field is, which is more use than a KeyError here.
    """
    parts = []
    for table, field_name in PAGE_FIELDS:
        values = descriptor.get(table)
        value = values.get(field_name) if isinstance(values, dict) else None
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def disagreements(readme_text: str | None, descriptor_text: str | None) -> list[str]:
    """Quantities the two sections do not both write, the same number of times, in order."""
    if readme_text is None or descriptor_text is None:
        return []
    in_readme = [(round(value, 6), written) for value, _, written in quantities(readme_text)]
    in_descriptor = [
        (round(value, 6), written) for value, _, written in quantities(descriptor_text)
    ]

    problems = []
    readme_counts = Counter(value for value, _ in in_readme)
    descriptor_counts = Counter(value for value, _ in in_descriptor)
    for value in sorted((readme_counts - descriptor_counts).elements()):
        problems.append(
            f"{README} claims {value} in its quality section and {DESCRIPTOR} does not, or does "
            f"not claim it as often. {DESCRIPTOR} is the registry page; the two say the same "
            f"thing about quality or neither can be trusted"
        )
    for value in sorted((descriptor_counts - readme_counts).elements()):
        problems.append(
            f"{DESCRIPTOR} claims {value} in its quality section and {README} does not, or does "
            f"not claim it as often"
        )
    if problems:
        return problems

    for index, (left, right) in enumerate(zip(in_readme, in_descriptor)):
        if left[0] != right[0]:
            problems.append(
                f"the two sections carry the same quantities in a different order: at position "
                f"{index + 1}, {README} writes `{left[1]}` where {DESCRIPTOR} writes "
                f"`{right[1]}`. A figure moved to another row or another corpus leaves the set "
                f"of quantities unchanged, which is how a swapped summary-table cell used to "
                f"pass this check in both directions"
            )
            break
    return problems


def revision_problems(source_text: str) -> list[str]:
    """The bundled model has not moved off the revision these figures were published against."""
    match = re.search(r"Revision:\s*`([0-9a-f]{40})`", source_text)
    if match is None:
        return [
            f"{MODEL_SOURCE} states no `Revision:` line. It is where the pinned model revision "
            f"is recorded, and without it nothing ties these figures to a model"
        ]
    if match.group(1) != MEASURED_ON_REVISION:
        return [
            f"the bundled model is at revision {match.group(1)} and the published figures were "
            f"transcribed against {MEASURED_ON_REVISION}. Vectors from two model versions are "
            f"not comparable: re-run the harness and update FIGURES, or re-bless them here "
            f"deliberately, rather than letting the page describe a model that is no longer in "
            f"the binary"
        ]
    return []


def run(root: pathlib.Path) -> int:
    """The whole check over one tree: read both pages, judge them, report, return an exit code.

    `main` calls this on the repository this file lives in, and `self_test`
    calls it on a staged tree with one defect planted at a time. That split is
    the point rather than a tidiness: every assertion above is wired together
    here and nowhere else, and a wiring line deleted here disables its assertion
    while the assertion itself — and every self-test case that calls it directly
    — stays green. Nine lines below were deletable one at a time with both CI
    commands at exit 0, `if problems:` among them, which reduced the gate to
    something that printed `ok:` and checked nothing.
    """
    try:
        import yaml
    except ImportError:
        print(
            "PyYAML is not installed. `extended_description` is a block scalar inside a YAML "
            "document and a regex cannot tell one from the prose around it: pip install pyyaml",
            file=sys.stderr,
        )
        return 2

    readme_path = root / README
    descriptor_path = root / DESCRIPTOR
    source_path = root / MODEL_SOURCE
    for path in (readme_path, descriptor_path, source_path):
        if not path.is_file():
            print(f"{path} is not in the tree", file=sys.stderr)
            return 2

    descriptor = yaml.safe_load(descriptor_path.read_text())
    extended = (descriptor.get("docs") or {}).get("extended_description")
    if not extended:
        print(
            f"{DESCRIPTOR} has no docs.extended_description. It is the registry page's body and "
            f"the copy of the quality position a stranger reads",
            file=sys.stderr,
        )
        return 1

    readme_text = readme_path.read_text()
    readme_section = section(readme_text, SECTION_HEADING)
    descriptor_section = section(extended, SECTION_HEADING)

    problems: list[str] = []
    problems += region_problems(README, readme_section)
    problems += region_problems(f"{DESCRIPTOR} (extended_description)", descriptor_section)
    problems += disagreements(readme_section, descriptor_section)
    problems += unused_figures(readme_section, descriptor_section)
    problems += unused_allowances(readme_section, descriptor_section)
    problems += page_problems(README, readme_text)
    problems += page_problems(f"{DESCRIPTOR} (the rendered page)", descriptor_page(descriptor))
    problems += revision_problems(source_path.read_text())

    if problems:
        print("FAIL: the published quality position does not hold up:", file=sys.stderr)
        for problem in problems:
            print(f"       {problem}", file=sys.stderr)
        return 1

    counted = len({round(value, 6) for value, _, _ in quantities(readme_section or "")})
    print(
        f"ok: {README} and {DESCRIPTOR} carry the same {counted} measured quantities under "
        f"`## {SECTION_HEADING}`, in the same order, every one of them registered in FIGURES "
        f"with its source; all {len(CLAIMS)} pinned claims present in both; every summary-table "
        f"cell the figure FIGURES registers for its row and column; no partial series in any "
        f"sentence; no universal quantifier outside the {len(ALLOWED_UNIVERSALS)} recorded in "
        f"ALLOWED_UNIVERSALS; no banned hedge in either section; no speed vocabulary in "
        f"{README} or in the {len(PAGE_FIELDS)} {DESCRIPTOR} fields the registry renders, their "
        f"SQL examples included; every one of the {len(FIGURES)} registered figures written on "
        f"at least one page; and the bundled model still at the revision they were published "
        f"against"
    )
    return 0


def stage_tree(root: pathlib.Path, tree: dict[str, str]) -> None:
    """Write a whole two-page tree under `root`, for the self-test to run `run` against.

    `run` reads a directory rather than a set of strings, so proving that it
    wires its assertions together at all needs a directory to plant a defect in.
    The descriptor is dumped and read back through PyYAML, so a case cannot pass
    by writing YAML the real loader would read differently.
    """
    import yaml

    if tree["readme"] is not None:
        (root / README).write_text(tree["readme"])
    source = root / MODEL_SOURCE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"Revision: `{tree['revision']}`\n")
    (root / DESCRIPTOR).write_text(
        yaml.safe_dump(
            {
                "extension": {"name": "subtoken", "description": tree["blurb"]},
                "docs": {
                    "hello_world": tree["hello_world"],
                    "extended_description": tree["extended"],
                },
            }
        )
    )


def staged_run(tree: dict[str, str]) -> tuple[int, str]:
    """`run` over a staged tree, as (exit code, everything it printed).

    A raise is reported as exit -1 and the exception in the text, rather than
    being allowed to leave `self_test` as a traceback. A check that raises has
    judged nothing, and a case whose only way of failing is somebody else's
    stack trace is a case that reports somebody else's message: two cases below
    plant a tree that raises rather than reports when the guard they name is
    removed, and this is what lets each of them say so itself.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        stage_tree(root, tree)
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = run(root)
        except Exception as raised:  # noqa: BLE001
            return -1, f"{out.getvalue()}{err.getvalue()}run() raised {raised!r}"
        return code, out.getvalue() + err.getvalue()


def staged_process(tree: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """This file, run as a process with no arguments, over a staged tree.

    `staged_run` calls `run` directly, which leaves `main`'s dispatch to it —
    the two lines the CI step actually reaches — with nothing behind them. The
    file is copied into the staged tree so its own `REPO_ROOT` resolves there,
    and then invoked the way CI invokes it.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "scripts").mkdir()
        script = root / "scripts" / pathlib.Path(__file__).name
        shutil.copy(pathlib.Path(__file__).resolve(), script)
        stage_tree(root, tree)
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True)


def self_test() -> int:  # noqa: C901
    """Plant each defect this is meant to catch, and require it to be reported.

    Each case plants one defect and names a needle the assertion it is written
    for produces. That is not a detail of style. A case whose needle some
    neighbouring assertion also satisfies proves nothing about its own, and
    this file shipped one: `("a reinstated hedge", good + " only a minority
    survive.", "a minority")` planted a hedge in a sentence carrying `only`, so
    the universal scan's message quoted the planted text back and satisfied the
    needle. `BANNED_IN_SECTION`'s enforcement could be deleted with every case
    here green.

    For the assertions named in `scripts/mutation_check.py` that is measured
    rather than asserted: each of those mutations disables one assertion and
    requires this self-test to redden. The rest of the assertions below have a
    case and no mutation, so for them this paragraph is a discipline and not a
    guarantee.

    A case that calls one helper directly says nothing about whether `run` calls
    that helper. The block at the foot stages a whole tree and drives `run` over
    it, and then the script as a process, which is what closes that: it is where
    a deleted wiring line reddens.
    """
    # FIGURES itself: assertion 5 reads one figure per (series, corpus), so two
    # entries claiming the same cell would make the table check pick arbitrarily.
    cells = [(figure.series, figure.corpus) for figure in FIGURES if figure.series]
    if len(cells) != len(set(cells)):
        print(f"self-test FAILED: FIGURES registers a (series, corpus) twice: {cells}", file=sys.stderr)
        return 1
    for series in (*TABLE_ROWS.values(), *DIRECTIONAL_SERIES):
        for corpus in CORPORA:
            if figure_for(series, corpus) is None:
                print(
                    f"self-test FAILED: FIGURES registers no {series!r} figure for {corpus!r}",
                    file=sys.stderr,
                )
                return 1

    # Where the figures come from. The three this page used to carry that were
    # not ours read as this extension's strength and were another serving
    # system's, on another benchmark, against another baseline. What stops the
    # next one is not vigilance: it is that a Figure's source has to be the one
    # committed file, at the one commit, and anything else is a self-test
    # failure before the pages are read at all.
    if MEASURED_AT_COMMIT not in FINETYPE_EVAL:
        print(
            f"self-test FAILED: FINETYPE_EVAL is {FINETYPE_EVAL!r} and does not name the "
            f"commit {MEASURED_AT_COMMIT!r} the figures were generated at. A source that names "
            "a file but not a revision cannot be re-read: the file moves and the figure stays",
            file=sys.stderr,
        )
        return 1
    foreign = sorted({figure.source for figure in FIGURES if figure.source != FINETYPE_EVAL})
    if foreign:
        print(
            f"self-test FAILED: FIGURES carries {len(foreign)} source(s) that are not "
            f"{FINETYPE_EVAL!r}: {foreign}. A figure this estate did not measure is a figure "
            "nobody here can re-run, and the page presents it in the same voice as one we did",
            file=sys.stderr,
        )
        return 1

    # The quantity parser: what it must see, and what it must not.
    seen = {written for _, _, written in quantities("13% and 0.3924 and 3,000 rows and 216 names")}
    if seen != {"13%", "0.3924", "3,000", "216"}:
        print(f"self-test FAILED: the quantity parser found {seen}", file=sys.stderr)
        return 1
    for identifier in (
        "potion-base-8M",
        "all-MiniLM-L6-v2",
        "`SELECT 41 + 1;`",
        "```sql\nSELECT 41 + 1;\n```",
        "measured on 2026-08-24",
    ):
        found = quantities(identifier)
        if found:
            print(
                f"self-test FAILED: {identifier} was read as a quantity: {found}", file=sys.stderr
            )
            return 1
    # A link whose target carries a bare number. Without LINK_TARGET stripping,
    # `1234.56789` is read as a quantity, reported unregistered, and the pages
    # cannot cite a source that has a number in its address.
    if quantities("[the committed results](https://example.invalid/abs/1234.56789)"):
        print("self-test FAILED: a link target was read as a quantity", file=sys.stderr)
        return 1

    # Rounding: a page may round a measurement, and may not restate it.
    if measured(*parse_quantity("13", "%")) is None:
        print("self-test FAILED: 13% did not match a measured 0.13185", file=sys.stderr)
        return 1
    if measured(*parse_quantity("14", "%")) is not None:
        print("self-test FAILED: 14% matched a measured figure", file=sys.stderr)
        return 1
    if measured(*parse_quantity("45", "%")) is not None:
        print(
            "self-test FAILED: 45% matched a measured figure — that is the value 'a minority' "
            "was compatible with, and the reason this check exists",
            file=sys.stderr,
        )
        return 1
    if measured(*parse_quantity("0.132", "")) is None:
        print("self-test FAILED: 0.132 did not match a measured 0.13185", file=sys.stderr)
        return 1
    for tie in ("0.1318", "0.1319"):
        if measured(*parse_quantity(tie, "")) is None:
            print(
                f"self-test FAILED: {tie} did not match a measured 0.13185, which is exactly "
                "between the two at four places",
                file=sys.stderr,
            )
            return 1
    for outside in ("0.1317", "0.1320", "0.140"):
        if measured(*parse_quantity(outside, "")) is not None:
            print(f"self-test FAILED: {outside} matched a measured figure", file=sys.stderr)
            return 1

    # Section extraction, including the two ways it must refuse to report clean.
    document = (
        "# title\n\nintro\n\n"
        f"## {SECTION_HEADING}\n\nthe claim, at 13%.\n\n### a subsection\n\nstill in it, at 28%.\n\n"
        "## something else\n\nnot in it, at 99%.\n"
    )
    body = section(document, SECTION_HEADING)
    if body is None or "13%" not in body or "28%" not in body:
        print(f"self-test FAILED: the section body was {body!r}", file=sys.stderr)
        return 1
    if "99%" in body:
        print("self-test FAILED: the section ran past the next level-2 heading", file=sys.stderr)
        return 1
    if section(document.replace(f"## {SECTION_HEADING}", "## renamed"), SECTION_HEADING) is not None:
        print("self-test FAILED: a renamed heading was not reported", file=sys.stderr)
        return 1
    if section(document + f"\n## {SECTION_HEADING}\n\nagain\n", SECTION_HEADING) is not None:
        print("self-test FAILED: a duplicated heading was not reported", file=sys.stderr)
        return 1
    # Both of these name the message they are written for. `== []` was enough
    # for the missing-section case and was not enough for the empty-section
    # one: an empty section fails every pin in CLAIMS as well, so the guard
    # could be deleted and this case still passed.
    if not any("it is not checking" in problem for problem in region_problems("a_file", None)):
        print(
            f"self-test FAILED: a missing section was not reported as missing: "
            f"{region_problems('a_file', None)}",
            file=sys.stderr,
        )
        return 1
    if not any("agrees with anything" in problem for problem in region_problems("a_file", "\n")):
        print(
            f"self-test FAILED: an empty section was not reported as empty: "
            f"{region_problems('a_file', chr(10))}",
            file=sys.stderr,
        )
        return 1

    # A summary table, then broken one cell and one header at a time.
    good_table = (
        "| | long-form prose | short text | very short strings |\n"
        "|---|---|---|---|\n"
        "| corpus | 20 Newsgroups posts | their subject lines | column names |\n"
        "| rows | 3,000 | 3,000 | 216 |\n"
        "| ranked lift, this model | 0.763 | 0.853 | 0.883 |\n"
        "| ranked lift, BM25 with no model | 0.746 | 0.694 | 0.588 |\n"
        "| nearest neighbours that survive | 13% | 28% | 40% |\n"
        "| region structure kept | 71% | 67% | 88% |\n"
    )
    # What `table` counts as a row. Neither line below is one, and each is a
    # line the real pages carry: prose with a pipe in it, and the separator
    # under the header.
    with_a_pipe = "a sentence with | a pipe in it\n| rows | 3,000 | 3,000 | 216 |\n"
    if table(with_a_pipe) != [["rows", "3,000", "3,000", "216"]]:
        print(
            f"self-test FAILED: a prose line carrying a pipe was read as a table row: "
            f"{table(with_a_pipe)}",
            file=sys.stderr,
        )
        return 1
    if ["---", "---", "---", "---"] in table(good_table):
        print(
            f"self-test FAILED: the separator under the header was read as a row of cells: "
            f"{table(good_table)}",
            file=sys.stderr,
        )
        return 1
    if table_problems("a_file", good_table) != []:
        print(
            f"self-test FAILED: a correct summary table was reported: "
            f"{table_problems('a_file', good_table)}",
            file=sys.stderr,
        )
        return 1
    for label, broken, needle in (
        (
            "two cells swapped between corpora",
            good_table.replace(
                "| nearest neighbours that survive | 13% | 28% | 40% |",
                "| nearest neighbours that survive | 28% | 13% | 40% |",
            ),
            "swapped between corpora",
        ),
        (
            "two cells transposed in the other row",
            good_table.replace(
                "| region structure kept | 71% | 67% | 88% |",
                "| region structure kept | 67% | 71% | 88% |",
            ),
            "swapped between corpora",
        ),
        (
            "a reversed header row",
            good_table.replace(
                "| | long-form prose | short text | very short strings |",
                "| | very short strings | short text | long-form prose |",
            ),
            "in that order",
        ),
        (
            "a deleted row",
            good_table.replace("| region structure kept | 71% | 67% | 88% |\n", ""),
            "needs exactly one",
        ),
        (
            "an emptied cell",
            good_table.replace(
                "| rows | 3,000 | 3,000 | 216 |", "| rows | 3,000 | | 216 |"
            ),
            "needs exactly one",
        ),
        # A column deleted from one row, which the emptied-cell case above
        # cannot stand in for: that one keeps three cells and this one has two.
        # `zip(CORPORA, cells)` truncates to the shorter of the two, so without
        # the width check the surviving cells are read against the corpora they
        # no longer sit under and the row is reported clean.
        (
            "a row with one of the three corpus columns deleted",
            good_table.replace(
                "| region structure kept | 71% | 67% | 88% |",
                "| region structure kept | 71% | 67% |",
            ),
            "2 cells under 3 corpora",
        ),
        ("no table at all", "prose with no table in it", "no single summary table"),
    ):
        found = table_problems("a_file", broken)
        if not any(needle in problem for problem in found):
            print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
            return 1

    # A whole series in one sentence, and the endpoints-only sentence that is
    # the defect: region structure is 71%, 67%, 88% and is not monotone.
    whole = (
        "It is worst on long prose and mildest on very short strings — 13%, then 28%, then 40% "
        "as the text gets shorter. Region structure is 71% on long-form prose, 67% on short "
        "text, 88% on very short strings."
    )
    if series_problems("a_file", whole) != []:
        print(
            f"self-test FAILED: two whole series were reported: "
            f"{series_problems('a_file', whole)}",
            file=sys.stderr,
        )
        return 1
    endpoints = "It is worst on long prose and mildest on short strings — 13% against 40% on neighbours, 71% against 88% on regions."
    found = series_problems("a_file", endpoints)
    if not any("kNN overlap" in problem for problem in found):
        print(f"self-test FAILED: an endpoints-only kNN sentence gave {found}", file=sys.stderr)
        return 1
    if not any("region retention" in problem for problem in found):
        print(f"self-test FAILED: an endpoints-only region sentence gave {found}", file=sys.stderr)
        return 1
    if series_problems("a_file", "216 rows is a small sample, and 3,000 is not.") != []:
        print(
            "self-test FAILED: corpus sizes were treated as a directional series; a sentence "
            "may cite one sample size without citing all three",
            file=sys.stderr,
        )
        return 1
    # A row writing two of the three region figures. The whole table cannot
    # fail this case — it writes every figure of both series, so it is not a
    # partial series however it is read — and a case that cannot fail is not a
    # case. The row is what `prose` has to remove.
    if series_problems("a_file", "| region structure kept | 71% | 67% |\n") != []:
        print(
            "self-test FAILED: a table row writing two of the three region figures was read as "
            "a sentence. The table's cells are checked against FIGURES by assertion 5, which is "
            "stronger, and a row is not a claim about a direction",
            file=sys.stderr,
        )
        return 1

    # The universal-quantifier scan, and its exception mechanism.
    for label, text, word in (
        ("every", "Every figure below compares the bundled model against MiniLM.", "every"),
        ("all", "All of these figures are ours.", "all"),
        ("only", "Only one of them is weak here.", "only"),
        ("never", "This never loses cluster structure.", "never"),
        ("everything", "Everything below that reads as a weakness is ranked retrieval.", "everything"),
        ("always", "It always returns a unit vector.", "always"),
        ("each", "Each figure here was measured on the bundled model.", "each"),
        ("none", "None of these came from a third party.", "none"),
    ):
        found = universal_problems("a_file", text)
        if not any(repr(word) in problem for problem in found):
            print(f"self-test FAILED: the universal {label!r} was not reported: {found}", file=sys.stderr)
            return 1
    if universal_problems("a_file", "the figures below compare this model against MiniLM") != []:
        print(
            "self-test FAILED: a sentence with no quantifier word was reported — the scan is "
            "documented as blind to a bare plural generic and must not pretend otherwise",
            file=sys.stderr,
        )
        return 1
    if universal_problems("a_file", "a vector from `all-MiniLM-L6-v2`") != []:
        print("self-test FAILED: a model name in a code span was read as a universal", file=sys.stderr)
        return 1
    # Both halves of what the header says this scan reaches. Neither is readable
    # off `UNIVERSAL_WORDS` by eye: `in each case` and `bar none` name no listed
    # word and carry `each` and `none` as whole words, and a header that said
    # they were invisible invited a maintainer to widen a list already covering
    # them.
    for unseen in ("invariably", "without exception", "universally"):
        if universal_problems("a_file", f"the projection {unseen} loses neighbourhoods") != []:
            print(
                f"self-test FAILED: {unseen!r} was reported as a universal, and the header says "
                f"this scan cannot see it",
                file=sys.stderr,
            )
            return 1
    for seen, word in (("in each case", "each"), ("bar none", "none")):
        found = universal_problems("a_file", f"the projection loses neighbourhoods {seen}")
        if not any(repr(word) in problem for problem in found):
            print(
                f"self-test FAILED: {seen!r} was not reported, and the header says this scan "
                f"sees it through the whole word {word!r}: {found}",
                file=sys.stderr,
            )
            return 1
    for allowance in ALLOWED_UNIVERSALS:
        if universal_problems("a_file", allowance.phrase) != []:
            print(
                f"self-test FAILED: the permitted universal {allowance.phrase!r} was reported: "
                f"{universal_problems('a_file', allowance.phrase)}",
                file=sys.stderr,
            )
            return 1
        if universal_problems("a_file", allowance.phrase.replace("shuffle", "reversal")) == []:
            print(
                f"self-test FAILED: {allowance.phrase!r} was permitted after its wording changed, "
                f"so the allowance is matching more than the sentence it was written for",
                file=sys.stderr,
            )
            return 1
    if unused_allowances(" ".join(a.phrase for a in ALLOWED_UNIVERSALS)) != []:
        print("self-test FAILED: a written allowance was reported unused", file=sys.stderr)
        return 1
    if unused_allowances("nothing here quantifies over anything") == []:
        print(
            "self-test FAILED: an allowance no page writes reported clean; a dead exception "
            "widens the scan without anyone deciding to",
            file=sys.stderr,
        )
        return 1

    # A section built from the claims themselves, then broken one way at a time.
    good = (
        " ".join(CLAIMS)
        + " 3,000 rows, 20 Newsgroups, 20 nearest neighbours, 0.3924 against 0.3510.\n"
        + good_table
        + "\nany phrase and its shuffle land in the same place.\n"
    )
    for label, text, needle in (
        ("a complete section reports nothing", good, None),
        (
            "an unmeasured quantity is caught",
            good + " and 45% of neighbours survive.",
            "not one of the measured figures",
        ),
        (
            "a dropped claim is caught",
            good.replace("pairwise judgement", "some other wording"),
            "does not carry the claim",
        ),
        (
            "two figures swapped between corpora in the prose are caught",
            good.replace(
                "13% are the same rows on long-form prose, 28% on short text",
                "28% are the same rows on long-form prose, 13% on short text",
            ),
            "does not carry the claim",
        ),
        (
            "two figures swapped between corpora in the table are caught",
            good.replace(
                "| nearest neighbours that survive | 13% | 28% | 40% |",
                "| nearest neighbours that survive | 28% | 13% | 40% |",
            ),
            "swapped between corpora",
        ),
        # Two hedges, each planted so that nothing but the BANNED_IN_SECTION
        # scan has anything to say about it: no universal, no quantity, no pin
        # broken. The first of these read "only a minority survive", and the
        # `only` meant the universal scan's message carried the needle `a
        # minority` in its quoted context — so the hedge ban could be deleted
        # and this whole suite stayed green. A needle a neighbouring assertion
        # can satisfy proves nothing about the assertion it is written for.
        ("a reinstated hedge is caught by the hedge ban", good + " a minority survive.", "a minority"),
        (
            "the hedge the region figures replaced is caught by the hedge ban",
            good + " it recovers most of the cluster structure.",
            "recovers most",
        ),
        (
            "the region sentence's two corpora swapped are caught",
            good.replace(
                "71% on long-form prose, 67% on short text",
                "71% on short text, 67% on long-form prose",
            ),
            "71% on long-form prose, 67% on short text",
        ),
        (
            "a blanket universal is caught",
            good + " Every figure below compares the bundled model against MiniLM.",
            "'every'",
        ),
        # Assertion 6 reached through `region_problems` rather than called
        # directly. The endpoints cases above drive `series_problems` and say
        # nothing about whether anything calls it: `problems +=
        # series_problems(...)` could be deleted with every case here green,
        # and a sentence quoting two of the three region figures would then
        # ship on both pages. Appended rather than substituted so that no pin
        # in CLAIMS breaks, and the two figures are registered, so this is the
        # only assertion with anything to say about it.
        (
            "a partial series reached through region_problems",
            good + " Region structure is 71% on long-form prose and 88% on very short strings.",
            "2 of the 3 region retention figures",
        ),
    ):
        found = region_problems("a_file", text)
        if needle is None:
            if found:
                print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
                return 1
        elif not any(needle in problem for problem in found):
            print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
            return 1

    # Assertion 8's page-wide half. Its defect was a speed figure planted three
    # sections below the quality claim, where a section-scoped ban never looked.
    if page_problems("a_file", "a page with no speed claim on it") != []:
        print("self-test FAILED: a clean page was reported", file=sys.stderr)
        return 1
    for label, text in (
        ("a speed multiple", "## The SQL surface\n\nit is 397 times faster than a transformer."),
        ("a row rate", "## The SQL surface\n\nit embeds 50,000 rows per second."),
        # Two cases rather than one. `a 397× speedup.` satisfies BANNED_ON_PAGE's
        # `speedup` and its `×` at once, so either could be deleted and the other
        # kept this case green. Each of these is reported by one phrase and no other.
        ("a speedup noun with no multiplier sign", "## The SQL surface\n\na 397x speedup."),
        ("a multiplier sign with no speed noun", "## The SQL surface\n\na 397× improvement."),
        ("a latency claim", "## Why this exists\n\nlatency is bounded."),
        # And the same figures written where the SQL examples live. That is the
        # surface this ban was widened to cover — `description.yml` publishes
        # every example in a fenced block and `README.md` carries six more —
        # and a ban read after `strip_noise` had already looked at none of it.
        (
            "a speed claim in a comment inside a fenced example",
            "## The SQL surface\n\n```sql\n-- 50,000 rows per second\nSELECT subtoken_embed(name) FROM t;\n```",
        ),
        (
            "a speed claim inside an inline code span",
            "## The SQL surface\n\nthe comment reads `-- 397\u00d7 faster` in that example.",
        ),
    ):
        if page_problems("a_file", text) == []:
            print(f"self-test FAILED: {label} outside the quality section reported clean", file=sys.stderr)
            return 1
    # Reading code is not reading addresses. A URL that spells a banned word is
    # an address, and `strip_addresses` is the only thing keeping it out now
    # that this scan no longer runs after code has been removed.
    for label, text in (
        # A relative target, not an http one: `BARE_URL` would strip an http
        # target on its own, so a case built from one leaves LINK_TARGET with
        # nothing only it can do. README.md links `models/potion-base-8M/
        # SOURCE.md` this way.
        ("a relative link target", "see [the note](notes/10x-faster-embeddings.md)"),
        ("a bare URL", "see https://example.invalid/faster-than-sbert for the write-up"),
        ("a fenced example with no claim in it", "```sql\nSELECT subtoken_embed('a') FROM t;\n```"),
    ):
        if page_problems("a_file", text) != []:
            print(
                f"self-test FAILED: {label} was read as a speed claim: "
                f"{page_problems('a_file', text)}",
                file=sys.stderr,
            )
            return 1

    # The two files against each other. A figure in one and not the other is
    # the defect this check is named for, and it must be caught both ways round.
    if disagreements("13% and 28%", "13% and 28%") != []:
        print("self-test FAILED: two agreeing sections were reported", file=sys.stderr)
        return 1
    # A section that could not be found is reported by `region_problems` as
    # missing, and comparing it against the other one would report the whole of
    # the other one as one-sided on top of that. Both directions, because the
    # guard is one condition covering two.
    for absent in ((None, "13%"), ("13%", None)):
        try:
            compared = disagreements(*absent)
        except Exception as raised:  # noqa: BLE001
            compared = [f"it raised {raised!r}"]
        if compared != []:
            print(
                f"self-test FAILED: a missing section was compared against a present one: "
                f"{compared}",
                file=sys.stderr,
            )
            return 1
    both_ways = disagreements("13% and 28%", "13%") + disagreements("13%", "13% and 28%")
    if len(both_ways) != 2:
        print(f"self-test FAILED: a one-sided figure gave {both_ways}", file=sys.stderr)
        return 1
    if disagreements("the overlap is 13%", "the overlap is 14%") == []:
        print("self-test FAILED: two sections disagreeing on a figure were not", file=sys.stderr)
        return 1
    # A count mismatch is reported as a count mismatch and not also as an
    # ordering one: `13% 28%` against `28%` differs in both, and reporting one
    # defect in two vocabularies sends a reader looking for a second edit.
    if len(disagreements("13% and 28%", "28%")) != 1:
        print(
            f"self-test FAILED: a one-sided figure was reported as a count mismatch and as an "
            f"ordering mismatch: {disagreements('13% and 28%', '28%')}",
            file=sys.stderr,
        )
        return 1
    if disagreements("13% then 28%", "28% then 13%") == []:
        print(
            "self-test FAILED: the same figures in a different order reported clean — a cell "
            "moved to another row keeps the set unchanged and is the defect this must see",
            file=sys.stderr,
        )
        return 1
    if disagreements("3,000 and 3,000", "3,000") == []:
        print(
            "self-test FAILED: a figure written twice in one section and once in the other "
            "reported clean; the summary table writes 3,000 in two columns",
            file=sys.stderr,
        )
        return 1

    # Assertion 10: a figure nothing writes any more.
    all_figures_written = " ".join(
        f"{figure.value}" for figure in FIGURES if figure.value not in (3000, 216, 12, 20)
    )
    all_figures_written += " 3,000 216 12 20"
    if unused_figures(all_figures_written) != []:
        print(
            f"self-test FAILED: a section writing every figure reported one unused: "
            f"{unused_figures(all_figures_written)}",
            file=sys.stderr,
        )
        return 1
    dropped = all_figures_written.replace("0.13185", "")
    if not any("0.13185" in problem for problem in unused_figures(dropped)):
        print(
            f"self-test FAILED: a figure no page writes was not reported: "
            f"{unused_figures(dropped)}",
            file=sys.stderr,
        )
        return 1

    # The model revision the figures belong to.
    if revision_problems(f"Revision: `{MEASURED_ON_REVISION}`") != []:
        print("self-test FAILED: the pinned revision was reported as a mismatch", file=sys.stderr)
        return 1
    if revision_problems("Revision: `" + "0" * 40 + "`") == []:
        print("self-test FAILED: a bumped model revision reported clean", file=sys.stderr)
        return 1
    if revision_problems("no revision line here") == []:
        print("self-test FAILED: a missing Revision: line reported clean", file=sys.stderr)
        return 1

    # ── the whole check, over a staged tree ──────────────────────────────────
    # Every case above calls one helper directly, which leaves `run` — the only
    # place the helpers are wired together — with nothing behind it. Nine of its
    # lines could each be deleted on their own with this self-test and the live
    # check both at exit 0, `if problems:` among them, which turned the gate
    # into something that printed `ok:` and checked nothing. One assertion per
    # line, written by hand, would repeat the shape that let that happen: these
    # drive `run` end to end over a temporary tree instead, one planted defect
    # at a time, and each names the message its wiring line produces.
    try:
        import yaml  # noqa: F401
    except ImportError:
        print(
            "self-test FAILED: PyYAML is not installed, and the staged-tree cases below write a "
            "description.yml and read it back through the loader the live check uses. A case "
            "that is skipped is a case that cannot fail: pip install pyyaml",
            file=sys.stderr,
        )
        return 1

    allowance_line = "any phrase and its shuffle land in the same place."
    readme_page = (
        "# subtoken\n\nA vector for a string, from a model bundled in the binary.\n\n"
        f"## {SECTION_HEADING}\n\n{good}\n"
        "## The SQL surface\n\n`subtoken_embed()` is a scalar function.\n"
    )
    extended_page = (
        "`subtoken` turns a string into a vector inside DuckDB.\n\n"
        f"## {SECTION_HEADING}\n\n{good}\n"
    )
    clean_tree = {
        "readme": readme_page,
        "blurb": "Static text embeddings as a DuckDB scalar, from a model in the binary",
        "hello_world": "-- 256 floats per row, for this model.\nSELECT subtoken_embed(name) FROM t;\n",
        "extended": extended_page,
        "revision": MEASURED_ON_REVISION,
    }
    code, report = staged_run(clean_tree)
    if code != 0:
        print(
            f"self-test FAILED: a staged tree with nothing wrong with it exited {code}. Every "
            f"case below plants one defect into this tree, so they prove nothing while it is "
            f"reported broken for some other reason: {report}",
            file=sys.stderr,
        )
        return 1

    # The two ways the tree is not readable at all, before the defects in it.
    # Both are reported rather than raised, and neither is exit 1: a tree this
    # cannot read is not a tree it has judged.
    for label, override, expected, needle in (
        ("one of the three files missing", {"readme": None}, 2, "is not in the tree"),
        (
            "a descriptor with an empty body",
            {"extended": ""},
            1,
            "has no docs.extended_description",
        ),
    ):
        code, report = staged_run({**clean_tree, **override})
        if code != expected:
            print(
                f"self-test FAILED: {label} — exited {code} rather than {expected}: {report}",
                file=sys.stderr,
            )
            return 1
        if needle not in report:
            print(
                f"self-test FAILED: {label} — nothing it reported carried {needle!r}: {report}",
                file=sys.stderr,
            )
            return 1

    hedged = good + " a minority survive.\n"
    a_whole_series = (
        good
        + " Overlap runs 13% on long-form prose, 28% on short text, 40% on very short strings.\n"
    )
    for label, override, needle in (
        (
            "a hedge on the README's side and not the registry entry's",
            {"readme": readme_page.replace(good, hedged)},
            f"{README}'s section contains 'a minority'",
        ),
        (
            "a hedge on the registry entry's side and not the README's",
            {"extended": extended_page.replace(good, hedged)},
            f"{DESCRIPTOR} (extended_description)'s section contains 'a minority'",
        ),
        (
            "a figure written on one page and not the other",
            {"readme": readme_page.replace(good, a_whole_series)},
            "does not, or does not claim it as often",
        ),
        (
            "a registered figure neither page writes any more",
            {
                "readme": readme_page.replace("0.3924 against 0.3510", "0.3924"),
                "extended": extended_page.replace("0.3924 against 0.3510", "0.3924"),
            },
            "A permitted value nothing uses",
        ),
        (
            "a permitted universal neither page writes any more",
            {
                "readme": readme_page.replace(allowance_line, ""),
                "extended": extended_page.replace(allowance_line, ""),
            },
            "A live exception to the universal-quantifier scan",
        ),
        (
            "a speed figure in the README, three headings below the quality section",
            {
                "readme": readme_page.replace(
                    "`subtoken_embed()` is a scalar function.",
                    "`subtoken_embed()` is a scalar function. It embeds 50,000 rows per second.",
                )
            },
            f"{README} contains 'per second'",
        ),
        # The two fields assertion 8 could not see. `docs.hello_world` is the
        # worked example the registry renders above the body, and it is raw SQL
        # rather than a fenced block: a rows-per-second comment in it puts the
        # embedding-only speed figure into the entry's own worked example, which
        # is the number this page leaves out on purpose, under a check printing
        # that there was no speed vocabulary anywhere on it.
        (
            "a speed figure in the registry entry's published SQL example",
            {
                "hello_world": "-- 256 floats per row, and 50,000 rows per second.\n"
                "SELECT subtoken_embed(name) FROM t;\n"
            },
            f"{DESCRIPTOR} (the rendered page) contains 'per second'",
        ),
        (
            "a speed figure in the registry entry's one-line blurb",
            {"blurb": "Static text embeddings as a DuckDB scalar, at 397x lower latency"},
            f"{DESCRIPTOR} (the rendered page) contains 'latency'",
        ),
        (
            "the bundled model moved off the revision the figures were published against",
            {"revision": "0" * 40},
            "re-run the harness and update FIGURES",
        ),
    ):
        code, report = staged_run({**clean_tree, **override})
        if code != 1:
            print(
                f"self-test FAILED: {label} — the staged tree exited {code} and a reported "
                f"problem is exit 1. What it printed: {report}",
                file=sys.stderr,
            )
            return 1
        if needle not in report:
            print(
                f"self-test FAILED: {label} — nothing it reported carried {needle!r}, which is "
                f"the message the assertion this case is written for produces. What it printed: "
                f"{report}",
                file=sys.stderr,
            )
            return 1

    # And the same check the way CI reaches it: as a process, through `main`,
    # with no arguments. Everything above calls `run` directly, so `main`'s
    # dispatch to it is the last wiring here with no case behind it.
    clean_process = staged_process(clean_tree)
    if clean_process.returncode != 0:
        print(
            f"self-test FAILED: this script, run as a process over a staged tree with nothing "
            f"wrong with it, exited {clean_process.returncode}: "
            f"{clean_process.stdout}{clean_process.stderr}",
            file=sys.stderr,
        )
        return 1
    broken_process = staged_process({**clean_tree, "revision": "0" * 40})
    if broken_process.returncode != 1:
        print(
            f"self-test FAILED: this script, run as a process over a staged tree with a bumped "
            f"model revision, exited {broken_process.returncode} rather than 1. `main` reaches "
            f"the check through one line and that line is what this case holds",
            file=sys.stderr,
        )
        return 1

    print(
        f"self-test ok: {len(FIGURES)} measured figures, {len(CLAIMS)} pinned claims, "
        f"{len(TABLE_ROWS)} summary-table rows read cell by cell, {len(DIRECTIONAL_SERIES)} "
        f"series that must be quoted whole, {len(UNIVERSAL_WORDS)} scanned quantifiers with "
        f"{len(ALLOWED_UNIVERSALS)} recorded exception(s), {len(BANNED_IN_SECTION)} banned "
        f"hedges and {len(BANNED_ON_PAGE)} page-wide speed phrases; 13% rounds onto 0.13185 and "
        f"14% and 45% do not; a renamed heading, an empty section reported as empty rather "
        f"than as its missing claims, an unmeasured quantity, a dropped claim, two figures "
        f"swapped between corpora in prose and in the table, the region sentence's two corpora "
        f"swapped, a reversed table header, a deleted table row, an emptied cell, a series "
        f"quoted at its endpoints only, an unpermitted universal, a permitted universal whose "
        f"wording moved, an exception no page writes, a hedge reinstated in a sentence with "
        f"nothing else wrong with it, a speed figure outside the quality section and a speed "
        f"figure inside a fenced example, a one-sided figure in either direction, the same "
        f"figures in a different order, a figure written a different number of times, a "
        f"registered figure no page writes and a bumped model revision are each reported; "
        f"`in each case` and `bar none` are seen through the whole words `each` and `none` "
        f"while `invariably` and `without exception` are not; and over a tree staged in a "
        f"temporary directory the whole of `run` reports a hedge on either page alone, a figure "
        f"on one page alone, a figure and a permitted universal neither page writes, a speed "
        f"figure below the quality section, in the registry entry's published SQL example and "
        f"in its one-line blurb, a file missing from the tree and a descriptor with an empty "
        f"body — with this file run as a process for `main`'s own dispatch to it; and a table "
        f"row with a corpus column deleted, a partial series reached through `region_problems` "
        f"rather than called directly, a prose line carrying a pipe and a separator row not "
        f"read as table rows, a missing section not compared against a present one, and a "
        f"count mismatch reported once rather than as an ordering mismatch as well"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    return run(REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
