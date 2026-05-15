"""
preprocessing.py — PDF-to-CSV pipeline for Neophytos the Recluse corpus.

Converts Stefanis 1996 edition (pp. 74–171) of the "Ten Discourses on
Christ's Commandments" into a structured CSV for computational analysis.

Usage:
    python preprocessing.py --pdf path/to/stefanis_1996_TEXT_ENTOLES.pdf
    python preprocessing.py --pdf ... --output corpus.csv --report quality_report.txt
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd
import pdfplumber

# ── constants ────────────────────────────────────────────────────────────────

VERSION = "1.0.0"
SOURCE_LABEL = "Stefanis 1996"
TOKEN_COUNT_WARN = 512

# Pages 74–171 in the Stefanis edition correspond to PDF indices 0–97
PDF_PAGE_OFFSET = 74          # printed page = PDF index + PDF_PAGE_OFFSET
PDF_FIRST_IDX   = 0
PDF_LAST_IDX    = 97

# Maps integer logos number → Greek ordinal label used in the edition
LOGOS_LABEL: dict[int, str] = {
    1: "Α΄",
    2: "Β΄",
    3: "Γ΄",
    4: "Δ΄",
    5: "Ε΄",
    6: "ΣΤ΄",
    7: "Ζ΄",
    8: "Η΄",
    9: "Θ΄",
    10: "Ι΄",
}

# Maps logos number → human-readable subtitle extracted from edition titles
LOGOS_SUBTITLE: dict[int, str] = {
    1: "Περὶ μετανοίας καὶ βασιλείας αἰωνίου",
    2: "Εἰς τὰ τῆς μετανοίας ἐπίλοιπα",
    3: "Περὶ τοῦ θείου βαπτίσματος καὶ τῆς ἁγίας ἀγάπης καὶ πλείστων ἑτέρων θείων ἐντολῶν",
    4: "Εἰς τὰς ἁγίας ἐντολὰς τοῦ Σωτῆρος",
    5: "Εἰς τὰς ἁγίας ἐντολὰς τοῦ Σωτῆρος καὶ περὶ τῶν πέντε τοῦ σώματος αἰσθήσεων",
    6: "Εἰς τὰς ἁγίας αὖθις ἐντολὰς τοῦ Κυρίου ἡμῶν Ἰησοῦ Χριστοῦ",
    7: "Εἰς τὰς ἁγίας ἐντολὰς τοῦ Κυρίου ἡμῶν Ἰησοῦ Χριστοῦ",
    8: "Εἰς τὰς σεπτὰς καὶ αὖθις θείας ἐντολάς καὶ εἰς τὴν δευτέραν παρουσίαν",
    9: "Εἰς τὰ ἐπίλοιπα τοῦ Μὴ κρίνετε καὶ εἰς τὰς λοιπὰς θείας ἐντολάς",
    10: "Εἰς τὰς ἁγίας καὶ αὖθις ἐντολάς καὶ εἰς τὴν ἔνδειαν τοῦ ἔτους",
}

# Running-header regex: logos, section, line, printed-page-number
_HEADER_RE = re.compile(
    r"^ΕΡΜΗΝΕΙΑ ΕΝΤΟΛΩΝ ΤΟΥ ΧΡΙΣΤΟΥ\s+(\d+),\s*(\d+),\s*(\d+)\s+(\d+)\s*$"
)
# Section-number marker at start of a paragraph, e.g. "12. Ἐπεὶ δέ..."
_SECTION_RE = re.compile(r"(?:^|\s)(\d+)\.\s+(?=\S)", re.UNICODE)
# Book of Scripture abbreviations used in inline citations
_BIBLE_ABBREV_RE = re.compile(
    r"(?:"
    r"Ματθ\.?\s*\d[\d,\s–-]*"
    r"|Μάρκ\.?\s*\d[\d,\s–-]*"
    r"|Λουκ\.?\s*\d[\d,\s–-]*"
    r"|Ἰωάν\.?\s*\d[\d,\s–-]*"
    r"|Πράξ\.?\s*\d[\d,\s–-]*"
    r"|Ῥωμ\.?\s*\d[\d,\s–-]*"
    r"|Κορ\.?\s*\d[\d,\s–-]*"
    r"|Γαλ\.?\s*\d[\d,\s–-]*"
    r"|Ἐφ\.?\s*\d[\d,\s–-]*"
    r"|Φιλ\.?\s*\d[\d,\s–-]*"
    r"|Κολ\.?\s*\d[\d,\s–-]*"
    r"|Θεσσ\.?\s*\d[\d,\s–-]*"
    r"|Τιμ\.?\s*\d[\d,\s–-]*"
    r"|Ἑβρ\.?\s*\d[\d,\s–-]*"
    r"|Ἰακ\.?\s*\d[\d,\s–-]*"
    r"|Πέτρ\.?\s*\d[\d,\s–-]*"
    r"|Ἀποκ\.?\s*\d[\d,\s–-]*"
    r"|Ψαλμ\.?\s*\d[\d,\s–-]*"
    r"|Γεν\.?\s*\d[\d,\s–-]*"
    r"|Ἔξοδ\.?\s*\d[\d,\s–-]*"
    r"|Ἰώβ\s*\d[\d,\s–-]*"
    r"|Παροιμ\.?\s*\d[\d,\s–-]*"
    r"|Σειρ\.?\s*\d[\d,\s–-]*"
    r"|Ἠσ\.?\s*\d[\d,\s–-]*"
    r"|Ἰερ\.?\s*\d[\d,\s–-]*"
    r"|Ἰεζ\.?\s*\d[\d,\s–-]*"
    r"|Δαν\.?\s*\d[\d,\s–-]*"
    r"|Μαλ\.?\s*\d[\d,\s–-]*"
    r")",
    re.UNICODE,
)

CSV_COLUMNS = [
    "id", "logos", "section", "paragraph", "text", "source",
    "page_start", "page_end", "sequence", "token_count", "text_hash",
    "biblical_citations", "incomplete", "partial",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class PageRecord:
    """Raw extraction result for one PDF page."""
    pdf_index: int
    printed_page: int
    logos_num: Optional[int]       # from running header or title detection
    body_text: str                 # header/separator stripped
    is_title_page: bool            # True when page carries a ΛΟΓΟΣ title heading


@dataclass
class SectionRecord:
    """One §-level unit ready for the CSV."""
    logos_num: int
    section_num: int
    text: str
    page_start: int
    page_end: int
    incomplete: bool = False
    partial: bool = False
    reason: str = ""


# ── Stage 1: text extraction ─────────────────────────────────────────────────

def stage1_extract_pages(pdf_path: Path) -> list[PageRecord]:
    """Extract raw text from every page in pp. 74–171.

    Args:
        pdf_path: Absolute path to the Stefanis 1996 PDF.

    Returns:
        List of PageRecord objects, one per PDF page (indices 0–97).
    """
    records: list[PageRecord] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        log.info("Stage 1 — PDF has %d pages; extracting indices %d–%d",
                 total, PDF_FIRST_IDX, PDF_LAST_IDX)
        for idx in range(PDF_FIRST_IDX, min(PDF_LAST_IDX + 1, total)):
            page = pdf.pages[idx]
            raw = page.extract_text() or ""
            printed = idx + PDF_PAGE_OFFSET
            records.append(PageRecord(
                pdf_index=idx,
                printed_page=printed,
                logos_num=None,
                body_text=raw,
                is_title_page=False,
            ))
    log.info("Stage 1 — extracted %d pages", len(records))
    return records


# ── Stage 2: source separation ───────────────────────────────────────────────

def stage2_separate_source(records: list[PageRecord]) -> list[PageRecord]:
    """Strip running headers, page-number lines, and separator rules.

    The Stefanis edition uses:
      • A running header:  ΕΡΜΗΝΕΙΑ ΕΝΤΟΛΩΝ ΤΟΥ ΧΡΙΣΤΟΥ {l},{s},{n} {page}
      • A full-width rule: ____________…____________  (≥20 underscores)
      • A bare page number on the first line of title pages.

    The critical apparatus (footnotes) does not appear in pdfplumber's
    extraction for this PDF — the lower page band is below pdfplumber's
    default crop.  No further separation is needed.

    Args:
        records: Output of stage1_extract_pages.

    Returns:
        Same list with body_text cleaned and logos_num populated.
    """
    for rec in records:
        # Page 74 is the critical apparatus introduction (Παράδοση, Εκδόσεις,
        # etc.) — not Neophytos's text.  Clear it so it never enters the corpus.
        if rec.printed_page == 74:
            rec.body_text = ""
            continue

        lines = rec.body_text.split("\n")
        clean: list[str] = []
        logos_from_header: Optional[int] = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Running header carries logos/section/page coordinates
            m = _HEADER_RE.match(stripped)
            if m:
                logos_from_header = int(m.group(1))
                continue

            # Full-width separator rule
            if re.fullmatch(r"[_\s]{15,}", stripped):
                continue

            # Bare page number on first non-empty line of a title page
            if i <= 1 and re.fullmatch(r"\d{2,3}", stripped):
                continue

            clean.append(line)

        rec.body_text = "\n".join(clean).strip()
        if logos_from_header is not None:
            rec.logos_num = logos_from_header

    log.info("Stage 2 — headers/separators removed from %d pages", len(records))
    return records


# ── Stage 3: encoding normalisation ──────────────────────────────────────────

def stage3_normalise_encoding(records: list[PageRecord]) -> list[PageRecord]:
    """Verify UTF-8 integrity and log any suspicious characters.

    Polytonic diacritics are deliberately preserved — no NFC/NFD
    normalisation is applied to the Greek text itself.

    Args:
        records: Output of stage2_separate_source.

    Returns:
        Same list; logs encoding warnings to stderr.
    """
    encoding_warnings: list[str] = []
    for rec in records:
        try:
            rec.body_text.encode("utf-8").decode("utf-8")
        except UnicodeError as exc:
            msg = f"p.{rec.printed_page}: UTF-8 error — {exc}"
            log.warning(msg)
            encoding_warnings.append(msg)

        # Flag control characters (except tab/newline)
        for ch in rec.body_text:
            cp = ord(ch)
            if cp < 0x20 and ch not in ("\t", "\n", "\r"):
                msg = (f"p.{rec.printed_page}: unexpected control char "
                       f"U+{cp:04X}")
                log.warning(msg)
                encoding_warnings.append(msg)

    log.info("Stage 3 — encoding check complete; %d warnings",
             len(encoding_warnings))
    return records


# ── Stage 4: structure detection ─────────────────────────────────────────────

def _detect_logos_title(text: str) -> Optional[int]:
    """Return the logos integer if text contains a ΛΟΓΟΣ heading, else None."""
    # Covers: ΛΟΓΟΣ Β΄  ΛΟΓΟΣ ΣΤ΄  ΛΟΓΟΣ\nς΄  (ς΄ is the PDF rendering of ΣΤ΄)
    m = re.search(
        r"ΛΟΓΟΣ\s*\n?\s*"
        r"(Α΄|Β΄|Γ΄|Δ΄|Ε΄|ς΄|ΣΤ΄|Ζ΄|Η΄|Θ΄|Ι΄)",
        text, re.UNICODE
    )
    if not m:
        return None
    label_map = {
        "Α΄": 1, "Β΄": 2, "Γ΄": 3, "Δ΄": 4, "Ε΄": 5,
        "ς΄": 6, "ΣΤ΄": 6, "Ζ΄": 7, "Η΄": 8, "Θ΄": 9, "Ι΄": 10,
    }
    return label_map.get(m.group(1))


def stage4_detect_structure(records: list[PageRecord]) -> list[PageRecord]:
    """Assign logos_num to every page and flag title pages.

    Uses two signals:
      1. Running header (already parsed in stage 2).
      2. ΛΟΓΟΣ heading text — used for title pages that carry no header,
         and as the authoritative value when a new logos begins mid-page.

    Logs any page whose logos assignment remains ambiguous.

    Args:
        records: Output of stage3_normalise_encoding.

    Returns:
        Same list with logos_num and is_title_page set on every record.
    """
    current_logos: int = 2   # Logos Α΄ is lost; corpus begins with Β΄

    for rec in records:
        title_logos = _detect_logos_title(rec.body_text)
        if title_logos is not None:
            rec.is_title_page = (rec.logos_num is None)  # no running header
            # A page may contain the END of one logos and START of the next.
            # We annotate it with the NEW logos so downstream segmentation
            # can detect the boundary.
            current_logos = title_logos

        if rec.logos_num is None:
            rec.logos_num = current_logos
        else:
            # Keep header value; update current tracker
            current_logos = rec.logos_num

        if rec.logos_num is None:
            log.warning("p.%d: logos assignment ambiguous — manual review needed",
                        rec.printed_page)

    log.info("Stage 4 — structure detection complete")
    return records


# ── Stage 5: segmentation ─────────────────────────────────────────────────────

_LOGOS_TITLE_SPLIT_RE = re.compile(
    r"(?=ΛΟΓΟΣ\s*\n?\s*(?:Α΄|Β΄|Γ΄|Δ΄|Ε΄|ς΄|ΣΤ΄|Ζ΄|Η΄|Θ΄|Ι΄))",
    re.UNICODE,
)
_LOGOS_LABEL_IN_CHUNK_RE = re.compile(
    r"ΛΟΓΟΣ\s*\n?\s*(Α΄|Β΄|Γ΄|Δ΄|Ε΄|ς΄|ΣΤ΄|Ζ΄|Η΄|Θ΄|Ι΄)",
    re.UNICODE,
)
_LABEL_TO_NUM: dict[str, int] = {
    "Α΄": 1, "Β΄": 2, "Γ΄": 3, "Δ΄": 4, "Ε΄": 5,
    "ς΄": 6, "ΣΤ΄": 6, "Ζ΄": 7, "Η΄": 8, "Θ΄": 9, "Ι΄": 10,
}


def _pages_to_token_stream(
    records: list[PageRecord],
) -> list[Tuple[int, int, str]]:
    """Flatten pages into a list of (logos_num, printed_page, text_chunk).

    Pages that span a Logos boundary are split at every ΛΟΓΟΣ heading they
    contain so each chunk is correctly attributed to one Logos.  A page may
    hold several Logos titles (e.g. p. 75 carries both Α΄ and Β΄).
    """
    chunks: list[Tuple[int, int, str]] = []

    for rec in records:
        text = rec.body_text
        if not text:
            continue

        # Split the page text at every ΛΟΓΟΣ title boundary (zero-width split)
        parts = _LOGOS_TITLE_SPLIT_RE.split(text)

        if len(parts) <= 1:
            # No title on this page — all text belongs to rec.logos_num
            chunks.append((rec.logos_num, rec.printed_page, text))
            continue

        # First part is text BEFORE any ΛΟΓΟΣ heading (tail of previous logos)
        if parts[0].strip():
            chunks.append((rec.logos_num, rec.printed_page, parts[0].strip()))

        # Remaining parts each start with a ΛΟΓΟΣ heading
        for part in parts[1:]:
            m = _LOGOS_LABEL_IN_CHUNK_RE.match(part)
            if m:
                logos_num = _LABEL_TO_NUM[m.group(1)]
            else:
                logos_num = rec.logos_num  # fallback
            chunks.append((logos_num, rec.printed_page, part.strip()))

    return chunks


def _split_into_sections(
    chunks: list[Tuple[int, int, str]],
) -> list[SectionRecord]:
    """Parse section markers from the flattened chunk stream.

    A section marker is a digit sequence followed by a period and a space
    at the start of a word boundary (e.g. "12. Ἐπεὶ δέ…").  Section numbers
    restart at 1 for each Logos.

    Cross-page sections are assembled by accumulating text until the next
    marker is encountered.  page_start is the page of the opening marker;
    page_end is the page of the last chunk before the next marker.

    Returns:
        List of SectionRecord, ordered globally across all Logoi.
    """
    sections: list[SectionRecord] = []

    # State for the in-progress section
    cur_logos: Optional[int]  = None
    cur_section: Optional[int] = None
    cur_texts: list[str]       = []
    cur_page_start: int        = 0
    cur_page_end: int          = 0
    expected_next: int         = 1   # expected section number

    def _flush() -> None:
        if cur_logos is None or cur_section is None:
            return
        joined = " ".join(cur_texts).strip()
        # Normalise whitespace
        joined = re.sub(r"\s+", " ", joined)
        sections.append(SectionRecord(
            logos_num=cur_logos,
            section_num=cur_section,
            text=joined,
            page_start=cur_page_start,
            page_end=cur_page_end,
        ))

    for logos_num, page_num, text in chunks:
        # When we enter a new logos, flush the previous section
        if logos_num != cur_logos:
            _flush()
            cur_logos    = logos_num
            cur_section  = None
            cur_texts    = []
            expected_next = 1

        # Find all section markers in this chunk
        markers = list(re.finditer(
            r"(?:(?:^|\n)\s*|(?<=\s))(\d+)\.\s+(?=\S)",
            text, re.UNICODE | re.MULTILINE
        ))

        if not markers:
            # No new section starts; accumulate into current section
            cur_texts.append(text)
            cur_page_end = page_num
            continue

        # Process text before the first marker (tail of previous section)
        pre = text[: markers[0].start()].strip()
        if pre:
            cur_texts.append(pre)
            cur_page_end = page_num

        for i, m in enumerate(markers):
            sec_num = int(m.group(1))
            end_pos = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            seg_text = text[m.end(): end_pos].strip()

            # Validate sequential numbering; log gaps
            if cur_section is not None and sec_num != cur_section + 1:
                if sec_num > cur_section + 1:
                    log.warning(
                        "Logos %d: section gap — expected §%d, found §%d "
                        "(p.%d) — flagging for review",
                        logos_num, cur_section + 1, sec_num, page_num,
                    )
                elif sec_num <= cur_section:
                    # False positive (number in body text) — skip
                    cur_texts.append(m.group(0) + seg_text)
                    continue

            # Flush the previous section before starting this one
            _flush()

            cur_section   = sec_num
            cur_texts     = [seg_text] if seg_text else []
            cur_page_start = page_num
            cur_page_end   = page_num
            expected_next  = sec_num + 1

    _flush()
    log.info("Stage 5 — segmented %d sections across all Logoi", len(sections))
    return sections


def stage5_segment(records: list[PageRecord]) -> list[SectionRecord]:
    """Orchestrate segmentation: flatten pages then split by section marker.

    Also inserts the Λόγος Α΄ placeholder row (text lost) and the Λόγος Β΄
    partial placeholder for the opening of that discourse (text missing).

    Args:
        records: Output of stage4_detect_structure.

    Returns:
        Ordered list of SectionRecord objects.
    """
    chunks = _pages_to_token_stream(records)
    sections = _split_into_sections(chunks)

    # Logos Α΄ — entirely lost; insert single placeholder
    logos_a_placeholder = SectionRecord(
        logos_num=1, section_num=0,
        text="",
        page_start=75, page_end=75,
        incomplete=True, partial=False,
        reason="lost",
    )

    # Logos Β΄ section 1 is partially preserved (opening lacuna marked ***])
    for sec in sections:
        if sec.logos_num == 2 and sec.section_num == 1:
            sec.incomplete = True
            sec.partial    = True
            sec.reason     = "lost"

    # Prepend the Α΄ placeholder
    sections.insert(0, logos_a_placeholder)

    return sections


# ── Stage 6: metadata attachment ──────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_count(text: str) -> int:
    """Whitespace-split approximation of token count."""
    return len(text.split()) if text else 0


def _extract_biblical_citations(text: str) -> str:
    """Return pipe-separated list of detected scripture references.

    Uses abbreviation patterns common in Byzantine Greek texts.  Results
    are best-effort; manual validation is recommended for the corpus.
    """
    hits = _BIBLE_ABBREV_RE.findall(text)
    cleaned = [re.sub(r"\s+", "", h.strip().rstrip(",.:;")) for h in hits]
    unique = list(dict.fromkeys(cleaned))   # deduplicate, preserve order
    return "|".join(unique) if unique else ""


def stage6_attach_metadata(sections: list[SectionRecord]) -> pd.DataFrame:
    """Build the final DataFrame from SectionRecord objects.

    Populates every column defined in CSV_COLUMNS.

    Args:
        sections: Output of stage5_segment.

    Returns:
        DataFrame with dtype-correct columns in canonical order.
    """
    rows = []
    for seq, sec in enumerate(sections, start=1):
        logos_label = LOGOS_LABEL.get(sec.logos_num, f"?{sec.logos_num}?")
        subtitle    = LOGOS_SUBTITLE.get(sec.logos_num, "")
        section_str = f"§{sec.section_num}" if sec.section_num else ""
        text_val    = sec.text if sec.text else None

        rows.append({
            "id":                seq,
            "logos":             logos_label,
            "section":           section_str,
            "paragraph":         subtitle,
            "text":              text_val,
            "source":            SOURCE_LABEL,
            "page_start":        sec.page_start,
            "page_end":          sec.page_end,
            "sequence":          seq,
            "token_count":       _token_count(sec.text),
            "text_hash":         _sha256(sec.text) if sec.text else "",
            "biblical_citations": _extract_biblical_citations(sec.text or ""),
            "incomplete":        sec.incomplete,
            "partial":           sec.partial,
        })

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    log.info("Stage 6 — metadata attached; %d rows total", len(df))
    return df


# ── Stage 7: quality checks ───────────────────────────────────────────────────

def stage7_quality_check(df: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    """Run quality checks and write a plain-text quality report.

    Checks:
      • Rows with token_count > TOKEN_COUNT_WARN (512).
      • Rows missing logos or section values.
      • Encoding: any non-UTF-8 text surviving to this point.

    Args:
        df:          DataFrame from stage6_attach_metadata.
        report_path: Path where quality_report.txt will be written.

    Returns:
        The same DataFrame (unchanged); side-effect is the report file.
    """
    lines: list[str] = [
        "Neophytos Corpus — Quality Report",
        "=" * 60,
        f"Pipeline version : {VERSION}",
        f"Source           : {SOURCE_LABEL}",
        f"Total rows       : {len(df)}",
        "",
    ]

    # Per-Logos row counts
    lines.append("Rows per Λόγος:")
    for logos_label, grp in df.groupby("logos", sort=False):
        lines.append(f"  {logos_label:6s}  {len(grp):4d} rows")
    lines.append("")

    # Oversized sections
    oversized = df[df["token_count"] > TOKEN_COUNT_WARN]
    lines.append(
        f"Sections with token_count > {TOKEN_COUNT_WARN}  "
        f"(exceeds typical model context limit):"
    )
    if oversized.empty:
        lines.append("  None.")
    else:
        for _, row in oversized.iterrows():
            lines.append(
                f"  id={row['id']}  logos={row['logos']}  "
                f"section={row['section']}  tokens={row['token_count']}"
            )
    lines.append("")

    # Traceability check
    missing_logos   = df[df["logos"].isna() | (df["logos"] == "")]
    missing_section = df[df["section"].isna() | (df["section"] == "")]
    lines.append("Traceability (logos + section must be populated):")
    if missing_logos.empty and missing_section.empty:
        lines.append("  All rows have logos and section. ✓")
    else:
        for _, row in missing_logos.iterrows():
            lines.append(f"  MISSING LOGOS  id={row['id']}")
        for _, row in missing_section.iterrows():
            lines.append(f"  MISSING SECTION  id={row['id']}  logos={row['logos']}")
    lines.append("")

    # Incomplete / partial rows
    inc = df[df["incomplete"] == True]   # noqa: E712
    lines.append(f"Incomplete rows (text=NULL or partial): {len(inc)}")
    for _, row in inc.iterrows():
        lines.append(
            f"  id={row['id']}  logos={row['logos']}  "
            f"section={row['section']}  partial={row['partial']}"
        )
    lines.append("")

    # Encoding spot-check
    enc_warnings: list[str] = []
    for _, row in df.iterrows():
        t = row["text"] if isinstance(row["text"], str) else ""
        try:
            t.encode("utf-8")
        except UnicodeError as e:
            enc_warnings.append(f"  id={row['id']}: {e}")
    lines.append(f"Encoding warnings: {len(enc_warnings)}")
    lines.extend(enc_warnings or ["  None."])

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Stage 7 — quality report written to %s", report_path)

    if not oversized.empty:
        log.warning(
            "Stage 7 — %d sections exceed %d tokens", len(oversized), TOKEN_COUNT_WARN
        )
    return df


# ── pipeline orchestration ────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: Path,
    output_path: Path,
    report_path: Path,
) -> pd.DataFrame:
    """Execute all seven stages and write outputs.

    Args:
        pdf_path:    Path to the source PDF.
        output_path: Destination for neophytos_corpus.csv.
        report_path: Destination for quality_report.txt.

    Returns:
        The final DataFrame (also written to output_path).
    """
    log.info("=== Neophytos preprocessing pipeline v%s ===", VERSION)

    records = stage1_extract_pages(pdf_path)
    records = stage2_separate_source(records)
    records = stage3_normalise_encoding(records)
    records = stage4_detect_structure(records)
    sections = stage5_segment(records)
    df = stage6_attach_metadata(sections)
    df = stage7_quality_check(df, report_path)

    df.to_csv(output_path, index=False, encoding="utf-8", quoting=1)
    log.info("Output written to %s  (%d rows)", output_path, len(df))
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Stefanis 1996 PDF to a structured CSV corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pdf", required=True, type=Path,
        help="Path to stefanis_1996_TEXT_ENTOLES.pdf",
    )
    p.add_argument(
        "--output", default=Path("neophytos_corpus.csv"), type=Path,
        help="Output CSV path",
    )
    p.add_argument(
        "--report", default=Path("quality_report.txt"), type=Path,
        help="Quality report path",
    )
    return p


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()
    if not args.pdf.exists():
        log.error("PDF not found: %s", args.pdf)
        sys.exit(1)
    run_pipeline(args.pdf, args.output, args.report)


if __name__ == "__main__":
    main()
