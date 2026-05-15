"""
tests/test_preprocessing.py — pytest unit tests for preprocessing.py

At least one test per pipeline stage (7 stages = 7+ tests).
Run with: pytest tests/ -v --cov=preprocessing --cov-report=term-missing
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing import (
    CSV_COLUMNS,
    LOGOS_LABEL,
    SOURCE_LABEL,
    TOKEN_COUNT_WARN,
    PageRecord,
    SectionRecord,
    _detect_logos_title,
    _extract_biblical_citations,
    _sha256,
    _token_count,
    stage1_extract_pages,
    stage2_separate_source,
    stage3_normalise_encoding,
    stage4_detect_structure,
    stage5_segment,
    stage6_attach_metadata,
    stage7_quality_check,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_record(
    pdf_index: int = 0,
    printed_page: int = 76,
    logos_num: int | None = None,
    body_text: str = "",
    is_title_page: bool = False,
) -> PageRecord:
    return PageRecord(
        pdf_index=pdf_index,
        printed_page=printed_page,
        logos_num=logos_num,
        body_text=body_text,
        is_title_page=is_title_page,
    )


def make_section(
    logos_num: int = 2,
    section_num: int = 1,
    text: str = "Ἐπεὶ δὲ παραδείσους παραδείσῳ συγκρίναντες.",
    page_start: int = 75,
    page_end: int = 75,
    incomplete: bool = False,
    partial: bool = False,
) -> SectionRecord:
    return SectionRecord(
        logos_num=logos_num,
        section_num=section_num,
        text=text,
        page_start=page_start,
        page_end=page_end,
        incomplete=incomplete,
        partial=partial,
    )


# ── Stage 1: text extraction ──────────────────────────────────────────────────

class TestStage1:
    """Stage 1 — PDF text extraction."""

    def test_extracts_correct_page_count(self, tmp_path):
        """Pipeline should request exactly PDF_LAST_IDX - PDF_FIRST_IDX + 1 pages."""
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "sample text"

        with patch("preprocessing.pdfplumber.open") as mock_open:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__  = MagicMock(return_value=False)
            ctx.pages     = [fake_page] * 98
            mock_open.return_value = ctx

            result = stage1_extract_pages(Path("dummy.pdf"))

        assert len(result) == 98

    def test_printed_page_mapping(self, tmp_path):
        """Printed page number = PDF index + 74."""
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "text"

        with patch("preprocessing.pdfplumber.open") as mock_open:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__  = MagicMock(return_value=False)
            ctx.pages     = [fake_page] * 98
            mock_open.return_value = ctx

            result = stage1_extract_pages(Path("dummy.pdf"))

        assert result[0].printed_page == 74
        assert result[97].printed_page == 171

    def test_empty_page_returns_empty_string(self):
        """extract_text returning None is handled gracefully."""
        fake_page = MagicMock()
        fake_page.extract_text.return_value = None

        with patch("preprocessing.pdfplumber.open") as mock_open:
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=ctx)
            ctx.__exit__  = MagicMock(return_value=False)
            ctx.pages     = [fake_page] * 98
            mock_open.return_value = ctx

            result = stage1_extract_pages(Path("dummy.pdf"))

        assert result[0].body_text == ""


# ── Stage 2: source separation ────────────────────────────────────────────────

class TestStage2:
    """Stage 2 — running header and separator removal."""

    def test_running_header_stripped(self):
        raw = (
            "ΕΡΜΗΝΕΙΑ ΕΝΤΟΛΩΝ ΤΟΥ ΧΡΙΣΤΟΥ 3, 6, 4 78\n"
            "__________________________________________________________________\n"
            "καὶ βρέχει ἐπὶ δικαίους καὶ ἀδίκους."
        )
        rec = make_record(body_text=raw)
        result = stage2_separate_source([rec])[0]
        assert "ΕΡΜΗΝΕΙΑ" not in result.body_text
        assert "καὶ βρέχει" in result.body_text

    def test_logos_num_extracted_from_header(self):
        raw = "ΕΡΜΗΝΕΙΑ ΕΝΤΟΛΩΝ ΤΟΥ ΧΡΙΣΤΟΥ 5, 11, 1 94\n______________\ntext"
        rec = make_record(body_text=raw)
        result = stage2_separate_source([rec])[0]
        assert result.logos_num == 5

    def test_separator_line_stripped(self):
        raw = "____________________________________\nsome text"
        rec = make_record(body_text=raw)
        result = stage2_separate_source([rec])[0]
        assert "_____" not in result.body_text

    def test_body_text_preserved(self):
        body = "1. Πρῶτον περὶ πρώτης θείας φωνῆς."
        raw  = (
            "ΕΡΜΗΝΕΙΑ ΕΝΤΟΛΩΝ ΤΟΥ ΧΡΙΣΤΟΥ 3, 1, 1 77\n"
            "__________________________________\n"
            f"{body}"
        )
        rec = make_record(body_text=raw)
        result = stage2_separate_source([rec])[0]
        assert result.body_text.strip() == body


# ── Stage 3: encoding normalisation ──────────────────────────────────────────

class TestStage3:
    """Stage 3 — UTF-8 integrity and polytonic preservation."""

    def test_valid_polytonic_greek_passes(self):
        text = "Εὐλόγησον, πάτερ. Ἐὰν μή τις γεννηθῇ ἄνωθεν."
        rec = make_record(body_text=text)
        result = stage3_normalise_encoding([rec])[0]
        assert result.body_text == text

    def test_diacritics_preserved(self):
        """Combining diacritics must survive the stage unchanged."""
        text = "ἄρρητον ἧς τεχνίτης καὶ δημιουργὸς ὁ Θεός"
        rec = make_record(body_text=text)
        result = stage3_normalise_encoding([rec])[0]
        assert result.body_text == text


# ── Stage 4: structure detection ─────────────────────────────────────────────

class TestStage4:
    """Stage 4 — Logos boundary and title-page detection."""

    def test_detects_logos_B(self):
        text = "ΛΟΓΟΣ Β΄\nΕΙΣ ΤΑ ΤΗΣ ΜΕΤΑΝΟΙΑΣ ΕΠΙΛΟΙΠΑ\nΕὐλόγησον, πάτερ."
        assert _detect_logos_title(text) == 2

    def test_detects_logos_ST_lowercase(self):
        """ς΄ (lowercase digamma rendering) should map to logos 6."""
        text = "ΛΟΓΟΣ\nς΄\nΕΙΣ ΤΑΣ ΑΓΙΑΣ ΑΥΘΙΣ ΕΝΤΟΛΑΣ"
        assert _detect_logos_title(text) == 6

    def test_detects_logos_I(self):
        text = "ΛΟΓΟΣ Ι΄\nΕΙΣ ΤΑΣ ΑΓΙΑΣ ΚΑΙ ΑΥΘΙΣ ΕΝΤΟΛΑΣ"
        assert _detect_logos_title(text) == 10

    def test_no_logos_title_returns_none(self):
        text = "καὶ βρέχει ἐπὶ δικαίους καὶ ἀδίκους."
        assert _detect_logos_title(text) is None

    def test_title_page_flag_set(self):
        """A page with ΛΟΓΟΣ heading and no prior logos_num → is_title_page=True."""
        text = "ΛΟΓΟΣ Γ΄\nΠΕΡΙ ΤΟΥ ΘΕΙΟΥ ΒΑΠΤΙΣΜΑΤΟΣ\n1. Πρῶτον."
        rec = make_record(body_text=text, logos_num=None)
        result = stage4_detect_structure([rec])[0]
        assert result.is_title_page is True
        assert result.logos_num == 3

    def test_propagates_logos_to_subsequent_pages(self):
        """logos_num from running header propagates to next pages missing a header."""
        rec1 = make_record(pdf_index=0, printed_page=77, logos_num=3,
                           body_text="continuation text")
        rec2 = make_record(pdf_index=1, printed_page=78, logos_num=None,
                           body_text="more text")
        result = stage4_detect_structure([rec1, rec2])
        assert result[1].logos_num == 3


# ── Stage 5: segmentation ─────────────────────────────────────────────────────

class TestStage5:
    """Stage 5 — § splitting and placeholder insertion."""

    def _make_body_records(self, logos_num: int, text: str,
                           printed_page: int = 80) -> list[PageRecord]:
        rec = make_record(logos_num=logos_num, body_text=text,
                          printed_page=printed_page)
        return [rec]

    def test_single_section_extracted(self):
        recs = self._make_body_records(
            3, "1. Πρῶτον περὶ πρώτης θείας φωνῆς καὶ ἐντολῆς."
        )
        # need to run through stage4 first to avoid propagation issues
        recs = stage4_detect_structure(recs)
        secs = stage5_segment(recs)
        # exclude Logos Α΄ placeholder (index 0)
        body = [s for s in secs if s.logos_num == 3]
        assert len(body) >= 1
        assert body[0].section_num == 1

    def test_logos_a_placeholder_present(self):
        recs = [make_record(logos_num=2, body_text="1. text here.")]
        recs = stage4_detect_structure(recs)
        secs = stage5_segment(recs)
        placeholder = [s for s in secs if s.logos_num == 1]
        assert len(placeholder) == 1
        assert placeholder[0].incomplete is True
        assert placeholder[0].partial is False

    def test_logos_b_section1_marked_partial(self):
        text = "1. ***] ὑπὲρ νοῦν θεαμάτων ἐκείνων."
        recs = [make_record(logos_num=2, body_text=text, printed_page=75)]
        recs = stage4_detect_structure(recs)
        secs = stage5_segment(recs)
        sec1 = next((s for s in secs if s.logos_num == 2 and s.section_num == 1), None)
        assert sec1 is not None
        assert sec1.incomplete is True
        assert sec1.partial is True

    def test_multiple_sections_in_single_page(self):
        text = (
            "3. Ἐπεὶ δὲ παραδείσους παραδείσῳ συγκρίναντες.\n"
            "4. Εἰ δέ τις ἡμᾶς ἀπαιτεῖ φάναι τί ἐστιν βασιλεία.\n"
            "5. Ὁρᾷς ὁσαχῶς ἐμνημόνευσε βασιλείας."
        )
        recs = [make_record(logos_num=2, body_text=text, printed_page=76)]
        recs = stage4_detect_structure(recs)
        secs = stage5_segment(recs)
        body = [s for s in secs if s.logos_num == 2]
        nums = [s.section_num for s in body]
        assert 3 in nums and 4 in nums and 5 in nums

    def test_cross_page_section_assembled(self):
        """Section text split across two pages should be joined."""
        rec1 = make_record(logos_num=3, body_text="1. Ἀρχή τοῦ λόγου",
                           printed_page=77)
        rec2 = make_record(logos_num=3, body_text="συνέχεια τοῦ λόγου.",
                           printed_page=78)
        recs = stage4_detect_structure([rec1, rec2])
        secs = stage5_segment(recs)
        body = [s for s in secs if s.logos_num == 3 and s.section_num == 1]
        assert len(body) == 1
        assert "Ἀρχή" in body[0].text
        assert "συνέχεια" in body[0].text
        assert body[0].page_start == 77
        assert body[0].page_end == 78


# ── Stage 6: metadata attachment ─────────────────────────────────────────────

class TestStage6:
    """Stage 6 — DataFrame schema, token count, SHA-256, citations."""

    def _minimal_df(self) -> pd.DataFrame:
        secs = [make_section(logos_num=2, section_num=2,
                             text="Ἐπεὶ δὲ παραδείσους παραδείσῳ.")]
        return stage6_attach_metadata(secs)

    def test_all_columns_present(self):
        df = self._minimal_df()
        assert list(df.columns) == CSV_COLUMNS

    def test_logos_label_mapped(self):
        df = self._minimal_df()
        assert df.iloc[0]["logos"] == "Β΄"

    def test_token_count_whitespace_split(self):
        assert _token_count("ἓν δύο τρία") == 3
        assert _token_count("") == 0

    def test_sha256_deterministic(self):
        t = "Ἐπεὶ δὲ παραδείσους παραδείσῳ συγκρίναντες."
        h1 = _sha256(t)
        h2 = _sha256(t)
        assert h1 == h2
        assert len(h1) == 64

    def test_source_label(self):
        df = self._minimal_df()
        assert df.iloc[0]["source"] == SOURCE_LABEL

    def test_biblical_citation_extraction(self):
        text = "ὡς ὁ θεῖος εἶπε· Ματθ. 5,3 καὶ Ψαλμ. 1,1"
        cits = _extract_biblical_citations(text)
        assert "Ματθ." in cits
        assert "Ψαλμ." in cits

    def test_null_text_for_placeholder(self):
        placeholder = SectionRecord(
            logos_num=1, section_num=0, text="",
            page_start=75, page_end=75,
            incomplete=True, partial=False, reason="lost",
        )
        df = stage6_attach_metadata([placeholder])
        assert df.iloc[0]["text"] is None


# ── Stage 7: quality checks ───────────────────────────────────────────────────

class TestStage7:
    """Stage 7 — quality report generation and flagging."""

    def _build_df(self, token_count_override: int = 10) -> pd.DataFrame:
        secs = [make_section(logos_num=2, section_num=1)]
        df = stage6_attach_metadata(secs)
        df.loc[0, "token_count"] = token_count_override
        return df

    def test_report_file_created(self, tmp_path):
        df = self._build_df()
        report = tmp_path / "quality_report.txt"
        stage7_quality_check(df, report)
        assert report.exists()

    def test_report_contains_row_count(self, tmp_path):
        df = self._build_df()
        report = tmp_path / "quality_report.txt"
        stage7_quality_check(df, report)
        text = report.read_text(encoding="utf-8")
        assert "Total rows" in text

    def test_oversized_section_flagged(self, tmp_path):
        df = self._build_df(token_count_override=600)
        report = tmp_path / "quality_report.txt"
        stage7_quality_check(df, report)
        text = report.read_text(encoding="utf-8")
        assert str(TOKEN_COUNT_WARN) in text
        # The oversized row should appear in the report
        assert "id=1" in text

    def test_missing_logos_flagged(self, tmp_path):
        secs = [make_section(logos_num=2, section_num=1)]
        df = stage6_attach_metadata(secs)
        df.loc[0, "logos"] = ""   # simulate missing
        report = tmp_path / "quality_report.txt"
        stage7_quality_check(df, report)
        text = report.read_text(encoding="utf-8")
        assert "MISSING LOGOS" in text

    def test_per_logos_counts_in_report(self, tmp_path):
        secs = [
            make_section(logos_num=2, section_num=1),
            make_section(logos_num=2, section_num=2),
            make_section(logos_num=3, section_num=1),
        ]
        df = stage6_attach_metadata(secs)
        report = tmp_path / "quality_report.txt"
        stage7_quality_check(df, report)
        text = report.read_text(encoding="utf-8")
        assert "Β΄" in text
        assert "Γ΄" in text
