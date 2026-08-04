# PostDoc — Saint Neophytos the Recluse: Computational Corpus

Preprocessing pipeline for the postdoctoral research project on Byzantine patristic texts.
Converts the Stefanis 1996 critical edition of Saint Neophytos the Recluse's
**"Ten Discourses on Christ's Commandments"** (*Δέκα Λόγοι περὶ τοῦ Χριστοῦ Ἐντολῶν*)
into a structured CSV corpus for computational emotional analysis using NLP and AI models.

---

## Primary Source

| Field | Value |
|---|---|
| Author | Neophytos the Recluse (Νεόφυτος ὁ Ἔγκλειστος, 1134 – c. 1214) |
| Edition | Stefanis, I. E. (ed.), *Νεοφύτου Ἐγκλείστου Συγγράμματα*, vol. 1, Paphos 1996 |
| Pages | 74–171 |
| Script | Byzantine polytonic Greek |
| Manuscript | cod. Paris. Coisl. gr. 287 (late 12th c.) |

The PDF is **not included** in this repository (copyright). The pipeline reads it
from a local path passed as a CLI argument.

---

## PhD Dissertation

This repository also hosts the author's PhD dissertation, which the postdoctoral
project builds upon. Unlike the Primary Source PDF above, this is the author's own
work and is therefore not subject to third-party copyright restrictions.

| Field | Value |
|---|---|
| Title | Ο Άγιος Νεόφυτος ο Έγκλειστος και η χρήση της Αγίας Γραφής στα συγγράμματά του — Συγκριτική αποτύπωση και στατιστική ανάλυση με τη χρήση υπολογιστικής επεξεργασίας |
| Author | Charalambos (Harris) Aristotelous |
| Institution | University of Nicosia |
| Year | 2023 |

The PDF (432 pages, including figures) is split into two parts in `docs/`:

- [Part 1 (pp. 1–216)](docs/PHD-AGIOS-NEOFYTOS-V5-part1.pdf)
- [Part 2 (pp. 217–432)](docs/PHD-AGIOS-NEOFYTOS-V5-part2.pdf)

---

## Corpus Overview

The work comprises 10 Discourses (Λόγοι), structured as numbered sections (§§):

| Λόγος | Label | Status | Subtitle |
|---|---|---|---|
| 1 | Α΄ | **Lost** | Περὶ μετανοίας καὶ βασιλείας αἰωνίου |
| 2 | Β΄ | Partially preserved (§1 lacuna) | Εἰς τὰ τῆς μετανοίας ἐπίλοιπα |
| 3 | Γ΄ | Complete | Περὶ τοῦ θείου βαπτίσματος καὶ τῆς ἁγίας ἀγάπης |
| 4 | Δ΄ | Complete | Εἰς τὰς ἁγίας ἐντολὰς τοῦ Σωτῆρος |
| 5 | Ε΄ | Complete | Εἰς τὰς ἁγίας ἐντολὰς … περὶ τῶν πέντε αἰσθήσεων |
| 6 | ΣΤ΄ | Complete | Εἰς τὰς ἁγίας αὖθις ἐντολὰς |
| 7 | Ζ΄ | Complete | Εἰς τὰς ἁγίας ἐντολὰς τοῦ Κυρίου |
| 8 | Η΄ | Complete | Εἰς τὰς σεπτὰς θείας ἐντολὰς … εἰς τὴν δευτέραν παρουσίαν |
| 9 | Θ΄ | Complete | Εἰς τὰ ἐπίλοιπα τοῦ Μὴ κρίνετε |
| 10 | Ι΄ | Complete | Εἰς τὰς ἁγίας ἐντολὰς … εἰς τὴν ἔνδειαν τοῦ ἔτους |

The corpus contains **457 rows** (one § per row).

---

## Requirements

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

Key dependencies (version-pinned):

```
pdfplumber==0.11.9
pandas==3.0.3
pytest==8.3.5
pytest-cov==6.1.0
```

---

## Running the Pipeline

```bash
python preprocessing.py --pdf /path/to/stefanis_1996_TEXT_ENTOLES.pdf
```

Optional arguments:

```
--output   Output CSV path          (default: neophytos_corpus.csv)
--report   Quality report path      (default: quality_report.txt)
```

Full example:

```bash
python preprocessing.py \
    --pdf ~/stefanis_1996_TEXT_ENTOLES.pdf \
    --output neophytos_corpus.csv \
    --report quality_report.txt
```

---

## Pipeline Stages

| Stage | Name | Description |
|---|---|---|
| 1 | Text extraction | pdfplumber extracts pp. 74–171 (98 pages) |
| 2 | Source separation | Strips running headers, separator rules, and the apparatus introduction (p. 74) |
| 3 | Encoding normalisation | UTF-8 integrity check; polytonic diacritics are never modified |
| 4 | Structure detection | Detects all 10 Λόγος boundaries; handles pages with multiple titles |
| 5 | Segmentation | Splits by § marker; inserts Α΄ placeholder; flags Β΄ §1 as partial |
| 6 | Metadata attachment | Populates all 14 CSV columns per row |
| 7 | Quality checks | Flags token_count > 512, missing logos/section, encoding warnings |

---

## CSV Schema

Output: `neophytos_corpus.csv` (UTF-8, comma-separated, all fields quoted)

| Column | Type | Description |
|---|---|---|
| `id` | int | Sequential row identifier |
| `logos` | str | Greek ordinal label (Α΄–Ι΄) |
| `section` | str | Section number within the Λόγος (e.g. §3) |
| `paragraph` | str | Λόγος subtitle from the edition |
| `text` | str / NULL | Polytonic Greek text of the § |
| `source` | str | `Stefanis 1996` |
| `page_start` | int | Printed page where the § begins |
| `page_end` | int | Printed page where the § ends |
| `sequence` | int | Global order across all Λόγοι |
| `token_count` | int | Whitespace-split token approximation |
| `text_hash` | str | SHA-256 of the UTF-8 encoded text |
| `biblical_citations` | str | Pipe-separated scripture references (best-effort) |
| `incomplete` | bool | True if text is NULL or partially lost |
| `partial` | bool | True if only part of the § is preserved |

### Example rows

```
id=1  logos=Α΄  section=      text=NULL           incomplete=True   partial=False
id=2  logos=Β΄  section=§1    text=***] ὑπὲρ νοῦν θεαμάτων ἐκείνων…   incomplete=True   partial=True
id=3  logos=Β΄  section=§2    text=Ἐπεὶ δὲ παραδείσους παραδείσῳ συγκρίναντες…
id=4  logos=Β΄  section=§3    text=Καὶ εἰ ταῦτα μὲν ῥητὰ καὶ αἰσθητὰ ὄντα…
id=5  logos=Β΄  section=§4    text=Εἰ δέ τις ἡμᾶς ἀπαιτεῖ φάναι τί ἐστιν βασιλεία…
```

---

## Quality Report

`quality_report.txt` is generated automatically and includes:

- Total row count and per-Λόγος breakdown
- Sections exceeding 512 tokens (model context limit warning)
- Traceability check (every row must have logos + section)
- Encoding warnings

Current corpus status: **0 encoding warnings**, **1 section >512 tokens** (Ε΄ §58, 517 tokens).

---

## Tests

```bash
pytest tests/ -v --cov=preprocessing --cov-report=term-missing
```

32 unit tests, minimum one per pipeline stage. Targets >80% coverage.

---

## Repository Structure

```
PostDoc-Sant-Neofytos/
├── preprocessing.py        # 7-stage pipeline
├── requirements.txt        # pinned dependencies
├── neophytos_corpus.csv    # output corpus (457 rows)
├── quality_report.txt      # quality report
├── tests/
│   └── test_preprocessing.py
└── README.md
```

The source PDF is excluded from the repository (copyright). Pass its local
path via `--pdf` when running the pipeline.

---

## License

MIT License. See source files for details.

---

## Citation

If you use this corpus or pipeline in your research, please cite:

> Aristotelous, H. (2026). *Computational corpus of Neophytos the Recluse's
> Ten Discourses on Christ's Commandments* (Stefanis 1996 edition).
> Preprocessing pipeline v1.0.0.
> https://github.com/harrisarist/PostDoc-Saint-Neofytos
