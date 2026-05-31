"""
Phase B — TEC Few-Shot Classification
======================================
Postdoctoral Research: Computational Emotional Analysis
Saint Neophytos the Recluse — Δέκα Λόγοι περί Χριστού Εντολών

Reads neophytos_corpus.csv, classifies each paragraph using
Claude API with few-shot prompting, and writes results to
neophytos_tec_results.csv.

Usage (from Claude Code / WSL):
    python3 tec_classify.py

Requirements:
    pip install anthropic pandas tqdm --break-system-packages

Environment variable:
    ANTHROPIC_API_KEY — set in WSL before running
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    raise SystemExit("Run: pip install anthropic --break-system-packages")

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL         = "claude-sonnet-4-6"
MAX_TOKENS    = 300
SLEEP_BETWEEN = 0.5   # seconds between API calls (rate limit safety)
INPUT_CSV     = "neophytos_corpus.csv"
OUTPUT_CSV    = "neophytos_tec_results.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ─── Few-Shot Examples (from TEC-Proposals-v1, approved) ─────────────────────
FEW_SHOT_EXAMPLES = [
    # ΑΓΑΠΗ / ΦΙΛΟΣΤΟΡΓΙΑ
    {
        "paragraph": "Ἠγάπησα ὑμᾶς ἀγάπην ἀπὸ καρδίας, ἀγάπην θεϊκήν, ἀγάπην σωτήριον, ἀγάπην πρὸς βασιλείαν, ὥστε καὶ τὸ αἷμά μου δίδωμι λύτρον ὑπὲρ τοῦ κόσμου.",
        "tec_category": "ΑΓΑΠΗ",
        "polarity": "Θετικό",
        "justification": "Τετραπλή επανάληψη ἀγάπην — παθιασμένη εκδήλωση θεϊκής αγάπης ως πρότυπο."
    },
    {
        "paragraph": "Ὥσπερ πατὴρ φιλόστοργος, εἰς εὐταξίας κοσμιότητα ῥυθμίζειν τοὺς παῖδας βουλόμενος, οὕτω καὶ ὁ Χριστός, βουλόμενος ἡμῶν γαληνιᾶν τὴν καρδίαν.",
        "tec_category": "ΑΓΑΠΗ",
        "polarity": "Θετικό",
        "justification": "Εικόνα πατρὸς φιλοστόργου — ήπια φροντίδα, γαλήνια εγγύτητα."
    },
    # ΜΕΤΑΝΟΙΑ / ΚΑΤΑΝΥΞΗ
    {
        "paragraph": "Οὐκ εἶπε· Συγχώρησόν μοι, Κύριε, τὰς λῃστρικὰς αἰσχρουργίας, τοὺς φόνους, τοὺς δόλους· ἀλλά· Μνήσθητί μου, Κύριε.",
        "tec_category": "ΜΕΤΑΝΟΙΑ",
        "polarity": "Αρνητικό",
        "justification": "Οξύ αλλά συντεταγμένο πένθος του λῃστοῦ — αρνητική φόρτιση, όχι απόγνωση."
    },
    {
        "paragraph": "Πηλίκας οὖν ἔχοντες μαρτυρίας μηδεὶς ἀπογνώσῃ· μόνον σπευσάτω ποιῆσαι καρπὸν ἄξιον τῆς μετανοίας. Αὕτη γὰρ καὶ μόνη μετὰ Θεὸν ἐλπὶς σωτηρίας.",
        "tec_category": "ΜΕΤΑΝΟΙΑ",
        "polarity": "Θετικό",
        "justification": "Μετάνοια που ανοίγει σε ελπίδα: μηδεὶς ἀπογνώσῃ — θετική φόρτιση κυριαρχεί."
    },
    # ΦΟΒΟΣ ΘΕΟΥ
    {
        "paragraph": "Κύριε, εἰσακήκοα τὴν ἀκοήν σου καὶ ἐφοβήθην, κατενόησα τὰ ἔργα σου καὶ ἐξέστην· ὁ γὰρ ἀκούων Θεοῦ ἀκοὴν ὡς δεῖ φοβεῖται πάνυ καὶ τρέμει.",
        "tec_category": "ΦΟΒΟΣ_ΘΕΟΥ",
        "polarity": "Αρνητικό",
        "justification": "Παράθεμα Αββακούμ με ερμηνεία: ἐξέστην, φοβεῖται, τρέμει — έντονο άγιο δέος."
    },
    # ΕΛΠΙΔΑ / ΠΑΡΑΚΛΗΣΗ
    {
        "paragraph": "Καὶ ὁ Κύριος· Ἀμήν, ἀμὴν λέγω σοι, μετ᾿ ἐμοῦ σήμερον ἔσῃ ἐν τῷ παραδείσῳ.",
        "tec_category": "ΕΛΠΙΔΑ",
        "polarity": "Θετικό",
        "justification": "Υπόσχεση Παραδείσου σήμερον — η πιο άμεση ελπίδα του corpus. Μέγιστη θετική φόρτιση."
    },
    {
        "paragraph": "Γένοιτο δὲ ἡμᾶς κατ᾿ ἀξίαν αἰτεῖν καὶ λαμβάνειν ζητεῖν τε καὶ εὑρίσκειν τὴν ὁδὸν τῆς ζωῆς εἰς τὴν αἰώνιον ζωὴν χάριτι καὶ φιλανθρωπίᾳ.",
        "tec_category": "ΕΛΠΙΔΑ",
        "polarity": "Θετικό",
        "justification": "Επευκτική φράση — ήρεμη, προσευχητική ελπίδα χωρίς δράμα."
    },
    # ΚΑΤΗΓΟΡΙΑ / ΕΠΙΚΡΙΣΗ
    {
        "paragraph": "Ὡσεὶ ἀσπίδος κωφῆς καὶ βυούσης τὰ ὦτα αὐτῆς, οἳ οὐκ εἰσακούσονται φωνῆς ἐπᾳδόντων τὰ θεῖα διδάγματα, καὶ φαρμακοῦνται παρὰ σοφοῦ δαίμονος τῇ κακίᾳ.",
        "tec_category": "ΚΑΤΗΓΟΡΙΑ",
        "polarity": "Αρνητικό",
        "justification": "Εικόνα κωφής ασπίδας — σκληρή κατηγορία κατά αδιαφόρων. Υψηλή αρνητική φόρτιση."
    },
    # ΔΟΞΟΛΟΓΙΑ / ΕΥΧΑΡΙΣΤΙΑ
    {
        "paragraph": "Καθάπερ γὰρ μύρον ἐκ μυροθήκης κενούμενον οὐ βίαν, οὐ μολυσμόν, οὐκ ἄλλην τινὰ κέκτηται δυσκολίαν, μόνους δὲ τοὺς μεθέξοντας θέλει τῆς τούτου εὐωδίας.",
        "tec_category": "ΔΟΞΟΛΟΓΙΑ",
        "polarity": "Θετικό",
        "justification": "Μεγαλειώδης εικόνα μύρου — αβίαστη αγαλλίαση. Μέγιστη θετική φόρτιση."
    },
    {
        "paragraph": "Εἴδομεν καὶ ταύτης τῆς αἰσθήσεως τὴν ὠφέλειαν· δόξα σοι ὁ Θεός, δόξα σοι τῷ εὐωδιάσαντι τὸν κόσμον τῇ σῇ παρουσίᾳ.",
        "tec_category": "ΔΟΞΟΛΟΓΙΑ",
        "polarity": "Θετικό",
        "justification": "Απλή δόξα σοι μετά από διδακτική παράγραφο — αυθόρμητη ευχαριστία."
    },
    # ΕΝΤΟΛΗ / ΥΠΟΧΡΕΩΣΗ
    {
        "paragraph": "Ἀγαπᾶτε τοὺς ἐχθροὺς ὑμῶν, εὐλογεῖτε τοὺς καταρωμένους ὑμᾶς, καλῶς ποιεῖτε τοῖς μισοῦσιν ὑμᾶς καὶ προσεύχεσθε ὑπὲρ τῶν ἐπηρεαζόντων καὶ διωκόντων ὑμᾶς.",
        "tec_category": "ΕΝΤΟΛΗ",
        "polarity": "Ουδέτερο",
        "justification": "Τέσσερις προστακτικές εγκλίσεις — καθαρά κανονιστικός τόνος χωρίς συναίσθημα."
    },
    # ΕΣΧΑΤΟΛΟΓΙΑ / ΠΡΟΕΙΔΟΠΟΙΗΣΗ
    {
        "paragraph": "Ὅταν ἔλθῃ, φησίν, ὁ υἱὸς τοῦ ἀνθρώπου ἐν τῇ δόξῃ αὐτοῦ, καὶ πάντες οἱ ἅγιοι ἄγγελοι μετ᾿ αὐτοῦ, τότε καθίσει ἐπὶ θρόνου δόξης αὐτοῦ.",
        "tec_category": "ΕΣΧΑΤΟΛΟΓΙΑ",
        "polarity": "Αρνητικό",
        "justification": "Εισαγωγή Δευτέρας Παρουσίας — δραματικός τόνος, ισχυρή εσχατολογική φόρτιση."
    },
    {
        "paragraph": "Γένοιτο δὲ πάντας ἡμᾶς εἰλικρινοῦς τυχεῖν μετανοίας καὶ τὰς θείας ἐντολὰς τηρεῖν μετὰ πόθου καὶ τῆς αἰωνίου ζωῆς κληρονόμους γενέσθαι.",
        "tec_category": "ΕΣΧΑΤΟΛΟΓΙΑ",
        "polarity": "Θετικό",
        "justification": "Κλείσιμο corpus με αναφορά αιώνιας ζωής ως ελπίδα — μεικτός χαρακτήρας."
    },
]

# ─── Build the master few-shot prompt ────────────────────────────────────────
def build_system_prompt():
    examples_text = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_text += f"""
ΠΑΡΑΓΡΑΦΟΣ: «{ex['paragraph']}»
→ ΚΑΤΗΓΟΡΙΑ_TEC: {ex['tec_category']}
→ ΠΟΛΙΚΟΤΗΤΑ: {ex['polarity']}
→ ΑΙΤΙΟΛΟΓΗΣΗ: {ex['justification']}
"""

    return f"""Είσαι ένα σύστημα κατηγοριοποίησης θεολογικών συναισθημάτων για βυζαντινά πατερικά κείμενα.

Αποστολή σου: να κατηγοριοποιήσεις παραγράφους από τους «Δέκα Λόγους περί Χριστού Εντολών» (ca. 1175/6) του Αγίου Νεοφύτου του Εγκλείστου σύμφωνα με το σύστημα TEC (Θεολογικές Συναισθηματικές Κατηγορίες).

## Κατηγορίες TEC

1. ΑΓΑΠΗ — Αγάπη και στοργή προς τον Θεό ή τον πλησίον. Θερμή εγγύτητα, πατρική διάθεση.
2. ΜΕΤΑΝΟΙΑ — Κατάνυξη. Πόνος για αμαρτία που οδηγεί σε μεταμέλεια, όχι απόγνωση.
3. ΦΟΒΟΣ_ΘΕΟΥ — Ιερό δέος. Σεβασμός μπροστά στο θεϊκό μεγαλείο, όχι τρόμος.
4. ΕΛΠΙΔΑ — Εσχατολογική παρηγοριά. Βεβαιότητα σωτηρίας ως κίνητρο ζωής.
5. ΚΑΤΗΓΟΡΙΑ — Αυστηρός έλεγχος αμαρτίας. Ηθική επίπληξη.
6. ΔΟΞΟΛΟΓΙΑ — Χαρά, ύμνος, ευχαριστία. Αγαλλίαση μπροστά στο θεϊκό μεγαλείο.
7. ΕΝΤΟΛΗ — Κανονιστικός, διδακτικός τόνος. Σαφής έκθεση εντολών.
8. ΕΣΧΑΤΟΛΟΓΙΑ — Κρίση, αιωνιότητα, θάνατος. Συνείδηση τέλους ως αφύπνιση.

## Παραδείγματα
{examples_text}

## Οδηγίες εξόδου

Απάντησε ΜΟΝΟ σε JSON, χωρίς καμία άλλη λέξη:
{{
  "tec_category": "μία από τις 8 κατηγορίες",
  "secondary_category": "δεύτερη κατηγορία αν υπάρχει ή null",
  "polarity": "Θετικό ή Αρνητικό ή Ουδέτερο",
  "confidence": 0.00,
  "justification": "μία πρόταση στα ελληνικά"
}}"""


def classify_paragraph(client, system_prompt, text):
    """Call Claude API for one paragraph. Returns dict or None on error."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[
                {"role": "user", "content": f"ΠΑΡΑΓΡΑΦΟΣ: «{text}»"}
            ]
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": raw}
    except Exception as e:
        return {"error": str(e)}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: Set ANTHROPIC_API_KEY environment variable first.\n"
            "In WSL: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Load corpus
    df = pd.read_csv(INPUT_CSV)
    df = df[df["incomplete"] == False].copy()
    df["text"] = df["text"].fillna("")
    print(f"Loaded {len(df)} paragraphs from {INPUT_CSV}")

    system_prompt = build_system_prompt()

    # Resume support: skip already classified
    results = []
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        done_ids = set(existing["id"].tolist())
        results = existing.to_dict("records")
        df = df[~df["id"].isin(done_ids)]
        print(f"Resuming — {len(done_ids)} already done, {len(df)} remaining")

    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        result = classify_paragraph(client, system_prompt, row["text"])

        record = {
            "id":                row["id"],
            "logos":             row["logos"],
            "section":           row["section"],
            "paragraph":         row["paragraph"],
            "tec_category":      result.get("tec_category", "ERROR"),
            "secondary_category":result.get("secondary_category"),
            "polarity":          result.get("polarity", "ERROR"),
            "confidence":        result.get("confidence"),
            "justification":     result.get("justification", ""),
            "error":             result.get("error"),
            "page_start":        row["page_start"],
            "biblical_citations":row["biblical_citations"],
        }
        results.append(record)

        if result.get("error"):
            errors += 1
            tqdm.write(f"  ⚠ Error at {row['logos']}{row['section']}: {result['error']}")

        # Save after every paragraph (crash-safe)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. {len(results)} paragraphs classified, {errors} errors.")
    print(f"Results saved to: {OUTPUT_CSV}")

    # Quick summary
    out_df = pd.DataFrame(results)
    if "tec_category" in out_df.columns:
        print("\nΚατανομή TEC κατηγοριών:")
        print(out_df["tec_category"].value_counts().to_string())
        print("\nΚατανομή πολικότητας:")
        print(out_df["polarity"].value_counts().to_string())


if __name__ == "__main__":
    main()
