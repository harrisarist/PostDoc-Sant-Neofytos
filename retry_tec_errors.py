"""
retry_tec_errors.py
====================
Phase B — Retry Failed TEC Classifications
Postdoctoral Research: Computational Emotional Analysis
Saint Neophytos the Recluse — Δέκα Λόγοι περί Χριστού Εντολών

Finds rows with errors in neophytos_tec_results.csv,
re-classifies them, and overwrites the fixed rows in-place.

Usage:
    python3 retry_tec_errors.py

Requirements:
    pip install anthropic pandas --break-system-packages
"""

import os
import json
import time
import pandas as pd

try:
    import anthropic
except ImportError:
    raise SystemExit("Run: pip install anthropic --break-system-packages")

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL         = "claude-sonnet-4-6"
MAX_TOKENS    = 400          # slightly more than original — gives JSON more room
SLEEP_BETWEEN = 1.0          # more generous delay on retry
CORPUS_CSV    = "neophytos_corpus.csv"
RESULTS_CSV   = "neophytos_tec_results.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ─── Same few-shot examples as tec_classify.py ───────────────────────────────
FEW_SHOT_EXAMPLES = [
    {"paragraph": "Ἠγάπησα ὑμᾶς ἀγάπην ἀπὸ καρδίας, ἀγάπην θεϊκήν, ἀγάπην σωτήριον, ἀγάπην πρὸς βασιλείαν, ὥστε καὶ τὸ αἷμά μου δίδωμι λύτρον ὑπὲρ τοῦ κόσμου.", "tec_category": "ΑΓΑΠΗ", "polarity": "Θετικό", "justification": "Τετραπλή επανάληψη ἀγάπην — παθιασμένη εκδήλωση θεϊκής αγάπης ως πρότυπο."},
    {"paragraph": "Ὥσπερ πατὴρ φιλόστοργος, εἰς εὐταξίας κοσμιότητα ῥυθμίζειν τοὺς παῖδας βουλόμενος, οὕτω καὶ ὁ Χριστός, βουλόμενος ἡμῶν γαληνιᾶν τὴν καρδίαν.", "tec_category": "ΑΓΑΠΗ", "polarity": "Θετικό", "justification": "Εικόνα πατρὸς φιλοστόργου — ήπια φροντίδα, γαλήνια εγγύτητα."},
    {"paragraph": "Οὐκ εἶπε· Συγχώρησόν μοι, Κύριε, τὰς λῃστρικὰς αἰσχρουργίας, τοὺς φόνους, τοὺς δόλους· ἀλλά· Μνήσθητί μου, Κύριε.", "tec_category": "ΜΕΤΑΝΟΙΑ", "polarity": "Αρνητικό", "justification": "Οξύ αλλά συντεταγμένο πένθος του λῃστοῦ."},
    {"paragraph": "Κύριε, εἰσακήκοα τὴν ἀκοήν σου καὶ ἐφοβήθην, κατενόησα τὰ ἔργα σου καὶ ἐξέστην.", "tec_category": "ΦΟΒΟΣ_ΘΕΟΥ", "polarity": "Αρνητικό", "justification": "ἐξέστην, φοβεῖται, τρέμει — έντονο άγιο δέος."},
    {"paragraph": "Καὶ ὁ Κύριος· Ἀμήν, ἀμὴν λέγω σοι, μετ᾿ ἐμοῦ σήμερον ἔσῃ ἐν τῷ παραδείσῳ.", "tec_category": "ΕΛΠΙΔΑ", "polarity": "Θετικό", "justification": "Υπόσχεση Παραδείσου σήμερον — μέγιστη θετική φόρτιση."},
    {"paragraph": "Ὡσεὶ ἀσπίδος κωφῆς καὶ βυούσης τὰ ὦτα αὐτῆς, οἳ οὐκ εἰσακούσονται φωνῆς ἐπᾳδόντων τὰ θεῖα διδάγματα.", "tec_category": "ΚΑΤΗΓΟΡΙΑ", "polarity": "Αρνητικό", "justification": "Εικόνα κωφής ασπίδας — σκληρή κατηγορία κατά αδιαφόρων."},
    {"paragraph": "Εἴδομεν καὶ ταύτης τῆς αἰσθήσεως τὴν ὠφέλειαν· δόξα σοι ὁ Θεός, δόξα σοι τῷ εὐωδιάσαντι τὸν κόσμον τῇ σῇ παρουσίᾳ.", "tec_category": "ΔΟΞΟΛΟΓΙΑ", "polarity": "Θετικό", "justification": "Απλή δόξα σοι — αυθόρμητη ευχαριστία."},
    {"paragraph": "Ἀγαπᾶτε τοὺς ἐχθροὺς ὑμῶν, εὐλογεῖτε τοὺς καταρωμένους ὑμᾶς, καλῶς ποιεῖτε τοῖς μισοῦσιν ὑμᾶς.", "tec_category": "ΕΝΤΟΛΗ", "polarity": "Ουδέτερο", "justification": "Τέσσερις προστακτικές — καθαρά κανονιστικός τόνος."},
    {"paragraph": "Ὅταν ἔλθῃ, φησίν, ὁ υἱὸς τοῦ ἀνθρώπου ἐν τῇ δόξῃ αὐτοῦ, καὶ πάντες οἱ ἅγιοι ἄγγελοι μετ᾿ αὐτοῦ, τότε καθίσει ἐπὶ θρόνου δόξης αὐτοῦ.", "tec_category": "ΕΣΧΑΤΟΛΟΓΙΑ", "polarity": "Αρνητικό", "justification": "Εισαγωγή Δευτέρας Παρουσίας — δραματικός εσχατολογικός τόνος."},
]


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

Κατηγορίες TEC: ΑΓΑΠΗ, ΜΕΤΑΝΟΙΑ, ΦΟΒΟΣ_ΘΕΟΥ, ΕΛΠΙΔΑ, ΚΑΤΗΓΟΡΙΑ, ΔΟΞΟΛΟΓΙΑ, ΕΝΤΟΛΗ, ΕΣΧΑΤΟΛΟΓΙΑ

## Παραδείγματα
{examples_text}

## ΚΡΙΣΙΜΟ: Μορφή εξόδου

Απάντησε ΜΟΝΟ με ένα έγκυρο JSON object. Χωρίς κείμενο πριν ή μετά. Χωρίς markdown. Χωρίς newlines εντός των string τιμών.

{{"tec_category": "μία κατηγορία", "secondary_category": null, "polarity": "Θετικό ή Αρνητικό ή Ουδέτερο", "confidence": 0.85, "justification": "μία σύντομη πρόταση"}}"""


def classify_with_retry(client, system_prompt, text, max_attempts=3):
    """Try up to max_attempts times, with increasing delay."""
    for attempt in range(1, max_attempts + 1):
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

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            result.pop("error", None)   # clear error field on success
            print(f"    ✅ Attempt {attempt} succeeded.")
            return result

        except json.JSONDecodeError as e:
            print(f"    ⚠️  Attempt {attempt} — JSON error: {e}")
            print(f"    Raw response: {raw[:200]}")
        except Exception as e:
            print(f"    ⚠️  Attempt {attempt} — API error: {e}")

        if attempt < max_attempts:
            wait = attempt * 2
            print(f"    Waiting {wait}s before retry...")
            time.sleep(wait)

    return {"error": f"Failed after {max_attempts} attempts", "tec_category": "ERROR", "polarity": "ERROR"}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Set ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=api_key)

    # Load files
    corpus  = pd.read_csv(CORPUS_CSV)
    results = pd.read_csv(RESULTS_CSV)

    # Find error rows
    error_mask = results["error"].notna() & (results["error"] != "") & (results["tec_category"] == "ERROR")
    error_rows = results[error_mask]

    if len(error_rows) == 0:
        print("✅ Δεν υπάρχουν errors στο results CSV. Τίποτα να διορθωθεί.")
        return

    print(f"Βρέθηκαν {len(error_rows)} εγγραφές με error:\n")
    for _, row in error_rows.iterrows():
        print(f"  ID {row['id']} — Λόγος {row.get('logos','?')} §{row.get('section','?')}")
        print(f"  Error: {row['error']}\n")

    # Get original text from corpus for each error row
    corpus_idx = corpus.set_index("id")
    system_prompt = build_system_prompt()

    fixed = 0
    for idx, row in error_rows.iterrows():
        pid = row["id"]
        try:
            text = corpus_idx.loc[pid, "text"]
        except KeyError:
            print(f"⚠️  ID {pid} δεν βρέθηκε στο corpus — παράλειψη.")
            continue

        print(f"Retry: ID {pid} — Λόγος {row.get('logos','?')} §{row.get('section','?')}")
        result = classify_with_retry(client, system_prompt, text)
        time.sleep(SLEEP_BETWEEN)

        # Update the row in results DataFrame
        results.at[idx, "tec_category"]       = result.get("tec_category", "ERROR")
        results.at[idx, "secondary_category"]  = result.get("secondary_category")
        results.at[idx, "polarity"]            = result.get("polarity", "ERROR")
        results.at[idx, "confidence"]          = result.get("confidence")
        results.at[idx, "justification"]       = result.get("justification", "")
        results.at[idx, "error"]               = result.get("error")  # None if success

        if not result.get("error"):
            fixed += 1
            print(f"  → {result.get('tec_category')} | {result.get('polarity')} | conf={result.get('confidence')}")
        else:
            print(f"  → Παραμένει ERROR μετά από 3 προσπάθειες.")

    # Save updated results
    results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"\nΑποθηκεύτηκε: {RESULTS_CSV}")
    print(f"Διορθώθηκαν: {fixed}/{len(error_rows)} εγγραφές.")

    # Remaining errors
    remaining = results[results["tec_category"] == "ERROR"]
    if len(remaining) > 0:
        print(f"\n⚠️  Παραμένουν {len(remaining)} ERROR — χειροκίνητη επέμβαση απαιτείται.")
    else:
        print("\n✅ Όλα τα errors διορθώθηκαν. Έτοιμο για validate_tec_results.py")


if __name__ == "__main__":
    main()
