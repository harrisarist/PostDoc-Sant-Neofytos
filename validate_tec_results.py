"""
validate_tec_results.py
========================
Phase B — Validation of TEC Classification Output
Postdoctoral Research: Computational Emotional Analysis
Saint Neophytos the Recluse — Δέκα Λόγοι περί Χριστού Εντολών

Reads neophytos_corpus.csv and neophytos_tec_results.csv,
runs a full suite of validation checks, and prints a report.

Usage (from Claude Code / WSL):
    python3 validate_tec_results.py

Requirements:
    pip install pandas --break-system-packages
"""

import sys
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
CORPUS_CSV  = "neophytos_corpus.csv"
RESULTS_CSV = "neophytos_tec_results.csv"

VALID_TEC = {
    "ΑΓΑΠΗ", "ΜΕΤΑΝΟΙΑ", "ΦΟΒΟΣ_ΘΕΟΥ",
    "ΕΛΠΙΔΑ", "ΚΑΤΗΓΟΡΙΑ", "ΔΟΞΟΛΟΓΙΑ",
    "ΕΝΤΟΛΗ", "ΕΣΧΑΤΟΛΟΓΙΑ", "ERROR"
}
VALID_POLARITY  = {"Θετικό", "Αρνητικό", "Ουδέτερο", "ERROR"}
CONFIDENCE_MIN  = 0.0
CONFIDENCE_MAX  = 1.0
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅"
WARN = "⚠️ "
FAIL = "❌"


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(label: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    msg = f"  {icon}  {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    return ok


def main():
    all_ok = True

    # ── Load files ────────────────────────────────────────────────────────────
    section("1. Φόρτωση αρχείων")
    try:
        corpus = pd.read_csv(CORPUS_CSV)
        print(f"  {PASS}  Corpus: {len(corpus)} εγγραφές")
    except FileNotFoundError:
        print(f"  {FAIL}  {CORPUS_CSV} δεν βρέθηκε — βεβαιώσου ότι τρέχεις από τον σωστό φάκελο.")
        sys.exit(1)

    try:
        results = pd.read_csv(RESULTS_CSV)
        print(f"  {PASS}  Results: {len(results)} εγγραφές")
    except FileNotFoundError:
        print(f"  {FAIL}  {RESULTS_CSV} δεν βρέθηκε — έχει ολοκληρωθεί το tec_classify.py;")
        sys.exit(1)

    # ── Κανονικοποίηση ────────────────────────────────────────────────────────
    corpus_complete = corpus[corpus["incomplete"] == False].copy()
    expected_ids    = set(corpus_complete["id"].tolist())
    result_ids      = set(results["id"].tolist())

    # ── 2. Πληρότητα ──────────────────────────────────────────────────────────
    section("2. Πληρότητα — Αριθμός Εγγραφών")

    expected_n = len(expected_ids)
    result_n   = len(results)

    ok = check(
        f"Σύνολο εγγραφών: {result_n} / {expected_n} αναμενόμενες",
        result_n == expected_n,
        "" if result_n == expected_n else f"Διαφορά: {expected_n - result_n} παράγραφοι λείπουν"
    )
    all_ok = all_ok and ok

    missing = expected_ids - result_ids
    if missing:
        print(f"       Λείπουν IDs: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}")

    extra = result_ids - expected_ids
    ok2 = check(
        "Δεν υπάρχουν επιπλέον IDs (εκτός corpus)",
        not extra,
        f"Επιπλέον IDs: {sorted(extra)}" if extra else ""
    )
    all_ok = all_ok and ok2

    duplicates = results[results.duplicated("id", keep=False)]
    ok3 = check(
        "Δεν υπάρχουν διπλές εγγραφές",
        len(duplicates) == 0,
        f"{len(duplicates)} διπλές εγγραφές" if len(duplicates) > 0 else ""
    )
    all_ok = all_ok and ok3

    # ── 3. Έλεγχος TEC κατηγοριών ─────────────────────────────────────────────
    section("3. Έγκυρες TEC Κατηγορίες")

    invalid_tec = results[~results["tec_category"].isin(VALID_TEC)]
    ok = check(
        f"Όλες οι tec_category είναι έγκυρες ({len(results) - len(invalid_tec)}/{len(results)})",
        len(invalid_tec) == 0,
        f"Μη έγκυρες τιμές: {invalid_tec['tec_category'].unique().tolist()}" if len(invalid_tec) > 0 else ""
    )
    all_ok = all_ok and ok

    # Έλεγχος secondary_category (null επιτρέπεται)
    sec = results["secondary_category"].dropna()
    invalid_sec = sec[~sec.isin(VALID_TEC)]
    ok2 = check(
        f"Έγκυρες secondary_category (null = OK)",
        len(invalid_sec) == 0,
        f"Μη έγκυρες: {invalid_sec.unique().tolist()}" if len(invalid_sec) > 0 else ""
    )
    all_ok = all_ok and ok2

    # ── 4. Πολικότητα ─────────────────────────────────────────────────────────
    section("4. Έγκυρη Πολικότητα")

    invalid_pol = results[~results["polarity"].isin(VALID_POLARITY)]
    ok = check(
        f"Όλες οι polarity είναι έγκυρες ({len(results) - len(invalid_pol)}/{len(results)})",
        len(invalid_pol) == 0,
        f"Μη έγκυρες: {invalid_pol['polarity'].unique().tolist()}" if len(invalid_pol) > 0 else ""
    )
    all_ok = all_ok and ok

    # ── 5. Confidence scores ───────────────────────────────────────────────────
    section("5. Confidence Scores")

    conf = pd.to_numeric(results["confidence"], errors="coerce")
    null_conf = conf.isna().sum()

    ok = check(
        f"Null confidence: {null_conf}/{len(results)}",
        null_conf == 0,
        "Παράγραφοι χωρίς confidence score" if null_conf > 0 else ""
    )
    all_ok = all_ok and ok

    out_of_range = conf[(conf < CONFIDENCE_MIN) | (conf > CONFIDENCE_MAX)].count()
    ok2 = check(
        f"Confidence εκτός [0,1]: {out_of_range}",
        out_of_range == 0,
    )
    all_ok = all_ok and ok2

    low_conf = conf[conf < 0.5].count()
    if low_conf > 0:
        pct = low_conf / len(results) * 100
        print(f"  {WARN}  Χαμηλό confidence (<0.5): {low_conf} παράγραφοι ({pct:.1f}%)")
        print(f"       → Αυτές χρήζουν προτεραιότητας στην ανθρώπινη επαλήθευση (Φάση Γ΄).")

    # ── 6. Σφάλματα API ───────────────────────────────────────────────────────
    section("6. Σφάλματα API")

    errors = results[results["error"].notna() & (results["error"] != "")]
    ok = check(
        f"Σφάλματα: {len(errors)}/{len(results)}",
        len(errors) == 0,
        ""
    )
    all_ok = all_ok and ok

    if len(errors) > 0:
        print(f"\n  Εγγραφές με σφάλματα:")
        for _, row in errors.iterrows():
            print(f"    ID {row['id']} ({row.get('logos','?')} {row.get('section','?')}): {row['error']}")

    # ── 7. Αδειες αιτιολογήσεις ───────────────────────────────────────────────
    section("7. Αιτιολογήσεις")

    empty_just = results[results["justification"].isna() | (results["justification"].str.strip() == "")]
    ok = check(
        f"Κενές αιτιολογήσεις: {len(empty_just)}/{len(results)}",
        len(empty_just) == 0,
        f"IDs: {empty_just['id'].tolist()[:10]}" if len(empty_just) > 0 else ""
    )
    all_ok = all_ok and ok

    # ── 8. Κατανομή TEC ───────────────────────────────────────────────────────
    section("8. Κατανομή TEC Κατηγοριών")

    tec_dist = results["tec_category"].value_counts()
    print()
    for cat, count in tec_dist.items():
        pct  = count / len(results) * 100
        bar  = "█" * int(pct / 2)
        print(f"  {cat:<20} {count:>4}  ({pct:5.1f}%)  {bar}")

    # Έλεγχος αν κάποια κατηγορία απουσιάζει τελείως
    missing_cats = (VALID_TEC - {"ERROR"}) - set(tec_dist.index)
    if missing_cats:
        print(f"\n  {WARN}  Κατηγορίες με μηδέν εγγραφές: {missing_cats}")
        print(f"       → Εξέτασε αν τα few-shot παραδείγματα καλύπτουν επαρκώς αυτές τις κατηγορίες.")

    # Έλεγχος class imbalance (>40% σε μία κατηγορία)
    dominant = tec_dist[tec_dist / len(results) > 0.40]
    if not dominant.empty:
        for cat, n in dominant.items():
            print(f"\n  {WARN}  Υπερεκπροσώπηση: {cat} = {n} ({n/len(results)*100:.1f}%)")
            print(f"       → Εξέτασε αν το few-shot prompt είναι μεροληπτικό προς αυτήν.")

    # ── 9. Κατανομή ανά Λόγο ─────────────────────────────────────────────────
    section("9. Κατανομή TEC ανά Λόγο (σύνοψη)")

    if "logos" in results.columns:
        logos_order = ["Γ", "Δ", "Ε", "ΣΤ", "Ζ", "Η", "Θ", "Ι"]
        print()
        for logos in logos_order:
            subset = results[results["logos"] == logos]
            if len(subset) == 0:
                continue
            top = subset["tec_category"].value_counts().head(2)
            top_str = ", ".join([f"{k}({v})" for k, v in top.items()])
            print(f"  Λόγος {logos:<4}  {len(subset):>3} παρ.  → {top_str}")

    # ── 10. Confidence ανά TEC κατηγορία ─────────────────────────────────────
    section("10. Μέσο Confidence ανά TEC Κατηγορία")

    conf_by_tec = results.groupby("tec_category")["confidence"].mean().sort_values()
    print()
    for cat, mean_conf in conf_by_tec.items():
        flag = f"  {WARN} χαμηλό" if mean_conf < 0.6 else ""
        print(f"  {cat:<20}  μέσο confidence: {mean_conf:.3f}{flag}")

    # ── Τελική κρίση ──────────────────────────────────────────────────────────
    section("ΤΕΛΙΚΗ ΑΞΙΟΛΟΓΗΣΗ")

    if all_ok:
        print(f"  {PASS}  Όλοι οι έλεγχοι πέρασαν.")
        print(f"       Το neophytos_tec_results.csv είναι έτοιμο για Φάση Γ΄ (expert annotation).")
    else:
        print(f"  {FAIL}  Υπάρχουν προβλήματα που χρειάζονται διόρθωση πριν τη Φάση Γ΄.")
        print(f"       Διόρθωσε τα ❌ παραπάνω και ξανατρέξε το script.")

    print()


if __name__ == "__main__":
    main()
