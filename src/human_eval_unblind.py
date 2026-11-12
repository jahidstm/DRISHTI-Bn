"""
Human evaluation — corrected un-blinding (DEFENSE_FIX_PLAN Phase 1)
===================================================================
Bug in the original pipeline:
  build_annotation_tool.py randomly swapped GT / model captions between
  positions A and B for every sample (per annotator), but the exported
  responses carry no record of the swap, and merge_and_run_iaa.py assumed
  A = model, B = ground truth for every row. Reported "model" and "GT"
  scores were therefore ~50/50 mixtures of both, and IAA compared
  ratings of different captions across annotators.

Fix:
  The true assignment ("hA": "gt" | "model") is embedded in each
  annotator's HTML tool. This script reads it, verifies it against the
  caption text in the responses, maps every score to model / GT, and
  recomputes means, preferences and IAA with the same formulas as
  merge_and_run_iaa.py. Original files are left untouched.

Usage (local, no GPU):  python src/human_eval_unblind.py
"""
import os, re, json, sys
from itertools import combinations
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR  = os.path.join(ROOT, "human_eval_outputs")
OUT_DIR = os.path.join(ROOT, "results", "defense_fixes", "phase1", "human_eval_unblinded")
os.makedirs(OUT_DIR, exist_ok=True)
ANNOTATORS = ["Annotator_1", "Annotator_2", "Annotator_3"]
DIMS       = ["fluency", "adequacy", "faithfulness"]
CLASSES    = ["Severe_Damage", "Humanitarian_Rescue", "Affected_People"]


# ── IAA formulas: identical to merge_and_run_iaa.py ─────────────────────────
def krippendorff_alpha(data):
    data = np.array(data, dtype=float)
    D_o, n_pairable = 0.0, 0
    for u in range(data.shape[1]):
        vals = data[:, u][~np.isnan(data[:, u])]
        for i, j in combinations(range(len(vals)), 2):
            D_o += (vals[i] - vals[j]) ** 2; n_pairable += 1
    if n_pairable == 0: return float("nan")
    D_o /= n_pairable
    all_vals = data.flatten(); all_vals = all_vals[~np.isnan(all_vals)]
    diffs = all_vals[:, None] - all_vals[None, :]
    iu = np.triu_indices(len(all_vals), k=1)
    D_e = float(np.mean(diffs[iu] ** 2))
    if D_e == 0: return 1.0 if D_o == 0 else float("nan")
    return 1.0 - D_o / D_e

def cohen_kappa(r1, r2):
    r1, r2 = np.array(r1, dtype=float), np.array(r2, dtype=float)
    m = ~(np.isnan(r1) | np.isnan(r2)); r1, r2 = r1[m], r2[m]
    if len(r1) == 0: return float("nan")
    cats = sorted(set(r1) | set(r2)); n = len(r1)
    p_o = np.mean(r1 == r2)
    p_e = sum((np.sum(r1 == c) / n) * (np.sum(r2 == c) / n) for c in cats)
    if p_e == 1.0: return 1.0 if p_o == 1.0 else float("nan")
    return (p_o - p_e) / (1.0 - p_e)

def weighted_kappa(r1, r2):
    """Quadratic-weighted Cohen's kappa (standard for ordinal 1-5 scales)."""
    from sklearn.metrics import cohen_kappa_score
    r1, r2 = np.array(r1, dtype=float), np.array(r2, dtype=float)
    m = ~(np.isnan(r1) | np.isnan(r2))
    return cohen_kappa_score(r1[m].astype(int), r2[m].astype(int), weights="quadratic")


# ── 1. Read true A/B assignment from each HTML tool ─────────────────────────
def read_assignment(ann_id):
    html = open(os.path.join(IN_DIR, f"annotation_tool_{ann_id}.html"), encoding="utf-8").read()
    m = re.search(r"const S=(\[.*?\]);\s*\n", html, re.S)
    if not m:
        m = re.search(r"const S=(\[.*?\]);", html, re.S)
    samples = json.loads(m.group(1))
    return pd.DataFrame([{"sample_id": s["sid"], "hA": s["hA"], "hB": s["hB"],
                          "cA_html": s["cA"], "cB_html": s["cB"]} for s in samples])

long_rows, pref_rows = [], []
for k, ann_id in enumerate(ANNOTATORS, start=1):
    resp = pd.read_csv(os.path.join(IN_DIR, f"annotator_{k}_responses.csv"), encoding="utf-8-sig")
    asg  = read_assignment(ann_id)
    df   = resp.merge(asg, on="sample_id", how="left", validate="one_to_one")
    assert df.hA.notna().all(), f"{ann_id}: samples missing from HTML"
    # Verify: caption text in the response must equal the HTML caption at the same position
    norm = lambda s: "" if pd.isna(s) else re.sub(r"\s+", " ", str(s)).strip()
    mism = ((df.caption_A.map(norm) != df.cA_html.map(norm)) |
            (df.caption_B.map(norm) != df.cB_html.map(norm))).sum()
    assert mism == 0, f"{ann_id}: {mism} caption mismatches - assignment not trustworthy"
    n_empty = ((df.hA == "gt") & (df.cA_html.map(norm) == "") |
               (df.hB == "gt") & (df.cB_html.map(norm) == "")).sum()
    if n_empty: print(f"    note: {n_empty} sample(s) with an empty GT caption")
    n_swap = (df.hA == "model").sum()
    print(f"[+] {ann_id}: {len(df)} rows | caption text verified | model shown as A in {n_swap}/{len(df)}")

    for _, r in df.iterrows():
        for pos in ["A", "B"]:
            src = r[f"h{pos}"]            # 'gt' or 'model'
            row = {"annotator_id": ann_id, "sample_id": r.sample_id,
                   "class_label": r.class_label, "source": src}
            for d in DIMS:
                row[d] = pd.to_numeric(r[f"{d}_{pos}"], errors="coerce")
            long_rows.append(row)
        p = str(r.preference).strip()
        pref = {"A": r.hA, "B": r.hB}.get(p, "equal" if p.lower().startswith("eq") else "missing")
        pref_rows.append({"annotator_id": ann_id, "sample_id": r.sample_id,
                          "class_label": r.class_label, "preferred": pref})

long_df = pd.DataFrame(long_rows)
pref_df = pd.DataFrame(pref_rows)
long_df.to_csv(os.path.join(OUT_DIR, "scores_long_unblinded.csv"), index=False, encoding="utf-8-sig")


# ── 2. Means: model vs GT (overall + per class) ─────────────────────────────
summary = []
for cls in CLASSES + ["Overall"]:
    sub = long_df if cls == "Overall" else long_df[long_df.class_label == cls]
    row = {"class": cls, "n_samples": sub.sample_id.nunique()}
    for src in ["model", "gt"]:
        for d in DIMS:
            row[f"{src}_{d}"] = round(sub[sub.source == src][d].mean(), 3)
    summary.append(row)
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_DIR, "human_eval_summary_unblinded.csv"), index=False, encoding="utf-8-sig")

# Paired significance test (per sample, averaged over annotators)
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None
paired = long_df.groupby(["sample_id", "source"])[DIMS].mean().unstack("source")
tests = []
for d in DIMS:
    m, g = paired[(d, "model")], paired[(d, "gt")]
    p = wilcoxon(m, g).pvalue if wilcoxon else float("nan")
    tests.append({"dimension": d, "model_mean": round(m.mean(), 3), "gt_mean": round(g.mean(), 3),
                  "diff_gt_minus_model": round((g - m).mean(), 3), "wilcoxon_p": p})
tests_df = pd.DataFrame(tests)
tests_df.to_csv(os.path.join(OUT_DIR, "model_vs_gt_tests.csv"), index=False, encoding="utf-8-sig")


# ── 3. IAA on correctly aligned ratings ─────────────────────────────────────
iaa = []
for src in ["model", "gt"]:
    for d in DIMS:
        piv = long_df[long_df.source == src].pivot_table(
            index="annotator_id", columns="sample_id", values=d, aggfunc="first")
        alpha = krippendorff_alpha(piv.values)
        pairs = list(combinations(piv.index, 2))
        kap  = np.nanmean([cohen_kappa(piv.loc[a].values, piv.loc[b].values) for a, b in pairs])
        wkap = np.nanmean([weighted_kappa(piv.loc[a].values, piv.loc[b].values) for a, b in pairs])
        iaa.append({"caption": src, "dimension": d, "krippendorff_alpha": round(alpha, 4),
                    "mean_cohen_kappa": round(kap, 4), "mean_weighted_kappa": round(wkap, 4)})
# Pooled: model + GT captions together (200 items) — do annotators agree on
# which captions are good vs bad? Within-condition IAA is depressed because
# model scores cluster near 1 (little variance to agree on).
for d in DIMS:
    tmp = long_df.assign(item=long_df.sample_id + "_" + long_df.source)
    piv = tmp.pivot_table(index="annotator_id", columns="item", values=d, aggfunc="first")
    pairs = list(combinations(piv.index, 2))
    iaa.append({"caption": "pooled", "dimension": d,
                "krippendorff_alpha": round(krippendorff_alpha(piv.values), 4),
                "mean_cohen_kappa": round(np.nanmean([cohen_kappa(piv.loc[a].values, piv.loc[b].values) for a, b in pairs]), 4),
                "mean_weighted_kappa": round(np.nanmean([weighted_kappa(piv.loc[a].values, piv.loc[b].values) for a, b in pairs]), 4)})
iaa_df = pd.DataFrame(iaa)
iaa_df.to_csv(os.path.join(OUT_DIR, "iaa_scores_unblinded.csv"), index=False, encoding="utf-8-sig")


# ── 4. Preference ───────────────────────────────────────────────────────────
pref_counts = pref_df.preferred.value_counts().reindex(["model", "gt", "equal", "missing"], fill_value=0)
pref_counts.to_csv(os.path.join(OUT_DIR, "preference_unblinded.csv"), encoding="utf-8-sig")


# ── 5. Print report ─────────────────────────────────────────────────────────
pd.set_option("display.width", 140)
print("\n=== Mean scores (1-5), correctly un-blinded ===")
print(summary_df.to_string(index=False))
print("\n=== Model vs GT, paired per sample (Wilcoxon signed-rank) ===")
print(tests_df.to_string(index=False))
print("\n=== Inter-annotator agreement (aligned) ===")
print(iaa_df.to_string(index=False))
within = iaa_df[iaa_df.caption != "pooled"]
pooled = iaa_df[iaa_df.caption == "pooled"]
print(f"\n  Within-condition mean: alpha = {within.krippendorff_alpha.mean():.3f} | "
      f"kappa = {within.mean_cohen_kappa.mean():.3f} | weighted kappa = {within.mean_weighted_kappa.mean():.3f}")
print(f"  Pooled (model+GT)   : alpha = {pooled.krippendorff_alpha.mean():.3f} | "
      f"kappa = {pooled.mean_cohen_kappa.mean():.3f} | weighted kappa = {pooled.mean_weighted_kappa.mean():.3f}")
print("\n=== Preference (300 judgements) ===")
print(pref_counts.to_string())
print("\n=== Preference per annotator ===")
print(pd.crosstab(pref_df.annotator_id, pref_df.preferred).to_string())
print("\n=== Per-annotator means (model vs GT) ===")
print(long_df.groupby(["annotator_id", "source"])[DIMS].mean().round(2).unstack("source").to_string())
print(f"\n[+] Outputs saved in {OUT_DIR}")
