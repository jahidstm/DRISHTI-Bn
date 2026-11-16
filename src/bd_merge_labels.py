"""
BD external test set — merge the labellers' files (2 or more labellers)
======================================================================
Put the files you received in  bd_external/  (labels_<name>.csv), then run:

    python src/bd_merge_labels.py

  * writes one column per labeller into bd_external/bd_labels.csv (label_a, label_b, label_c, ...)
  * final_label = majority vote (a label chosen by more than half of the labellers);
    with 2 labellers that means both must agree
  * agreement: pairwise Cohen's kappa, Fleiss' kappa, per-class counts per labeller
  * bd_external/disagreements.html shows only the images WITHOUT a majority, with
    everybody's answer — decide those together and type the code into final_label
  * an existing non-empty final_label is never overwritten
"""
import os, sys, glob, html, string
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "bd_external")
CSV = os.path.join(D, "bd_labels.csv")
CATS = ["SD", "HR", "AP", "EX"]


def fleiss_kappa(table):
    """table: items x categories counts, same number of raters per item."""
    n = table.sum(1)[0]
    p_j = table.sum(0) / table.sum()
    P_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e)


files = sorted(glob.glob(os.path.join(D, "labels_*.csv")))
if len(files) < 2:
    sys.exit(f"Need at least 2 files bd_external/labels_<name>.csv, found {len(files)}")
df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
cols, names = [], []
for k, f in enumerate(files):
    lab = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    col = f"label_{string.ascii_lowercase[k]}"
    df[col] = df.image_id.map(dict(zip(lab.image_id, lab.label.str.strip().str.upper()))).fillna("")
    cols.append(col); names.append(lab.labeller.iloc[0])
    bad = set(df[col]) - set(CATS) - {""}
    if bad: sys.exit(f"{f}: unknown codes {bad}")

# majority vote
def majority(row):
    votes = [row[c] for c in cols if row[c]]
    if len(votes) < len(cols): return ""
    vc = pd.Series(votes).value_counts()
    return vc.index[0] if vc.iloc[0] > len(cols) / 2 else ""
maj = df.apply(majority, axis=1)
fill = (df.final_label == "") & (maj != "")
df.loc[fill, "final_label"] = maj[fill]
# put the per-labeller columns right after the metadata
front = [c for c in df.columns if c not in cols + ["final_label", "description_bn", "notes"]]
df = df[front + cols + ["final_label", "description_bn", "notes"]]
df.to_csv(CSV, index=False, encoding="utf-8-sig")

full = df[(df[cols] != "").all(axis=1)]
print(f"Labellers: " + " | ".join(f"{c[-1].upper()} = {n}" for c, n in zip(cols, names)))
print(f"Images labelled by everyone: {len(full)} / {len(df)}\n")
print("Label counts per labeller:")
print(pd.DataFrame({n: full[c].value_counts().reindex(CATS, fill_value=0) for c, n in zip(cols, names)}).T.to_string())
print("\nPairwise agreement:")
for (c1, n1), (c2, n2) in combinations(zip(cols, names), 2):
    print(f"  {n1} vs {n2}: {(full[c1] == full[c2]).mean() * 100:.1f}% | Cohen's kappa = "
          f"{cohen_kappa_score(full[c1], full[c2]):.3f}")
if len(cols) >= 3:
    table = np.stack([(full[cols] == cat).sum(axis=1).values for cat in CATS], 1)
    print(f"Fleiss' kappa (all {len(cols)} labellers): {fleiss_kappa(table):.3f}")
    unanimous = (full[cols].nunique(axis=1) == 1).mean() * 100
    print(f"Unanimous: {unanimous:.1f}% | majority exists: {(maj[full.index] != '').mean() * 100:.1f}%")

dis = df[(df.final_label == "") & (df[cols] != "").all(axis=1)]
print(f"\nfinal_label set: {(df.final_label != '').sum()} | still to decide together: {len(dis)}")
print("final_label counts: " + str(df.final_label.replace("", "(empty)").value_counts().to_dict()))
cards = "\n".join(
    f'<div class="c"><img src="../data/bd_external/images/{r.file}"><b>{r.image_id}</b><span>'
    + " · ".join(f"{html.escape(n)}: <b>{getattr(r, c)}</b>" for c, n in zip(cols, names))
    + "</span></div>" for r in dis.itertuples())
open(os.path.join(D, "disagreements.html"), "w", encoding="utf-8").write(
    f"""<!doctype html><meta charset="utf-8"><title>মতভেদ</title><style>body{{font-family:sans-serif;margin:16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}.c{{border:1px solid #ccc;padding:6px}}
img{{width:100%;height:220px;object-fit:contain;background:#eee}}span{{display:block;font-size:13px}}</style>
<h2>{len(dis)}টা ছবিতে সংখ্যাগরিষ্ঠ মত নেই — একসঙ্গে দেখে bd_labels.csv-এর final_label-এ SD/HR/AP/EX লিখুন</h2>
<div class="g">{cards}</div>""")
print(f"[+] Updated {CSV}\n[+] Open bd_external/disagreements.html for the images without a majority")
