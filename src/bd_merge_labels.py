"""
BD external test set — merge the two labellers' files
=====================================================
Put the two files you received in  bd_external/  (labels_<name>.csv), then run:

    python src/bd_merge_labels.py

  * fills label_a / label_b in bd_external/bd_labels.csv
  * where both agree -> final_label is set automatically
  * prints Cohen's kappa and writes bd_external/disagreements.html showing only the
    images where they disagree, with both answers — decide those together and type
    the agreed code into the final_label column of bd_labels.csv (Excel is fine).
"""
import os, sys, glob, html
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "bd_external")
CSV = os.path.join(D, "bd_labels.csv")

files = sorted(glob.glob(os.path.join(D, "labels_*.csv")))
if len(files) != 2:
    sys.exit(f"Need exactly 2 files bd_external/labels_<name>.csv, found {len(files)}: {files}")
a, b = (pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig") for f in files)
na, nb = a.labeller.iloc[0], b.labeller.iloc[0]

df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
df["label_a"] = df.image_id.map(dict(zip(a.image_id, a.label))).fillna("")
df["label_b"] = df.image_id.map(dict(zip(b.image_id, b.label))).fillna("")
agree = (df.label_a != "") & (df.label_a == df.label_b)
df.loc[agree & (df.final_label == ""), "final_label"] = df.label_a
df.to_csv(CSV, index=False, encoding="utf-8-sig")

both = df[(df.label_a != "") & (df.label_b != "")]
from sklearn.metrics import cohen_kappa_score
print(f"Labeller A = {na} ({(df.label_a != '').sum()} labelled) | Labeller B = {nb} ({(df.label_b != '').sum()} labelled)")
print(f"Agreement: {(both.label_a == both.label_b).mean() * 100:.1f}% | Cohen's kappa = "
      f"{cohen_kappa_score(both.label_a, both.label_b):.3f} (n = {len(both)})")
dis = df[(df.label_a != "") & (df.label_b != "") & (df.label_a != df.label_b) & (df.final_label == "")]
print(f"Disagreements still to resolve: {len(dis)}")
cards = "\n".join(
    f'<div class="c"><img src="../data/bd_external/images/{r.file}"><b>{r.image_id}</b>'
    f'<span>{html.escape(na)}: <b>{r.label_a}</b> · {html.escape(nb)}: <b>{r.label_b}</b></span></div>'
    for r in dis.itertuples())
open(os.path.join(D, "disagreements.html"), "w", encoding="utf-8").write(
    f"""<!doctype html><meta charset="utf-8"><title>মতভেদ</title><style>body{{font-family:sans-serif;margin:16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}.c{{border:1px solid #ccc;padding:6px}}
img{{width:100%;height:220px;object-fit:contain;background:#eee}}</style>
<h2>{len(dis)}টা ছবিতে মতভেদ — একসঙ্গে দেখে bd_labels.csv-এর final_label-এ SD/HR/AP/EX লিখুন</h2>
<div class="g">{cards}</div>""")
print(f"[+] Updated {CSV}\n[+] Open bd_external/disagreements.html to resolve the disagreements")
