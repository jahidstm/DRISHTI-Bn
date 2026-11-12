"""
Phase 3B.6 — standard caption metrics for the multi-task v3 model
=================================================================
`metrics.py` computes BLEU as the mean of smoothed sentence-level scores,
which makes BLEU-1..4 nearly identical when overlap is low. This script
recomputes the caption metrics with standard tools on the saved outputs
(no GPU, no model needed):

  * corpus BLEU (sacreBLEU, 13a tokenisation) and BLEU-1..4 precisions
  * chrF and chrF++ (character n-grams; better suited to Bengali morphology)
  * the same after removing tweet boilerplate (URLs, RT, @mentions, '#')
  * diversity: unique captions, distinct-1/2, most frequent caption share
  * two reference points on the same test references:
      - random train tweet  (seeded): what unrelated in-domain text scores
      - constant caption    : most frequent train tweet for every sample

Input : results/defense_fixes/phase3b/phase3b_per_sample.csv
        results/defense_fixes/phase3/grouped_split.csv
        data/processed/master_dataset_translated.csv
Output: results/defense_fixes/phase3b/phase3b_caption_metrics_standard.csv

Usage : python src/caption_metrics_v3.py
"""
import os, re, sys
import numpy as np
import pandas as pd
import sacrebleu

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3B  = os.path.join(ROOT, "results", "defense_fixes", "phase3b")
per   = pd.read_csv(os.path.join(P3B, "phase3b_per_sample.csv"), keep_default_na=False)
split = pd.read_csv(os.path.join(ROOT, "results", "defense_fixes", "phase3", "grouped_split.csv"))
data  = pd.read_csv(os.path.join(ROOT, "data", "processed", "master_dataset_translated.csv"))
assert len(per) == 375 and (split.image_path.values == data.image_path.values).all()

URL = re.compile(r"h\s*t\s*t\s*p\s*s?\s*:\s*/\s*/\s*\S+|https?\s*:?\s*/?\s*/?\s*t\s*\.\s*co\s*/\s*\S+|t\s*\.\s*co\s*/\s*\S+|https?")
def clean(s):
    s = URL.sub(" ", str(s))
    s = re.sub(r"\bRT\b|আরটি", " ", s)            # retweet marker (also its transliteration)
    s = re.sub(r"@\s*\w+", " ", s)                # @mentions, incl. "@ name"
    s = s.replace("#", " ")
    return re.sub(r"\s+", " ", s).strip()

refs = per.reference.astype(str).tolist()
hyps = per.generated.astype(str).tolist()

train_caps = data.bengali_caption.fillna("").astype(str).values[split.split.values == "train"]
rng = np.random.default_rng(42)
rand_hyps  = list(rng.choice(train_caps[train_caps != ""], size=len(refs), replace=True))
const_hyps = [pd.Series(train_caps[train_caps != ""]).value_counts().index[0]] * len(refs)

def scores(h, r):
    b = sacrebleu.corpus_bleu(h, [r], tokenize="13a")
    return dict(bleu=b.score, bleu1=b.precisions[0], bleu2=b.precisions[1], bleu3=b.precisions[2],
                bleu4=b.precisions[3], bp=b.bp,
                chrf=sacrebleu.corpus_chrf(h, [r]).score,
                chrfpp=sacrebleu.corpus_chrf(h, [r], word_order=2).score)

rows = []
for system, h in [("DRISHTI-Bn v3 (beam 3)", hyps), ("Random train tweet", rand_hyps),
                  ("Constant (most frequent train tweet)", const_hyps)]:
    for cond, fh, fr in [("raw", h, refs), ("clean (no URL/RT/@/#)", [clean(x) for x in h], [clean(x) for x in refs])]:
        rows.append(dict(system=system, references=cond, **scores(fh, fr)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(P3B, "phase3b_caption_metrics_standard.csv"), index=False, encoding="utf-8-sig")

def diversity(caps):
    toks = [c.split() for c in caps]
    uni = [t for s in toks for t in s]
    bi  = [tuple(s[i:i + 2]) for s in toks for i in range(len(s) - 1)]
    vc  = pd.Series(caps).value_counts()
    return dict(unique=f"{len(vc)}/{len(caps)}", top1_share=round(vc.iloc[0] / len(caps) * 100, 1),
                distinct1=round(len(set(uni)) / max(len(uni), 1), 3),
                distinct2=round(len(set(bi)) / max(len(bi), 1), 3),
                mean_words=round(np.mean([len(t) for t in toks]), 1), top1=vc.index[0])

pd.set_option("display.width", 160)
print(f"sacreBLEU {sacrebleu.__version__} | test n = {len(refs)}\n")
print(res.round(2).to_string(index=False))
print("\n=== Diversity ===")
for name, caps in [("generated raw", hyps), ("generated clean", [clean(x) for x in hyps]),
                   ("references clean", [clean(x) for x in refs])]:
    print(f"  {name:<17}", diversity(caps))
boiler = lambda pat: np.mean([bool(re.search(pat, x)) for x in hyps]) * 100
print("\n=== Tweet boilerplate in generated captions ===")
print(f"  URL fragment {boiler(r't\s*\.\s*co|https'):.1f}% | RT/আরটি {boiler(r'RT|আরটি'):.1f}% | "
      f"@mention {boiler(r'@'):.1f}% | hashtag {boiler(r'#'):.1f}%")
ev = {"Harvey": r"Harvey|হার্ভি|হ্যারাভি|হারভে", "Maria": r"Maria|মারিয়া", "Irma": r"Irma|ইরমা|ইরামা"}
n_ev = [sum(bool(re.search(p, x)) for p in ev.values()) for x in hyps]
print(f"  captions naming >=2 different hurricanes: {np.mean([n >= 2 for n in n_ev]) * 100:.1f}%")
print(f"\n[+] Saved {os.path.join(P3B, 'phase3b_caption_metrics_standard.csv')}")
