# THESIS_SUMMARY.md
> **Context document for agents and planning** — rewritten 2026-11-16 after the pre-defense fixes.
> Numbers here are copied from [`CLAUDE.md`](CLAUDE.md); how each claim may be phrased is in
> [`docs/corrected_claims.md`](docs/corrected_claims.md). If anything here disagrees with those two files, they win.
> The previous version (Sept 2026: Macro F1 66.67% on n = 2, BLEU-4 5.14, "human-verified" translation) is obsolete.

---

## Thesis

**DRISHTI-Bn: An Explainable Multimodal Deep Learning Framework for Bengali Disaster Image Captioning and Classification**

Jahid Hasan (ID 0242) · Supervisor: Md. Fahad Hossain · B.Sc. in Software Engineering, Daffodil International University · codename DisasterNet-Bangla

## One-paragraph summary

A ViT-Base/16 image encoder and a BanglaBERT text encoder, both adapted with LoRA (r = 8, α = 16), classify
disaster images into Severe Damage / Humanitarian Rescue / Affected People. On a leakage-free, tweet-grouped
test set (n = 375) the classification-only model reaches **Macro F1 75.69 ± 1.33%** (3 seeds) while training
1.38M parameters (≈ 0.70%) — on par with a ViT-Large baseline that trains 38× more. The classifier is effectively
image-driven: the Bengali text adds almost nothing. Grad-CAM explanations are faithful in a deletion test
(≈ 20× the drop of random patches), and the model is well calibrated (ECE 2.57%). A multi-task variant with a
3-layer caption decoder generates Bengali text, but it imitates tweets rather than describing images. On 325
independently labelled Bangladeshi flood/cyclone photos the classifier scores Macro F1 81.98% (CI 76.1–87.2),
an annotator-dependent, preliminary external check.

---

## Data

| Item | Detail |
|------|--------|
| Source | CrisisMMD v2.0 (Alam et al.), humanitarian-category images |
| Size | 3,755 images: Severe_Damage 2,240 · Humanitarian_Rescue 1,185 · Affected_People 330 (8.8%) |
| Events | 95.2% hurricanes (Harvey, Maria, Irma, US/Caribbean), 4.8% Sri Lanka floods; **no Bangladeshi images** in training |
| Bengali text | Machine translation of the tweets (`monirbishal/en-bn-nmt`), **not human-verified**; 8.9% untranslated, 97.3% contain URLs |
| Split | Tweet/caption-grouped 80/10/10 → train 3,007 / val 373 / test 375 (`results/defense_fixes/phase3/grouped_split.csv`) |
| Old split | Un-grouped (test n = 376): 21.3% of test texts also in train → inflated numbers, superseded |
| External test | 325 Bangladeshi images from Wikimedia Commons, 3 annotators, Fleiss' κ 0.613 (`bd_external/`) |

## Models

| Model | What it is | Trainable |
|-------|-----------|-----------|
| **CLS-Only (main)** | ViT-B/16 + BanglaBERT, LoRA on q/v; concat(ViT pooler, BERT [CLS]) → MLP 1536→512→3 | 1,379,331 (≈ 0.70%) |
| Multi-Task v3 | same encoders + 3-layer Transformer decoder (8 heads) cross-attending to ViT patches; loss = CE + 0.5·NLL | 74.79M |

Training: AdamW, lr 1e-4, weight decay 0.01, cosine schedule, batch 16; CLS-Only 10 epochs (seeds 42/123/2024),
Multi-Task 15 epochs (seed 42); single T4 GPU (Colab). Checkpoints (Drive): `models/cls_only_grouped_s42.pt`,
`models/drishti_multitask_v3.pt`.

---

## Results (grouped test set, n = 375)

**Classification**

| Model | Macro F1 | Trainable |
|-------|----------|-----------|
| **DRISHTI-Bn CLS-Only** (3 seeds) | **75.69 ± 1.33%** | 1.38M |
| CrisisViT-style (ViT-L, our re-implementation, 1 seed) | 75.87% | 51.96M |
| ViT-Only (3 seeds) | 73.27 ± 0.39% | 15.16M |
| Multi-Task v3, image + tweet (1 seed) | 77.29% | 74.79M |
| Multi-Task v3, image + its own generated caption | 73.63% | 74.79M |

Per-class F1 (CLS-Only): SD 93.28 · HR 88.31 · **AP 45.48 ± 4.63** (weakest class).
Text ablation (seed 42): tweet 74.83 · placeholder text 76.52 · another sample's tweet 76.63 → image-driven.

**Explainability** — deletion test, drop in predicted-class probability after removing the top 50% patches:
Grad-CAM 21.27 pp · Attention Rollout 7.35 pp · random 4.16 pp. Decoder cross-attention overlaps Grad-CAM
only at chance level (IoU 0.059 vs 0.054), so it explains caption generation only.

**Calibration** — ECE 2.57% (T = 1); temperature scaling (T* = 1.1349) gives 2.99% → no improvement needed.

**Caption generation (Multi-Task v3)** — sacreBLEU 14.38 / chrF 14.19 on raw tweet references, 0.11 / 8.46
after removing URLs, hashtags and mentions; below a random-tweet baseline. 96.8% of outputs contain URL
fragments. Human evaluation (100 samples, 3 annotators, old v2 captions): model 1.24–1.58 / 5 vs reference
3.08–3.42 / 5; reference preferred in 94.9% of judgements.

**Bangladesh external test (preliminary)** — Macro F1 81.98% (95% CI 76.1–87.2), accuracy 89.85%,
AP F1 58.8; against single annotators 81.65 / 81.36 / 55.82% → label-dependent, not a domain-shift measure.

---

## What can be claimed

- Parameter-efficient: matches a ViT-Large baseline with 38× fewer trainable parameters (not "beats SOTA").
- Faithful visual explanations (Grad-CAM deletion test) and good calibration out of the box.
- Honest, leakage-free evaluation; no performance drop on a small Bangladeshi external set.
- A Bengali text-generation branch exists, but it does **not** produce image descriptions.

Never claim: "human-verified translation", "text and image are complementary", "multi-task improves
classification", "generalizes to Bangladesh", 83.21%, "XAI accuracy 81%".

## Main limitations

Leakage in the original split (fixed) · seed variance ≈ ±1.3 pp · AP class small and weak · machine-translated
tweets as text · classifier ignores text · captions imitate tweets · human eval used old v2 captions ·
ablation only on the old split, one seed · external test small and annotator-dependent.

---

## Key files

| Path | Purpose |
|------|---------|
| `CLAUDE.md` | Approved numbers (source of truth) |
| `docs/corrected_claims.md` | Claim wording, dropped claims, limitations |
| `DEFENSE_FIX_PLAN.md` | Phase-by-phase fix log |
| `docs/defense_qa.md` | Likely defense questions + answers |
| `src/cls_only_ablation.py` · `src/model.py` | CLS-Only and Multi-Task models |
| `colab/phase*.py` | GPU experiments (Phases 0–4) |
| `src/bd_*.py` | Bangladesh external test set |
| `figures/` | Final thesis figures |
| `app.py` | Gradio demo (CLS-Only + Grad-CAM; v3 text shown with a warning) |
