# DRISHTI-Bn Thesis — Claude Code Context

## Project Identity
- **Thesis:** "DRISHTI-Bn: An Explainable Multimodal Deep Learning Framework for Bengali Disaster Image Captioning and Classification"
- **Student:** Jahid Hasan | ID: 0242 | DIU SWE
- **Supervisor:** Md. Fahad Hossain

## Response Rules
- Explain in **Bengali**, code/technical content in **English**
- Be concise — minimize token usage
- Max 1–3 `[REF]` per paragraph (APA format, NOT after every sentence)
- Figure captions: **BELOW** images | Table captions: **ABOVE** tables

## Formatting (DIU Guidelines)
- Font: Times New Roman 12pt, 1.5 line spacing, 4cm left margin
- Writing tool: Google Docs (DIU Template)
- Citations: APA `(Author, Year)` — NOT IEEE `[1]` style

---

## Actual Experiment Results (NEVER change these numbers)

> Numbers below were updated on 2026-11-12 with the user's approval (DEFENSE_FIX_PLAN decisions 3.5 and 3B.5).
> Main results use the **leakage-free grouped split** (no tweet / identical caption shared across train/val/test;
> train 3007 / val 373 / **test 375**). How to phrase each claim: `docs/corrected_claims.md`.
> Old numbers from the original un-grouped split (test n=376) were inflated by leakage and are listed only as "superseded".

### Classification — main result (grouped split, test n=375)
| Model | Seeds | Macro F1 | Precision | Recall | Trainable |
|-------|-------|----------|-----------|--------|-----------|
| **DRISHTI-Bn CLS-Only (LoRA r=8)** | 42, 123, 2024 | **75.69 ± 1.33%** | 77.09 ± 1.31% | 74.93 ± 1.59% | 1.38M (≈0.70%) |
| Multi-Task v3, image + original tweet | 42 | 77.29% | 79.03% | 76.12% | 74.79M |
| Multi-Task v3, image + generated caption (image-only) | 42 | 73.63% | 76.77% | 72.12% | 74.79M |

**CLS-Only per-class F1 (mean of 3 seeds):** Severe_Damage=93.28% | Humanitarian_Rescue=88.31% | Affected_People=45.48 ± 4.63%

**Multi-Task v3 per-class F1:** (tweet) SD 92.00 / HR 87.24 / AP 52.63 · (generated caption) SD 92.51 / HR 86.07 / AP 42.31

**Multi-Task v3 confusion (rows SD, HR, AP):** tweet [207,10,7] [10,106,3] [9,8,15] · generated [210,10,4] [9,105,5] [11,10,11]

Multi-task 77.29% is a single seed within CLS-Only seed variance — do NOT claim multi-task improves classification.
Image-only 73.63% ≈ ViT-Only — the generated caption adds no information beyond the image.

Superseded (original split, leakage): CLS-Only 83.21% · Multi-Task v2 77.95% (P 93.28, R 73.03).

### Caption Generation (Multi-Task v3, grouped test n=375, Beam Width=3, sacreBLEU 2.6.0)
| Metric | Raw references | Clean (no URL/RT/@/#) |
|--------|----------------|------------------------|
| Corpus BLEU | 14.38 | 0.11 |
| chrF | 14.19 | 8.46 |
| BERTScore-F1 (mBERT) | 69.66% | **62.27%** (URLs removed) |

Reference points (same refs, chrF raw / clean): random train tweet 21.58 / 11.93 · constant tweet 20.46 / 15.19 → the model scores BELOW both.
Generated captions imitate tweets (96.8% contain URL fragments, 99.2% hashtags) — they are not image descriptions (C18).

Superseded (v2, leakage + unshifted teacher forcing, non-standard BLEU): BLEU-1..4 24.85/14.20/8.95/6.13, METEOR 0.32, ROUGE-L 0.56, BERTScore 60.35%.

### Baselines (grouped split, test n=375)
| Model | Seeds | Macro F1 | Trainable |
|-------|-------|----------|-----------|
| ViT-Only (last 2 blocks + pooler) | 3 | 73.27 ± 0.39% | 15.16M |
| CrisisViT-style (our re-implementation, ViT-L, last 4 blocks + pooler) | 1 | 75.87% | 51.96M |
| **DRISHTI-Bn CLS-Only** | 3 | **75.69 ± 1.33%** | **1.38M** |

Claim: matches the ViT-Large baseline with 38× fewer trainable parameters — NOT "beats SOTA".

Original split only (not re-run; label as such): Stratified Random 39.95% · BanglaBERT-Only 43.02% · Full Fine-Tune 74.19%.
Superseded (pooler-only bug, C17): ViT-Only 78.64% · CrisisViT 78.15%.

### Ablation Study (original un-grouped split, single seed — NOT re-run)
| Config | Macro F1 | Note |
|--------|----------|------|
| Caption-Only | 16.90% | untrained classifier head (C6) — exclude or footnote |
| No-LoRA | 77.74% | |
| LoRA r=4 | 79.41% | |
| LoRA r=16 | 79.14% | |
| LoRA r=8 (rank ablation run) | 81.47% | was mislabelled "r=8 Multi-Task" (C5) |
| No-CrossAttn | 81.88% | difference = run-to-run variance (C4) |
| CLS-Only | 83.21% | same config as r=8 above → ~1.7pp variance |

Differences < ~2.5pp are within seed variance (grouped split CLS-Only range 74.83–77.22%).

### Text ablation (Phase 2, CLS-Only seed 42, grouped test n=375)
Original tweet 74.83% · v3 generated caption 76.41% · placeholder "বন্যা পরিস্থিতি" 76.52% · another sample's tweet 76.63% (96.5% same predictions) · empty text 61.95% (out-of-distribution)
→ the classifier is effectively **image-driven**; never claim "text and image are complementary" (C19).

### Calibration (CLS-Only seed 42; T* fitted on grouped val n=373; grouped test n=375; 10 bins)
| Metric | Before TS | After TS (T* = 1.1349) |
|--------|-----------|------------------------|
| ECE | **2.57%** | 2.99% |
| MCE | 19.06% | 20.60% |
| NLL | 0.3223 | 0.3228 |
| Brier | 0.1681 | 0.1688 |
| Mean confidence | 88.84% | 87.03% |
| Accuracy | 88.27% | 88.27% |

Already well calibrated; temperature scaling gives no improvement. Superseded: T* = 1.1113 / 0.6669, ECE 2.16% (old split, old model).

### XAI (Phase 4, CLS-Only seed 42 + v3 decoder)
**Qualitative samples:** 21 random grouped-test images (7/class, seed 42, not cherry-picked) → 15/21 classified correctly | SD 6/7 | HR 7/7 | AP 2/7 (errors: mean confidence 0.58 vs 0.89). Never call this "XAI accuracy".

**Faithfulness — deletion test (all 375 test images; drop in predicted-class probability, accuracy before = 88.27%):**
| Method | Top 10% removed | Top 25% | Top 50% | Accuracy after 50% |
|--------|-----------------|---------|---------|--------------------|
| **Grad-CAM** | **6.92 pp** | **14.27 pp** | **21.27 pp** | **73.87%** |
| Attention Rollout | 0.85 pp | 3.26 pp | 7.35 pp | 81.60% |
| Random | 0.34 pp | 1.57 pp | 4.16 pp | 86.93% |

Top-10% patch IoU: Grad-CAM vs Rollout 0.148 (2.7× chance) · Grad-CAM vs v3 cross-attention 0.059 (≈ chance 0.054) → cross-attention is auxiliary (explains caption generation only).
Grad-CAM hooks the INPUT of the last ViT block (old code hooked the output → zero gradient → blank maps, C20).
Superseded: 17/21 = 81% "XAI accuracy" (old split; samples partly seen in training).

### Human Evaluation (100 samples, 3 annotators, blind A/B — correctly un-blinded)
> Corrected 2026-11-12 with the user's approval (DEFENSE_FIX_PLAN 1.20). Source: `src/human_eval_unblind.py` →
> `results/defense_fixes/phase1/human_eval_unblinded/`. Model captions are from the old v2 checkpoint, not v3
> (decision 3B.4: kept as a limitation; v3 assessed with automatic metrics only).

| Dimension | Model | Ground Truth | Wilcoxon p |
|-----------|-------|--------------|------------|
| Fluency | **1.58/5** | **3.42/5** | 2.1 × 10⁻¹⁷ |
| Adequacy | **1.24/5** | **3.08/5** | 9.7 × 10⁻¹⁸ |
| Faithfulness | **1.23/5** | **3.23/5** | 1.8 × 10⁻¹⁷ |

**Per-class (model F / A / Fa):** SD(1.63, 1.28, 1.29) | HR(1.53, 1.21, 1.16) | AP(1.46, 1.08, 1.08)

**Preference:** GT preferred in 225 of 237 judgements (94.9%)

**IAA:** pooled (model + GT, 200 items) Krippendorff's α = **0.38**, quadratic-weighted κ = 0.40 · within caption type ≈ 0

Superseded (merge bug — A/B swap ignored): model 2.49 / 2.11 / 2.17, GT 2.51 / 2.21 / 2.30, κ ≈ −0.000, α ≈ −0.052.

### Translation
Bengali texts are **machine translations** of the CrisisMMD tweets (`monirbishal/en-bn-nmt`), NOT human-verified
(user confirmed 2026-11-12). 8.9% were left untranslated; 97.3% contain URLs. Never write "human-verified translation".

---

## Key Files
| File | Purpose |
|------|---------|
| `DEFENSE_FIX_PLAN.md` | **Active:** phase-by-phase pre-defense fix plan (update after each phase, then commit + push) |
| `docs/corrected_claims.md` | **Reference for writing:** how each claim must be phrased, what is dropped, limitations |
| `PHASE4_WRITING_PLAN.md` | Old writing plan — **ignored** (new prompts will be built from `docs/corrected_claims.md`) |
| `Thesis Writing/` | DIU guidelines, SWE thesis template, defense slide templates |
| `DRISHTI_XAI_CHECKLIST.md` | XAI task checklist |
| `test_set_indices_grouped.csv` | Grouped (leakage-free) test set, n=375 — use this |
| `test_set_indices.csv` | Old un-grouped test set (n=376) — superseded |
| `docs/defense_qa.md` | Likely defense questions + answers |
| `human_eval_outputs/human_eval_report.md` | Human eval full report |
| `human_eval_outputs/iaa_scores.csv` | IAA scores |

## Generated Figures (all in project root) — use the `_grouped` / new versions in the thesis
| File | Content |
|------|---------|
| `figure_1_1_flood_stats.png` | Flood statistics (introduction) |
| `figure_1_2_system_pipeline.png` | System pipeline (introduction) |
| `figure_2_1_lora.png` | LoRA injection diagram |
| `figure_3_1_architecture.png` | DRISHTI-Bn architecture |
| `figure_3_3_xai_pipeline.png` | XAI pipeline flowchart |
| `figure_4_1_confusion_matrix_grouped.png` | **Fig 4.1** — CLS-Only seed 42, grouped test n=375, Macro F1 74.83% |
| `figure_4_2_baseline_comparison_grouped.png` | **Fig 4.2** — baselines on grouped split (mean ± std) |
| `figure_4_3_ablation_original_split.png` | **Fig 4.3** — ablation (original split, single seed, relabelled per C4–C6) |
| `figure_4_4_gradcam_grouped.png` | **Fig 4.4** — fixed Grad-CAM (first random sample per class) |
| `figure_4_5_crossattn_grouped.png` | **Fig 4.5** — v3 decoder cross-attention (auxiliary) |
| `figure_4_6_rollout_grouped.png` | **Fig 4.6** — Attention Rollout |
| `figure_4_7_integrated_xai_grouped.png` | **Fig 4.7** — Grad-CAM / Rollout / cross-attention side by side |
| `figure_4_8_calibration_diagram_grouped.png` | **Fig 4.8** — reliability diagram, T* = 1.1349 |
| `figure_4_9_faithfulness_deletion.png` | **Fig 4.9** — deletion-test curves (Grad-CAM / Rollout / Random) |
| `results/defense_fixes/phase3b/phase3b_confusion_matrix.png` | Multi-task v3 confusion matrices (tweet vs generated caption) |

Superseded (do NOT use): `figure_4_1_confusion_matrix.png` (v2), `figure_4_2_baseline_comparison.png`, `figure_4_3_ablation_study.png`,
`figure_4_4_gradcam_examples.png` (blank Grad-CAM), `figure_4_5_crossattn_examples.png` (untrained decoder),
`figure_4_6_rollout_examples.png`, `figure_4_7_integrated_xai.png`, `figure_4_8_calibration_diagram.png` — kept under `figures_superseded/`.

---

## Model & Dataset Info
- **Model:** ViT-Base/16 + BanglaBERT + LoRA (r=8, α=16; LoRA = 0.30% of encoder params; CLS-Only trains 1,379,331 ≈ 0.70% incl. head; Multi-Task v3 trains 74.79M incl. decoder + ViT pooler)
- **Dataset:** CrisisMMD-derived Bengali flood corpus, 3755 samples (95.2% hurricane: Harvey/Maria/Irma + Sri Lanka floods, C16), tweet-grouped 80/10/10 split (`results/defense_fixes/phase3/grouped_split.csv`); checkpoints on Drive: `models/cls_only_grouped_s42.pt`, `models/drishti_multitask_v3.pt`
- **Hardware:** Single NVIDIA T4 GPU (16GB VRAM), Colab Pro
- **Google Drive folder:** `My Drive > DisasterNet` (src/ NOT uploaded to Drive)
- **Thesis draft:** old drafts (`DRISHTI_Bn_Thesis_Final_Working_Draft.docx`, `..._v1.txt`) were intentionally deleted (pre-results work). A fresh draft will be written in the DIU template after `DEFENSE_FIX_PLAN.md` is complete.
- **Env note:** local venv uses transformers 5.x, where ViT attention modules are `q_proj/v_proj` (v4: `query/value`). `src/model.py` auto-detects, so checkpoint keys stay compatible.

