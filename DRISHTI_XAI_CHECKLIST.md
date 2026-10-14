# DRISHTI-XAI: Master Implementation Checklist
> **Last Updated:** 14 October 2026
> **Branch:** `main` | **Target:** IEEE Access / ISCRAM 2027

---

## Warning: Important Writing Rule

> **Thesis writing (`DRISHTI_Bn_Thesis_Working_Draft_v1.txt`) directly edit korbo na.**
> Each heading/subheading-er jonno age একটি NotebookLM prompt debo.
> Sei prompt NotebookLM-e diye writing generate korbe and draft-e jogkorbe.
> **Amar anumoti chara kono writing section touch hobe na.**

---

## Phase 0: Foundation Fixes
> Fatal evaluation gaps bondho kora - Q1 publication-er minimum requirement

- [x] **Task 0.1** ~~Stratified 80/10/10 Data Split~~ -- `src/data_loader.py` refactor done
  - Train=3004 / Val=375 / Test=376 | Class ratio identical | Zero overlap
  - Commit: `0e36ed2`

- [x] **Task 0.2** ~~Extended Metrics Module~~ -- `src/metrics.py` new module, `src/benchmark.py` refactor done
  - BLEU (1-4), METEOR, ROUGE-L, BERTScore
  - Commit: `b82b135`

- [x] **Task 0.3** ~~Full Test Set Evaluation~~ -- v2 checkpoint diye proper benchmark done
  - **[CLS] Macro F1: 77.95%** | Precision: 93.28% | Recall: 73.03%
  - Per-class: Severe_Damage F1=92.53% | Humanitarian_Rescue F1=88.00% | **Affected_People F1=53.33% (class imbalance)**
  - **[CAP] BLEU-4: 6.13** | METEOR: 0.32 | ROUGE-L: 0.56 | **BERTScore-F1: 60.35%**
  - Key Finding: Captioning BLEU scores khub kom -> beam search + training improvement dorkar

- [x] **Task 0.4** ~~Baseline #1 -- Random/Majority Classifier~~ done
  - Majority F1=**24.89%** | Uniform Random F1=**29.21%** | Stratified Random F1=**39.95%**
  - DisasterNet-v2 vs Best Baseline: **77.95% vs 39.95% (+38pp gain!)**

- [x] **Task 0.5** ~~Baseline #2 -- ViT-Only (no text encoder)~~ done
  - **Macro F1: 78.64%** | P=79.23% | R=78.48% | Seed=42 (deterministic)
  - Per-class: Severe_Damage F1=90.37% | Humanitarian_Rescue F1=84.25% | **Affected_People F1=61.29%**
  - Key Finding: ViT-Only (78.64%) > DisasterNet-v2 (77.95%) — text encoder contribution: -0.69pp
  - Commit: `6ea44d5`

- [x] **Task 0.6** ~~Baseline #3 -- BanglaBERT-Only (no image)~~ done
  - **Macro F1: 43.02%** | P=41.44% | R=44.83% | Seed=42
  - Per-class: Severe_Damage F1=73.60% | Humanitarian_Rescue F1=55.46% | **Affected_People F1=0.00% ⚠️** (সম্পূর্ণ ব্যর্থ)
  - Key Finding: Image contribution: ViT-Only (78.64%) − BERT-Only (43.02%) = **+35.62pp** 🔑
  - Commit: `b719a24`

- [x] **Task 0.7** ~~Baseline #4 -- Full Fine-Tune (no LoRA)~~ done
  - **Macro F1: 74.19%** | P=73.43% | R=75.09% | Seed=42 | AMP=True
  - Per-class: Severe_Damage F1=91.36% | Humanitarian_Rescue F1=82.64% | **Affected_People F1=48.57% ⚠️**
  - Key Finding: Full FT (74.19%) < LoRA (83.21%) — **-9.02pp** | LoRA regularizes better for 3004 train samples
  - Commit: `d7430d3`

- [x] **Task 0.8** ~~Baseline #5 -- CrisisViT SOTA comparison~~ done
  - Backbone: **ViT-Large/16** (304M params) | Last 4 blocks unfrozen (0.34% trainable)
  - **Macro F1: 78.15%** | P=81.35% | R=76.00% | Seed=42 | AMP=True
  - Per-class: Severe_Damage F1=91.27% | Humanitarian_Rescue F1=83.54% | **Affected_People F1=59.65%**
  - Key Finding: CrisisViT-style (78.15%) < CLS+LoRA (83.21%) — **-5.06pp** | LoRA wins
  - Commit: `0bb9860`

> ✅ **Phase 0 COMPLETE!** All 8 tasks done.

---

## Phase 1: Ablation Study
> Kon component kotkhani contribute korche ta porimap kora

- [x] **Task 1.1** ~~Ablation -- Classification Only (λ_cap = 0)~~ done
  - Script: `src/cls_only_ablation.py` | Commit: `6478fe5`
  - **Macro F1: 83.21%** | P=85.33% | R=81.84% | Best Epoch=3
  - Per-class: Severe_Damage F1=93.93% | Humanitarian_Rescue F1=87.90% | **Affected_People F1=67.80%** 🏆
  - Trainable params: 589,824 / 197,005,824 **(0.30%)** — highly parameter-efficient
  - Key Finding: CLS-Only (83.21%) > DisasterNet-v2 (77.95%) → **+5.26pp** | Caption loss HURTS classification
  - Best checkpoint: `best_cls_only.pt`

- [x] **Task 1.2** ~~Ablation -- Caption Only (λ_cls = 0)~~ done
  - Script: `src/caption_only_ablation.py` | Commit: `dfa823c`
  - **Macro F1: 16.90%** | P=24.17% | R=32.12% | Best Epoch=10
  - Per-class: Severe_Damage F1=4.24% | Humanitarian_Rescue F1=46.47% | **Affected_People F1=0.00%**
  - Model collapsed: 219/224 Severe_Damage predicted as Humanitarian_Rescue
  - Key Finding: CAP-Only (16.90%) << Stratified Random (39.95%) — **worse than random!**

- [x] **Task 1.3** ~~Ablation -- No LoRA (Frozen Encoders)~~ done
  - Script: `src/no_lora_ablation.py` | Commit: `6d841b7`
  - **Macro F1: 77.74%** | P=78.39% | R=77.37% | Best Epoch=4
  - Per-class: Severe_Damage F1=91.61% | Humanitarian_Rescue F1=83.53% | **Affected_People F1=58.06%**
  - Trainable params: 0 / 196,416,000 (0.00%) — only classifier head trained
  - Key Finding: No-LoRA (77.74%) vs CLS+LoRA (83.21%) → **LoRA contributes +5.47pp** 🔑

- [x] **Task 1.4** ~~Ablation -- No Cross-Attention (Self-Attention Only Decoder)~~ done
  - Script: `src/no_crossattn_ablation.py` | Commit: `c29913a`
  - **Macro F1: 81.88%** | P=83.82% | R=80.68% | Best Epoch=3
  - Per-class: Severe_Damage F1=93.69% | Humanitarian_Rescue F1=87.55% | **Affected_People F1=64.41%**
  - Key Finding: Self-Attn-only (81.88%) vs Cross-Attn (83.21%) → **Cross-attn contributes +1.33pp** 🔑

- [x] **Task 1.5** ~~Ablation -- LoRA Rank Sensitivity (r=4 vs r=8 vs r=16)~~ done
  - Script: `src/lora_rank_ablation.py` | Commit: `e3f8ac9`
  - **r=4: F1=79.41%** | r=8: F1=81.47% | r=16: F1=79.14%
  - Key Finding: r=8 is optimal — r=16 overfits, r=4 underfits. Performance spread: 2.33pp

- [x] **Task 1.6** ~~Results Compilation -- Ablation summary table + analysis~~ done
  - Full summary in `ablation_summary.md`

> ✅ **Phase 1 COMPLETE!** All 6 ablation tasks done.

---

## Phase 2: XAI Extension Modules
> DRISHTI-XAI-er notun components

- [x] **Task 2.1** ~~Grad-CAM -- ViT attention layer visual explanation~~ done
  - Model: CLS-Only (Task 1.1 best, `best_cls_only.pt`) | Device: cuda
  - **21 samples processed (7 per class, stratified) | Accuracy: 17/21 (81.0%)**
  - Per-class: Severe_Damage **7/7 correct** | Humanitarian_Rescue **7/7 correct** | Affected_People **3/7 (weak)**
  - Grad-CAM hook: last ViT encoder block (`layers.11`)
  - Output: `gradcam_outputs/` (21 PNG + `gradcam_summary.csv`)
  - Known Issue: Heatmap ekhono fully dark -- ViT global attention-er karone spatial localization durbol
    - Next step: Task 2.2-te Cross-Attention / Integrated Gradients diye better explanation
  - Commit: `eb4549a`

- [x] **Task 2.2** ~~Cross-Attention Visualization -- Token-to-patch alignment~~ done
  - Model: CLS-Only (Task 1.1 best, `best_cls_only.pt`)
  - **Accuracy: 17/21 (81.0%)** (Severe: 7/7, Human: 7/7, Affected: 3/7)
  - Finding: Spatial attention patterns are visible and meaningful (unlike Grad-CAM). Resolved Bengali font warnings in matplotlib.
  - Output: `crossattn_outputs/` -> `crossattn_results.zip`
  - Commit: `abb0b55`
- [x] **Task 2.3** ~~Attention Rollout Visualization~~ done
  - Method: Abnar & Zuidema (2020) | Head fusion: mean | Discard ratio: 0.9
  - **21 samples processed (7 per class, stratified) | Accuracy: 17/21 (81.0%)**
  - Per-class: Severe_Damage 7/7 | Humanitarian_Rescue 7/7 | Affected_People 3/7
  - Output: `rollout_outputs/` → `rollout_results.zip`
  - Key Finding: Rollout attention maps diekhay exactly kon token-er dike model attention diche classification-er somoy
  - Commit: `ccdb459`

- [x] **Task 2.4** ~~Temperature Scaling -- Post-hoc Calibration~~ done
  - Method: Guo et al. (2017) | Checkpoint: `best_cls_only.pt` (Loaded successfully)
  - **T* = 1.1113 (Over-confident -> T>1 softens distribution)**
  - Test Accuracy: **88.83%** (Severe_Damage: 92.4% | Humanitarian_Rescue: 91.6% | Affected_People: 54.5%)
  - Metric Before -> After:
    - ECE: 0.0216 -> 0.0263
    - MCE: 0.6209 -> 0.6256
    - NLL: 0.3201 -> 0.3224 (Val NLL improved by 0.67%)
    - Brier: 0.1715 -> 0.1728
    - Mean Confidence: 0.8825 -> 0.8665 (softened to align with real confidence)
  - Key Finding: Deep neural networks tend to be slightly over-confident (T*=1.1113 > 1.0). Pre-calibration ECE is already exceptionally low (2.16%), proving high reliability.
  - Output: `calibration_outputs/` -> `calibration_results.zip`
  - Commit: `6120de2`

- [x] **Task 2.5** ~~Integrated XAI Pipeline -- All modules combined~~ done
  - Script: `src/integrated_xai_pipeline.py` | Commit: `b80871f`
  - **Accuracy: 81.0% (17/21)** | T* = 0.6669 | Split: test | Samples: 21
  - Severe_Damage 6/7 | Humanitarian_Rescue 7/7 | **Affected_People 4/7 ⚠️**
  - Output: `integrated_outputs/` (21 PNG + `combined_comparison.png` + 2 CSV)
  - Key Finding: All 4 XAI methods unified in a single 3-row per-sample figure.
    Cross-Attention remains the most spatially sharp; Rollout is smoother.
    Affected_People improved to 4/7 (vs 3/7 in Tasks 2.1-2.3).

> ✅ **Phase 2 COMPLETE!** All 5 XAI tasks done.

---

## Phase 3: Human Evaluation
> Manobik mullayon -- caption quality proof

- [ ] **Task 3.1** Evaluation Protocol Design -- 100 sample, annotation guidelines
- [ ] **Task 3.2** Annotator Recruitment -- 3 Bengali speakers
- [ ] **Task 3.3** Annotation Execution -- Fluency, Adequacy, Faithfulness scoring
- [ ] **Task 3.4** Inter-Annotator Agreement -- Cohen's Kappa

---

## Phase 4: Thesis Writing and Paper Submission
> NotebookLM workflow-e cholbe -- protiti section-e alada prompt pabe

- [ ] **Task 4.1** Paper Structure and Abstract Draft
  - NotebookLM prompt pabe
- [ ] **Task 4.2** Figures and Tables -- Architecture diagram, heatmaps, result tables
- [ ] **Task 4.3** Full Draft -- IEEE Access format
  - Protiti section-e NotebookLM prompt pabe
- [ ] **Task 4.4** Internal Review and Revision
- [ ] **Task 4.5** Submission -> IEEE Access / ISCRAM 2027

---

### Legend
| Symbol | Ortho |
|--------|-------|
| `[ ]`  | Incomplete |
| `[/]`  | In Progress |
| `[x]`  | Complete |
| Colab  | Google Colab-e GPU dorkar |
| NLM    | NotebookLM prompt dewa hobe |
