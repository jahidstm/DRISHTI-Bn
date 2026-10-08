# DRISHTI-XAI: Master Implementation Checklist
> **Last Updated:** 8 October 2026
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
  - **Macro F1: 78.64%** | Precision: 79.04% | Recall: 78.38%
  - Per-class: Severe_Damage F1=87.17% | Humanitarian_Rescue F1=83.90% | **Affected_People F1=64.84%**
  - Finding: Image alone besh bhalo kaj kore (78.64%)

- [x] **Task 0.6** ~~Baseline #3 -- BanglaBERT-Only (no image)~~ done
  - **Macro F1: 43.02%** | Text-only model onek durbol
  - Finding: Visual features chara classification onek kothin (-35.62pp vs ViT-Only)

- [ ] **Task 0.7** Baseline #4 -- Full Fine-Tune (no LoRA) [Colab GPU needed]
- [ ] **Task 0.8** Baseline #5 -- CrisisViT SOTA comparison [Colab GPU needed]

---

## Phase 1: Ablation Study
> Kon component kotkhani contribute korche ta porimap kora

- [x] **Task 1.1** ~~Ablation -- Classification Only (lambda_cap = 0)~~ done
  - **Macro F1: 80.79%** | Precision: 83.03% | Recall: 79.52%
  - Per-class: Severe_Damage F1=93.45% | Humanitarian_Rescue F1=86.85% | **Affected_People F1=62.07%**
  - Best checkpoint: Epoch 3 (Val F1=78.36%) -> saved as `best_cls_only.pt`
  - Key Finding: lambda_cap=0 (captioning OFF) korle F1 **+2.84pp** bare -> captioning loss classification ke hurt kore
  - Commit: `eb4549a`

- [ ] **Task 1.2** Ablation -- Caption Only (lambda_cls = 0) [Colab GPU needed]
- [ ] **Task 1.3** Ablation -- No LoRA (Frozen Encoders) [Colab GPU needed]
- [ ] **Task 1.4** Ablation -- No Cross-Attention Decoder [Colab GPU needed]
- [ ] **Task 1.5** Ablation -- LoRA Rank Sensitivity (r=4, r=8, r=16) [Colab GPU needed]
- [ ] **Task 1.6** Results Compilation -- Ablation summary table

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

- [ ] **Task 2.2** Cross-Attention Visualization -- Token-to-patch alignment [Colab GPU needed]
- [ ] **Task 2.3** Temperature Scaling -- Post-hoc confidence calibration [Colab GPU needed]
- [ ] **Task 2.4** MC Dropout Uncertainty -- Predictive entropy [Colab GPU needed]
- [ ] **Task 2.5** Integrated XAI Pipeline -- All modules combined [Colab GPU needed]

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
