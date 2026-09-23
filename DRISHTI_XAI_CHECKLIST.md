# DRISHTI-XAI: Master Implementation Checklist
> **Last Updated:** ২৩ সেপ্টেম্বর ২০২৬
> **Branch:** `main` | **Target:** IEEE Access / ISCRAM 2027

---

## ⚠️ গুরুত্বপূর্ণ Writing Rule (ভুলবে না!)

> **Thesis writing (`DRISHTI_Bn_Thesis_Working_Draft_v1.txt`) কখনো সরাসরি edit করব না।**
> প্রতিটি heading/subheading-এর জন্য আগে তোমাকে একটি **NotebookLM prompt** দেব।
> তুমি সেটা NotebookLM-এ দিয়ে writing generate করবে এবং draft-এ যোগ করবে।
> **আমার অনুমতি ছাড়া কোনো writing section touch হবে না।**

---

## Phase 0: Foundation Fixes
> Fatal evaluation gaps বন্ধ করা — Q1 publication-এর minimum requirement

- [x] **Task 0.1** ~~Stratified 80/10/10 Data Split~~ — `src/data_loader.py` রিফ্যাক্টর ✅
  - Train=3004 / Val=375 / Test=376 | Class ratio identical | Zero overlap ✅
  - Commit: `0e36ed2`

- [x] **Task 0.2** ~~Extended Metrics Module — `src/metrics.py` নতুন module, `src/benchmark.py` refactor~~ ✅
  - BLEU (1-4), METEOR, ROUGE-L, BERTScore 
  - `metrics.py` imported to `benchmark.py`
  - Commit: `b82b135`
- [x] **Task 0.3** ~~Full Test Set Evaluation — v2 checkpoint দিয়ে proper benchmark~~ ✅
  - **[CLS] Macro F1: 77.95%** | Precision: 93.28% | Recall: 73.03%
  - Per-class: Severe_Damage F1=92.53% ✅ | Humanitarian_Rescue F1=88.00% ✅ | **Affected_People F1=53.33% ⚠️ (class imbalance)**
  - **[CAP] BLEU-4: 6.13** | METEOR: 0.32 | ROUGE-L: 0.56 | **BERTScore-F1: 60.35%**
  - ⚠️ Key Finding: Captioning BLEU scores খুবই কম → beam search + training improvement দরকার
- [x] **Task 0.4** ~~Baseline #1 — Random/Majority Classifier~~ ✅
  - Majority F1=**24.89%** | Uniform Random F1=**29.21%** | Stratified Random F1=**39.95%**
  - DisasterNet-v2 vs Best Baseline: **77.95% vs 39.95% (+38pp gain!)** ✅
- [ ] **Task 0.5** Baseline #2 — ViT-Only (no text encoder)
- [ ] **Task 0.6** Baseline #3 — BanglaBERT-Only (no image)
- [ ] **Task 0.7** Baseline #4 — Full Fine-Tune (no LoRA) ⚡ Colab
- [ ] **Task 0.8** Baseline #5 — CrisisViT SOTA comparison ⚡ Colab

---

## Phase 1: Ablation Study
> কোন component কতটুকু contribute করছে তা পরিমাপ

- [ ] **Task 1.1** Ablation — Classification Only (λ_cap = 0) ⚡ Colab
- [ ] **Task 1.2** Ablation — Caption Only (λ_cls = 0) ⚡ Colab
- [ ] **Task 1.3** Ablation — No LoRA (Frozen Encoders) ⚡ Colab
- [ ] **Task 1.4** Ablation — No Cross-Attention Decoder ⚡ Colab
- [ ] **Task 1.5** Ablation — LoRA Rank Sensitivity (r=4, r=8, r=16) ⚡ Colab
- [ ] **Task 1.6** Results Compilation — Ablation summary table

---

## Phase 2: XAI Extension Modules
> DRISHTI-XAI-এর নতুন components

- [ ] **Task 2.1** Grad-CAM — ViT attention layer visual explanation
- [ ] **Task 2.2** Cross-Attention Visualization — Token-to-patch alignment
- [ ] **Task 2.3** Temperature Scaling — Post-hoc confidence calibration
- [ ] **Task 2.4** MC Dropout Uncertainty — Predictive entropy
- [ ] **Task 2.5** Integrated XAI Pipeline — সব module একত্রিত

---

## Phase 3: Human Evaluation
> মানবিক মূল্যায়ন — caption quality proof

- [ ] **Task 3.1** Evaluation Protocol Design — ১০০ sample, annotation guidelines
- [ ] **Task 3.2** Annotator Recruitment — ৩ জন Bengali speaker
- [ ] **Task 3.3** Annotation Execution — Fluency, Adequacy, Faithfulness scoring
- [ ] **Task 3.4** Inter-Annotator Agreement — Cohen's Kappa

---

## Phase 4: Thesis Writing & Paper Submission
> NotebookLM workflow-এ চলবে — প্রতিটি section-এ আলাদা prompt পাবে

- [ ] **Task 4.1** Paper Structure & Abstract Draft
  - 🗒️ *NotebookLM prompt পাবে*
- [ ] **Task 4.2** Figures & Tables — Architecture diagram, heatmaps, result tables
- [ ] **Task 4.3** Full Draft — IEEE Access format
  - 🗒️ *প্রতিটি section-এ NotebookLM prompt পাবে*
- [ ] **Task 4.4** Internal Review & Revision
- [ ] **Task 4.5** Submission → IEEE Access / ISCRAM 2027

---

### Legend
| Symbol | অর্থ |
|--------|------|
| `[ ]` | অসম্পন্ন |
| `[/]` | চলমান |
| `[x]` | সম্পন্ন ✅ |
| ⚡ Colab | Google Colab-এ GPU দরকার |
| 🗒️ | NotebookLM prompt দেওয়া হবে |
