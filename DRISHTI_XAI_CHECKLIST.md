# DRISHTI-XAI: Master Implementation Checklist
> **Project:** DisasterNet-Bangla (DRISHTI-Bn)  
> **Extension:** DRISHTI-XAI (Explainable AI & Uncertainty Calibration for Multimodal Disaster Assessment)  
> **Last Updated:** ১২ সেপ্টেম্বর ২০২৬ | **Status:** 🟡 In Progress

---

## 📊 Quick Progress Overview
- **Phase 0:** 1 / 8 Completed (12.5%)
- **Phase 1:** 0 / 6 Completed (0%)
- **Phase 2:** 0 / 5 Completed (0%)
- **Phase 3:** 0 / 4 Completed (0%)
- **Phase 4:** 0 / 5 Completed (0%)
- **Overall Progress:** 1 / 28 Tasks (3.6%)

---

## Phase 0: Foundation Fixes (ন্যূনতম প্রকাশনা যোগ্যতা)
> *এই Phase সম্পন্ন না হলে কোনো Extension বা Ablation পেপারের রিভিউতে টিকবে না।*

- [x] **Task 0.1:** Stratified 80/10/10 Data Split — `src/data_loader.py` রিফ্যাক্টর (✅ Commit `c52a989`)
- [/] **Task 0.2:** Extended Metrics Module — `src/metrics.py` (METEOR, ROUGE-L, BERTScore যোগ)
- [ ] **Task 0.3:** Full Test Set Evaluation — ৩৭৬ স্যাম্পলের টেস্ট সেটে বর্তমান v2 মডেল বেঞ্চমার্ক ⚡ Colab
- [ ] **Task 0.4:** Baseline #1 — Random & Majority Class Baseline
- [ ] **Task 0.5:** Baseline #2 — ViT-Only Classifier (Unimodal Image Baseline)
- [ ] **Task 0.6:** Baseline #3 — BanglaBERT-Only Classifier (Unimodal Text Baseline)
- [ ] **Task 0.7:** Baseline #4 — Full Fine-Tuning (Without LoRA) ⚡ Colab
- [ ] **Task 0.8:** Baseline #5 — CrisisViT / SOTA Architecture Comparison ⚡ Colab

---

## Phase 1: Ablation Study (Component Contribution Analysis)
> *মডেলের প্রতিটি কম্পোনেন্টের একক ও যৌথ কার্যকারিতা প্রমাণ।*

- [ ] **Task 1.1:** Ablation — Classification Only ($\lambda_{cap} = 0$) ⚡ Colab
- [ ] **Task 1.2:** Ablation — Caption Only ($\lambda_{cls} = 0$) ⚡ Colab
- [ ] **Task 1.3:** Ablation — Frozen Encoders (No LoRA adaptation) ⚡ Colab
- [ ] **Task 1.4:** Ablation — No Cross-Attention (Self-Attention Decoder Only) ⚡ Colab
- [ ] **Task 1.5:** Ablation — LoRA Rank Sensitivity Analysis ($r \in \{4, 8, 16\}$) ⚡ Colab
- [ ] **Task 1.6:** Results Compilation — Comprehensive Ablation Table & Trade-off Analysis

---

## Phase 2: XAI Extension Modules (DRISHTI-XAI Core Innovation)
> *ডিসিশন-সাপোর্ট এবং রেসপন্সিবল এআই আর্কিটেকচার।*

- [ ] **Task 2.1:** Grad-CAM Implementation — ViT visual attention heatmaps (`src/xai/gradcam.py`)
- [ ] **Task 2.2:** Cross-Attention Heatmaps — Multimodal Token-to-Patch Alignment Map (`src/xai/cross_attn.py`)
- [ ] **Task 2.3:** Temperature Scaling — Post-hoc Confidence Calibration with ECE metric (`src/xai/calibration.py`)
- [ ] **Task 2.4:** MC Dropout Uncertainty — Predictive Entropy & Confidence Bounds (`src/xai/uncertainty.py`)
- [ ] **Task 2.5:** Integrated XAI Pipeline — Single-call Explanation & Reliability Generator

---

## Phase 3: Human Evaluation (মানবিক মূল্যায়ন)
> *Q1 পেপারের অন্যতম প্রধান মানদণ্ড।*

- [ ] **Task 3.1:** Protocol Design — ১০০ স্যাম্পল বাছাই ও মানদণ্ড গাইডলাইন (Fluency, Adequacy, Faithfulness)
- [ ] **Task 3.2:** Annotator Recruitment & Setup — ৩ জন নেটিভ বাংলা স্পিকার
- [ ] **Task 3.3:** Annotation Execution & Data Collection
- [ ] **Task 3.4:** Inter-Annotator Agreement — Fleiss' Kappa / Krippendorff's Alpha

---

## Phase 4: Paper Writing & Submission
> *টার্গেট: IEEE Access (Q1) / ISCRAM 2027*

- [ ] **Task 4.1:** Paper Structure & Abstract Draft
- [ ] **Task 4.2:** High-Resolution Figures & LaTeX Tables
- [ ] **Task 4.3:** Full Manuscript Draft (IEEE Template)
- [ ] **Task 4.4:** Supervisor Review & Iterative Polishing
- [ ] **Task 4.5:** Final Submission & Open-Source Artifact Release

---

### Legend
- `[ ]` = অসম্পন্ন (Pending)
- `[/]` = চলমান (In Progress)
- `[x]` = সম্পন্ন (Completed)
- ⚡ Colab = GPU প্রয়োজন (Google Colab স্ক্রিপ্ট সরবরাহ করা হবে)
