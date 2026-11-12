# Phase 4 — XAI + calibration on the final models (Colab T4, 2026-11-12)

Script: `colab/phase4_xai_calibration.py` (repo @ `6f55911`)
Models: CLS-Only seed 42 (`cls_only_grouped_s42.pt`) for classification XAI and calibration; multi-task v3
(`drishti_multitask_v3.pt`) for decoder cross-attention. Grouped split (val 373, test 375), original tweet text.

## 4.2 Calibration (T* fitted on grouped val, evaluated on grouped test; 10 equal-width bins)
| Metric | Before (T = 1) | After (T* = 1.1349) |
|--------|----------------|---------------------|
| ECE | **2.57%** | 2.99% |
| MCE | 19.06% | 20.60% |
| NLL | 0.3223 | 0.3228 |
| Brier | 0.1681 | 0.1688 |
| Mean confidence | 88.84% | 87.03% |
| Accuracy | 88.27% | 88.27% |

Val ECE before scaling = 3.39%. The model is already well calibrated (mean confidence 88.84% vs accuracy 88.27%);
temperature scaling slightly softens confidence but does not improve test calibration — same pattern as the old
analysis (2.16% → 2.63%). Old T* values 1.1113 and 0.6669 are superseded; **T* = 1.1349 is the only value.**

## 4.1 / 4.3 XAI samples (21 test images, 7 per class, `random.Random(42)`, not cherry-picked)
- Classified correctly: **15/21** — Severe_Damage 6/7 · Humanitarian_Rescue 7/7 · Affected_People 2/7
- Fixed Grad-CAM: all-zero maps 0/21 (old implementation: every map was zero, C20)
- Top-10% patch overlap (IoU, mean over 21; random ≈ 0.054):
  Grad-CAM vs Rollout **0.148** (≈ 2.7× chance) · Grad-CAM vs decoder cross-attention **0.059** (≈ chance)

| n | true | pred | ok | conf | conf (T*) | IoU GC-RO | IoU GC-CA |
|---|------|------|----|------|-----------|-----------|-----------|
| 1 | SD | SD | ✓ | 0.993 | 0.987 | 0.081 | 0.081 |
| 2 | SD | AP | ✗ | 0.561 | 0.548 | 0.538 | 0.212 |
| 3 | SD | SD | ✓ | 0.949 | 0.926 | 0.000 | 0.000 |
| 4 | SD | SD | ✓ | 0.880 | 0.842 | 0.000 | 0.000 |
| 5 | SD | SD | ✓ | 0.911 | 0.882 | 0.176 | 0.111 |
| 6 | SD | SD | ✓ | 0.745 | 0.704 | 0.026 | 0.081 |
| 7 | SD | SD | ✓ | 0.780 | 0.742 | 0.026 | 0.176 |
| 8 | HR | HR | ✓ | 0.996 | 0.991 | 0.000 | 0.026 |
| 9 | HR | HR | ✓ | 0.943 | 0.919 | 0.000 | 0.000 |
| 10 | HR | HR | ✓ | 0.997 | 0.994 | 0.026 | 0.053 |
| 11 | HR | HR | ✓ | 0.969 | 0.950 | 0.290 | 0.000 |
| 12 | HR | HR | ✓ | 0.664 | 0.628 | 0.250 | 0.026 |
| 13 | HR | HR | ✓ | 0.915 | 0.886 | 0.000 | 0.000 |
| 14 | HR | HR | ✓ | 0.838 | 0.803 | 0.333 | 0.176 |
| 15 | AP | HR | ✗ | 0.420 | 0.409 | 0.429 | 0.053 |
| 16 | AP | SD | ✗ | 0.689 | 0.661 | 0.176 | 0.143 |
| 17 | AP | SD | ✗ | 0.519 | 0.500 | 0.000 | 0.026 |
| 18 | AP | AP | ✓ | 0.923 | 0.892 | 0.429 | 0.000 |
| 19 | AP | AP | ✓ | 0.780 | 0.740 | 0.333 | 0.000 |
| 20 | AP | SD | ✗ | 0.516 | 0.499 | 0.000 | 0.081 |
| 21 | AP | SD | ✗ | 0.786 | 0.744 | 0.000 | 0.000 |

Errors have lower confidence (mean 0.58 for the 6 errors vs 0.88 for the 15 correct).

## 4.5 Faithfulness — deletion test (all 375 test images)
Top-k% of the 196 patches (ranked by each map) set to the dataset mean; drop in the probability of the originally
predicted class. Accuracy before masking = 88.27%.

| Method | 10% deleted: prob drop / acc | 25%: prob drop / acc | 50%: prob drop / acc |
|--------|------------------------------|----------------------|----------------------|
| **Grad-CAM** | **6.92 pp / 85.60%** | **14.27 pp / 78.93%** | **21.27 pp / 73.87%** |
| Rollout | 0.85 pp / 88.53% | 3.26 pp / 84.00% | 7.35 pp / 81.60% |
| Random | 0.34 pp / 88.27% | 1.57 pp / 89.60% | 4.16 pp / 86.93% |

## Findings
1. **Fixed Grad-CAM is faithful.** Deleting the top 10% Grad-CAM patches lowers the predicted-class probability ~20× more than deleting random patches (6.92 vs 0.34 pp); at 50% accuracy falls to 73.87% vs 86.93% for random.
2. **Attention Rollout is only weakly faithful** (≈ 1.8–2.5× random). It is class-agnostic and diffuse, so it shows where the ViT routes information in general rather than the evidence for the predicted class.
3. **Decoder cross-attention is unrelated to the classification evidence** (IoU with Grad-CAM 0.059 ≈ chance), consistent with C18/C19: it explains the (tweet-like) caption generation only, so it is reported as auxiliary.
4. This reverses the old XAI conclusion ("Grad-CAM dark/uninformative, cross-attention most meaningful"): the old Grad-CAM was blank because of a hook bug (C20), and the old cross-attention came from an untrained decoder (C3).
5. Calibration: already good without scaling; temperature scaling gives no benefit on the test set.
