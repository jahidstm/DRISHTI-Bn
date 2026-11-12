# Phase 2 — Fair evaluation of the CLS-Only model (Colab T4, 2026-11-12)

Script: `colab/phase2_fair_evaluation.py` (repo @ `e14b2f5`) · checkpoint `models/cls_only_grouped_s42.pt` (Phase 3, seed 42)
Grouped test set, n = 375 · fp32 inference with `cls_only_ablation.evaluate` (same path as Phase 3).

## Results
| Condition | Text input | Macro F1 | Macro P | Macro R | Acc | F1 SD | F1 HR | F1 AP | Agree with A |
|-----------|-----------|----------|---------|---------|-----|-------|-------|-------|--------------|
| A | original tweet | **74.83** | 77.82 | 73.52 | 88.27 | 93.13 | 89.07 | 42.31 | — |
| B | v3 generated caption | 76.41 | 81.99 | 74.41 | 88.27 | 93.30 | 86.96 | 48.98 | 95.2% |
| C | empty text | 61.95 | 90.64 | 63.85 | 86.93 | 92.47 | 87.30 | 6.06 | 92.5% |
| D | placeholder "বন্যা পরিস্থিতি" (demo fallback) | 76.52 | 79.50 | 74.44 | 87.47 | 91.85 | 87.72 | 50.00 | 93.3% |
| E | another test sample's tweet (shuffled, seed 42) | 76.63 | 79.25 | 75.19 | 88.53 | 93.57 | 88.16 | 48.15 | 96.5% |

Confusion matrices (rows SD, HR, AP):
```
A [[210, 8, 6], [6, 110, 3], [11, 10, 11]]
B [[209, 12, 3], [7, 110, 2], [8, 12, 12]]
C [[215, 9, 0], [9, 110, 0], [17, 14, 1]]
D [[214, 4, 6], [15, 100, 4], [13, 5, 14]]
E [[211, 7, 6], [8, 108, 3], [8, 11, 13]]
```
Condition A reproduces Phase 3 seed 42 exactly (74.83%, identical confusion matrix) → the evaluation is consistent.

## Findings
1. **The classifier barely uses the tweet's content.** Giving every image a *different* sample's tweet (E) does not lower Macro F1 (76.63 vs 74.83) and leaves 96.5% of predictions unchanged. A generated caption (B) or a fixed two-word placeholder (D) behave the same.
2. **Differences between A, B, D and E (≤ 1.8pp) are noise.** They come almost entirely from the Affected_People class (32 test samples, where one sample moves F1 by several points), and they are smaller than the seed spread of CLS-Only (74.83–77.22).
3. **Empty text (C) is out-of-distribution, not "image-only".** A sequence of just [CLS][SEP] was never seen in training; the model then almost never predicts Affected_People (F1 6.06). Any ordinary Bengali sentence restores normal behaviour.
4. **Consequence for the "multimodal" claim.** CLS-Only's +2.4pp over ViT-Only (Phase 3) cannot be attributed to tweet semantics. It more likely comes from the image branch (LoRA on all ViT attention layers vs fine-tuning the last 2 blocks) and/or the BanglaBERT [CLS] vector acting as an almost constant extra input. The model is effectively image-driven.
5. **Consequence for deployment.** In an image-only setting the CLS-Only model with the demo placeholder (76.52%) is at least as good as with the real tweet. The image-only multi-task pipeline (v3 generated caption → its own classifier) gives 73.63% (Phase 3B), so for the demo the CLS-Only checkpoint + placeholder is the better classifier.
