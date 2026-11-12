# Phase 3B — Multi-task retrain (v3) results (Colab T4, 2026-11-12)

Script: `colab/phase3b_multitask_retrain.py` (repo @ `671ecb9`, code identical to `36c3cfb`)
Checkpoint: Drive `models/drishti_multitask_v3.pt` (v2 untouched)
Setup: Phase 3 grouped split (3007 / 373 / 375), seed 42, 15 epochs, bs 16, AdamW 1e-4 (wd 0.01), cosine, AMP,
L = 1.0·L_cls + 0.5·L_cap, right-shifted teacher forcing; best epoch = lowest val total loss → **epoch 4**.

## Diagnostics
```
[diag] tokenizer: vocab 32000 | PAD 0 | CLS 2 | SEP 3
[diag] trainable: ViT-LoRA 885,504 | BERT-LoRA 294,912 | cls head 789,507 | decoder 72,817,664 | total 74,787,587 of 271,203,587
```
ViT-LoRA 885,504 = LoRA 294,912 + trainable ViT pooler copy 590,592 (`modules_to_save=["pooler"]` in `src/model.py`).
The CLS-Only ablation does not train the pooler, so the two models differ by 590,592 trainable parameters besides the decoder.

## Training log
| ep | train total | train cls | train cap | val total | val cls | val cap | val F1 |
|----|-------------|-----------|-----------|-----------|---------|---------|--------|
| 1 | 3.1890 | 0.5301 | 5.3178 | 2.6525 | 0.4003 | 4.5044 | 77.63 |
| 2 | 2.4092 | 0.3578 | 4.1026 | 2.4857 | 0.3582 | 4.2549 | 78.57 |
| 3 | 2.1686 | 0.2983 | 3.7405 | 2.4314 | 0.3562 | 4.1504 | 76.58 |
| **4** | 1.9640 | 0.2344 | 3.4593 | **2.4051** | 0.3546 | 4.1011 | 78.41 |
| 5 | 1.7950 | 0.1837 | 3.2225 | 2.4183 | 0.3817 | 4.0731 | 74.02 |
| 6 | 1.6432 | 0.1352 | 3.0159 | 2.4063 | 0.3713 | 4.0700 | 76.91 |
| 7 | 1.5404 | 0.1204 | 2.8400 | 2.4463 | 0.4116 | 4.0694 | 76.33 |
| 8 | 1.4340 | 0.0916 | 2.6847 | 2.4641 | 0.4222 | 4.0837 | 75.69 |
| 9 | 1.3481 | 0.0721 | 2.5522 | 2.4585 | 0.4108 | 4.0953 | 75.53 |
| 10 | 1.2844 | 0.0629 | 2.4428 | 2.4970 | 0.4409 | 4.1122 | 75.86 |
| 11 | 1.2366 | 0.0581 | 2.3570 | 2.5236 | 0.4589 | 4.1295 | 76.29 |
| 12 | 1.1933 | 0.0464 | 2.2938 | 2.5074 | 0.4378 | 4.1392 | 75.89 |
| 13 | 1.1714 | 0.0475 | 2.2479 | 2.5069 | 0.4377 | 4.1382 | 77.18 |
| 14 | 1.1587 | 0.0477 | 2.2219 | 2.5079 | 0.4357 | 4.1444 | 76.27 |
| 15 | 1.1474 | 0.0423 | 2.2100 | 2.5134 | 0.4405 | 4.1458 | 76.98 |

Train caption loss keeps falling (5.32 → 2.21) while val caption loss plateaus at ~4.07 from epoch 5 (perplexity ≈ 59): the decoder overfits after ~5 epochs.

## Classification (test n = 375, seed 42)
| Text input | Macro F1 | Macro P | Macro R | Acc | F1 SD | F1 HR | F1 AP | Confusion (rows SD, HR, AP) |
|------------|----------|---------|---------|-----|-------|-------|-------|-----------------------------|
| (a) original tweet text | **77.29** | 79.03 | 76.12 | 87.47 | 92.00 | 87.24 | 52.63 | [207,10,7] [10,106,3] [9,8,15] |
| (b) generated caption (image-only) | **73.63** | 76.77 | 72.12 | 86.93 | 92.51 | 86.07 | 42.31 | [210,10,4] [9,105,5] [11,10,11] |

Context (Phase 3, same split): CLS-Only 75.69 ± 1.33 (seed 42: 74.83) · ViT-Only 73.27 ± 0.39 · CrisisViT-style 75.87.

## Captions (beam 3, scores ×100, `src/metrics.py`)
| References | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | BERTScore P | R | F1 |
|------------|--------|--------|--------|--------|--------|---------|-------------|---|----|
| raw | 7.42 | 6.82 | 6.80 | 6.93 | 1.10 | 1.80 | 71.54 | 67.98 | **69.66** |
| URLs stripped | 7.60 | 6.99 | 6.97 | 7.11 | 1.25 | 1.90 | 61.24 | 63.42 | **62.27** |

Diversity: 321/375 unique; most common caption 2.1% ("RT @ HurricanMaria : # Harvey ত্রাণের জন্য দান করা হয়েছে ।"); mean 15.4 words; 0 empty.

Example outputs (all classes look alike):
```
GEN: # Hurrican Harvey ' ss : / t. co / jEMABCHOTJ https : Rico Irma এর মাধ্যমে হ্যারিকেন মারিয়া
GEN: আরটি @ সিএনএন : # হুরিকানে হ্যারাভি ত্রাণের জন্য সরবরাহ করেছে https : / t. co / qJKBCHRMA
GEN: RT @ HurricanMaria : # Harvey ' র পর, কিন্তু হ্যারিকেন মারিয়া https : / t. co / qPSLCVYR
GEN: # Hurrican Harvey ত্রাণের জন্য দান করা হয়েছে । https : / t. co / cCSPRMA
```

## Findings
1. **Captions are tweet imitations, not image descriptions.** The decoder learned the surface style of the references (machine-translated tweets, C10): hashtags, `RT @handle`, fake `t.co` URLs, and mixed-up event names (Harvey/Maria/Irma in one sentence). Content barely depends on the image or the class.
2. **BERTScore is inflated by shared tweet boilerplate.** Removing URLs drops it from 69.66 to 62.27. BERTScore with mBERT gives ~60+ even for unrelated Bengali sentences (no baseline rescaling for `bn`), so it should not be presented as evidence of caption quality.
3. **`metrics.py` BLEU is non-standard.** Sentence-level BLEU with +1 smoothing, averaged over sentences, makes BLEU-1..4 nearly equal (7.4 / 6.8 / 6.8 / 6.9), which is impossible for standard BLEU. To be recomputed locally with sacreBLEU (corpus BLEU) + chrF from `phase3b_per_sample.csv`.
4. **Uniqueness (321/375) overstates diversity.** Most of the variation is in the random URL suffix; to be re-measured after URL removal.
5. **Multi-task classification with tweet text (77.29%) is a single seed.** It lies about 1.2 std above the CLS-Only mean (75.69 ± 1.33) and 2.5pp above CLS-Only seed 42, but checkpoint selection (val loss vs val F1) and the trainable pooler also differ. It does **not** show that captioning helps classification.
6. **Image-only deployment (generated caption) = 73.63%, the same as ViT-Only (73.27 ± 0.39).** The generated text adds no information beyond the image, as expected for text that is derived from the image and carries little image content.
7. Leakage-free numbers are far below the old v2 claims (77.95% F1, BERTScore 60.35%, BLEU-1 24.85) — those came from a model that had seen most of the old test set and used unshifted teacher forcing (C1).
