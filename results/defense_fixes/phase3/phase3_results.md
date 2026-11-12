# Phase 3 — Grouped split + multi-seed results (Colab T4, 2026-11-12)

Script: `colab/phase3_grouped_split_seeds.py` @ `12bf43f` · transformers 5.22.1 · torch 2.14.1+cu130 · peft 0.21.2

## Split (leakage-free, class-balanced)
3,755 samples → 3,189 leakage groups (union-find over tweet_id OR identical caption).
GroupShuffleSplit best-of-200: random_state test = 91, val = 197. Zero tweet / caption overlap.

| split | n | Severe_Damage | Humanitarian_Rescue | Affected_People |
|-------|---|---------------|---------------------|-----------------|
| train | 3007 | 1794 (59.7%) | 949 (31.6%) | 264 (8.8%) |
| val | 373 | 222 (59.5%) | 117 (31.4%) | 34 (9.1%) |
| test | 375 | 224 (59.7%) | 119 (31.7%) | 32 (8.5%) |

## Diagnostics
```
[diag] ViT param names look like: layers.0.attention.q_proj.bias
[diag] Original Task 0.5 rule unfreezes 590,592 params | robust rule unfreezes 14,766,336
ViT LoRA targets: ['q_proj','v_proj'] | BERT LoRA targets: ['query','value']
LoRA trainable 589,824 (0.30% of encoders); with fusion head 1,379,331
CrisisViT-style trainable 51,961,859 of 304,351,232 (ViT-Large)
```

## Test results (n = 375; checkpoint = best val Macro F1 epoch, 10 epochs)

| Model | Seeds | Macro F1 | Macro P | Macro R | Accuracy | F1 SD | F1 HR | F1 AP | Trainable |
|-------|-------|----------|---------|---------|----------|-------|-------|-------|-----------|
| CLS-Only (ViT+BanglaBERT+LoRA r=8) | 42, 123, 2024 | **75.69 ± 1.33** | 77.09 ± 1.31 | 74.93 ± 1.59 | 88.00 ± 0.27 | 93.28 ± 0.15 | 88.31 ± 0.82 | 45.48 ± 4.63 | 1.38M |
| ViT-Only (last 2 blocks + pooler) | 42, 123, 2024 | 73.27 ± 0.39 | 74.64 ± 0.87 | 72.38 ± 0.75 | 85.60 ± 0.27 | 91.46 ± 0.27 | 84.69 ± 1.06 | 43.66 ± 2.01 | 15.16M |
| CrisisViT-style (ViT-L, last 4 blocks + pooler) | 42 | 75.87 | 76.27 | 75.53 | 86.93 | 93.63 | 83.98 | 50.00 | 51.96M |

(std = sample std over seeds)

Per-run Macro F1 — CLS-Only: 74.83 / 75.02 / 77.22 · ViT-Only: 73.69 / 73.21 / 72.91
Paired difference CLS − ViT per seed: +1.14 / +1.81 / +4.31 (mean +2.42; positive in 3/3 seeds)

### Confusion matrices (rows = true SD, HR, AP)
| Run | Matrix |
|-----|--------|
| CLS s42 | [210,8,6] [6,110,3] [11,10,11] |
| CLS s123 | [209,5,10] [7,107,5] [8,11,13] |
| CLS s2024 | [213,3,8] [11,101,7] [8,8,16] |
| ViT s42 | [206,8,10] [13,100,6] [9,9,14] |
| ViT s123 | [207,11,6] [13,102,4] [8,12,12] |
| ViT s2024 | [208,7,9] [12,102,5] [10,10,12] |
| CrisisViT s42 | [213,5,6] [12,97,10] [6,10,16] |

## Comparison with previously reported (leaky split, single seed, pooler-only baselines)

| Model | Old (CLAUDE.md) | Phase 3 | Δ |
|-------|-----------------|---------|---|
| CLS-Only | 83.21 | 75.69 ± 1.33 | −7.5 |
| ViT-Only | 78.64 (pooler only) | 73.27 ± 0.39 | −5.4 |
| CrisisViT-style | 78.15 (pooler only) | 75.87 (1 seed) | −2.3 |

## Findings
1. Removing tweet/caption leakage lowers CLS-Only by ~7.5pp. The old 83.21% is not a valid generalisation estimate.
2. CLS-Only beats the corrected ViT-Only baseline in all 3 seeds (+2.4pp mean). Gains are in SD (+1.8) and HR (+3.6); AP is similar. Note CLS-Only uses the original tweet text at test time (C13).
3. CLS-Only and CrisisViT-style are tied (75.69 ± 1.33 vs 75.87, within one std), while CLS-Only trains 38× fewer parameters (1.38M vs 51.96M) on a smaller backbone. The supportable claim is parameter efficiency, not superiority.
4. Affected_People remains the weak class (F1 ≈ 41–51%, n = 32 in test); its seed variance (± 4.6) dominates the Macro F1 variance.
5. Old pooler-only baselines were *higher* on the leaky split than the corrected ones here, so the leakage effect is larger than the unfreeze-bug effect.
