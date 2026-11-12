# Phase 0.7 — Checkpoint audit (Colab, 2026-11-12)

Environment: torch 2.14.1+cu130 | transformers 5.22.1 | peft 0.21.2

## Raw output

```
=== 3 checkpoint file(s) found ===

/disasternet_multitask_v2.pth
  modified=2026-06-25 15:08 | size=1086.7 MB | md5=16bb292e5d62115cc5c966dc70c120d4  <-- SAME as laptop v2
  keys=562 | vit_lora=48 ['q_proj', 'v_proj'] | bert_lora=48 | decoder_keys=54
  vocab_projector.weight std=0.02539 (default init ~0.0208)

/models/disasternet_multitask_v2.pth
  modified=2026-06-25 15:08 | size=1086.7 MB | md5=16bb292e5d62115cc5c966dc70c120d4  <-- SAME as laptop v2
  keys=562 | vit_lora=48 ['q_proj', 'v_proj'] | bert_lora=48 | decoder_keys=54
  vocab_projector.weight std=0.02539 (default init ~0.0208)

/models/best_cls_only.pt
  modified=2026-10-08 07:32 | size=1084.2 MB | md5=f90d14e9824316e54f6570519ed405f6
  keys=560 | vit_lora=48 ['q_proj', 'v_proj'] | bert_lora=48 | decoder_keys=54
  vocab_projector.weight std=0.02084 (default init ~0.0208)

=== Result / log files (csv, txt, json, md) ===
(none)
```

## Findings

1. **Only one multi-task checkpoint exists** — `disasternet_multitask_v2.pth`, trained 2026-06-25 (identical MD5 on laptop and Drive; the root-level copy is a duplicate).
2. **It predates every training fix.** `train.py` at that date (commit `21f1652`):
   - `LAMBDA_CAP = 0.2` (not 0.5 as documented)
   - unshifted teacher forcing (decoder input == target → autoencoding identity bias)
   - old `data_loader.py`: unseeded `random_split` 80/20 (no test split)
3. **Consequence — test contamination.** The 376-sample test set was defined later (2026-09-12). Under an unseeded 80/20 random split, each test sample had an 80% chance of being in v2's training data → **~300 of 376 test samples (est.) were seen in training.** The exact overlap cannot be reconstructed (no seed).
   Affected numbers: multi-task Macro F1 77.95% (P 93.28%, R 73.03%, per-class F1, confusion matrix / Figure 4.1), all caption metrics (BLEU-1..4, METEOR, ROUGE-L, BERTScore-F1 60.35%), and the model captions used in human evaluation.
4. **`best_cls_only.pt` decoder is untrained — confirmed.** `vocab_projector.weight` std = 0.02084 ≈ PyTorch default init 0.0208 (v2, whose decoder was trained, = 0.02539). Cross-attention maps from this checkpoint (Figure 4.5, Cross-Attention column of Figure 4.7) come from random weights.
5. CLS-only, ablation, baseline, XAI and calibration numbers are **not** affected by finding 3 — those scripts used the seeded 80/10/10 split.
6. No result/log files are kept in Drive; all result CSVs live in the repo under `results/`.
