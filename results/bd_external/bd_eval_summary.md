# BD external test set — results

Candidates in CSV: 419 | labellers: 3 | labelled by all: 419
  Cohen's kappa A vs B = 0.856 (raw agreement 90.2%)
  Cohen's kappa A vs C = 0.505 (raw agreement 62.3%)
  Cohen's kappa B vs C = 0.521 (raw agreement 63.5%)
  Fleiss' kappa (all 3) = 0.613 | unanimous 59.2%
Excluded (EX): 94 | unlabelled: 0
Test images per class: Severe_Damage 102 | Humanitarian_Rescue 188 | Affected_People 35
Near-duplicates of CrisisMMD images (dHash Hamming <= 5): 0 ✔

## Classification (image + placeholder text, CLS-Only seed 42)
Macro F1 = 81.98%  (95% bootstrap CI 76.1–87.2, 2000 resamples, n = 325)
Accuracy = 89.85%  (95% CI 86.5–92.9)
Macro P = 81.91% | Macro R = 82.22%
Per-class F1: Severe_Damage 93.9 | Humanitarian_Rescue 93.2 | Affected_People 58.8
Confusion (rows true SD, HR, AP; cols pred): [[100, 1, 1], [4, 172, 12], [7, 8, 20]]
Mean confidence: correct 0.884 | wrong 0.726
Reference — same model on the CrisisMMD grouped test set: 74.83% (difference +7.15 pp; not a pure domain-shift measure — label source and image style differ)

## Sensitivity to the reference labels (same predictions)
  vs annotator 1 only: Macro F1 = 81.65% (n = 319)
  vs annotator 2 only: Macro F1 = 81.36% (n = 322)
  vs annotator 3 only: Macro F1 = 55.82% (n = 321)
  unanimous images only: Macro F1 = 86.93% (n = 179)
