<div align="center">

# DRISHTI-Bn

### An Explainable Multimodal Deep Learning Framework for Bengali Disaster Image Classification and Text Generation

B.Sc. thesis (Software Engineering), Daffodil International University · Jahid Hasan · Supervisor: Md. Fahad Hossain

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers%205.x-FFD21F)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2)](https://github.com/huggingface/peft)

</div>

---

## Overview

DRISHTI-Bn classifies disaster images from social media into three humanitarian categories
(**Severe Damage**, **Humanitarian Rescue**, **Affected People**), explains each decision with
Grad-CAM, and includes an experimental branch that generates Bengali text from the image.

- **Encoders:** ViT-Base/16 (`google/vit-base-patch16-224-in21k`) and BanglaBERT (`csebuetnlp/banglabert`),
  both adapted with **LoRA** (r = 8, α = 16, on the attention query/value projections).
- **Classifier:** concatenation of the ViT pooler output and the BanglaBERT `[CLS]` vector → MLP → 3 classes.
- **Text branch (multi-task model):** 3-layer Transformer decoder cross-attending to ViT patches,
  trained with right-shifted teacher forcing; L = 1.0·L_cls + 0.5·L_cap.
- **Explainability:** Grad-CAM, Attention Rollout, decoder cross-attention, temperature scaling,
  and a deletion-based faithfulness test.
- **Hardware:** a single NVIDIA T4 (16 GB), Google Colab.

> **About the numbers in this repository.** All headline results come from a **leakage-free,
> tweet-grouped split** and are averaged over **three seeds**. Earlier versions of this README and
> project reported higher numbers that were inflated by data leakage and several bugs; these were
> identified and corrected before the defense. The full audit trail is in
> [`DEFENSE_FIX_PLAN.md`](DEFENSE_FIX_PLAN.md), and the claim-by-claim corrections are in
> [`docs/corrected_claims.md`](docs/corrected_claims.md).

---

## Dataset

A Bengali version of four [CrisisMMD](https://crisisnlp.qcri.org/crisismmd) events (Alam et al., 2018).

| Event | Images | Share |
|-------|-------:|------:|
| Hurricane Harvey | 1,416 | 37.7% |
| Hurricane Maria | 1,149 | 30.6% |
| Hurricane Irma | 1,010 | 26.9% |
| Sri Lanka floods | 180 | 4.8% |
| **Total** | **3,755** | 3,189 unique tweets |

- Classes: Severe Damage 2,240 · Humanitarian Rescue 1,185 · Affected People 330.
- The Bengali texts are **machine translations** of the original tweets (`monirbishal/en-bn-nmt`);
  they were not human-verified. About 8.9% remained untranslated and 97.3% contain URLs.
- **Split:** 80/10/10, grouped so that no tweet and no identical text appears in more than one split
  (union-find over tweet ID and caption; best class balance among 200 `GroupShuffleSplit` draws).
  Train 3,007 · val 373 · **test 375**. File: [`results/defense_fixes/phase3/grouped_split.csv`](results/defense_fixes/phase3/grouped_split.csv).

---

## Results (grouped test set, n = 375)

### Classification

| Model | Seeds | Macro F1 (%) | Trainable params |
|-------|-------|--------------|------------------|
| ViT-Only (ViT-Base, last 2 blocks + pooler) | 3 | 73.27 ± 0.39 | 15.16M |
| CrisisViT-style (our re-implementation, ViT-Large, last 4 blocks) | 1 | 75.87 | 51.96M |
| **DRISHTI-Bn (CLS-Only, LoRA r = 8)** | 3 | **75.69 ± 1.33** | **1.38M** |

DRISHTI-Bn matches the ViT-Large baseline while training 38× fewer parameters.
Per-class F1 (mean of 3 seeds): Severe Damage 93.28 · Humanitarian Rescue 88.31 · Affected People 45.48 ± 4.63
(the smallest class, 32 test images, remains the hardest).

**Text ablation.** Replacing each tweet with another sample's tweet does not lower Macro F1
(76.63% vs 74.83% for seed 42; 96.5% of predictions unchanged), and a fixed placeholder sentence gives 76.52%.
The classifier is therefore effectively **image-driven**; the demo uses the image with a placeholder text.

### Explainability

Deletion test on all 375 test images — the top-k% patches ranked by each method are removed and the drop
in the predicted-class probability is measured:

| Method | Top 10% removed | Top 50% removed | Accuracy after 50% |
|--------|-----------------|-----------------|--------------------|
| **Grad-CAM** | **6.92 pp** | **21.27 pp** | **73.87%** |
| Attention Rollout | 0.85 pp | 7.35 pp | 81.60% |
| Random patches | 0.34 pp | 4.16 pp | 86.93% |

Grad-CAM is computed on the input of the last ViT block (the last block's output patch tokens receive
zero gradient because the pooler reads only the `[CLS]` token).

**Calibration:** ECE 2.57% without post-hoc scaling (mean confidence 88.84% vs accuracy 88.27%);
temperature scaling (T* = 1.1349) does not improve it (ECE 2.99%).

### Bengali text generation (multi-task model, experimental)

| Metric (sacreBLEU 2.6.0, beam 3) | Raw references | URLs/mentions/hashtags removed |
|----------------------------------|----------------|--------------------------------|
| Corpus BLEU | 14.38 | 0.11 |
| chrF | 14.19 | 8.46 |

Because the references are translated tweets, the decoder learned to imitate tweet conventions
(hashtags, mentions, URL fragments) instead of describing images. On chrF it scores below a random
training tweet (21.58) and a single constant tweet (20.46). A human evaluation of an earlier checkpoint
(100 samples, 3 annotators) rated generated text 1.58 / 1.24 / 1.23 out of 5 (fluency / adequacy /
faithfulness) versus 3.42 / 3.08 / 3.23 for the references. **The text branch is a limitation of this work,
not a usable captioning system.**

---

## Reproducing the results

Heavy steps run on Google Colab (T4). Each script is self-contained: paste it into one cell. It clones this
repository, reads `processed.zip` and the checkpoints from `My Drive/DisasterNet/`, and writes results back to Drive.

| Step | Script | Output |
|------|--------|--------|
| Grouped split, CLS-Only ×3 seeds, ViT-Only ×3, CrisisViT-style | [`colab/phase3_grouped_split_seeds.py`](colab/phase3_grouped_split_seeds.py) | `models/cls_only_grouped_s42.pt`, seed results |
| Multi-task model + benchmark | [`colab/phase3b_multitask_retrain.py`](colab/phase3b_multitask_retrain.py) | `models/drishti_multitask_v3.pt` |
| Text-ablation evaluation | [`colab/phase2_fair_evaluation.py`](colab/phase2_fair_evaluation.py) | Figure 4.1 |
| XAI, faithfulness, calibration | [`colab/phase4_xai_calibration.py`](colab/phase4_xai_calibration.py) | Figures 4.4–4.8 |

Local (CPU) steps:

```bash
python src/caption_metrics_v3.py      # sacreBLEU / chrF with trivial baselines
python src/human_eval_unblind.py      # human-evaluation scores with the correct A/B mapping
python src/make_phase5_figures.py     # Figures 4.2, 4.3, 4.9
```

Results and logs: [`results/defense_fixes/`](results/defense_fixes/).

---

## Demo

```bash
pip install -r requirements.txt
# copy cls_only_grouped_s42.pt and drishti_multitask_v3.pt from Drive into models/
python app.py        # http://127.0.0.1:7860
```

The demo shows the predicted class with probabilities, a Grad-CAM overlay, and the **unedited**
output of the text decoder.

---

## Project structure

```
src/
  model.py                 multi-task architecture (ViT + BanglaBERT + LoRA + decoder)
  cls_only_ablation.py     classification-only model (main model) and training loop
  train.py                 multi-task training loop (right-shifted teacher forcing)
  data_loader.py           dataset and tokenisation
  evaluate.py              predictor (beam search) used by the benchmark
  metrics.py               legacy caption metrics (BLEU here is non-standard; see caption_metrics_v3.py)
  caption_metrics_v3.py    standard sacreBLEU / chrF evaluation
  human_eval_unblind.py    corrected human-evaluation analysis
  make_phase5_figures.py   thesis figures 4.2, 4.3, 4.9
colab/                     self-contained Colab scripts for every GPU experiment
results/defense_fixes/     all corrected results, by phase
docs/corrected_claims.md   what may and may not be claimed, with wording
figures_superseded/        figures from before the corrections (do not use)
app.py                     Gradio demo
```

---

## Limitations

- Mostly hurricane imagery from the US and Caribbean; no images from Bangladesh, so generalisation to
  Bangladeshi floods is untested.
- Bengali texts are unverified machine translations of tweets.
- The classifier makes little use of the text, so the gain from multimodality is not demonstrated.
- The text-generation branch does not produce image descriptions.
- Affected People is small (330 images) and has the lowest F1.
- The ablation study (LoRA rank, No-LoRA, No-CrossAttn) was run once on the original split only.

---

## References

- Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. *ACL*.
- Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter datasets from natural disasters. *ICWSM*.
- Bhattacharjee, A., et al. (2022). BanglaBERT: Language model pretraining and benchmarks for low-resource language understanding evaluation in Bangla. *Findings of NAACL*.
- Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.

---

## Citation

```bibtex
@thesis{hasan2026drishti,
  author = {Jahid Hasan},
  title  = {DRISHTI-Bn: An Explainable Multimodal Deep Learning Framework for
            Bengali Disaster Image Captioning and Classification},
  school = {Daffodil International University},
  year   = {2026},
  type   = {B.Sc. Thesis}
}
```
