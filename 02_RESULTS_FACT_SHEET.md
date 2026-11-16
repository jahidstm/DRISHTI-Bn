# 02_RESULTS_FACT_SHEET — DRISHTI-Bn

> **Single source of truth for every number and claim about this study** (version 16 November 2026).
> It replaces the old `02_BENCHMARK_RAW_DATA.txt` (n = 2, Macro F1 66.67%, BLEU-4 5.14) and the old
> `human_eval_report.md` (merge bug) — both are wrong and must not be used.
> If any other source in the notebook shows a different number for this study, this file is correct.

---

## 1. Study at a glance

| Item | Value |
|------|-------|
| Title | DRISHTI-Bn: An Explainable Multimodal Deep Learning Framework for Bengali Disaster Image Captioning and Classification |
| Task 1 | Three-class humanitarian image classification: Severe Damage (SD), Humanitarian Rescue (HR), Affected People (AP) |
| Task 2 | Bengali text generation from the image (multi-task variant; experimental) |
| Main model | CLS-Only: ViT-Base/16 + BanglaBERT, both with LoRA (r = 8), concatenation + MLP head |
| Main result | Macro F1 **75.69 ± 1.33%** (3 seeds, leakage-free grouped test set, n = 375) |
| Trainable parameters | **1,379,331 (≈ 0.70%)** of the full model |
| Hardware | One NVIDIA T4 GPU (16 GB VRAM), Google Colab Pro; Bangladesh evaluation on a laptop CPU |
| Software | PyTorch 2.14.1, Hugging Face Transformers 5.22.1, PEFT 0.21.2, scikit-learn, sacreBLEU 2.6.0 |

---

## 2. Dataset

### 2.1 Source and classes
- Source: **CrisisMMD v2.0** (Alam et al., 2018) — tweets with images from 2017 disasters, annotated with humanitarian categories.
- Events in the final corpus: **95.2% hurricanes** (Harvey, Irma, Maria; United States and Caribbean) and **4.8% Sri Lanka floods**. **No images from Bangladesh** in training data.
- Class mapping from CrisisMMD humanitarian categories:

| Class (this study) | CrisisMMD categories merged | Images | Share |
|--------------------|-----------------------------|--------|-------|
| Severe_Damage (SD) | infrastructure_and_utility_damage, vehicle_damage | 2,240 | 59.7% |
| Humanitarian_Rescue (HR) | rescue_volunteering_or_donation_effort | 1,185 | 31.6% |
| Affected_People (AP) | affected_individuals, injured_or_dead_people, missing_or_found_people | 330 | 8.8% |
| **Total** | (not_humanitarian, other_relevant_information excluded) | **3,755** | |

### 2.2 Bengali text
- The Bengali text of each sample is a **machine translation of the original English tweet** produced with the open-source English→Bengali model `monirbishal/en-bn-nmt` (Hugging Face).
- **Not human-verified.** 8.9% of texts remained (partly) untranslated English; 97.3% contain URL fragments; most contain hashtags and mentions.
- The text is a tweet, **not a description of the image**. This matters for caption evaluation.

### 2.3 Leakage-free (grouped) split
- Problem found in the original random split: **80 of 376 test samples (21.3%) had the identical text of a training sample** (several images of the same tweet, retweets, identical captions) → inflated scores.
- Fix: samples linked by the same tweet ID **or** identical text were merged into **3,189 leakage groups** (union-find). Groups were split 80/10/10 with GroupShuffleSplit; the most class-balanced of 200 random splits was kept. **Zero tweet or text overlap** between splits.

| Split | n | SD | HR | AP |
|-------|---|----|----|----|
| Train | 3,007 | 1,794 (59.7%) | 949 (31.6%) | 264 (8.8%) |
| Validation | 373 | 222 (59.5%) | 117 (31.4%) | 34 (9.1%) |
| Test | 375 | 224 (59.7%) | 119 (31.7%) | 32 (8.5%) |

### 2.4 Preprocessing
- Images: resized to 224 × 224, ImageNet mean/std normalization (no augmentation).
- Text: BanglaBERT tokenizer (`csebuetnlp/banglabert`, 32,000 sub-word vocabulary), padded/truncated to 128 tokens.

---

## 3. Model

### 3.1 CLS-Only model (main)
- **Vision encoder:** ViT-Base/16 (`google/vit-base-patch16-224-in21k`): 12 layers, 12 heads, hidden size 768, 196 patch tokens + [CLS]. Pre-trained weights frozen.
- **Text encoder:** BanglaBERT (ELECTRA discriminator, 12 layers, hidden size 768). Pre-trained weights frozen.
- **LoRA:** rank r = 8, α = 16, dropout 0.1, on the query and value projections of every attention layer in both encoders. Update: h = W₀x + (α/r)·BAx, A ∈ ℝ^(r×d), B ∈ ℝ^(d×r), B initialized to zero.
- **Fusion and head:** concatenate ViT pooled [CLS] vector and BanglaBERT [CLS] vector (768 + 768 = 1536) → Linear(1536, 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512, 3) → softmax.
- **Trainable parameters:** LoRA 589,824 (ViT 294,912 + BanglaBERT 294,912; 0.30% of the encoders) + head 789,507 = **1,379,331 ≈ 0.70%** of all parameters.

### 3.2 Multi-Task v3 model (caption branch)
- Same encoders and head, plus a **caption decoder**: 3 Transformer decoder layers, 8 heads, feed-forward size 2048, dropout 0.1, sinusoidal positional encoding, token embedding 32,000 × 768, output projection 768 → 32,000.
- The decoder **cross-attends to the 196 ViT patch tokens** and generates BanglaBERT sub-word tokens autoregressively (causal mask, right-shifted teacher forcing).
- Decoding: beam search, beam width 3, maximum 40 tokens, length normalization exponent 0.7, repetition penalty and bigram blocking.
- Trainable: 74.79 M parameters (decoder 72.82 M, head 0.79 M, LoRA + trainable ViT pooler 1.18 M).

### 3.3 Training
| Setting | CLS-Only | Multi-Task v3 |
|---------|----------|---------------|
| Loss | cross-entropy | L = 1.0·L_cls + 0.5·L_cap (token negative log-likelihood) |
| Optimizer | AdamW, lr 1 × 10⁻⁴, weight decay 0.01 | same |
| Schedule | cosine annealing | cosine annealing |
| Batch size / epochs | 16 / 10 | 16 / 15 |
| Checkpoint | epoch with best validation Macro F1 | epoch with lowest validation total loss (epoch 4) |
| Seeds | 42, 123, 2024 | 42 |
| Precision | mixed precision (AMP) | mixed precision (AMP) |

---

## 4. Results — classification (grouped test set, n = 375)

### 4.1 Main result and baselines
| Model | Seeds | Macro F1 | Macro Precision | Macro Recall | Accuracy | Trainable params |
|-------|-------|----------|-----------------|--------------|----------|------------------|
| **DRISHTI-Bn CLS-Only (LoRA r = 8)** | 42 / 123 / 2024 | **75.69 ± 1.33%** | 77.09 ± 1.31% | 74.93 ± 1.59% | 88.00 ± 0.27% | 1.38 M (≈ 0.70%) |
| ViT-Only (ViT-Base, last 2 blocks + pooler fine-tuned) | 3 | 73.27 ± 0.39% | 74.64 ± 0.87% | 72.38 ± 0.75% | 85.60 ± 0.27% | 15.16 M |
| CrisisViT-style (our re-implementation: ViT-Large, last 4 blocks + pooler) | 1 (42) | 75.87% | 76.27% | 75.53% | 86.93% | 51.96 M |
| Multi-Task v3, image + original tweet | 1 (42) | 77.29% | 79.03% | 76.12% | — | 74.79 M |
| Multi-Task v3, image + its own generated caption (image-only use) | 1 (42) | 73.63% | 76.77% | 72.12% | — | 74.79 M |

- Per-seed Macro F1 — CLS-Only: 74.83 / 75.02 / 77.22; ViT-Only: 73.69 / 73.21 / 72.91. CLS-Only is higher than ViT-Only in all three seeds (mean +2.42 pp).
- CLS-Only and CrisisViT-style are **tied** (within one standard deviation); CLS-Only trains **38× fewer parameters** on a smaller backbone. Supported claim: parameter efficiency, not superiority.
- Multi-Task 77.29% is a single seed inside the CLS-Only seed range → **no evidence that multi-task learning improves classification.**
- Image-only multi-task (73.63%) ≈ ViT-Only → the generated caption adds no information beyond the image.

### 4.2 Per-class results
| Class | CLS-Only F1 (mean of 3 seeds) | ViT-Only F1 | CrisisViT-style F1 | MT v3 (tweet) | MT v3 (generated) |
|-------|------------------------------|-------------|--------------------|---------------|-------------------|
| Severe Damage | 93.28 ± 0.15% | 91.46% | 93.63% | 92.00% | 92.51% |
| Humanitarian Rescue | 88.31 ± 0.82% | 84.69% | 83.98% | 87.24% | 86.07% |
| Affected People | **45.48 ± 4.63%** | 43.66% | 50.00% | 52.63% | 42.31% |

Affected People is the weakest class in every model (only 32 test images); its seed variance drives the Macro F1 variance.

### 4.3 Confusion matrix — CLS-Only seed 42 (Macro F1 74.83%; rows = true, columns = predicted; order SD, HR, AP)
| True \ Predicted | SD | HR | AP |
|------------------|----|----|----|
| SD (224) | 210 (94%) | 8 | 6 |
| HR (119) | 6 | 110 (92%) | 3 |
| AP (32) | 11 | 10 | 11 (34%) |

Affected People images are confused with both other classes in roughly equal parts.

### 4.4 Does the Bengali text help? (text ablation, CLS-Only seed 42, same test images)
| Text given to the model at test time | Macro F1 |
|---------------------------------------|----------|
| Original translated tweet | 74.83% |
| Caption generated by Multi-Task v3 | 76.41% |
| Fixed placeholder sentence "বন্যা পরিস্থিতি" ("flood situation") | 76.52% |
| Tweet of a **different, random** test sample | 76.63% (96.5% of predictions unchanged) |
| Empty text | 61.95% (out-of-distribution input) |

→ The classifier is **effectively image-driven**: the content of the text does not matter. "Multimodal fusion improves classification" cannot be claimed.

### 4.5 Ablation study (ORIGINAL un-grouped split, single seed — not re-run; label as such)
| Configuration | Macro F1 | Note |
|---------------|----------|------|
| No LoRA (frozen encoders, head only) | 77.74% | |
| LoRA r = 4 | 79.41% | |
| LoRA r = 8 (rank-ablation run) | 81.47% | same configuration as CLS-Only below |
| LoRA r = 16 | 79.14% | |
| Decoder without cross-attention | 81.88% | decoder is not used for classification → difference = run-to-run variance |
| CLS-Only (r = 8) | 83.21% | leaky split |
| Caption-only (λ_cls = 0) | 16.90% | classifier head never trained → exclude or footnote |

Two runs of the identical r = 8 configuration gave 83.21% and 81.47% (≈ 1.7 pp run-to-run variance). Differences below ~2.5 pp are not meaningful. LoRA helps over no LoRA (77.74% vs 79–83%), but the rank choice (4/8/16) is not resolved.
Also original split only: stratified random baseline 39.95%, BanglaBERT-only (text only) 43.02%, full fine-tuning 74.19%.

### 4.6 Effect of data leakage (for Section 4.2 only)
| Model | Original split (leaky, n = 376) | Grouped split (n = 375) | Change |
|-------|--------------------------------|-------------------------|--------|
| CLS-Only | 83.21% (1 seed) | 75.69 ± 1.33% (3 seeds) | −7.5 pp |
| ViT-Only | 78.64%* | 73.27 ± 0.39% | −5.4 pp |
| CrisisViT-style | 78.15%* | 75.87% | −2.3 pp |
\* earlier baselines also had a parameter-freezing bug (only the pooler was trained); corrected in the grouped runs.

---

## 5. Results — Bengali text generation (Multi-Task v3, grouped test n = 375, beam width 3)

| Metric (sacreBLEU 2.6.0) | Raw references (tweets) | Cleaned references (no URL, RT, @, #) |
|--------------------------|-------------------------|----------------------------------------|
| Corpus BLEU | 14.38 | 0.11 |
| chrF | 14.19 | 8.46 |
| BERTScore-F1 (multilingual BERT) | 69.66% | 62.27% |

Reference points on the same references (chrF raw / clean): a random training tweet 21.58 / 11.93; one constant tweet for all images 20.46 / 15.19 → **the model scores below both trivial baselines.**
- 96.8% of generated outputs contain URL fragments and 99.2% contain hashtags: the decoder **learned to imitate tweets**, not to describe images. The raw BLEU of 14.38 comes from matching shared tweet patterns (URLs, hashtags, event names).
- Example outputs (reference → generated):
  1. Ref: "ঘূর্ণিঝড় #মোরা #আরসিওয়াই রাঙ্গামাটি ফায়ার সার্ভিস বিডি দিয়ে কাজ করছে। https://t.co/…" → Gen: "# Hurrican Harvey ' ss : / t. co / … https : Rico Irma এর মাধ্যমে হ্যারিকেন মারিয়া"
  2. Ref: "#পুয়ের্টোরিকো #হুরিকানেমারিয়াকে মুছে ফেলেছে বলে মনে হচ্ছে …" → Gen: "# HurricanMaria এর পরে : https : / t. co / …"
  3. Ref: "পুয়ের্তো রিকোতে, ১০,০০০ রেডিও … সাহায্য করেছে https://t.co/…" → Gen: "# HurricanMaria এর পর, কিন্তু হ্যারিকেন হার্ভি ত্রাণের জন্য সরবরাহ করেছে https : / t. co / …"
- Root cause: the training references are tweets (C10/C18). Image description requires human-written descriptive references (future work).

## 6. Human evaluation (earlier v2 captions)
- Design: 100 test samples (stratified by class), 3 native Bengali-speaking annotators, **blind A/B presentation** (model caption and reference tweet in random order), each rated on a 1–5 scale for **Fluency**, **Adequacy** (information content) and **Faithfulness** (no false content), plus a preference judgement.
- The captions rated were produced by the **earlier v2 checkpoint**, not v3 (time constraint; v3 assessed only with automatic metrics). An un-blinding bug in the first analysis was fixed (A/B order had been ignored).

| Dimension | Model | Reference (tweet) | Wilcoxon signed-rank p |
|-----------|-------|-------------------|------------------------|
| Fluency | 1.58 / 5 | 3.42 / 5 | 2.1 × 10⁻¹⁷ |
| Adequacy | 1.24 / 5 | 3.08 / 5 | 9.7 × 10⁻¹⁸ |
| Faithfulness | 1.23 / 5 | 3.23 / 5 | 1.8 × 10⁻¹⁷ |

- Per class (model Fluency / Adequacy / Faithfulness): SD 1.63 / 1.28 / 1.29 · HR 1.53 / 1.21 / 1.16 · AP 1.46 / 1.08 / 1.08.
- Preference: reference preferred in **225 of 237** judgements (94.9%).
- Inter-annotator agreement: pooled over model + reference (200 items) Krippendorff's α = **0.38**, quadratic-weighted κ = 0.40 (fair); within one caption type agreement ≈ 0 (scores cluster at the low end).

---

## 7. Explainability (CLS-Only seed 42; decoder cross-attention from Multi-Task v3)

### 7.1 Methods
- **Grad-CAM:** gradients of the predicted-class logit with respect to the patch tokens **entering the last ViT block**; channel weights = mean gradient; map = ReLU of the weighted sum → 14 × 14 → upsampled to 224 × 224. (Hooking the output of the last block gives zero gradients for patch tokens because only [CLS] feeds the classifier; an earlier version had this bug and produced blank maps.)
- **Attention Rollout:** attention of all 12 ViT layers, averaged over heads, lowest 90% of weights discarded, identity added, re-normalized, multiplied through the layers; [CLS]-to-patch row used.
- **Decoder cross-attention:** mean over 3 layers × 8 heads × generated tokens of the attention to the 196 patches. Explains caption generation only.

### 7.2 Qualitative samples
21 test images drawn at random (7 per class, seed 42, not cherry-picked): 15 / 21 classified correctly (SD 6/7, HR 7/7, AP 2/7). Mean confidence of the errors 0.58 vs 0.89 for correct predictions. **Never call 15/21 an "XAI accuracy".**
Figures 4.4–4.7 show, from these 21 images, the most confident correct prediction per class (#1 SD, #10 HR, #18 AP) and the most confident error (#21, AP predicted as SD). Observation: for #18 Grad-CAM highlights the flood water rather than the people.

### 7.3 Faithfulness — deletion test (all 375 test images)
The 16 × 16 patches ranked highest by each map are removed (set to zero after normalization, i.e. the ImageNet mean color); the drop in the predicted-class probability is measured. Accuracy before removal: 88.27%.

| Method | Top 10% removed | Top 25% | Top 50% | Accuracy after 50% removed |
|--------|-----------------|---------|---------|----------------------------|
| **Grad-CAM** | **6.92 pp** | **14.27 pp** | **21.27 pp** | **73.87%** |
| Attention Rollout | 0.85 pp | 3.26 pp | 7.35 pp | 81.60% |
| Random patches | 0.34 pp | 1.57 pp | 4.16 pp | 86.93% |

→ Grad-CAM identifies the regions the classifier relies on (≈ 20× the effect of random patches at 10%); Rollout is weak.

### 7.4 Agreement between methods (top-10% patch IoU)
Grad-CAM vs Rollout 0.148 (2.7× chance) · Grad-CAM vs decoder cross-attention 0.059 (≈ chance level 0.054) → cross-attention does not explain the classification decision.

## 8. Calibration (CLS-Only seed 42; temperature fitted on validation n = 373 by NLL; test n = 375; 10 bins)
| Metric | Before (T = 1) | After temperature scaling (T* = 1.1349) |
|--------|----------------|------------------------------------------|
| ECE | **2.57%** | 2.99% |
| MCE | 19.06% | 20.60% |
| NLL | 0.3223 | 0.3228 |
| Brier score | 0.1681 | 0.1688 |
| Mean confidence | 88.84% | 87.03% |
| Accuracy | 88.27% | 88.27% |

→ The model is already well calibrated; temperature scaling gives no improvement (T* close to 1).

---

## 9. External test on Bangladeshi images (preliminary)
- **Collection:** 419 candidate images downloaded from 6 Wikimedia Commons categories (Floods in Bangladesh, Cyclone Sidr, Cyclone Amphan, Cyclone Aila, Cyclone Remal, 2017 South Asian floods), Creative Commons or public-domain licenses; source URL, author and license recorded for each. Images are not redistributed.
- **Labelling:** 3 annotators labelled all 419 images **independently** with a written guideline (classes SD / HR / AP, or EX = exclude: maps, satellite images, non-Bangladesh, unrelated, sensitive). One annotator is the researcher. Agreement: pairwise Cohen's κ 0.856, 0.505, 0.521; **Fleiss' κ 0.613** (substantial); unanimous on 59.2%. Final label = majority vote; 10 images without a majority were decided jointly. No AI labels.
- **Final set:** 94 excluded → **n = 325** (HR 188, SD 102, AP 35). Near-duplicate check against all CrisisMMD images (perceptual dHash, Hamming distance ≤ 5): **0 duplicates**.
- **Evaluation:** CLS-Only seed 42, image + placeholder text "বন্যা পরিস্থিতি", no tuning on this set; 95% CI from 2,000 bootstrap resamples.

| Metric | Value |
|--------|-------|
| Macro F1 | **81.98%** (95% CI 76.1–87.2) |
| Accuracy | 89.85% (95% CI 86.5–92.9) |
| Macro precision / recall | 81.91% / 82.22% |
| Per-class F1 | SD 93.9 · HR 93.2 · AP 58.8 |
| Confusion (rows SD, HR, AP) | [100, 1, 1] · [4, 172, 12] · [7, 8, 20] |
| Mean confidence correct / wrong | 0.884 / 0.726 |

Sensitivity to the reference labels (same predictions): annotator 1 only 81.65% · annotator 2 only 81.36% · **annotator 3 only 55.82%** · unanimous images only 86.93% (n = 179). Most disagreement is on the Humanitarian Rescue vs Affected People boundary.
Interpretation: **no performance drop is observed** on this small set, but the estimate depends strongly on the annotator; it is **not a measure of domain shift** (label source and image style differ from CrisisMMD; Wikimedia photos are more professional than social-media images).

---

## 10. Contributions that may be claimed
1. A Bengali-text version of the CrisisMMD humanitarian image subset (3,755 images, machine-translated tweets) with a **leakage-free tweet-grouped split** that removes the 21.3% text overlap of a random split.
2. A **parameter-efficient** classifier (LoRA on ViT-Base and BanglaBERT, 1.38 M trainable parameters ≈ 0.70%) that matches a ViT-Large baseline trained with 38× more parameters and beats a ViT-Base baseline in every seed.
3. A **careful evaluation protocol**: multiple seeds, leakage analysis, a text ablation that shows the classifier is image-driven, calibration analysis.
4. **Quantitatively checked explanations**: a deletion test showing Grad-CAM is faithful, Rollout weak, decoder cross-attention unrelated to the classification decision.
5. A documented **negative result** on Bengali caption generation from tweet references, supported by automatic metrics, trivial baselines and human evaluation.
6. A **preliminary external test** on 325 independently labelled Bangladeshi disaster images.

## 11. DO NOT CLAIM
- "human-verified" or "manually verified" translation
- a "new annotated Bengali flood dataset" (the labels are CrisisMMD's; the text is machine-translated)
- "text and image are complementary" / "multimodal fusion improves accuracy"
- "multi-task learning improves classification"
- "the model generates accurate Bengali image descriptions / captions"
- "beats / outperforms the state of the art" or "outperforms CrisisViT"
- "the model generalizes to Bangladesh" or "+7.15 pp domain shift"
- "XAI accuracy 81%" or "17/21"
- any number from the original leaky split as a main result (83.21%, 77.95%, 78.64%, 78.15%, BLEU-4 6.13, METEOR 0.32, BERTScore 60.35%, human-eval 2.49/2.11/2.17, ECE 2.16%)
- "0.1%" or "0.30%" as the trainable share of the classifier (correct: ≈ 0.70%)
- "flood images" for the whole dataset (95.2% hurricanes; use "disaster images")

## 12. Limitations (for Section 5.4)
1. Original split leaked text between train and test; fixed with the grouped split, but the ablation study and three baselines exist only on the old split, with one seed.
2. Seed variance ≈ ±1.3 pp Macro F1; Affected People F1 varies ±4.6 pp (32 test images).
3. Bengali text is unverified machine translation of tweets; 8.9% untranslated.
4. The classifier makes little use of the text; the benefit of multimodality is not demonstrated.
5. Generated "captions" imitate tweets; no human-written image descriptions were available for training.
6. Human evaluation used the earlier v2 captions; 3 annotators without a calibration session; agreement fair (α = 0.38).
7. Class imbalance: Affected People only 8.8% of the data.
8. Temperature scaling fitted on a small validation set (n = 373).
9. Training data are mostly US/Caribbean hurricanes; the Bangladesh test is small (n = 325, AP 35), uses Wikimedia rather than social-media images, has no text, and depends on the annotator (one annotator is the researcher).
10. Single dataset, single GPU budget; no comparison with large vision-language models.

## 13. Figures available (file → number in the thesis)
| Thesis number | File | Content |
|---------------|------|---------|
| Figure 1.1 | figures/figure_1_1_flood_stats.png | Population of Bangladesh exposed to flood risk: 94.42 M of 164.2 M (57.5%) (Rentschler & Salhab, 2020) |
| Figure 1.2 | figures/figure_1_2_system_pipeline.png | Where DRISHTI-Bn sits in a disaster-information workflow; responder use is out of scope |
| Figure 2.1 | figures/figure_2_1_lora.png | LoRA update h = W₀x + (α/r)·BAx |
| Figure 3.1 | figures/figure_3_1_architecture.png | CLS-Only architecture (solid) and the multi-task caption branch (dashed) |
| Figure 3.2 | figures/figure_3_3_xai_pipeline.png | Explanation and reliability analysis (Grad-CAM, Rollout, cross-attention, deletion test, temperature scaling) |
| Figure 4.1 | figures/figure_4_1_confusion_matrix_grouped.png | Confusion matrix, CLS-Only seed 42, grouped test set |
| Figure 4.2 | figures/figure_4_2_baseline_comparison_grouped.png | Baselines on the grouped split, seed points + mean ± std |
| Figure 4.3 | phase3b_confusion_matrix.png | Multi-Task v3 confusion matrices (tweet vs generated caption) |
| Figure 4.4 | figures/figure_4_3_ablation_original_split.png | Ablation (original split, single seed) |
| Figure 4.5 | figures/figure_4_4_gradcam_grouped.png | Grad-CAM examples |
| Figure 4.6 | figures/figure_4_6_rollout_grouped.png | Attention Rollout examples |
| Figure 4.7 | figures/figure_4_5_crossattn_grouped.png | Decoder cross-attention examples |
| Figure 4.8 | figures/figure_4_7_integrated_xai_grouped.png | Side-by-side comparison of the three maps |
| Figure 4.9 | figures/figure_4_9_faithfulness_deletion.png | Deletion curves: Grad-CAM vs Rollout vs Random |
| Figure 4.10 | figures/figure_4_8_calibration_diagram_grouped.png | Reliability diagram before/after temperature scaling |
| Appendix B | xai_all_21_samples.png | All 21 random XAI samples |

File-name numbers differ from thesis numbers for some figures; always use the thesis number above.
