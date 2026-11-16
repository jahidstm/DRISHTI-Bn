# থিসিস সারসংক্ষেপ (THESIS_SUMMARY — বাংলা সংস্করণ)
> **প্রসঙ্গ ডকুমেন্ট** — pre-defense সংশোধনের পর নতুন করে লেখা, ১৬ নভেম্বর ২০২৬।
> সংখ্যাগুলো [`CLAUDE.md`](CLAUDE.md) থেকে নেওয়া; কোন দাবি কীভাবে লিখতে হবে তা [`docs/corrected_claims.md`](docs/corrected_claims.md)-এ।
> এই ফাইলের সঙ্গে ওই দুটোর অমিল হলে ওই দুটোই সঠিক। ইংরেজি সংস্করণ: [`THESIS_SUMMARY.md`](THESIS_SUMMARY.md)।
> আগের সংস্করণ (সেপ্টেম্বর ২০২৬: n = 2-এ Macro F1 66.67%, BLEU-4 5.14, "human-verified" অনুবাদ) বাতিল।

---

## থিসিস

**DRISHTI-Bn: An Explainable Multimodal Deep Learning Framework for Bengali Disaster Image Captioning and Classification**

জাহিদ হাসান (ID 0242) · তত্ত্বাবধায়ক: মো. ফাহাদ হোসেন · B.Sc. in Software Engineering, ড্যাফোডিল ইন্টারন্যাশনাল ইউনিভার্সিটি

## এক অনুচ্ছেদে

ViT-Base/16 (ছবি) আর BanglaBERT (বাংলা text) — দুটোতেই LoRA (r = 8, α = 16) বসিয়ে দুর্যোগের ছবিকে
Severe Damage / Humanitarian Rescue / Affected People-এ ভাগ করা হয়। Leakage-মুক্ত, tweet-grouped test set-এ
(n = 375) classification-only মডেলের **Macro F1 75.69 ± 1.33%** (৩ seed); train হয় মাত্র 1.38M parameter
(≈ 0.70%) — ৩৮ গুণ বেশি parameter train করা ViT-Large baseline-এর সমান। মডেল কার্যত ছবি দেখেই সিদ্ধান্ত নেয়;
বাংলা text প্রায় কিছু যোগ করে না। Grad-CAM ব্যাখ্যা deletion test-এ নির্ভরযোগ্য (random-এর ~২০ গুণ), আর মডেল
এমনিতেই ভালোভাবে calibrated (ECE 2.57%)। Multi-task সংস্করণ বাংলা text তৈরি করে, কিন্তু তা tweet-এর অনুকরণ,
ছবির বর্ণনা নয়। বাংলাদেশের ৩২৫টা আলাদাভাবে লেবেল করা বন্যা/ঘূর্ণিঝড়ের ছবিতে Macro F1 81.98% (CI 76.1–87.2) —
প্রাথমিক যাচাই, যার ফল annotator-এর ওপর নির্ভর করে।

---

## ডেটা

| বিষয় | বিবরণ |
|------|--------|
| উৎস | CrisisMMD v2.0 (Alam et al.), humanitarian category-র ছবি |
| আকার | ৩,৭৫৫টা ছবি: Severe_Damage 2,240 · Humanitarian_Rescue 1,185 · Affected_People 330 (8.8%) |
| ঘটনা | 95.2% hurricane (Harvey, Maria, Irma — যুক্তরাষ্ট্র/ক্যারিবিয়ান), 4.8% Sri Lanka-র বন্যা; training-এ **বাংলাদেশের কোনো ছবি নেই** |
| বাংলা text | Tweet-এর মেশিন অনুবাদ (`monirbishal/en-bn-nmt`), **মানুষ দিয়ে যাচাই করা নয়**; 8.9% অনুবাদ হয়নি, 97.3%-এ URL |
| Split | Tweet/caption-grouped 80/10/10 → train 3,007 / val 373 / test 375 |
| পুরনো split | Un-grouped (test n = 376): test-এর 21.3% text train-এও ছিল → ফুলানো সংখ্যা, বাতিল |
| External test | Wikimedia Commons-এর বাংলাদেশের ৩২৫টা ছবি, ৩ জন annotator, Fleiss' κ 0.613 (`bd_external/`) |

## মডেল

| মডেল | কী | Trainable |
|-------|-----------|-----------|
| **CLS-Only (মূল)** | ViT-B/16 + BanglaBERT, q/v-তে LoRA; concat(ViT pooler, BERT [CLS]) → MLP 1536→512→3 | 1,379,331 (≈ 0.70%) |
| Multi-Task v3 | একই encoder + ৩-layer Transformer decoder (৮ head), ViT patch-এর ওপর cross-attention; loss = CE + 0.5·NLL | 74.79M |

Training: AdamW, lr 1e-4, weight decay 0.01, cosine schedule, batch 16; CLS-Only ১০ epoch (seed 42/123/2024),
Multi-Task ১৫ epoch (seed 42); একটা T4 GPU (Colab)।

---

## ফলাফল (grouped test set, n = 375)

**Classification**

| মডেল | Macro F1 | Trainable |
|-------|----------|-----------|
| **DRISHTI-Bn CLS-Only** (৩ seed) | **75.69 ± 1.33%** | 1.38M |
| CrisisViT-style (ViT-L, আমাদের পুনর্নির্মাণ, ১ seed) | 75.87% | 51.96M |
| ViT-Only (৩ seed) | 73.27 ± 0.39% | 15.16M |
| Multi-Task v3, ছবি + tweet (১ seed) | 77.29% | 74.79M |
| Multi-Task v3, ছবি + নিজের তৈরি caption | 73.63% | 74.79M |

Class অনুযায়ী F1 (CLS-Only): SD 93.28 · HR 88.31 · **AP 45.48 ± 4.63** (সবচেয়ে দুর্বল)।
Text ablation (seed 42): tweet 74.83 · placeholder 76.52 · অন্য sample-এর tweet 76.63 → মডেল image-driven।

**Explainability** — deletion test, শীর্ষ ৫০% patch সরালে predicted class-এর probability কতটা কমে:
Grad-CAM 21.27 pp · Attention Rollout 7.35 pp · random 4.16 pp। Decoder-এর cross-attention Grad-CAM-এর সঙ্গে
শুধু chance মাত্রায় মেলে (IoU 0.059 বনাম 0.054) — তাই এটা শুধু caption তৈরির ব্যাখ্যা।

**Calibration** — ECE 2.57% (T = 1); temperature scaling (T* = 1.1349)-এ 2.99% → উন্নতির দরকার নেই।

**Caption (Multi-Task v3)** — raw tweet reference-এ sacreBLEU 14.38 / chrF 14.19; URL/hashtag/mention বাদে
0.11 / 8.46 — random-tweet baseline-এর চেয়েও কম। 96.8% output-এ URL-এর টুকরো। Human evaluation (১০০ sample,
৩ annotator, পুরনো v2 caption): মডেল 1.24–1.58 / 5, reference 3.08–3.42 / 5; ৯৪.৯% বিচারে reference পছন্দ।

**বাংলাদেশ external test (প্রাথমিক)** — Macro F1 81.98% (95% CI 76.1–87.2), accuracy 89.85%, AP F1 58.8;
একেকজন annotator-এর লেবেলে 81.65 / 81.36 / 55.82% → লেবেল-নির্ভর, domain shift-এর মাপ নয়।

---

## যা দাবি করা যাবে

- Parameter-দক্ষ: ৩৮ গুণ কম trainable parameter-এ ViT-Large baseline-এর সমান (SOTA ছাড়িয়ে যাওয়া নয়)।
- Grad-CAM ব্যাখ্যা নির্ভরযোগ্য (deletion test), calibration এমনিতেই ভালো।
- সৎ, leakage-মুক্ত মূল্যায়ন; বাংলাদেশের ছোট external set-এ পারফরম্যান্স কমেনি।
- বাংলা text তৈরির অংশ আছে, কিন্তু তা ছবির বর্ণনা **নয়**।

কখনো লেখা যাবে না: "human-verified translation", "text আর image পরস্পরের পরিপূরক", "multi-task classification
উন্নত করে", "বাংলাদেশে generalize করে", 83.21%, "XAI accuracy 81%"।

## প্রধান সীমাবদ্ধতা

মূল split-এ leakage (ঠিক করা হয়েছে) · seed-ভেদে ±1.3 pp · AP class ছোট ও দুর্বল · text হলো tweet-এর মেশিন অনুবাদ ·
classifier text প্রায় ব্যবহার করে না · caption tweet-অনুকরণ · human eval পুরনো v2 caption-এ · ablation শুধু পুরনো split-এ,
এক seed-এ · external test ছোট ও annotator-নির্ভর।

---

## গুরুত্বপূর্ণ ফাইল

| Path | কাজ |
|------|---------|
| `CLAUDE.md` | অনুমোদিত সংখ্যা (মূল উৎস) |
| `docs/corrected_claims.md` | দাবির ভাষা, বাদ দেওয়া দাবি, সীমাবদ্ধতা |
| `DEFENSE_FIX_PLAN.md` | Phase-ভিত্তিক সংশোধনের লগ |
| `docs/defense_qa.md` | Defense-এর সম্ভাব্য প্রশ্নোত্তর |
| `src/cls_only_ablation.py` · `src/model.py` | CLS-Only ও Multi-Task মডেল |
| `colab/phase*.py` | GPU পরীক্ষা (Phase 0–4) |
| `src/bd_*.py` | বাংলাদেশ external test set |
| `figures/` | থিসিসের চূড়ান্ত figure |
| `app.py` | Gradio demo |
