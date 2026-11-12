# DRISHTI-Bn — Corrected Claims Reference

> **উদ্দেশ্য:** নতুন thesis draft লেখার সময় কোন দাবি কীভাবে লিখতে হবে, কোনটা বাদ যাবে আর কোনটা limitation হবে, তার একমাত্র reference।
> **তৈরি:** Phase 1, DEFENSE_FIX_PLAN.md (2026-11-12)
> **নিয়ম:** writing prompt বানানোর সময় এই ফাইল আগে পড়তে হবে। "⏳ pending" চিহ্নিত item-এর চূড়ান্ত লেখা পরের phase-এর ফল এলে update হবে।

**Status চিহ্ন:** ✅ চূড়ান্ত · ⏳ pending (কোন phase-এর ফলের অপেক্ষা) · ❓ তোমার উত্তর লাগবে

---

## A. যে সংখ্যা এখন ব্যবহার করা যাবে না

### C1. Multi-task মডেলের সব সংখ্যা — ⏳ Phase 3B
- **পুরনো দাবি:** Multi-task Macro F1 77.95% (P 93.28%, R 73.03%), per-class F1, confusion matrix (Figure 4.1), BLEU-1..4 = 24.85/14.20/8.95/6.13, METEOR 0.32, ROUGE-L 0.56, BERTScore-F1 60.35%; "λ_cap = 0.5"।
- **সমস্যা:** এগুলো এসেছে `disasternet_multitask_v2.pth` (২৫ জুন) থেকে, যা train হয়েছিল λ_cap = 0.2, unshifted teacher forcing আর seed ছাড়া 80/20 split দিয়ে। Test set-এর আনুমানিক ~300/376 sample মডেল training-এ দেখেছে। (প্রমাণ: `results/defense_fixes/phase0/checkpoint_audit.md`)
- **করণীয়:** Phase 3B-এর নতুন মডেলের সংখ্যা দিয়ে প্রতিস্থাপন। ততদিন এই সংখ্যাগুলো কোথাও লেখা যাবে না।

### C2. Human evaluation-এর model caption — ⏳ Phase 3B (সিদ্ধান্ত 3B.4)
- **সমস্যা:** ১০০টা model caption v2 থেকে, অর্থাৎ C1-এর একই সমস্যা।
- **দুই বিকল্প:** (ক) আগের ফল রেখে limitation হিসেবে লেখা; (খ) নতুন মডেলের caption দিয়ে আবার human eval।
- **(ক) বেছে নিলে লেখা (English):**
  > "The human evaluation was conducted on captions from an earlier multi-task checkpoint trained before the final data split was fixed; a portion of the evaluated samples may therefore have been seen during training. The scores should be read as an upper bound on caption quality."

---

## B. যে দাবি বাদ যাবে বা নতুনভাবে লিখতে হবে

### C3. Cross-Attention "সবচেয়ে অর্থপূর্ণ ব্যাখ্যা" — ⏳ Phase 4
- **পুরনো দাবি:** "Cross-Attention provides the most spatially meaningful explanations."
- **সমস্যা:** Map নেওয়া হয়েছে `best_cls_only.pt`-এর decoder থেকে, যা কখনো train হয়নি (λ_cap = 0)। প্রমাণ: `vocab_projector.weight` std = 0.02084 ≈ PyTorch default init 0.0208। তার ওপর classification-এ decoder ব্যবহারই হয় না।
- **আপাতত:** Figure 4.5 আর Figure 4.7-এর Cross-Attention column ব্যবহার করা যাবে না।
- **Phase 4-এর পর:** নতুন multi-task মডেলের trained decoder থেকে map বানিয়ে সেটাকে **caption generation-এর ব্যাখ্যা** হিসেবে উপস্থাপন (classification-এর নয়)। Classification-এর ব্যাখ্যা হিসেবে থাকবে শুধু Grad-CAM আর Attention Rollout।

### C4. No-CrossAttn ablation (+1.33pp) — ✅
- **পুরনো দাবি:** "Cross-attention contributes +1.33pp (81.88% → 83.21%)."
- **সমস্যা:** দুটো রানেই λ_cap = 0, আর classification path-এ decoder ব্যবহার হয় না। দুই মডেলের classification অংশ হুবহু এক; পার্থক্য শুধু random initialization-এর।
- **নতুন লেখা (English):**
  > "Replacing the cross-attention decoder with a self-attention-only decoder yielded 81.88% Macro F1. Since the decoder does not participate in the classification path when λ_cap = 0, this difference reflects run-to-run variance rather than a contribution of cross-attention."
- **বিকল্প:** ablation table থেকে সরিয়ে দেওয়া (সবচেয়ে পরিষ্কার)।

### C5. "r=8 Multi-Task (81.47%)" লেবেল — ✅
- **সমস্যা:** `lora_rank_ablation.py`-তে `LAMBDA_CAP = 0.0`, তাই এটা multi-task নয়; এটা CLS-only-র হুবহু একই config।
- **নতুন লেবেল:** "LoRA r = 8 (rank ablation run)"
- **গুরুত্বপূর্ণ পর্যবেক্ষণ:** একই config দুবার চালিয়ে 83.21% আর 81.47% → **~1.7pp run-to-run variance।** তাই single-seed ফলে ~2pp-এর কম পার্থক্য (r = 4: 79.41%, r = 16: 79.14%, No-CrossAttn: 81.88%) অর্থবহ বলা যাবে না।
- **নতুন লেখা (English):**
  > "Two independent runs of the identical r = 8 configuration produced 83.21% and 81.47%, indicating run-to-run variance of roughly 1.7 percentage points. Differences smaller than this should not be interpreted as meaningful."
- ⏳ Phase 3-এর ৩-seed ফল এলে mean ± std দিয়ে এটা প্রতিস্থাপিত হবে।

### C6. Caption-Only (16.90%) — ✅
- **সমস্যা:** λ_cls = 0, তাই classifier head কখনো train হয়নি। 16.90% একটা random head-এর ফল, কোনো তথ্য দেয় না।
- **করণীয়:** Classification ablation table থেকে সরানো। রাখতে চাইলে footnote:
  > "With λ_cls = 0 the classification head receives no gradient; its 16.90% Macro F1 corresponds to an untrained head and is reported only for completeness."

### C7. "XAI accuracy 81% (17/21)" — ✅
- **সমস্যা:** এটা XAI-এর কোনো metric নয়; ২১টা নির্বাচিত sample-এ classification accuracy।
- **নতুন লেখা:** "Of the 21 samples selected for qualitative XAI analysis (7 per class), the model classified 17 correctly (Severe_Damage 7/7, Humanitarian_Rescue 7/7, Affected_People 3/7)." — কোনো "XAI accuracy" শব্দ নয়।

---

## C. সঠিক কিন্তু সৎভাবে/সম্পূর্ণভাবে লিখতে হবে

### C8. Calibration — ✅ (T* অংশ ⏳ Phase 4)
- **পুরনো দাবি:** "Temperature Scaling confirms high reliability (ECE = 2.16%)."
- **সমস্যা:** TS-এর পর সব metric সামান্য **খারাপ** হয়েছে, কিন্তু শুধু "before" সংখ্যা দেখানো হচ্ছিল।
- **পুরো table (test set, `best_cls_only.pt`, T* = 1.1113):**

  | Metric | Before TS | After TS |
  |--------|-----------|----------|
  | ECE | 2.16% | 2.63% |
  | MCE | 62.09% | 62.56% |
  | NLL | 0.3201 | 0.3224 |
  | Brier | 0.1715 | 0.1728 |
  | Mean confidence | 88.25% | 86.65% |
  | Accuracy | 88.83% | 88.83% |

- **নতুন লেখা (English):**
  > "The uncalibrated model was already well calibrated (ECE = 2.16%, mean confidence 88.25% vs. accuracy 88.83%). Temperature scaling (T* = 1.1113, fitted on the validation set) slightly softened the predictions but did not improve test calibration (ECE 2.63%), consistent with the small validation set (n = 375)."
- ⏳ Integrated pipeline-এ T* = 0.6669 এসেছিল — Phase 4-এ কারণ বের করে একটা সংখ্যায় স্থির হবে। ততদিন শুধু 1.1113 ব্যবহার।

### C9. Trainable parameter — ✅ (CLAUDE.md-এর সংখ্যা বদল Phase 5-এ, অনুমতি সাপেক্ষে)
- **সমস্যা:** "0.30%" গোনা হয়েছিল classifier head তৈরির **আগে** (`cls_only_ablation.py:212`), তাই head বাদ পড়েছে। No-LoRA-তে "0 trainable" লেখা, অথচ head train হয়েছে। README-তে আবার "~0.1%"।
- **সঠিক হিসাব (CLS-only path):**

  | অংশ | Parameter |
  |-----|-----------|
  | LoRA (ViT 294,912 + BanglaBERT 294,912) | 589,824 |
  | Fusion classifier head | 789,507 |
  | **মোট trainable** | **1,379,331** |
  | মোট (encoders + LoRA + head) | 197,795,331 |
  | **Trainable %** | **≈ 0.70%** |

- No-LoRA-তে trainable = 789,507 (শুধু head, ≈ 0.40%)।
- Multi-task মডেলে caption decoder (≈ 72.8M parameter, পুরোটা trainable) যোগ হয়, তাই সেখানে "0.30%" বা "0.70%" কোনোটাই খাটে না।
- **নতুন লেখা (English):**
  > "LoRA adds 589,824 trainable parameters to the two encoders (0.30% of the 197.0M encoder parameters); together with the fusion head, 1.38M parameters (≈0.70%) are trained for classification."

### C10. Ground truth "caption" আসলে tweet-এর অনুবাদ — ✅ / ❓
- **সমস্যা:** "Caption" শব্দে মনে হয় ছবির বর্ণনা। আসলে এগুলো CrisisMMD tweet text-এর মেশিন অনুবাদ।
- **তথ্য (`master_dataset_translated.csv`):**
  - 3,755টা ছবি, কিন্তু unique tweet মাত্র 3,189টা; 566টা caption duplicate (একই tweet-এর একাধিক ছবি)
  - English text-এর 100%-এ আর Bengali text-এর 97.3%-এ URL (`https://t.co/...`) আছে
- **নতুন লেখা (English):**
  > "The Bengali reference texts are machine translations of the original tweets accompanying each image, not purpose-written image descriptions. They frequently contain information not visible in the image (locations, organisations, URLs). This explains the modest BLEU scores and the low human ratings of the references themselves (2.21–2.51/5)."
- ❓ README-তে লেখা "machine-translated … + human verification"। **সত্যিই কি human verification হয়েছিল?** না হলে এই দাবি বাদ যাবে। (Notebook অনুযায়ী অনুবাদ মডেল: `monirbishal/en-bn-nmt`।)

### C11. Human eval-এর IAA (κ ≈ 0) — ✅
- **তথ্য:** Annotator-দের গড় score অনেক আলাদা (caption A ও B মিলিয়ে গড়):

  | Annotator | Fluency | Adequacy | Faithfulness |
  |-----------|---------|----------|--------------|
  | 1 | 1.77 | 1.78 | 1.86 |
  | 2 | 2.68 | 2.60 | 2.74 |
  | 3 | 3.06 | 2.10 | 2.10 |

- **নতুন লেখা (English):**
  > "Inter-annotator agreement was near chance (mean Cohen's κ ≈ 0.00, Krippendorff's α = −0.052). Annotators differed systematically in their use of the scale (mean fluency from 1.77 to 3.06), and no calibration session was held beforehand. The human scores should therefore be treated as indicative only."
- Annotator যে বন্ধু ছিলেন, সেটাও limitation-এ উল্লেখ।

### C12. Figure 4.1 (confusion matrix) — ⏳ Phase 2 / 3B
- **সমস্যা:** Matrix-টা v2 মডেলের (77.95%), সেরা মডেলের (83.21%) নয়; আর C1 অনুযায়ী v2-এর সংখ্যা ব্যবহারযোগ্যই নয়।
- **করণীয়:** Phase 2-এ সেরা মডেলের নতুন matrix, Phase 3B-এ নতুন multi-task মডেলের matrix। পুরনো figure ব্যবহার করা যাবে না।

### C13. 83.21% কোন শর্তে মাপা — ⏳ Phase 2
- **সমস্যা:** CLS-only মডেল test-এ **আসল tweet text** দেখে। Deployment-এ (demo) শুধু ছবি থাকে।
- **আপাতত লেখা (English):**
  > "The reported 83.21% Macro F1 is obtained with the original (translated) tweet text as the text input. In an image-only deployment the tweet is unavailable; performance under that setting is reported in Section X."
- Phase 2-এর ফল এলে Section X-এর সংখ্যা যোগ হবে।

### C14. "CrisisViT (SOTA)" — ✅
- **সমস্যা:** এটা আমাদের নিজস্ব CrisisViT-style re-implementation (ViT-Large/16, শেষ ৪টা block unfrozen), মূল paper-এর রিপোর্ট করা ফল নয়।
- **নতুন লেবেল:** "CrisisViT-style baseline (our re-implementation, ViT-Large/16)"। "SOTA" শব্দ বাদ।

### C15. "First Bengali multimodal …" ধরনের দাবি — ✅
- **সমস্যা:** README-তে "First Bengali Multimodal Flood Intelligence System"। কিন্তু `literature/1. base_papers/`-এ *BanglaMM-Disaster* (Bangla multimodal disaster classification) আর *BiCap* (Bangla image captioning) আছে।
- **নতুন লেখা:** "first" এড়িয়ে নির্দিষ্টভাবে লেখা, যেমন: "To our knowledge, this is one of the first works to jointly perform flood-damage classification and Bengali text generation from disaster images with explainability analysis." — আর related work-এ এই দুটো paper থেকে পার্থক্য স্পষ্ট করা।

### C16. "Flood corpus" — dataset আসলে প্রধানত hurricane-এর — ✅ / ❓
- **পুরনো দাবি:** "CrisisMMD-derived Bengali **flood** corpus"; README: "Flood-only subset extracted; earthquake, hurricane, wildfire samples removed."
- **সমস্যা:** প্রতিটা ছবিকে CrisisMMD-এর raw event ফোল্ডারের সঙ্গে মিলিয়ে দেখা গেছে (৩,৭৫৫টার সবগুলো মিলেছে):

  | Event | ছবি | % | Severe_Damage | Humanitarian_Rescue | Affected_People |
  |-------|-----|---|---------------|---------------------|-----------------|
  | Hurricane Harvey | 1,416 | 37.7% | 774 | 503 | 139 |
  | Hurricane Maria | 1,149 | 30.6% | 700 | 344 | 105 |
  | Hurricane Irma | 1,010 | 26.9% | 687 | 277 | 46 |
  | Sri Lanka Floods | 180 | 4.8% | 79 | 61 | 40 |

  অর্থাৎ **95.2% ছবি hurricane-এর**; earthquake আর wildfire সত্যিই বাদ গেছে, কিন্তু hurricane বাদ যায়নি। Tweet text-এর মাত্র 8.3%-এ "flood" শব্দ আছে।
- **নতুন লেখা (English):**
  > "The corpus is derived from four CrisisMMD events involving severe flooding and storm damage: Hurricanes Harvey, Irma and Maria and the 2017 Sri Lanka floods (earthquake and wildfire events were excluded). Hurricane events account for 95.2% of the samples."
- Title, abstract, objective-এ "flood" শব্দের বদলে "flood and storm" / "hydro-meteorological disaster" জাতীয় শব্দ বিবেচনা করা। Harvey-তে ব্যাপক বন্যা হয়েছিল, তাই "flood-related" বলা যায়, কিন্তু "flood-only" নয়।
- ❓ Title-এ "Flood" শব্দ থাকবে কিনা, সেটা তোমার (ও supervisor-এর) সিদ্ধান্ত।

---

## D. Limitations অংশে যা থাকবে (তালিকা)

1. ⏳ **Data leakage:** একই tweet-এর একাধিক ছবি train ও test-এ ভাগ হয়েছিল — 376টার মধ্যে 80টা (21.3%) test sample-এর হুবহু একই text train-এ আছে। (Phase 3-এর পর: grouped split-এর ফল দিয়ে প্রতিস্থাপন বা পাশে রাখা।)
2. ⏳ **Single-seed ফল / run-to-run variance (~1.7pp)।** (Phase 3-এর পর mean ± std।)
3. ✅ Reference text হলো tweet-এর মেশিন অনুবাদ, ছবির বর্ণনা নয় (C10)।
4. ✅ Human eval-এ IAA প্রায় শূন্য, calibration session ছাড়া, annotator বন্ধু (C11)।
5. ✅ Class imbalance: Affected_People মাত্র 330টা (8.8%) — সব মডেলে এই class-এর F1 সবচেয়ে কম।
6. ✅ Temperature scaling-এ উন্নতি হয়নি; validation set ছোট (C8)।
7. ⏳ Human eval-এর model caption পুরনো checkpoint থেকে (C2, যদি বিকল্প (ক))।
8. ✅ Dataset-এর 95.2% ছবি যুক্তরাষ্ট্র ও ক্যারিবিয়ানের hurricane থেকে, বাকি 4.8% Sri Lanka-র বন্যা; বাংলাদেশের নিজস্ব কোনো ছবি নেই (C16)। বাংলাদেশের বন্যায় generalization যাচাই করা হয়নি।
