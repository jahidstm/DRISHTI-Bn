# DRISHTI-Bn — Final Defense Fix Plan

> **শুরু:** 12 November 2026
> **লক্ষ্য:** Final defense-এর আগে থিসিসের সব methodological ও documentation সমস্যা ধাপে ধাপে ঠিক করা
> **অবস্থা:** 🟡 চলমান — Phase 0 ✅ · Phase 1 ✅ · Phase 3 ✅ · Phase 3B ✅ · Phase 2 ✅ · Phase 4 ✅ · Phase 5 ✅ (৫.৭ তোমার) → **পরের: নতুন writing prompt দিয়ে থিসিস লেখা**

---

## কাজের নিয়ম (Workflow)

1. **একবারে একটা phase।** আগের phase শেষ না হলে পরেরটা শুরু হবে না (Phase 1 আর Phase 2 একসঙ্গে চলতে পারে)।
2. **ভারী কাজ Colab T4-এ।** লোকাল PC-তে শুধু হালকা কাজ (file edit, git, ছোট script)। GPU লাগে এমন প্রতিটা কাজের জন্য Claude পুরো Colab code দেবে।
3. **Colab result → Claude।** Colab-এর output (printed log, CSV-র লেখা) chat-এ paste করবে। Claude সেটা `results/defense_fixes/<phase>/`-এ সংরক্ষণ করবে আর এই ফাইলের **Result Log**-এ লিখবে।
4. **Image/figure download।** কোন ফাইল Colab থেকে download করে কোন ফোল্ডারে রাখতে হবে, প্রতিটা task-এ **📥 Download** অংশে লেখা থাকবে।
5. **Phase শেষ হলে:** Claude এই ফাইলের checkbox আর Progress Log update করবে, তারপর `git commit` + `git push` করবে।
6. **সংখ্যা বদলানোর নিয়ম।** CLAUDE.md-এর "NEVER change these numbers" নিয়ম বহাল থাকবে। নতুন experiment-এ নতুন সংখ্যা এলে, **তোমার অনুমতি নিয়ে** তবেই CLAUDE.md আর থিসিসে সংখ্যা বদলানো হবে।
7. **Writing পরে।** এই plan-এ শুধু fix; সব phase শেষ হলে [docs/corrected_claims.md](docs/corrected_claims.md) মেনে নতুন writing prompt বানানো হবে। পুরনো `PHASE4_WRITING_PLAN.md` ব্যবহার হবে না।

**Status চিহ্ন:** `[ ]` বাকি · `[/]` চলছে · `[x]` শেষ · `[-]` বাদ দেওয়া হয়েছে (কারণসহ)

---

## Phase overview

| Phase | বিষয় | কোথায় চলবে | আনুমানিক সময় | Status |
|-------|------|-------------|---------------|--------|
| ক্রম | Phase | বিষয় | কোথায় চলবে | আনুমানিক সময় | Status |
|------|-------|------|-------------|---------------|--------|
| 1 | 0 | Repo সুরক্ষা ও পরিচ্ছন্নতা | Local + Colab (ছোট check) | 1–2 ঘণ্টা | ✅ শেষ |
| 2 | 1 | লেখার সংশোধন (নতুন experiment ছাড়া) | Local | 1 দিন | ✅ শেষ |
| 3 | 3 | Leakage fix: tweet-grouped split + 3 seeds + baseline fix | Colab T4 | ~3–3.5 ঘণ্টা | ✅ শেষ (৭২ মিনিট) |
| 4 | **3B** | **Multi-task মডেল নতুন করে train (বাধ্যতামূলক — Phase 0.7-এর ফল)** | Colab T4 | ~45–60 মিনিট | ✅ শেষ |
| 5 | 2 | Fair evaluation: CLS-only মডেল generated caption দিয়ে (3B-এর মডেল লাগবে) | Colab T4 | ~10 মিনিট | ✅ শেষ |
| 6 | 4 | XAI fix: trained decoder থেকে cross-attention + T* মেলানো (3B-এর মডেল লাগবে) | Colab T4 | ~1 ঘণ্টা | ✅ শেষ |
| 7 | 5 | চূড়ান্ত সমন্বয়: সংখ্যা, figure, README, demo | Local | ½ দিন | ✅ শেষ |
| ➡️ | — | **এরপর:** DIU নির্দেশনা অনুযায়ী নতুন thesis draft লেখা | Google Docs (DIU template) | — | ⬜ plan শেষে |

> ⚠️ **ক্রম বদলেছে (Phase 0.7-এর পর):** একমাত্র multi-task checkpoint (v2) সব fix-এর আগের, আর test set-এর আনুমানিক ~300/376 sample সে training-এ দেখেছে। তাই আগের ঐচ্ছিক Phase 6 এখন বাধ্যতামূলক **Phase 3B**, আর Phase 2 ও 4 এর ওপর নির্ভরশীল। বিস্তারিত: [results/defense_fixes/phase0/checkpoint_audit.md](results/defense_fixes/phase0/checkpoint_audit.md)

> সব Colab script থাকবে [colab/](colab/) ফোল্ডারে (Colab-এ এক cell-এ paste করে চালানো যায়)।

---

## Phase 0 — Repo সুরক্ষা ও পরিচ্ছন্নতা

**উদ্দেশ্য:** কিছু হারানোর ঝুঁকি দূর করা আর ভাঙা কোড ঠিক করা, যাতে পরের phase-গুলো নিরাপদে চলে।

- [-] **0.1 Draft ফেরত আনা।** বাদ: পুরনো draft দুটো (`DRISHTI_Bn_Thesis_Final_Working_Draft.docx`, `..._v1.txt`) তুমি ইচ্ছা করে মুছেছ, কারণ সেগুলো result-এর আগের কাজ। মুছে ফেলাটা commit করা হয়েছে (git history-তে থেকে যাবে)। নতুন draft লেখা হবে এই plan শেষ হলে, DIU-র নির্দেশনা অনুযায়ী।
- [x] **0.2 CLAUDE.md ঠিক করা ও commit করা।** পুরনো draft-এর উল্লেখ সরানো হয়েছে, `figure_3_2_lora.png` → `figure_2_1_lora.png`, `DEFENSE_FIX_PLAN.md` আর `Thesis Writing/` যোগ হয়েছে, transformers ভার্সনের নোট যোগ হয়েছে।
- [x] **0.3 [src/train.py](src/train.py) ঠিক করা।** এখন `train_loader, val_loader, _ = get_dataloaders(...)`। Import যাচাই করা হয়েছে।
- [x] **0.4 [src/model.py](src/model.py) version-robust করা।** ⚠️ আগের বিশ্লেষণে ভুল ছিল: `q_proj/v_proj` বাগ নয়। transformers 5.x-এ ViT-এর module নামই `q_proj/v_proj`, আর v2 checkpoint-এ ViT-এর LoRA weight (48টা key) ঠিকঠাক আছে। তাই নাম বদলানো হয়নি; ablation script-এর মতো auto-detect যোগ করা হয়েছে, যাতে transformers 4.x (`query/value`)-এও চলে। যাচাই: v2 checkpoint `strict=True`-এ load হয় ("All keys matched")।
- [x] **0.5 অপ্রয়োজনীয় ফাইল সরানো।** `temp_crop.png`, `temp_sample01_top.png` (integrated XAI figure-এর crop), `fix_nb.py`, `append_dl_test.py` (একবার চালানো notebook-edit script) মুছে ফেলা হয়েছে।
- [x] **0.6 `results/gradcam/` পরিষ্কার।** `gradcam_summary.csv`-এ নেই এমন ১১টা পুরনো রানের ছবি সরানো হয়েছে। এখন ২১টা ছবি, summary-র সঙ্গে হুবহু মেলে (বাকি ৪টা `_ERR` আসল ভুল prediction: sample 017, 019, 020, 021)।
- [x] **0.7 🖥️ Colab check: কোন checkpoint থেকে কোন সংখ্যা এসেছে।** ✅ ফল: [checkpoint_audit.md](results/defense_fixes/phase0/checkpoint_audit.md)
  - Drive-এ multi-task checkpoint মাত্র একটা: v2 (২৫ জুন, laptop-এর সঙ্গে MD5 হুবহু এক)। তখনকার `train.py`-তে λ_cap = 0.2, unshifted teacher forcing, আর seed ছাড়া 80/20 `random_split` ছিল। → 77.95%, সব caption metric আর human eval-এর model caption এই দূষিত checkpoint থেকে।
  - `best_cls_only.pt`-এর decoder untrained — **নিশ্চিত** (`vocab_projector` std 0.02084 ≈ init 0.0208)।
  - CLS-only, ablation, baseline, XAI, calibration-এর সংখ্যা এতে প্রভাবিত নয়। Drive-এর সব checkpoint-এর তারিখ, size, MD5, LoRA key আর decoder-এর অবস্থা বের করা। উদ্দেশ্য: 77.95% যে checkpoint থেকে এসেছে সেটা teacher-forcing fix (৭ আগস্ট) আর 80/10/10 split (১২ সেপ্টেম্বর)-এর পরে train হয়েছে কিনা নিশ্চিত করা, আর `best_cls_only.pt`-এর decoder সত্যিই untrained কিনা দেখা।
  - Script: [colab/phase0_checkpoint_audit.py](colab/phase0_checkpoint_audit.py) (CPU runtime-ই যথেষ্ট)

**শেষ হওয়ার শর্ত:** draft ফেরত এসেছে, `train.py`/`model.py` import হলে error নেই, checkpoint-এর ইতিহাস জানা গেছে।

---

## Phase 1 — লেখার সংশোধন (নতুন experiment ছাড়া)

**উদ্দেশ্য:** ভুল বা অতিরঞ্জিত দাবি ঠিক করা। ফল: **[docs/corrected_claims.md](docs/corrected_claims.md)** — কোন দাবি কীভাবে লিখতে হবে, কোনটা বাদ যাবে, কোনটা limitation। সব phase শেষ হলে নতুন writing prompt বানানো হবে এই ফাইল মেনে (পুরনো `PHASE4_WRITING_PLAN.md` আর ব্যবহার হবে না)।

- [x] **1.1 Cross-Attention XAI-এর দাবি** → C3 (চূড়ান্ত লেখা ⏳ Phase 4)
- [x] **1.2 No-CrossAttn ablation (+1.33pp)** → C4
- [x] **1.3 "r=8 Multi-Task" লেবেল ও ~1.7pp variance** → C5
- [x] **1.4 Caption-Only (16.90%)** → C6
- [x] **1.5 Calibration-এর পুরো before/after table** → C8 (T* ⏳ Phase 4)
- [x] **1.6 "XAI accuracy 81%"** → C7
- [x] **1.7 Trainable parameter** → C9: সঠিক হিসাব LoRA 589,824 + head 789,507 = 1,379,331 (≈ 0.70%); 0.30% শুধু encoder-এর LoRA অংশ
- [x] **1.8 Reference text = tweet-এর মেশিন অনুবাদ** → C10
- [x] **1.9 Human eval-এর IAA** → C11
- [x] **1.10 Figure 4.1** → C12 (নতুন figure ⏳ Phase 2/3B)
- [x] **1.11 Leakage ও seed variance** → Limitations তালিকা (D), ⏳ Phase 3
- [x] **1.12 (নতুন) Multi-task-এর সব সংখ্যা ব্যবহার স্থগিত** → C1, C2 (⏳ Phase 3B)
- [x] **1.13 (নতুন) 83.21% কোন শর্তে মাপা** → C13 (⏳ Phase 2)
- [x] **1.14 (নতুন) "CrisisViT (SOTA)" লেবেল** → C14
- [x] **1.15 (নতুন) "First …" দাবি** → C15 (*BanglaMM-Disaster*, *BiCap* আগে থেকেই আছে)
- [x] **1.16 (নতুন) Dataset "flood-only" নয়** → C16: 95.2% hurricane (Harvey/Irma/Maria), মাত্র 4.8% (180টা) Sri Lanka flood
- [x] **1.17 তোমার উত্তর:** Title-এ আপাতত "Flood" থাকবে, supervisor-এর সঙ্গে কথা বলে চূড়ান্ত (C16)। অনুবাদ যাচাই → 1.19 দেখো।
- [x] **1.18 (নতুন) ViT-Only ও CrisisViT baseline-এ শুধু pooler train হয়েছিল** → C17 (Phase 3-এ আবার চালানো হচ্ছে)
- [x] **1.19 (নতুন) Human eval-এর un-blinding bug ঠিক করা** → C2, C11। HTML tool A/B এলোমেলো করেছিল, merge script সব row-এ A = model ধরেছিল। [src/human_eval_unblind.py](src/human_eval_unblind.py) HTML থেকে আসল assignment পড়ে (৩০০ row-এর caption text যাচাই করা) আবার হিসাব করেছে → model 1.58 / 1.24 / 1.23 বনাম GT 3.42 / 3.08 / 3.23 (p < 10⁻¹⁶); GT পছন্দ 94.9%; pooled α = 0.38।
- [x] **1.20 তোমার উত্তর (2026-11-12):**
  - (ক) অনুবাদ আলাদাভাবে যাচাই হয়নি → "human-verified translation" দাবি বাদ, শুধু "machine-translated" (C10; CLAUDE.md-এ Translation অংশ যোগ)।
  - (খ) অনুমতি দেওয়া হয়েছে → CLAUDE.md-এর Human Evaluation অংশে সংশোধিত সংখ্যা বসানো হয়েছে (C2, C11)।

**শেষ হওয়ার শর্ত:** 1.20-এর দুটো উত্তর।

---

## Phase 2 — Fair evaluation (Colab T4)

**সমস্যা:** CLS-Only (83.21%) test-এ **আসল tweet text** দেখে, Multi-Task (77.95%) দেখে **মডেলের বানানো caption**। দুটো তুলনা একই শর্তে নয়। আর demo-তে শুধু ছবি থাকে, tweet থাকে না।

> **Phase 3B-এর পর অবস্থা:** Multi-task v3-এর দুই অবস্থা (আসল tweet 77.29%, generated caption 73.63%) 3B-তেই মাপা হয়ে গেছে। Phase 2-এ বাকি শুধু **CLS-only মডেল**।

**Script:** [colab/phase2_fair_evaluation.py](colab/phase2_fair_evaluation.py) — training নেই, ~১০ মিনিট। লাগবে Drive-এ `models/cls_only_grouped_s42.pt` (Phase 3), `defense_fixes/phase3/grouped_split.csv`, `defense_fixes/phase3b/phase3b_per_sample.csv`।

- [x] **2.1 🖥️ `cls_only_grouped_s42.pt` পাঁচ অবস্থায় evaluate করা** → A 74.83 (Phase 3-এর সঙ্গে হুবহু মেলে) · B 76.41 · C 61.95 · D 76.52 · **E 76.63** (৯৬.৫% prediction অপরিবর্তিত) → classifier tweet-এর বিষয়বস্তু প্রায় ব্যবহার করে না (C19)। বিস্তারিত: [results/defense_fixes/phase2/phase2_results.md](results/defense_fixes/phase2/phase2_results.md) (grouped test set, n = 375; পুরনো `best_cls_only.pt` / n = 376 নয়):
  - (A) আসল tweet text — sanity check: Phase 3-এর 74.83% আবার আসা উচিত
  - (B) Phase 3B-এর v3 generated caption (image-only pipeline)
  - (C) খালি text (image-only, কোনো text নেই)
  - (D) demo-র placeholder "বন্যা পরিস্থিতি" (`app.py` caption না পেলে এটা দেয়)
  - (E) এলোমেলো tweet (অন্য sample-এর tweet) — মডেল text-এর ওপর কতটা নির্ভর করে
- [x] **2.2 🖥️ নতুন Figure 4.1** — CLS-Only, অবস্থা A, grouped test set। Drive-এ তৈরি; 📥 download বাকি।
- [x] **2.3 ফল সংরক্ষণ** → `results/defense_fixes/phase2/` + Result Log।

📥 **Download (Drive-এর `DisasterNet/defense_fixes/phase2/` থেকে):**
- `phase2_eval_summary.csv` → `results/defense_fixes/phase2/`
- `phase2_per_sample.csv` → `results/defense_fixes/phase2/`
- `figure_4_1_confusion_matrix_grouped.png` → project root (পুরনো `figure_4_1_confusion_matrix.png` মুছবে না, Phase 5-এ সিদ্ধান্ত)

**শেষ হওয়ার শর্ত:** তিন অবস্থার Macro F1 জানা গেছে; থিসিসে কোন সংখ্যা কোন শর্তে, সেটা স্পষ্ট।

---

## Phase 3 — Leakage fix: tweet-grouped split + multi-seed (Colab T4)

**সমস্যা:**
- 376টা test sample-এর **80টার (21.3%)** হুবহু একই tweet/caption train set-এ আছে (একই tweet-এর একাধিক ছবি, split হয়েছে sample ধরে)।
- একই config দুবার চালিয়ে 83.21% ও 81.47% → ~1.7pp variance; single-seed ফলে ২pp-র কম পার্থক্য অর্থবহ নয়।

**আরেকটা সমস্যা (Phase 1-এ ধরা পড়েছে, C17):** ViT-Only আর CrisisViT baseline-এ block unfreeze করার নিয়ম (`"encoder.layer.N"`) transformers 5.x-এ কিছুই মেলায় না, তাই শুধু pooler train হয়েছিল। দুটো baseline-ই এখানে ঠিক নিয়মে আবার চালানো হচ্ছে।

**Script:** [colab/phase3_grouped_split_seeds.py](colab/phase3_grouped_split_seeds.py) — এক cell-এ পুরোটা। Disconnect হলে আবার চালালে শেষ হওয়া run বাদ দিয়ে বাকিগুলো চলবে।

- [x] **3.1 Tweet/caption-grouped, class-balanced split** (80/10/10; union-find দিয়ে একই tweet বা হুবহু একই caption এক group-এ; `GroupShuffleSplit`-এর ২০০টা চেষ্টা থেকে যেটার class অনুপাত পুরো dataset-এর সবচেয়ে কাছে)। যাচাই: 3,755 sample → 3,189 group; train 3,007 · val 373 · test 375; class অনুপাত তিনটাতেই ≈ 59.7 / 31.6 / 8.8%; tweet ও caption overlap = 0।
  - প্রথম চেষ্টায় (`StratifiedGroupKFold`) test-এ class অনুপাত বাঁকা ছিল (Severe 52% বনাম train 60%) আর Colab-এ `torchao` 0.10 / peft-এর অমিলে crash হয়েছিল (কোনো training হয়নি) → দুটোই script-এ ঠিক করা হয়েছে।
  - Resume করলে Drive-এ সেভ করা `grouped_split.csv` আবার ব্যবহার হয়, তাই মাঝপথে split বদলাবে না। **পরের সব phase এই ফাইল থেকেই split নেবে।**
- [x] **3.2 🖥️ CLS-only (r = 8) ৩টা seed-এ (42, 123, 2024)** → **75.69 ± 1.33%** Macro F1 (74.83 / 75.02 / 77.22)। পুরনো 83.21%-এর চেয়ে ~7.5pp কম → leakage ফল ফুলিয়েছিল। Seed 42-এর checkpoint Drive-এ `models/cls_only_grouped_s42.pt` (Phase 2/4-এ লাগবে)।
- [x] **3.3 🖥️ ViT-Only baseline ৩টা seed-এ, ঠিক unfreeze নিয়মে** (14.77M ViT + head = 15.16M trainable) → **73.27 ± 0.39%**। CLS-Only তিনটা seed-এই বেশি (গড়ে +2.4pp)।
- [x] **3.4 🖥️ CrisisViT-style baseline seed 42-এ, ঠিক unfreeze নিয়মে** (51.96M trainable) → **75.87%**। CLS-Only-র সঙ্গে কার্যত সমান (এক std-এর মধ্যে), কিন্তু আমাদের মডেল 38 গুণ কম parameter train করে। বিস্তারিত: [results/defense_fixes/phase3/phase3_results.md](results/defense_fixes/phase3/phase3_results.md)
- [x] **3.5 সিদ্ধান্ত (তোমার, 2026-11-12): হ্যাঁ।** থিসিসের মূল classification সংখ্যা **75.69 ± 1.33%** (grouped split, ৩ seed)। থিসিসে সৎভাবে লেখা হবে যে পুরনো 83.21% ছিল un-grouped split-এর leakage-এর প্রভাব। CLAUDE.md update হবে Phase 5.1-এ (খসড়া সেখানে)।
- [-] LoRA r = 4 / 16 grouped split-এ — বাদ (সময় বাঁচাতে)। Rank ablation পুরনো split-এর ফল হিসেবে থাকবে, C5-এর variance সতর্কতাসহ।

**Colab-এ চালানোর আগে:**
1. Runtime → Change runtime type → **T4 GPU**।
2. Drive-এর `DisasterNet/` ফোল্ডারে **`processed.zip`** আছে কিনা দেখো। না থাকলে laptop-এর `data/processed.zip` (475 MB) সেখানে upload করো। (Phase 0.7-এর audit-এ Drive-এ কোনো CSV পাওয়া যায়নি।)
3. Script চালাও। প্রায় ৩–৩.৫ ঘণ্টা লাগবে।
4. শেষে `PHASE 3 FINAL SUMMARY` থেকে নিচের পুরো অংশ Claude-কে paste করো। শুরুর দিকের `[diag]` লাইন দুটোও paste করো।

📥 **Download (Drive-এর `DisasterNet/defense_fixes/phase3/` থেকে laptop-এ):**
- `test_set_indices_grouped.csv` → project root
- `grouped_split.csv` → `results/defense_fixes/phase3/`
- `phase3_seed_results.csv` → `results/defense_fixes/phase3/` (paste করলে Claude নিজেও বানাতে পারবে)
- Checkpoint **Drive-এই থাকবে**

**শেষ হওয়ার শর্ত:** leakage-মুক্ত split-এ mean ± std জানা গেছে; ViT-Only-র সঙ্গে তুলনা নিশ্চিত।

---

## Phase 3B — Multi-task মডেল নতুন করে train (Colab T4, বাধ্যতামূলক)

**সমস্যা (Phase 0.7-এ ধরা পড়েছে):** একমাত্র multi-task checkpoint `disasternet_multitask_v2.pth` (২৫ জুন) train হয়েছিল:
- λ_cap = 0.2 (README/থিসিসে লেখা 0.5)
- unshifted teacher forcing (decoder-এর input আর target একই → autoencoding identity bias)
- seed ছাড়া 80/20 `random_split` → বর্তমান 376-sample test set-এর আনুমানিক ~300টা সে training-এ দেখেছে

ফলে 77.95% Macro F1, BLEU 1–4, METEOR, ROUGE-L, BERTScore 60.35% আর human eval-এর model caption — সবই leakage-যুক্ত, fix-এর আগের মডেল থেকে।

**Script:** [colab/phase3b_multitask_retrain.py](colab/phase3b_multitask_retrain.py) — এক cell-এ পুরোটা; training শেষ হলে আবার চালালে training বাদ দিয়ে সরাসরি benchmark।
- `src/model.py`-এর একই architecture (তাই checkpoint `evaluate.py` / `app.py`-তে সরাসরি load হয়)। Speed-এর জন্য দুই task-এ encoder একবার চালানো হয়। লোকালি যাচাই: `model.forward`-এর সঙ্গে logit পার্থক্য 0.0।
- Beam search `evaluate.py`-র হুবহু একই algorithm (repetition penalty, bigram block, length penalty 0.7); শুধু ViT memory একবার হিসাব হয়। লোকালি যাচাই: একই caption।
- Hyper-parameter: 15 epoch, bs 16, AdamW lr 1e-4 (wd 0.01), cosine (Phase 3-এর CLS-only-র মতো), AMP, seed 42।

- [x] **3B.1 🖥️ ঠিক করা `train.py`-এর logic দিয়ে multi-task retrain** — λ_cls = 1.0, λ_cap = 0.5, right-shifted teacher forcing, Phase 3-এর grouped split, seed 42, checkpoint বাছাই validation total loss দিয়ে → best epoch 4 (১৫ মিনিট)। Val caption loss epoch 5-এর পর ~4.07-এ আটকে, train loss নামতেই থাকে → decoder overfit।
- [x] **3B.2 🖥️ নতুন test set-এ (n = 375) পুরো benchmark** → classification (ক) ছবি + আসল tweet **77.29%**, (খ) ছবি + generated caption **73.63%** (= ViT-Only)। Caption: BERTScore-F1 69.66 (URL বাদে 62.27); caption আসলে tweet-অনুকরণ (C18)। বিস্তারিত: [results/defense_fixes/phase3b/phase3b_results.md](results/defense_fixes/phase3b/phase3b_results.md)
- [x] **3B.3 নতুন checkpoint Drive-এ রাখা** → `models/drishti_multitask_v3.pt` (v2 অক্ষত)।
- [x] **3B.6 (নতুন) Caption metric লোকালি আবার হিসাব** ([src/caption_metrics_v3.py](src/caption_metrics_v3.py), sacreBLEU 2.6.0) → v3: BLEU 14.38 / chrF **14.19** (boilerplate বাদে 0.11 / 8.46)। তুলনা: random train tweet chrF 21.58, একটাই constant tweet 20.46 → মডেল দুটো তুচ্ছ baseline-এর চেয়েও কম। Boilerplate বাদে ১২৪/৩৭৫ আলাদা caption।
- [x] **3B.4 সিদ্ধান্ত (তোমার, 2026-11-12): (ক)** — জমার ১৫ দিন বাকি, human eval আবার হবে না; v2-এর human eval limitation হিসেবে, v3 শুধু automatic metric দিয়ে (C2)। AI দিয়ে annotator form পূরণ করা হবে না (সেটা বানানো human data হতো)। বিকল্পগুলো ছিল:
  - (ক) আগেরটা রেখে limitation হিসেবে স্বীকার করা (সময় লাগবে না)
  - (খ) নতুন মডেলের caption দিয়ে আবার করানো, এবার annotator-দের নিয়ে calibration session সহ (IAA-ও ভালো হওয়ার সম্ভাবনা)
- [x] **3B.5 সিদ্ধান্ত (তোমার, 2026-11-12): হ্যাঁ।** CLAUDE.md update করা হয়েছে (৩.৫ + ৩B.৫-এর অনুমোদিত সংখ্যা; পুরনো সংখ্যা "superseded" চিহ্নসহ)।

**Colab-এ চালানোর আগে:** Runtime → **T4 GPU**। Drive-এ `DisasterNet/defense_fixes/phase3/grouped_split.csv` থাকতে হবে (Phase 3-এ তৈরি হয়েছে)। নতুন session-এ চালানো ভালো; `peft` আগে load হয়ে থাকলে script restart করতে বলবে।

📥 **Download (Drive-এর `DisasterNet/defense_fixes/phase3b/` থেকে → `results/defense_fixes/phase3b/`):**
- `phase3b_benchmark_summary.csv`
- `phase3b_per_sample.csv` (৩৭৫টা reference + generated caption; human eval আবার করলে লাগবে)
- `phase3b_train_log.csv`
- `phase3b_confusion_matrix.png`
- Checkpoint `models/drishti_multitask_v3.pt` **Drive-এই থাকবে**

**শেষ হওয়ার শর্ত:** fix-করা, leakage-মুক্ত multi-task মডেলের সব classification ও caption metric জানা গেছে।

---

## Phase 4 — XAI ও calibration fix (Colab T4)

**সমস্যা:**
- Cross-attention map এসেছে untrained decoder (`best_cls_only.pt`) থেকে।
- Integrated pipeline-এ T* = 0.6669, আলাদা script-এ T* = 1.1113 — অমিল। **কারণ পাওয়া গেছে:** `integrated_xai_pipeline.py` নিজের আলাদা split বানায় (90/10, তারপর 10/90), তাই তার "val" set-এর বেশিরভাগই মডেলের training data — দেখা data-য় মডেল অতি-নিশ্চিত ও সঠিক, তাই T* < 1। `temp_scaling_xai.py`-এর 1.1113 সঠিক val set-এ fit হয়েছিল। একই কারণে integrated pipeline-এর ২১টা "test" sample-এর অনেকগুলোও training-এ দেখা।
- **(নতুন, 2026-11-12) পুরনো Grad-CAM ফাঁকা ছিল:** `gradcam_xai.py` ও `integrated_xai_pipeline.py` শেষ ViT block-এর **output** patch token-এ hook করে। Pooler শুধু CLS token পড়ে, তাই ওখানে gradient ঠিক শূন্য (লোকালি যাচাই: patch grad max = 0.0, আগের block-এ 0.023)। ফলে map সবসময় শূন্য — পুরনো "Dark heatmap; lacks spatial localization" পর্যবেক্ষণের আসল কারণ এটাই। Figure 4.4 অবৈধ।
- Phase 2-এর ফল (C19): classifier কার্যত image-driven → XAI-এর মূল ভার Grad-CAM + Rollout; cross-attention শুধু সহায়ক (তোমার সিদ্ধান্ত, 2026-11-12)।

**Script:** [colab/phase4_xai_calibration.py](colab/phase4_xai_calibration.py) — training নেই, ~১৫–২০ মিনিট। মূল মডেল `cls_only_grouped_s42.pt` (XAI + calibration), `drishti_multitask_v3.pt` (cross-attention)। Grad-CAM, Rollout, cross-attention, calibration-এর function লোকালি CPU-তে যাচাই করা।

- [x] **4.1 🖥️ Cross-attention trained v3 decoder থেকে** → Grad-CAM-এর সঙ্গে শীর্ষ-10% IoU 0.059 ≈ random (0.054): classification-এর প্রমাণের সঙ্গে সম্পর্কহীন, তাই শুধু সহায়ক (C3)। — মডেলের নিজের generated caption teacher-force করে, ২১টা grouped test sample (৭/class, `random.Random(42)`, বাছাই করা নয়)। Caption generation-এর ব্যাখ্যা হিসেবে, classification-এর নয়।
- [x] **4.2 🖥️ Calibration নতুন মডেলে** → **T* = 1.1349**; ECE 2.57% → 2.99% (TS-এ উন্নতি নেই, মডেল আগে থেকেই ভালো calibrated) (C8)। — T* grouped val (373)-এ fit, grouped test (375)-এ ECE / MCE / NLL / Brier আগে-পরে + reliability diagram। পুরনো 1.1113 ও 0.6669 দুটোই বাতিল।
- [x] **4.3 🖥️ Grad-CAM ঠিক করা + সব XAI figure নতুন করে** → ০/২১টা map শূন্য; ২১টার মধ্যে ১৫টা সঠিক (SD 6/7, HR 7/7, AP 2/7)। 📥 figure download বাকি। — Grad-CAM শেষ block-এর input-এ (ReLU, channel-wise gradient weight)। Figure 4.4–4.7 (প্রতি class-এর প্রথম sample) + ২১টা sample-এর পূর্ণ grid।
- [x] **4.5 (নতুন) 🖥️ Faithfulness / deletion test** → শীর্ষ 10% মুছলে prob কমে Grad-CAM 6.92 pp · Rollout 0.85 · Random 0.34; 50%-এ accuracy 73.87 / 81.60 / 86.93% → Grad-CAM শক্তভাবে faithful (C7)। বিস্তারিত: [results/defense_fixes/phase4/phase4_results.md](results/defense_fixes/phase4/phase4_results.md) — পুরো test set (375)-এ Grad-CAM / Rollout / random অনুযায়ী শীর্ষ 10 / 25 / 50% patch মুছে predicted-class probability কতটা কমে। "XAI accuracy"-র বদলে এটাই XAI-এর আসল সংখ্যাগত মূল্যায়ন।
- [x] **4.4 সিদ্ধান্ত (তোমার, 2026-11-12):** cross-attention সংক্ষেপে সহায়ক বিশ্লেষণ হিসেবে; মূল ফোকাস Grad-CAM ও Rollout।

📥 **Download (Drive-এর `DisasterNet/defense_fixes/phase4/` থেকে):**
- `figure_4_4_gradcam_grouped.png`, `figure_4_5_crossattn_grouped.png`, `figure_4_6_rollout_grouped.png`, `figure_4_7_integrated_xai_grouped.png`, `figure_4_8_calibration_diagram_grouped.png` → project root (পুরনো figure মুছবে না; Phase 5-এ সিদ্ধান্ত)
- `xai_all_21_samples.png`, `phase4_calibration.csv`, `phase4_xai_samples.csv`, `phase4_faithfulness.csv` → `results/defense_fixes/phase4/`

**শেষ হওয়ার শর্ত:** সব XAI figure ও calibration সংখ্যা চূড়ান্ত মডেল আর grouped split থেকে; T* একটাই; XAI-এর একটা সংখ্যাগত মূল্যায়ন (faithfulness)।

---

## Phase 5 — চূড়ান্ত সমন্বয় (Local)

- [x] **5.1 CLAUDE.md-এর সংখ্যা update** — তোমার অনুমোদনে সব অংশ বসানো: classification (75.69 ± 1.33%), baseline, ablation-এর চিহ্ন, multi-task v3, caption metric (sacreBLEU/chrF/BERTScore), human eval (1.58/1.24/1.23 বনাম 3.42/3.08/3.23, α 0.38), অনুবাদ (machine-translated), text ablation, calibration (T* 1.1349, ECE 2.57→2.99%), XAI (15/21 + faithfulness), figure table। পুরনো সংখ্যা "superseded" চিহ্নসহ।
- [x] **5.2 [README.md](README.md) পুনর্লিখন** — সব পুরনো সংখ্যা (66.67%, BLEU-4 5.14, 80/20, ~0.1%, "first", "human verification", flood-only) সরিয়ে grouped split-এর ফল, সৎ limitation, Colab reproduction table। MIT badge বাদ (LICENSE ফাইল নেই)।
- [x] **5.3 [DRISHTI_XAI_CHECKLIST.md](DRISHTI_XAI_CHECKLIST.md)** — শুরুতে "SUPERSEDED — historical record only" সতর্কবার্তা।
- [x] **5.4 Demo ([app.py](app.py)) — তোমার সিদ্ধান্ত (খ):** classification = `cls_only_grouped_s42.pt` + placeholder "বন্যা পরিস্থিতি"; text = v3 decoder (অপরিবর্তিত output, সতর্কবার্তাসহ); নতুন Grad-CAM overlay। লোকালি random weight দিয়ে চালিয়ে যাচাই (~১২ সেকেন্ড/ছবি, CPU)।
  - ⚠️ **ধরা পড়েছে:** পুরনো `app.py`-তে caption **hardcoded** ছিল ("বন্যার পানিতে নিমজ্জিত ঘরবাড়ি…", comment: "Hardcoded for thesis screenshot"), আর `docs/demo_screenshot.png` (README-তে public) সেই বানানো caption দেখাচ্ছিল। Code থেকে সরানো হয়েছে; screenshot `figures_superseded/demo_screenshot_HARDCODED_CAPTION_do_not_use.png`-এ — থিসিস বা slide-এ কখনো ব্যবহার করা যাবে না।
  - 📥 Demo চালাতে Drive থেকে `models/cls_only_grouped_s42.pt` আর `models/drishti_multitask_v3.pt` laptop-এর `models/`-এ নামাতে হবে (`.gitignore`-এ, GitHub-এ যাবে না)।
- [x] **5.5 Figure মিলানো** — নতুন: 4.1 / 4.4–4.8 (`_grouped`), 4.2 (`figure_4_2_baseline_comparison_grouped.png`), 4.3 (`figure_4_3_ablation_original_split.png`), **4.9 নতুন** (`figure_4_9_faithfulness_deletion.png`) — [src/make_phase5_figures.py](src/make_phase5_figures.py)। পুরনো ৮টা figure আর ২টা demo screenshot `figures_superseded/`-এ। CLAUDE.md-এর figure table আপডেট।
- [x] **5.6 Defense প্রশ্নোত্তর** → [docs/defense_qa.md](docs/defense_qa.md) (২৪টা প্রশ্ন: data/leakage, multimodality, text generation, XAI/calibration, অবদান)।
- [ ] **5.7 (তোমার কাজ) portfolio পেজ** — README-র পুরনো link (`jahidstm.lovable.app/research/drishti-bn`) সরানো হয়েছে; ওই পেজে পুরনো সংখ্যা বা hardcoded screenshot থাকলে নিজে update করবে।

**শেষ হওয়ার শর্ত:** repo, CLAUDE.md, README আর থিসিসে একই সংখ্যা; demo-র অবস্থা জানা।

---

## Phase 6 (ঐচ্ছিক) — BD external test set (domain shift)

**উদ্দেশ্য:** US/Caribbean hurricane-এ train করা মডেল বাংলাদেশের বন্যা/ঘূর্ণিঝড়ের ছবিতে কতটা কাজ করে — শুধু test, কোনো training নয়। Paper-এর জন্য মূল নতুন অবদান; থিসিসে "preliminary external validation" (সময় থাকলে)।

- [x] **6.1 ছবি সংগ্রহ (automatic)** — [src/bd_fetch_wikimedia.py](src/bd_fetch_wikimedia.py): Wikimedia Commons-এর ৬টা category থেকে **৪১৯টা CC/public-domain candidate** (Sidr 212, Floods in Bangladesh 105, Amphan 49, Aila 22, 2017 South Asian floods 22, Remal 9), উৎস/লেখক/লাইসেন্সসহ → `bd_external/bd_labels.csv` + `bd_external/review.html`। ছবি `data/bd_external/images/` (git-এ নয়)। অনুমান: satellite/মানচিত্র/ভারতের ছবি বাদে ব্যবহারযোগ্য ~২০০–২৫০; `AP` কম পড়তে পারে।
- [ ] **6.2 🧑 Labelling (দুজন বন্ধু, দুজনেই সব ছবি)** — একই ফাইল (`data/bd_external/BD_labelling_tool.html`, ছবি ভেতরে, ক্লিক-ভিত্তিক) দুজনকে; ফেরত `labels_<নাম>.csv` → `src/bd_merge_labels.py`। — [docs/bd_external_labelling_guide.md](docs/bd_external_labelling_guide.md); দুজন আলাদাভাবে `label_a` / `label_b`, মতভেদ মিটিয়ে `final_label`; ঐচ্ছিক `description_bn` (মানুষের লেখা)। **AI দিয়ে লেবেল বা বর্ণনা নয়।** জনপ্রতি ~১–১.৫ ঘণ্টা।
- [ ] **6.3 Evaluation (laptop, CPU)** — [src/bd_external_eval.py](src/bd_external_eval.py): κ, CrisisMMD-duplicate check (dHash), Macro F1 + 95% bootstrap CI, CrisisMMD-এর 74.83%-এর সঙ্গে পার্থক্য, Grad-CAM figure, ঐচ্ছিক caption-বনাম-মানুষের-বর্ণনা (sacreBLEU/chrF + shuffled baseline)। নকল label আর random weight দিয়ে চালিয়ে script যাচাই করা। লাগবে Drive থেকে `models/cls_only_grouped_s42.pt` (+ caption-এর জন্য `drishti_multitask_v3.pt`)।
- [ ] **6.4 সিদ্ধান্ত:** labelling ~১ সপ্তাহের মধ্যে শেষ হলে থিসিসে যোগ; না হলে future work, defense-এর পর paper-এর জন্য।

---

## Result Log

> Colab থেকে paste করা ফল এখানে সংক্ষেপে জমা হবে। পূর্ণ output থাকবে `results/defense_fixes/<phase>/`-এ।

| তারিখ | Phase/Task | ফল (সংক্ষেপে) | সংরক্ষিত ফাইল |
|-------|-----------|---------------|---------------|
| 2026-11-12 | 0.7 | একমাত্র multi-task checkpoint v2 (২৫ জুন) সব fix-এর আগের; test set-এর ~300/376 training-এ দেখা (আনুমানিক)। `best_cls_only.pt`-এর decoder untrained (std 0.02084 ≈ init) — নিশ্চিত। | `results/defense_fixes/phase0/checkpoint_audit.md` |
| 2026-11-12 | 1.19 | Human eval সঠিকভাবে un-blind: model 1.58/1.24/1.23 বনাম GT 3.42/3.08/3.23 (fluency/adequacy/faithfulness), Wilcoxon p < 10⁻¹⁶; GT পছন্দ 225/237; IAA pooled α 0.38, within-type ≈ 0 | `results/defense_fixes/phase1/human_eval_unblinded/` |
| 2026-11-12 | 1.18 | পুরনো ViT unfreeze নিয়ম transformers 5.x-এ মাত্র 590,592 param (pooler) unfreeze করে, ঠিক নিয়মে 14,766,336; CrisisViT-এর 0.34% = ViT-L pooler | `docs/corrected_claims.md` (C17) |
| 2026-11-12 | 3.1 | Grouped split (Colab-এ চূড়ান্ত): 3,189 group; 3,007 / 373 / 375; class অনুপাত ≈ 59.7/31.6/8.8%; overlap 0 | `colab/phase3_grouped_split_seeds.py` |
| 2026-11-12 | 4.1–4.5 | T* = 1.1349 (ECE 2.57→2.99%) · XAI ১৫/২১ সঠিক · deletion test শীর্ষ-10%: Grad-CAM 6.92 / Rollout 0.85 / Random 0.34 pp · IoU GC–RO 0.148, GC–CA 0.059 | `results/defense_fixes/phase4/phase4_results.md` |
| 2026-11-12 | 2.1–2.3 | CLS-Only s42: tweet 74.83 · generated caption 76.41 · placeholder 76.52 · অন্য sample-এর tweet 76.63 · খালি text 61.95 → মডেল কার্যত image-driven (C19) | `results/defense_fixes/phase2/phase2_results.md` |
| 2026-11-12 | 3B.6 | sacreBLEU: v3 BLEU 14.38 / chrF 14.19 (clean 0.11 / 8.46); random train tweet chrF 21.58, constant tweet 20.46 → মডেল তুচ্ছ baseline-এর নিচে | `results/defense_fixes/phase3b/phase3b_caption_metrics_standard.csv` |
| 2026-11-12 | 3B.1–3B.3 | Multi-task v3 (epoch 4): ছবি + tweet 77.29% · ছবি + generated caption 73.63% · BERTScore-F1 69.66 (URL বাদে 62.27) · caption = tweet-অনুকরণ | `results/defense_fixes/phase3b/phase3b_results.md` |
| 2026-11-12 | 3.2–3.4 | Grouped split, test n = 375: CLS-Only 75.69 ± 1.33 · ViT-Only 73.27 ± 0.39 · CrisisViT-style 75.87 (1 seed); AP-এর F1 ≈ 45% | `results/defense_fixes/phase3/` |
| 2026-11-12 | 1.16 | Dataset-এর event ভাগ: Harvey 1,416 · Maria 1,149 · Irma 1,010 · Sri Lanka floods 180 → 95.2% hurricane | `docs/corrected_claims.md` (C16) |

---

## Progress Log

| তারিখ | Phase | কী করা হলো | Commit |
|-------|-------|------------|--------|
| 2026-11-12 | — | Fix plan তৈরি | `6c266db` |
| 2026-11-12 | 0 (আংশিক) | 0.1 বাদ (draft ইচ্ছাকৃতভাবে মোছা), 0.2–0.6 শেষ, 0.7-এর Colab script তৈরি | `c380372` |
| 2026-11-12 | 0 ✅ | 0.7 checkpoint audit শেষ; Phase 3B (multi-task retrain) বাধ্যতামূলক হিসেবে যোগ, phase-এর ক্রম বদল | `47db4f5` |
| 2026-11-12 | 1 (প্রায়) | `docs/corrected_claims.md` তৈরি (C1–C16 + Limitations); নতুন ফল: dataset-এর 95.2% hurricane; PHASE4_WRITING_PLAN.md বাদ | `df140a8` |
| 2026-11-12 | 1 ✅ · 3 (script) | তোমার দুটো উত্তর যোগ; C17 (baseline-এ শুধু pooler train) যোগ; Phase 3 Colab script তৈরি ও split লোকালি যাচাই | `cf8d45b` |
| 2026-11-12 | 1 (আবার খোলা) | Human eval un-blinding bug ধরা ও ঠিক করা (C2, C11); অনুবাদ যাচাইয়ের দাবির প্রমাণ মেলেনি (C10) | `c5a21d2` |
| 2026-11-12 | 3 (script fix) | Colab-এ torchao crash ঠিক; split-এ class-balance উন্নত; resume-এ saved split ব্যবহার | `12bf43f` |
| 2026-11-12 | 3 (run শেষ) | ৭টা run-এর ফল সংরক্ষণ (3.2–3.4); corrected_claims-এ C5, C13, C14, C17, D1, D2 update | `4f29b97` |
| 2026-11-12 | 3 ✅ · 3B (script) | ৩.৫ সিদ্ধান্ত: মূল সংখ্যা 75.69 ± 1.33%; Phase 5.1-এ CLAUDE.md-এর খসড়া; Phase 3B Colab script তৈরি ও লোকালি যাচাই | `36c3cfb` |
| 2026-11-12 | 3B (run শেষ) | v3-এর ফল সংরক্ষণ (3B.1–3B.3); C1 প্রতিস্থাপন, নতুন C18 (caption = tweet-অনুকরণ), C9-এ multi-task parameter; 3B.6 (metric পুনর্হিসাব) যোগ | `0288960` |
| 2026-11-12 | 3B ✅ · 5.1 (আংশিক) | 3B.6 শেষ (sacreBLEU/chrF + তুচ্ছ baseline); 3B.4 = (ক), 3B.5 = হ্যাঁ; CLAUDE.md-এ অনুমোদিত সংখ্যা বসানো; Colab-এর download করা ফাইল যোগ | `14ccbeb` |
| 2026-11-12 | 1 ✅ · 2 (script) | ১.২০-এর উত্তর: human eval-এর সঠিক সংখ্যা CLAUDE.md-এ, "human-verified translation" দাবি বাদ; Phase 2 Colab script তৈরি | `9da91da` |
| 2026-11-12 | 2 ✅ | Phase 2-এর ফল সংরক্ষণ; নতুন C19 (classifier text প্রায় ব্যবহার করে না), C13 চূড়ান্ত; demo-র পরামর্শ 5.4-এ | `8b629b2` |
| 2026-11-12 | 4 (script) | T*-এর অমিলের কারণ (integrated pipeline-এর নিজস্ব split) আর পুরনো Grad-CAM-এর শূন্য-gradient bug ধরা; Phase 4 Colab script (calibration, ঠিক Grad-CAM, Rollout, v3 cross-attention, faithfulness test) | `6f55911` |
| 2026-11-12 | 4 ✅ | Phase 4-এর ফল: T* = 1.1349, ECE 2.57→2.99%; ঠিক করা Grad-CAM faithful (random-এর ~২০ গুণ), Rollout দুর্বল, cross-attention ≈ random; C3, C7, C8, C20 চূড়ান্ত | `59ae7a2` |
| 2026-11-12 | 5 ✅ | CLAUDE.md চূড়ান্ত; README পুনর্লিখন; demo নতুন (CLS-only + v3 + Grad-CAM, hardcoded caption সরানো); Figure 4.2 / 4.3 / 4.9; পুরনো figure `figures_superseded/`-এ; defense Q&A | `6698abd` |
| 2026-11-13 | 6 (ঐচ্ছিক, শুরু) | BD external test set: Wikimedia downloader (৪১৯ candidate), labelling নির্দেশিকা, CPU evaluation script | (এই commit) |
