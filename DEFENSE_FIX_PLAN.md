# DRISHTI-Bn — Final Defense Fix Plan

> **শুরু:** 12 November 2026
> **লক্ষ্য:** Final defense-এর আগে থিসিসের সব methodological ও documentation সমস্যা ধাপে ধাপে ঠিক করা
> **অবস্থা:** 🟡 চলমান — Phase 0 শেষ · Phase 1 (১.২০-এর উত্তর বাকি) · **Phase 3 Colab run শেষ** (৩.৫-এর সিদ্ধান্ত বাকি) · পরের: **Phase 3B**

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
| 2 | 1 | লেখার সংশোধন (নতুন experiment ছাড়া) | Local | 1 দিন | 🟡 প্রায় শেষ (১টা প্রশ্ন + ১টা অনুমতি বাকি) |
| 3 | 3 | Leakage fix: tweet-grouped split + 3 seeds + baseline fix | Colab T4 | ~3–3.5 ঘণ্টা | 🟡 run শেষ (৭২ মিনিট); ৩.৫-এর সিদ্ধান্ত বাকি |
| 4 | **3B** | **Multi-task মডেল নতুন করে train (বাধ্যতামূলক — Phase 0.7-এর ফল)** | Colab T4 | ~2 ঘণ্টা | ⬜ বাকি |
| 5 | 2 | Fair evaluation: CLS-only মডেল generated caption দিয়ে (3B-এর মডেল লাগবে) | Colab T4 | ~1 ঘণ্টা | ⬜ বাকি |
| 6 | 4 | XAI fix: trained decoder থেকে cross-attention + T* মেলানো (3B-এর মডেল লাগবে) | Colab T4 | ~1 ঘণ্টা | ⬜ বাকি |
| 7 | 5 | চূড়ান্ত সমন্বয়: সংখ্যা, figure, README, demo | Local | ½ দিন | ⬜ বাকি |
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
- [ ] **1.20 তোমার উত্তর/অনুমতি লাগবে:**
  - (ক) `annotator_*_responses.csv` আসলে caption-এর human eval, অনুবাদ যাচাই নয়। June-এ আলাদা কোনো অনুবাদ যাচাই হয়েছিল কি? (C10)
  - (খ) CLAUDE.md-এর Human Evaluation অংশের পুরনো (ভুল) সংখ্যা সংশোধিত সংখ্যা দিয়ে বদলানোর অনুমতি।

**শেষ হওয়ার শর্ত:** 1.20-এর দুটো উত্তর।

---

## Phase 2 — Fair evaluation (Colab T4)

**সমস্যা:** CLS-Only (83.21%) test-এ **আসল tweet text** দেখে, Multi-Task (77.95%) দেখে **মডেলের বানানো caption**। দুটো তুলনা একই শর্তে নয়। আর demo-তে শুধু ছবি থাকে, tweet থাকে না।

- [ ] **2.1 🖥️ `best_cls_only.pt` তিনটা অবস্থায় evaluate করা** (একই test set, n = 376):
  - (ক) আসল tweet text দিয়ে — 83.21% আবার পাওয়া যায় কিনা (sanity check)
  - (খ) **Phase 3B-এর নতুন** multi-task মডেলের generated caption দিয়ে (beam = 3) — পুরনো v2 নয়
  - (গ) text ছাড়া / fixed placeholder text দিয়ে (image-only deployment-এর অবস্থা)
- [ ] **2.2 🖥️ সেরা মডেলের confusion matrix** (অবস্থা ক) → নতুন figure।
- [ ] **2.3 ফল সংরক্ষণ** → `results/defense_fixes/phase2/` + Result Log।

📥 **Download (Colab থেকে):**
- `phase2_eval_summary.csv` → `results/defense_fixes/phase2/`
- `phase2_per_sample.csv` → `results/defense_fixes/phase2/`
- `figure_4_1b_confusion_matrix_best.png` → project root

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
- [ ] **3.5 সিদ্ধান্ত (তোমার):** থিসিসে মূল সংখ্যা হিসেবে grouped split-এর mean ± std ব্যবহার করবে কিনা (Claude-এর পরামর্শ: হ্যাঁ)। সিদ্ধান্ত অনুযায়ী CLAUDE.md update।
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

- [ ] **3B.1 🖥️ ঠিক করা `train.py` দিয়ে multi-task retrain** — λ_cap = 0.5, right-shifted teacher forcing, Phase 3-এর grouped split, seed 42, checkpoint বাছাই validation loss দিয়ে।
- [ ] **3B.2 🖥️ নতুন test set-এ পুরো benchmark** — Macro F1, per-class F1, confusion matrix, BLEU 1–4, METEOR, ROUGE-L, BERTScore (beam = 3)।
- [ ] **3B.3 নতুন checkpoint Drive-এ রাখা** → `models/drishti_multitask_v3.pt` (পুরনো v2 মুছবে না)।
- [ ] **3B.4 সিদ্ধান্ত (তোমার): human evaluation।** আগের ১০০টা model caption v2 থেকে। বিকল্প:
  - (ক) আগেরটা রেখে limitation হিসেবে স্বীকার করা (সময় লাগবে না)
  - (খ) নতুন মডেলের caption দিয়ে আবার করানো, এবার annotator-দের নিয়ে calibration session সহ (IAA-ও ভালো হওয়ার সম্ভাবনা)
- [ ] **3B.5 সিদ্ধান্ত (তোমার):** নতুন সংখ্যা CLAUDE.md আর থিসিসে বসানো।

📥 **Download (Colab থেকে):**
- `phase3b_benchmark_summary.csv` → `results/defense_fixes/phase3b/`
- `phase3b_per_sample.csv` (reference + generated caption) → `results/defense_fixes/phase3b/`
- `phase3b_confusion_matrix.png` → `results/defense_fixes/phase3b/`
- Checkpoint **Drive-এই থাকবে**

**শেষ হওয়ার শর্ত:** fix-করা, leakage-মুক্ত multi-task মডেলের সব classification ও caption metric জানা গেছে।

---

## Phase 4 — XAI fix (Colab T4)

**সমস্যা:**
- Cross-attention map এসেছে untrained decoder (`best_cls_only.pt`) থেকে।
- Integrated pipeline-এ T* = 0.6669, আলাদা script-এ T* = 1.1113 — অমিল।

- [ ] **4.1 🖥️ Cross-attention map trained decoder থেকে আবার বানানো** (**Phase 3B-এর নতুন** multi-task checkpoint; v2-এর decoder unshifted teacher forcing-এ train হওয়ায় সেটা চলবে না; একই ২১টা sample)। এটা caption generation-এর ব্যাখ্যা হিসেবে উপস্থাপন — classification-এর নয়।
- [ ] **4.2 🖥️ T*-এর অমিলের কারণ বের করা** (কোন checkpoint/split-এ fit হয়েছে) এবং একটা সংখ্যায় স্থির করা।
- [ ] **4.3 Figure 4.5 ও 4.7 আবার বানানো** (প্রয়োজনে)।
- [ ] **4.4 সিদ্ধান্ত (তোমার):** নতুন map যথেষ্ট অর্থবহ না হলে cross-attention অংশ ছোট করে limitation হিসেবে রাখা।

📥 **Download (Colab থেকে):**
- `figure_4_5_crossattn_examples.png` (নতুন) → project root (পুরনোটা প্রতিস্থাপন)
- `figure_4_7_integrated_xai.png` (নতুন, প্রয়োজনে) → project root
- `crossattn_summary_v2.csv` → `results/defense_fixes/phase4/`

**শেষ হওয়ার শর্ত:** cross-attention figure trained decoder থেকে; T* একটাই সংখ্যা।

---

## Phase 5 — চূড়ান্ত সমন্বয় (Local)

- [ ] **5.1 CLAUDE.md-এর সংখ্যা update** (শুধু তোমার অনুমোদিত নতুন সংখ্যা)।
- [ ] **5.2 [README.md](README.md) পুনর্লিখন।** পুরনো 66.67% F1, BLEU-4 5.14, 80/20 split, 0.1% সরিয়ে বর্তমান ফল।
- [ ] **5.3 [DRISHTI_XAI_CHECKLIST.md](DRISHTI_XAI_CHECKLIST.md)-এ সংশোধিত দাবি প্রতিফলিত করা।** (`PHASE4_WRITING_PLAN.md` বাদ — writing prompt নতুন করে বানানো হবে।)
- [ ] **5.4 Demo ([app.py](app.py)) নিয়ে সিদ্ধান্ত।** এখন পুরনো v2 checkpoint চালায়। বিকল্প: (ক) যেমন আছে রেখে ডিফেন্সে স্পষ্ট বলা, (খ) caption-এর জন্য multi-task + classification-এর জন্য CLS-only checkpoint।
- [ ] **5.5 সব figure-এর তালিকা মিলিয়ে দেখা** (CLAUDE.md-এর figure table বনাম root-এ আসল ফাইল)।
- [ ] **5.6 ডিফেন্সের সম্ভাব্য প্রশ্ন ও উত্তরের তালিকা** (leakage, fair comparison, IAA, calibration, caption-এর মান)।

**শেষ হওয়ার শর্ত:** repo, CLAUDE.md, README আর থিসিসে একই সংখ্যা; demo-র অবস্থা জানা।

---

## Result Log

> Colab থেকে paste করা ফল এখানে সংক্ষেপে জমা হবে। পূর্ণ output থাকবে `results/defense_fixes/<phase>/`-এ।

| তারিখ | Phase/Task | ফল (সংক্ষেপে) | সংরক্ষিত ফাইল |
|-------|-----------|---------------|---------------|
| 2026-11-12 | 0.7 | একমাত্র multi-task checkpoint v2 (২৫ জুন) সব fix-এর আগের; test set-এর ~300/376 training-এ দেখা (আনুমানিক)। `best_cls_only.pt`-এর decoder untrained (std 0.02084 ≈ init) — নিশ্চিত। | `results/defense_fixes/phase0/checkpoint_audit.md` |
| 2026-11-12 | 1.19 | Human eval সঠিকভাবে un-blind: model 1.58/1.24/1.23 বনাম GT 3.42/3.08/3.23 (fluency/adequacy/faithfulness), Wilcoxon p < 10⁻¹⁶; GT পছন্দ 225/237; IAA pooled α 0.38, within-type ≈ 0 | `results/defense_fixes/phase1/human_eval_unblinded/` |
| 2026-11-12 | 1.18 | পুরনো ViT unfreeze নিয়ম transformers 5.x-এ মাত্র 590,592 param (pooler) unfreeze করে, ঠিক নিয়মে 14,766,336; CrisisViT-এর 0.34% = ViT-L pooler | `docs/corrected_claims.md` (C17) |
| 2026-11-12 | 3.1 | Grouped split (Colab-এ চূড়ান্ত): 3,189 group; 3,007 / 373 / 375; class অনুপাত ≈ 59.7/31.6/8.8%; overlap 0 | `colab/phase3_grouped_split_seeds.py` |
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
| 2026-11-12 | 3 (run শেষ) | ৭টা run-এর ফল সংরক্ষণ (3.2–3.4); corrected_claims-এ C5, C13, C14, C17, D1, D2 update; ৩.৫-এর সিদ্ধান্ত বাকি | (এই commit) |
