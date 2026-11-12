# DRISHTI-Bn — Final Defense Fix Plan

> **শুরু:** 12 November 2026
> **লক্ষ্য:** Final defense-এর আগে থিসিসের সব methodological ও documentation সমস্যা ধাপে ধাপে ঠিক করা
> **অবস্থা:** 🟡 চলমান — বর্তমান phase: **Phase 0**

---

## কাজের নিয়ম (Workflow)

1. **একবারে একটা phase।** আগের phase শেষ না হলে পরেরটা শুরু হবে না (Phase 1 আর Phase 2 একসঙ্গে চলতে পারে)।
2. **ভারী কাজ Colab T4-এ।** লোকাল PC-তে শুধু হালকা কাজ (file edit, git, ছোট script)। GPU লাগে এমন প্রতিটা কাজের জন্য Claude পুরো Colab code দেবে।
3. **Colab result → Claude।** Colab-এর output (printed log, CSV-র লেখা) chat-এ paste করবে। Claude সেটা `results/defense_fixes/<phase>/`-এ সংরক্ষণ করবে আর এই ফাইলের **Result Log**-এ লিখবে।
4. **Image/figure download।** কোন ফাইল Colab থেকে download করে কোন ফোল্ডারে রাখতে হবে, প্রতিটা task-এ **📥 Download** অংশে লেখা থাকবে।
5. **Phase শেষ হলে:** Claude এই ফাইলের checkbox আর Progress Log update করবে, তারপর `git commit` + `git push` করবে।
6. **সংখ্যা বদলানোর নিয়ম।** CLAUDE.md-এর "NEVER change these numbers" নিয়ম বহাল থাকবে। নতুন experiment-এ নতুন সংখ্যা এলে, **তোমার অনুমতি নিয়ে** তবেই CLAUDE.md আর থিসিসে সংখ্যা বদলানো হবে।
7. **থিসিস draft-এ সরাসরি হাত দেওয়া হবে না।** লেখার পরিবর্তন NotebookLM prompt বা নির্দিষ্ট replacement text হিসেবে দেওয়া হবে; draft-এ বসাবে তুমি।

**Status চিহ্ন:** `[ ]` বাকি · `[/]` চলছে · `[x]` শেষ · `[-]` বাদ দেওয়া হয়েছে (কারণসহ)

---

## Phase overview

| Phase | বিষয় | কোথায় চলবে | আনুমানিক সময় | Status |
|-------|------|-------------|---------------|--------|
| 0 | Repo সুরক্ষা ও পরিচ্ছন্নতা | Local + Colab (ছোট check) | 1–2 ঘণ্টা | 🟡 চলমান (শুধু 0.7 বাকি) |
| 1 | লেখার সংশোধন (নতুন experiment ছাড়া) | Local | 1 দিন | ⬜ বাকি |
| 2 | Fair evaluation: CLS-only মডেল generated caption দিয়ে | Colab T4 | ~1 ঘণ্টা | ⬜ বাকি |
| 3 | Leakage fix: tweet-grouped split + 3 seeds | Colab T4 | ~2.5–3 ঘণ্টা | ⬜ বাকি |
| 4 | XAI fix: trained decoder থেকে cross-attention + T* মেলানো | Colab T4 | ~1 ঘণ্টা | ⬜ বাকি |
| 5 | চূড়ান্ত সমন্বয়: সংখ্যা, figure, README, demo | Local | ½ দিন | ⬜ বাকি |
| 6 | (ঐচ্ছিক) Multi-task মডেল grouped split-এ retrain | Colab T4 | ~2 ঘণ্টা | ⬜ ঐচ্ছিক |
| ➡️ | **এরপর:** DIU নির্দেশনা অনুযায়ী নতুন thesis draft লেখা | Google Docs (DIU template) | — | ⬜ plan শেষে |

> সব Colab script থাকবে [colab/](colab/) ফোল্ডারে (Colab-এ এক cell-এ paste করে চালানো যায়)।

---

## Phase 0 — Repo সুরক্ষা ও পরিচ্ছন্নতা

**উদ্দেশ্য:** কিছু হারানোর ঝুঁকি দূর করা আর ভাঙা কোড ঠিক করা, যাতে পরের phase-গুলো নিরাপদে চলে।

- [-] **0.1 Draft ফেরত আনা।** বাদ: পুরনো draft দুটো (`DRISHTI_Bn_Thesis_Final_Working_Draft.docx`, `..._v1.txt`) তুমি ইচ্ছা করে মুছেছ, কারণ সেগুলো result-এর আগের কাজ। মুছে ফেলাটা commit করা হয়েছে (git history-তে থেকে যাবে)। নতুন draft লেখা হবে এই plan শেষ হলে, DIU-র নির্দেশনা অনুযায়ী।
- [x] **0.2 CLAUDE.md ঠিক করা ও commit করা।** পুরনো draft-এর উল্লেখ সরানো হয়েছে, `figure_3_2_lora.png` → `figure_2_1_lora.png`, `DEFENSE_FIX_PLAN.md` আর `Thesis Writing/` যোগ হয়েছে, transformers ভার্সনের নোট যোগ হয়েছে।
- [x] **0.3 [src/train.py](src/train.py) ঠিক করা।** এখন `train_loader, val_loader, _ = get_dataloaders(...)`। Import যাচাই করা হয়েছে।
- [x] **0.4 [src/model.py](src/model.py) version-robust করা।** ⚠️ আগের বিশ্লেষণে ভুল ছিল: `q_proj/v_proj` বাগ নয়। transformers 5.x-এ ViT-এর module নামই `q_proj/v_proj`, আর v2 checkpoint-এ ViT-এর LoRA weight (48টা key) ঠিকঠাক আছে। তাই নাম বদলানো হয়নি; ablation script-এর মতো auto-detect যোগ করা হয়েছে, যাতে transformers 4.x (`query/value`)-এও চলে। যাচাই: v2 checkpoint `strict=True`-এ load হয় ("All keys matched")।
- [x] **0.5 অপ্রয়োজনীয় ফাইল সরানো।** `temp_crop.png`, `temp_sample01_top.png` (integrated XAI figure-এর crop), `fix_nb.py`, `append_dl_test.py` (একবার চালানো notebook-edit script) মুছে ফেলা হয়েছে।
- [x] **0.6 `results/gradcam/` পরিষ্কার।** `gradcam_summary.csv`-এ নেই এমন ১৩টা পুরনো রানের ছবি সরানো হয়েছে। এখন ২১টা ছবি, summary-র সঙ্গে হুবহু মেলে (বাকি ৪টা `_ERR` আসল ভুল prediction: sample 017, 019, 020, 021)।
- [/] **0.7 🖥️ Colab check: কোন checkpoint থেকে কোন সংখ্যা এসেছে।** Drive-এর সব checkpoint-এর তারিখ, size, MD5, LoRA key আর decoder-এর অবস্থা বের করা। উদ্দেশ্য: 77.95% যে checkpoint থেকে এসেছে সেটা teacher-forcing fix (৭ আগস্ট) আর 80/10/10 split (১২ সেপ্টেম্বর)-এর পরে train হয়েছে কিনা নিশ্চিত করা, আর `best_cls_only.pt`-এর decoder সত্যিই untrained কিনা দেখা।
  - Script: [colab/phase0_checkpoint_audit.py](colab/phase0_checkpoint_audit.py) (CPU runtime-ই যথেষ্ট)
  - Laptop-এর v2-এর MD5: `16bb292e5d62115cc5c966dc70c120d4` (2026-06-25)
  - 📥 Download: কিছু না, শুধু printed output paste করবে।

**শেষ হওয়ার শর্ত:** draft ফেরত এসেছে, `train.py`/`model.py` import হলে error নেই, checkpoint-এর ইতিহাস জানা গেছে।

---

## Phase 1 — লেখার সংশোধন (নতুন experiment ছাড়া)

**উদ্দেশ্য:** ভুল বা অতিরঞ্জিত দাবি ঠিক করা। পুরনো draft মুছে ফেলা হয়েছে, তাই এই phase-এর ফল হবে একটা **"সংশোধিত দাবি" reference** (`docs/corrected_claims.md`): কোন দাবি কীভাবে লিখতে হবে, কোনটা বাদ যাবে, কোনটা limitation। পরে নতুন draft লেখার সময় এটা মেনে লেখা হবে, আর [PHASE4_WRITING_PLAN.md](PHASE4_WRITING_PLAN.md)-এর prompt-গুলোও সেই অনুযায়ী ঠিক করা হবে।

- [ ] **1.1 Cross-Attention XAI-এর দাবি।** "Cross-Attention সবচেয়ে অর্থপূর্ণ ব্যাখ্যা দেয়" — `best_cls_only.pt`-এর decoder কখনো train হয়নি (λ_cap = 0), তাই map random weight থেকে। Phase 4 পর্যন্ত দাবিটা স্থগিত/নরম করা।
- [ ] **1.2 No-CrossAttn ablation (+1.33pp)।** Classification path-এ decoder ব্যবহারই হয় না → পার্থক্যটা random variance। "কোনো উল্লেখযোগ্য প্রভাব নেই" হিসেবে পুনর্লিখন।
- [ ] **1.3 "r=8 Multi-Task" লেবেল।** Script-এ `LAMBDA_CAP = 0.0` → এটা CLS-only config। টেবিলের লেবেল ঠিক করা, আর লক্ষ্য করা যে একই config দুবার চালিয়ে 83.21% ও 81.47% এসেছে (≈1.7pp variance)।
- [ ] **1.4 Caption-Only (16.90%)।** Classifier head train হয়নি (λ_cls = 0) → সংখ্যাটা অর্থহীন। টেবিল থেকে সরানো বা footnote দিয়ে ব্যাখ্যা।
- [ ] **1.5 Calibration-এর সৎ রিপোর্ট।** ECE 2.16% → 2.63% (TS-এর পর খারাপ), NLL ও Brier-ও সামান্য খারাপ। দুটোই রিপোর্ট করা; ব্যাখ্যা: মডেল আগে থেকেই well-calibrated। [PHASE4_WRITING_PLAN.md](PHASE4_WRITING_PLAN.md)-এর সংশ্লিষ্ট লাইনও ঠিক করা।
- [ ] **1.6 "XAI accuracy 81%"।** এটা ২১টা sample-এ classification accuracy, XAI metric নয় → নাম বদল।
- [ ] **1.7 Trainable parameter-এর হিসাব।** 0.30% (589,824) হিসাবে classifier head (~0.79M) ধরা হয়নি; No-LoRA-তে "0 trainable" অথচ head train হয়েছে; README-তে 0.1%। একটা সামঞ্জস্যপূর্ণ সংজ্ঞা ঠিক করা ("encoder-এর trainable parameter" বনাম "মোট trainable")।
- [ ] **1.8 Ground truth "caption" আসলে tweet অনুবাদ।** 97%-এ URL আছে, এগুলো ছবির বর্ণনা নয়। Dataset অংশে স্পষ্ট করা; কম BLEU আর GT-র কম human score (2.2–2.5)-এর ব্যাখ্যা হিসেবে ব্যবহার।
- [ ] **1.9 Human eval-এর IAA (κ ≈ 0)।** Annotator-দের গড় fluency 1.7 / 2.7 / 3.1 — calibration session না হওয়া, annotator বন্ধু হওয়া → Limitations-এ।
- [ ] **1.10 Figure 4.1-এর caption।** এই confusion matrix multi-task মডেলের (77.95%), সেরা মডেলের নয় → caption-এ স্পষ্ট লেখা। (সেরা মডেলের matrix আসবে Phase 2-এ।)
- [ ] **1.11 Leakage ও seed variance-কে Limitations-এ রাখা** (Phase 3-এর ফল আসার আগ পর্যন্ত placeholder)।

**শেষ হওয়ার শর্ত:** `docs/corrected_claims.md` তৈরি ও তোমার অনুমোদিত; PHASE4_WRITING_PLAN.md-এর সংশ্লিষ্ট prompt সংশোধিত।

---

## Phase 2 — Fair evaluation (Colab T4)

**সমস্যা:** CLS-Only (83.21%) test-এ **আসল tweet text** দেখে, Multi-Task (77.95%) দেখে **মডেলের বানানো caption**। দুটো তুলনা একই শর্তে নয়। আর demo-তে শুধু ছবি থাকে, tweet থাকে না।

- [ ] **2.1 🖥️ `best_cls_only.pt` তিনটা অবস্থায় evaluate করা** (একই test set, n = 376):
  - (ক) আসল tweet text দিয়ে — 83.21% আবার পাওয়া যায় কিনা (sanity check)
  - (খ) multi-task মডেলের generated caption দিয়ে (beam = 3)
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

- [ ] **3.1 Tweet-ID-grouped stratified split** (80/10/10, `StratifiedGroupKFold`/সমতুল্য) — train/test-এ কোনো tweet শেয়ার হবে না। Overlap = 0 যাচাই।
- [ ] **3.2 🖥️ CLS-only (r = 8) grouped split-এ ৩টা seed-এ (42, 123, 2024) train** → mean ± std Macro F1।
- [ ] **3.3 🖥️ ViT-Only baseline একই grouped split-এ ৩টা seed-এ** → মূল তুলনা (multimodal বনাম image-only) টিকে থাকে কিনা।
- [ ] **3.4 (সময় থাকলে) 🖥️ LoRA r = 4 / 16 grouped split-এ ১টা করে seed।**
- [ ] **3.5 নতুন split-এর index সংরক্ষণ** → `test_set_indices_grouped.csv`।
- [ ] **3.6 সিদ্ধান্ত (তোমার):** থিসিসে মূল সংখ্যা হিসেবে পুরনো split রাখবে, নাকি grouped split-এর mean ± std। সিদ্ধান্ত অনুযায়ী CLAUDE.md update।

📥 **Download (Colab থেকে):**
- `phase3_seed_results.csv` → `results/defense_fixes/phase3/`
- `test_set_indices_grouped.csv` → project root
- Checkpoint-গুলো **Drive-এই থাকবে** (লোকালে download লাগবে না)

**শেষ হওয়ার শর্ত:** leakage-মুক্ত split-এ mean ± std জানা গেছে; ViT-Only-র সঙ্গে তুলনা নিশ্চিত।

---

## Phase 4 — XAI fix (Colab T4)

**সমস্যা:**
- Cross-attention map এসেছে untrained decoder (`best_cls_only.pt`) থেকে।
- Integrated pipeline-এ T* = 0.6669, আলাদা script-এ T* = 1.1113 — অমিল।

- [ ] **4.1 🖥️ Cross-attention map trained decoder থেকে আবার বানানো** (Phase 0.7-এ যাচাই করা multi-task checkpoint, একই ২১টা sample)। এটা caption generation-এর ব্যাখ্যা হিসেবে উপস্থাপন — classification-এর নয়।
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
- [ ] **5.3 [DRISHTI_XAI_CHECKLIST.md](DRISHTI_XAI_CHECKLIST.md) ও [PHASE4_WRITING_PLAN.md](PHASE4_WRITING_PLAN.md)-এ সংশোধিত দাবি প্রতিফলিত করা।**
- [ ] **5.4 Demo ([app.py](app.py)) নিয়ে সিদ্ধান্ত।** এখন পুরনো v2 checkpoint চালায়। বিকল্প: (ক) যেমন আছে রেখে ডিফেন্সে স্পষ্ট বলা, (খ) caption-এর জন্য multi-task + classification-এর জন্য CLS-only checkpoint।
- [ ] **5.5 সব figure-এর তালিকা মিলিয়ে দেখা** (CLAUDE.md-এর figure table বনাম root-এ আসল ফাইল)।
- [ ] **5.6 ডিফেন্সের সম্ভাব্য প্রশ্ন ও উত্তরের তালিকা** (leakage, fair comparison, IAA, calibration, caption-এর মান)।

**শেষ হওয়ার শর্ত:** repo, CLAUDE.md, README আর থিসিসে একই সংখ্যা; demo-র অবস্থা জানা।

---

## Phase 6 — (ঐচ্ছিক) Multi-task মডেল grouped split-এ retrain

শুধু সময় থাকলে। Multi-task (λ_cap = 0.5) মডেল grouped split-এ train করে BLEU/BERTScore আর classification আবার মাপা, যাতে caption metric-ও leakage-মুক্ত হয়।

- [ ] **6.1 🖥️ Multi-task retrain (grouped split, seed 42)**
- [ ] **6.2 🖥️ Caption metric (BLEU 1–4, METEOR, ROUGE-L, BERTScore) আবার মাপা**

📥 **Download:** `phase6_caption_metrics.csv` → `results/defense_fixes/phase6/`

---

## Result Log

> Colab থেকে paste করা ফল এখানে সংক্ষেপে জমা হবে। পূর্ণ output থাকবে `results/defense_fixes/<phase>/`-এ।

| তারিখ | Phase/Task | ফল (সংক্ষেপে) | সংরক্ষিত ফাইল |
|-------|-----------|---------------|---------------|
| — | — | — | — |

---

## Progress Log

| তারিখ | Phase | কী করা হলো | Commit |
|-------|-------|------------|--------|
| 2026-11-12 | — | Fix plan তৈরি | `6c266db` |
| 2026-11-12 | 0 (আংশিক) | 0.1 বাদ (draft ইচ্ছাকৃতভাবে মোছা), 0.2–0.6 শেষ, 0.7-এর Colab script তৈরি | (এই commit) |
