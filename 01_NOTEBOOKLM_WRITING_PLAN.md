# 01_NOTEBOOKLM_WRITING_PLAN — DRISHTI-Bn Thesis

> **Thesis:** DRISHTI-Bn: An Explainable Multimodal Deep Learning Framework for Bengali Disaster Image Captioning and Classification
> **Student:** Jahid Hasan (ID 0242) · **Supervisor:** Md. Fahad Hossain · DIU, Department of Software Engineering
> **Version:** 16 November 2026 — replaces `PHASE4_WRITING_PLAN.md` (outdated numbers; do not use)
> **Target:** ≈ 20,000 words main text (Chapters 1–5 + Abstract); DIU maximum 30,000 · Turnitin ≤ 40% total, < 3% per source

---

## Part A — কীভাবে কাজ করবে

1. **Part B** অনুযায়ী NotebookLM-এর source ঠিক করো: পুরনো ভুল ফাইল সরাও, নতুন ফাইল আর paper যোগ করো।
2. Word-এ `Thesis Writing/SWE_Thesis Template_2024.docx`-এর একটা কপি খোলো। **Part C**-এর heading-গুলো Heading 1/2/3 style দিয়ে আগে বসিয়ে নাও।
3. **Part F**-এর prompt একটা একটা করে NotebookLM chat-এ paste করো। একটা prompt = একটা section/sub-section।
4. Output পড়ো, সংখ্যাগুলো `02_RESULTS_FACT_SHEET.md`-এর সঙ্গে মিলিয়ে নাও, তারপর Word-এ বসাও (style: DIU Paragraph)।
5. `[FIGURE …]` জায়গায় `figures/` থেকে ছবি insert করো, caption **নিচে**। `[TABLE …]` জায়গায় Word table, caption **উপরে**। `[EQUATION …]` Word equation editor-এ টাইপ করো।
6. সব শেষে `[REF: …]`-গুলো Zotero দিয়ে APA citation-এ বদলাও, তারপর reference list তৈরি করো।
7. **Part D**-এর tracker-এ টিক দাও।

**লেখার ক্রম (পরামর্শ):** Chapter 3 → Chapter 4 → Chapter 2 → Chapter 1 → Chapter 5 → Abstract।
Method আর result আগে লিখলে introduction আর conclusion সঠিক ও সামঞ্জস্যপূর্ণ হয়।

**প্রতিটা output-এর পর নিজে দেখবে:**
- সংখ্যা fact sheet-এর সঙ্গে মেলে কি না (NotebookLM মাঝে মাঝে paper থেকে অন্য সংখ্যা নিয়ে আসে)।
- কোনো "DO NOT CLAIM" কথা (fact sheet §11) ঢুকে গেছে কি না।
- [REF]-এর সংখ্যা prompt-এর সীমার মধ্যে কি না।
- লেখাটা নিজের ভাষায় একবার পড়ে ঠিক করো। Turnitin-এর জন্যও এটা সবচেয়ে কাজের, আর defense-এ প্রতিটা বাক্যের দায় তোমার।

---

## Part B — NotebookLM sources

### B.1 এখনই মুছে ফেলো (ভুল তথ্য আছে)
| ফাইল | কেন |
|------|-----|
| পুরনো `00_MASTER_WRITING_GUIDELINES.txt` | ভুল শিরোনাম ("DRISTHI-Bn … Flood Damage Assessment"), IEEE citation, ভুল contribution ("new annotated dataset") |
| `02_BENCHMARK_RAW_DATA.txt` | n = 2-এ Macro F1 66.67%, BLEU-4 5.14, 4-layer decoder — সব বাতিল |
| `human_eval_report.md` | merge bug-এর পুরনো ফল (2.49 / 2.11 / 2.17, κ ≈ 0) — সংশোধিত ফল fact sheet §6-এ |
| পুরনো figure image (যদি আগে দিয়ে থাকো) | পুরনো architecture/pipeline-এ ভুল আছে (BanglaBERT-কে decoder দেখানো, ECE 2.16%, ফাঁকা Grad-CAM) |
| পুরনো draft / `PHASE4_WRITING_PLAN.md` (যদি থাকে) | পুরনো সংখ্যা (83.21%, 17/21 ইত্যাদি) |

### B.2 নতুন করে আপলোড করো
| # | ফাইল (project root থেকে) | কাজ |
|---|------|-----|
| 1 | `00_MASTER_WRITING_GUIDELINES.txt` (নতুন সংস্করণ) | লেখার নিয়ম |
| 2 | `01_NOTEBOOKLM_WRITING_PLAN.md` (এই ফাইল) | কাঠামো আর কোন section কী বলবে |
| 3 | `02_RESULTS_FACT_SHEET.md` | **একমাত্র সংখ্যার উৎস** |
| 4 | `figures/` ফোল্ডারের ১৪টা PNG + `results/defense_fixes/phase3b/phase3b_confusion_matrix.png` | কোন figure কী দেখায় বোঝার জন্য (NotebookLM image নিতে না পারলে বাদ দাও; fact sheet §13-এ বর্ণনা আছে) |

### B.3 আগে থেকে আছে এমন paper (১৮টা) — রেখে দাও
| Folder | Paper | কোথায় কাজে লাগবে |
|--------|-------|-------------------|
| 1. base_papers | BanglaMM-Disaster | 2.2.2, 2.8 |
| | BiCap (Bangla image captioning) | 2.4.2, 2.8 |
| | Damage analysis via bidirectional multi-task cascaded multimodal fusion (BiMCF, 2025) | 2.2.2, 2.3.2, 2.8 |
| | Explainable flood damage assessment (multi-atrous self-attention + VLM) | 2.2.1, 2.6.1, 2.8 |
| | SARChat-Bench-2M | 2.3.2 |
| 2. related_work | AI and NLP in Social Good (emergency response text analytics) | 1.1.2, 2.4.1 |
| | Co-Training Vision-Language Models for Remote Sensing Multi-Task Learning | 2.3.2 |
| | CrisisViT | 2.2.1, 3.7, 4.4, 2.8 |
| | Image captioning in Bengali language using visual attention | 2.4.2, 2.8 |
| | ReliefNet (2026) | 2.2.2, 2.8 |
| 3. background | Deep learning satellite image processing, Ganges Delta flood | 1.1.1, 2.2.1 |
| | PetroMind | 2.3.2 |
| | Transfer Learning for Low-Resource Bengali Image Captioning | 2.4.2 |
| 4. foundational | ViT — An Image is Worth 16x16 Words (2021) | 2.3.1, 3.4.1 |
| | BanglaBERT (2022) | 2.4.1, 3.4.2 |
| | CrisisMMD (2018) | 2.2.2, 3.3.1 |
| | LoRA (2022) | 2.5, 3.5 |

### B.4 নতুন করে নামাতে হবে (PDF → `literature/` ফোল্ডারে রাখো → NotebookLM-এ আপলোড)
বেশিরভাগই 2020-এর পরের। পুরনোগুলো (★) foundational method, এগুলো ছাড়া চলবে না।

**অবশ্যই লাগবে (১২টা)**
| # | Paper | কোথায় পাবে | কোন section |
|---|-------|-------------|-------------|
| 1 | Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS ★ | arXiv 1706.03762 | 2.3.1, 3.4.4 |
| 2 | Selvaraju et al. (2017/2020). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017 / IJCV 2020 ★ | arXiv 1610.02391 | 2.6.1, 3.8.1 |
| 3 | Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.* ACL | arXiv 2005.00928 | 2.6.1, 3.8.2 |
| 4 | Guo, Pleiss, Sun & Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML ★ | arXiv 1706.04599 | 2.6.3, 3.8.4 |
| 5 | Petsiuk, Das & Saenko (2018). *RISE: Randomized Input Sampling for Explanation of Black-box Models.* BMVC ★ (deletion metric) | arXiv 1806.07421 | 2.6.2, 3.8.3 |
| 6 | Kapoor & Narayanan (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns | journal site / arXiv 2207.07048 | 2.7.1, 3.3.3, 4.2 |
| 7 | Ofli, Alam & Imran (2020). *Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response.* ISCRAM | arXiv 2004.11838 | 2.2.2 |
| 8 | Abavisani, Wu, Hu, Tetreault & Jaimes (2020). *Multimodal Categorization of Crisis Events in Social Media.* CVPR | arXiv 2004.04917 | 2.2.2 |
| 9 | Hasan et al. (2020). *Not Low-Resource Anymore: Aligner Ensembling, Batch Filtering, and New Datasets for Bengali-English Machine Translation.* EMNLP | arXiv 2009.09359 | 2.4.1, 3.3.2 |
| 10 | Zhang, Kishore, Wu, Weinberger & Artzi (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR | arXiv 1904.09675 | 2.7.2, 3.9.2 |
| 11 | van der Lee, Gatt, van Miltenburg & Krahmer (2021). *Human evaluation of automatically generated text: Current trends and best practice guidelines.* Computer Speech & Language | journal site (open access) | 2.7.2, 3.9.3 |
| 12 | Rentschler, Salhab & Jafino (2022). *Flood exposure and poverty in 188 countries.* Nature Communications (অথবা এর আগের World Bank working paper: Rentschler & Salhab, 2020, *People in Harm's Way*) | nature.com (open access) | 1.1.1, Figure 1.1 |

> Figure 1.1-এর সংখ্যা (94.42 million, 57.5%) Rentschler & Salhab (2020)-এর World Bank working paper থেকে নেওয়া। Paper খুলে Bangladesh-এর সংখ্যাটা একবার মিলিয়ে নিও। 2022-এর paper-এ সংখ্যা আলাদা হলে caption-এ যে paper থেকে সংখ্যা নিয়েছ সেটাই cite করবে।

**ভালো হয় (৬টা)**
| # | Paper | কোথায় পাবে | কোন section |
|---|-------|-------------|-------------|
| 13 | Alam et al. (2023). *MEDIC: A Multi-Task Learning Dataset for Disaster Image Classification.* Neural Computing and Applications | arXiv 2108.12828 | 2.2.1, 2.3.2 |
| 14 | Chefer, Gur & Wolf (2021). *Transformer Interpretability Beyond Attention Visualization.* CVPR | arXiv 2012.09838 | 2.6.1 |
| 15 | Post (2018). *A Call for Clarity in Reporting BLEU Scores.* WMT | arXiv 1804.08771 | 2.7.2, 3.9.2 |
| 16 | Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation.* WMT ★ | ACL Anthology W15-3049 | 2.7.2, 3.9.2 |
| 17 | Han et al. (2024). *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR | arXiv 2403.14608 | 2.5 |
| 18 | Clark, Luong, Le & Manning (2020). *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.* ICLR | arXiv 2003.10555 | 2.3.1, 3.4.2 |

মোট ≈ ৩৬টা paper। Bachelor thesis-এর জন্য ৩০–৪০টা reference যথেষ্ট। NotebookLM-এর ৫০-source সীমার মধ্যেও থাকবে।

---

## Part C — Thesis structure and word budget (≈ 20,400 words)

DIU template অনুযায়ী: প্রতি chapter X.1 Introduction দিয়ে শুরু (Chapter 1 বাদে), আর Chapter 2–4 শেষ হয় Summary দিয়ে। Sub-section (X.Y.Z) শুধু যেখানে বিষয়টা সত্যিই ভাগ করা যায়।

```
Front matter (word count-এ ধরা হয় না): Title page · Declarations (template) · Acknowledgements (নিজে লিখবে)
  · Abstract (280) · Table of Contents · List of Tables · List of Figures · List of Abbreviations

CHAPTER 1  INTRODUCTION ................................................ 2,550
  1.1 Background of the Study
      1.1.1 Disasters in Bangladesh and the Need for Rapid Information ... 320   Fig 1.1
      1.1.2 Social Media Imagery and Deep Learning for Crisis Response .. 330
  1.2 Problem Statement ................................................. 400
  1.3 Research Objectives ............................................... 280
  1.4 Research Questions ................................................ 200
  1.5 Scope of the Study ................................................ 250   Fig 1.2
  1.6 Significance of the Study ......................................... 300
  1.7 Research Contributions ............................................ 300
  1.8 Thesis Structure .................................................. 170

CHAPTER 2  LITERATURE REVIEW ........................................... 4,700
  2.1 Introduction ...................................................... 180
  2.2 Deep Learning for Disaster Image Analysis
      2.2.1 Unimodal Crisis Image Classification ......................... 450
      2.2.2 Multimodal Image–Text Fusion for Crisis Data ................ 450
  2.3 Transformers and Vision-Language Models
      2.3.1 Transformer Encoders for Vision and Language ................ 330
      2.3.2 Multi-Task Vision-Language Learning ......................... 330
  2.4 Bengali Language Processing
      2.4.1 Bengali Language Models and Machine Translation ............. 320
      2.4.2 Bengali Image Captioning .................................... 350
  2.5 Parameter-Efficient Fine-Tuning ................................... 380   Fig 2.1
  2.6 Explainability and Reliability of Deep Models
      2.6.1 Visual Explanation Methods for Vision Transformers .......... 350
      2.6.2 Evaluating the Faithfulness of Explanations ................. 200
      2.6.3 Confidence Calibration ...................................... 230
  2.7 Evaluation Practice
      2.7.1 Data Leakage and Reproducibility ............................ 200
      2.7.2 Automatic and Human Evaluation of Generated Text ............ 280
  2.8 Comparative Analysis of Reviewed Studies .......................... 350   Table 2.1
  2.9 Research Gap and Summary .......................................... 300

CHAPTER 3  METHODOLOGY ................................................. 5,400
  3.1 Introduction ...................................................... 150
  3.2 Overview of the Proposed Framework ................................ 260   Fig 3.1
  3.3 Dataset Construction
      3.3.1 Source Data and Class Definition ............................ 330   Table 3.1
      3.3.2 Bengali Text through Machine Translation .................... 260
      3.3.3 Leakage-Free Grouped Data Split ............................. 300   Table 3.2
      3.3.4 Image and Text Preprocessing ................................ 150
  3.4 Model Architecture
      3.4.1 Vision Encoder: ViT-Base/16 ................................. 280   Eq 3.1
      3.4.2 Text Encoder: BanglaBERT .................................... 220
      3.4.3 Feature Fusion and Classification Head ...................... 250   Eq 3.2
      3.4.4 Caption Decoder of the Multi-Task Variant ................... 260   Eq 3.3
  3.5 Parameter-Efficient Adaptation with LoRA .......................... 300   Eq 3.4, Table 3.3
  3.6 Training Procedure
      3.6.1 Loss Functions .............................................. 220   Eq 3.5–3.6
      3.6.2 Optimization and Hyperparameters ............................ 250   Table 3.4
  3.7 Baselines and Ablation Design ..................................... 300   Table 3.5
  3.8 Explainability and Reliability Analysis
      3.8.1 Grad-CAM for the Vision Transformer ......................... 250   Fig 3.2, Eq 3.7
      3.8.2 Attention Rollout and Decoder Cross-Attention ............... 220   Eq 3.8
      3.8.3 Deletion-Based Faithfulness Test ............................ 170
      3.8.4 Temperature Scaling and Calibration Metrics ................. 200   Eq 3.9–3.10
  3.9 Evaluation Protocol
      3.9.1 Classification Metrics and Seed Protocol .................... 170   Eq 3.11
      3.9.2 Caption Metrics and Reference Baselines ..................... 200
      3.9.3 Human Evaluation Design ..................................... 220
      3.9.4 External Test Set from Bangladesh ........................... 200
  3.10 Implementation Environment ....................................... 120
  3.11 Summary .......................................................... 130

CHAPTER 4  RESULTS AND DISCUSSION ...................................... 5,540
  4.1 Introduction ...................................................... 130
  4.2 Effect of Data Leakage on Reported Performance .................... 380   Table 4.1
  4.3 Classification Performance
      4.3.1 Main Results Across Seeds ................................... 300   Table 4.2
      4.3.2 Per-Class Performance and Error Patterns .................... 380   Fig 4.1, Table 4.3
      4.3.3 Contribution of the Bengali Text ............................ 330   Table 4.4
  4.4 Comparison with Baselines ......................................... 450   Table 4.5, Fig 4.2, Fig 4.3
  4.5 Ablation Study .................................................... 350   Table 4.6, Fig 4.4
  4.6 Bengali Text Generation
      4.6.1 Automatic Evaluation ........................................ 330   Table 4.7
      4.6.2 Qualitative Analysis of Generated Text ...................... 280   Table 4.8
  4.7 Human Evaluation .................................................. 420   Table 4.9
  4.8 Explainability Analysis
      4.8.1 Qualitative Analysis of Explanation Maps .................... 400   Fig 4.5–4.8
      4.8.2 Faithfulness of the Explanations ............................ 300   Table 4.10, Fig 4.9
      4.8.3 Agreement Between Explanation Methods ....................... 180
  4.9 Confidence Calibration ............................................ 300   Table 4.11, Fig 4.10
  4.10 External Validation on Bangladeshi Images ........................ 430   Table 4.12, Table 4.13
  4.11 Discussion ....................................................... 450
  4.12 Summary .......................................................... 130

CHAPTER 5  CONCLUSION .................................................. 1,900
  5.1 Introduction ...................................................... 100
  5.2 Summary of Findings ............................................... 450
  5.3 Answers to the Research Questions ................................. 300
  5.4 Limitations of the Study .......................................... 400
  5.5 Recommendations ................................................... 200
  5.6 Future Work ....................................................... 300
  5.7 Concluding Remarks ................................................ 150

REFERENCES (Zotero, APA 7) · APPENDICES
  Appendix A: Human evaluation guideline (docs/annotation_guidelines_bn.md, summary + rating scale)
  Appendix B: All 21 random XAI samples (results/defense_fixes/phase4/xai_all_21_samples.png)
  Appendix C: Bangladesh test set — labelling guideline and image sources/licenses (bd_external/bd_labels.csv)
```

### Figure / Table / Equation map
| Thesis no. | File / content | Section |
|-----------|----------------|---------|
| Figure 1.1 | `figures/figure_1_1_flood_stats.png` | 1.1.1 |
| Figure 1.2 | `figures/figure_1_2_system_pipeline.png` | 1.5 |
| Figure 2.1 | `figures/figure_2_1_lora.png` | 2.5 |
| Figure 3.1 | `figures/figure_3_1_architecture.png` | 3.2 |
| Figure 3.2 | `figures/figure_3_3_xai_pipeline.png` | 3.8.1 |
| Figure 4.1 | `figures/figure_4_1_confusion_matrix_grouped.png` | 4.3.2 |
| Figure 4.2 | `figures/figure_4_2_baseline_comparison_grouped.png` | 4.4 |
| Figure 4.3 | `results/defense_fixes/phase3b/phase3b_confusion_matrix.png` | 4.4 |
| Figure 4.4 | `figures/figure_4_3_ablation_original_split.png` | 4.5 |
| Figure 4.5 | `figures/figure_4_4_gradcam_grouped.png` | 4.8.1 |
| Figure 4.6 | `figures/figure_4_6_rollout_grouped.png` | 4.8.1 |
| Figure 4.7 | `figures/figure_4_5_crossattn_grouped.png` | 4.8.1 |
| Figure 4.8 | `figures/figure_4_7_integrated_xai_grouped.png` | 4.8.1 |
| Figure 4.9 | `figures/figure_4_9_faithfulness_deletion.png` | 4.8.2 |
| Figure 4.10 | `figures/figure_4_8_calibration_diagram_grouped.png` | 4.9 |
| Table 2.1 | Comparison of reviewed studies | 2.8 |
| Table 3.1 – 3.5 | Class mapping · split · trainable parameters · hyperparameters · baselines/ablations | 3.3.1, 3.3.3, 3.5, 3.6.2, 3.7 |
| Table 4.1 – 4.13 | leakage · seeds · per-class · text ablation · baselines · ablation · caption metrics · caption examples · human eval · faithfulness · calibration · Bangladesh results · Bangladesh sensitivity | Chapter 4 |
| Eq 3.1 – 3.11 | patch embedding · fusion/head · decoder attention · LoRA · CE loss · multi-task loss · Grad-CAM · rollout · temperature softmax · ECE · Macro F1 | Chapter 3 |

> **Note:** figure-এর file name-এর নম্বর আর thesis-এর নম্বর সবসময় এক নয় (যেমন `figure_4_3_…` → Figure 4.4)। Thesis-এ উপরের টেবিলের নম্বরই ব্যবহার করবে।

---

## Part D — Progress tracker

| ✔ | Prompt | Words | ✔ | Prompt | Words | ✔ | Prompt | Words |
|---|--------|-------|---|--------|-------|---|--------|-------|
| ☐ | 0 Abstract | 280 | ☐ | 3.3.4 | 150 | ☐ | 4.4 | 450 |
| ☐ | 1.1.1 | 320 | ☐ | 3.4.1 | 280 | ☐ | 4.5 | 350 |
| ☐ | 1.1.2 | 330 | ☐ | 3.4.2 | 220 | ☐ | 4.6.1 | 330 |
| ☐ | 1.2 | 400 | ☐ | 3.4.3 | 250 | ☐ | 4.6.2 | 280 |
| ☐ | 1.3 | 280 | ☐ | 3.4.4 | 260 | ☐ | 4.7 | 420 |
| ☐ | 1.4 | 200 | ☐ | 3.5 | 300 | ☐ | 4.8.1 | 400 |
| ☐ | 1.5 | 250 | ☐ | 3.6.1 | 220 | ☐ | 4.8.2 | 300 |
| ☐ | 1.6 | 300 | ☐ | 3.6.2 | 250 | ☐ | 4.8.3 | 180 |
| ☐ | 1.7 | 300 | ☐ | 3.7 | 300 | ☐ | 4.9 | 300 |
| ☐ | 1.8 | 170 | ☐ | 3.8.1 | 250 | ☐ | 4.10 | 430 |
| ☐ | 2.1 | 180 | ☐ | 3.8.2 | 220 | ☐ | 4.11 | 450 |
| ☐ | 2.2.1 | 450 | ☐ | 3.8.3 | 170 | ☐ | 4.12 | 130 |
| ☐ | 2.2.2 | 450 | ☐ | 3.8.4 | 200 | ☐ | 5.1 | 100 |
| ☐ | 2.3.1 | 330 | ☐ | 3.9.1 | 170 | ☐ | 5.2 | 450 |
| ☐ | 2.3.2 | 330 | ☐ | 3.9.2 | 200 | ☐ | 5.3 | 300 |
| ☐ | 2.4.1 | 320 | ☐ | 3.9.3 | 220 | ☐ | 5.4 | 400 |
| ☐ | 2.4.2 | 350 | ☐ | 3.9.4 | 200 | ☐ | 5.5 | 200 |
| ☐ | 2.5 | 380 | ☐ | 3.10 | 120 | ☐ | 5.6 | 300 |
| ☐ | 2.6.1 | 350 | ☐ | 3.11 | 130 | ☐ | 5.7 | 150 |
| ☐ | 2.6.2 | 200 | ☐ | 4.1 | 130 | ☐ | X.1 Abbreviations | — |
| ☐ | 2.6.3 | 230 | ☐ | 4.2 | 380 | ☐ | X.2 Consistency check | — |
| ☐ | 2.7.1 | 200 | ☐ | 4.3.1 | 300 | | | |
| ☐ | 2.7.2 | 280 | ☐ | 4.3.2 | 380 | | | |
| ☐ | 2.8 | 350 | ☐ | 4.3.3 | 330 | | | |
| ☐ | 2.9 | 300 | ☐ | | | | | |
| ☐ | 3.1 | 150 | | | | | | |
| ☐ | 3.2 | 260 | | | | | | |
| ☐ | 3.3.1 | 330 | | | | | | |
| ☐ | 3.3.2 | 260 | | | | | | |
| ☐ | 3.3.3 | 300 | | | | | | |

---

## Part E — Common rules for every prompt

Every prompt below starts with the same first line. **Copy the whole code block**, including that line.
If a parent section (e.g. 3.3) has sub-sections, the first sub-section prompt also writes the parent heading
and a two-sentence lead-in; the other sub-section prompts write only their own heading.

---

## Part F — Prompts

### ━━━━━━━━━━ ABSTRACT (write last) ━━━━━━━━━━

#### PROMPT 0 — Abstract
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: ABSTRACT
TARGET LENGTH: 260–300 words, one paragraph, no headings.
CONTENT (in this order):
1. Problem (2 sentences): disaster images on social media could support response in Bangladesh, but Bengali-language tools are scarce and many reported results are not checked for leakage, explanation quality or calibration.
2. Approach: ViT-Base/16 and BanglaBERT adapted with LoRA (r = 8; 1.38 M trainable parameters, ≈ 0.70%), three classes, CrisisMMD-derived corpus of 3,755 images with machine-translated Bengali text, leakage-free tweet-grouped split; a multi-task variant adds a Bengali text decoder.
3. Results with numbers: Macro F1 75.69 ± 1.33% (3 seeds, n = 375), on par with a ViT-Large baseline trained with 38× more parameters; leakage had inflated an earlier figure from 75.69% to 83.21%; text ablation shows the classifier is image-driven; Grad-CAM faithful in a deletion test (21.27 vs 4.16 percentage-point probability drop for random patches at 50% removal); ECE 2.57% without post-hoc scaling; generated Bengali text imitates tweets (human rating 1.24–1.58 of 5); 81.98% Macro F1 (95% CI 76.1–87.2) on 325 independently labelled Bangladeshi images, with strong annotator dependence.
4. Contribution (1–2 sentences): a parameter-efficient, carefully evaluated baseline and an honest account of what the Bengali text and caption branch do and do not contribute.
CITATIONS: Do not include any [REF] in the Abstract.
```

### ━━━━━━━━━━ CHAPTER 1: INTRODUCTION ━━━━━━━━━━

#### PROMPT 1.1.1 — Disasters in Bangladesh and the Need for Rapid Information
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Begin with "CHAPTER 1 INTRODUCTION", then "1.1 Background of the Study" with a two-sentence lead-in, then "1.1.1 Disasters in Bangladesh and the Need for Rapid Information".
TARGET LENGTH: 300–340 words (including the lead-in).
CONTENT:
- Bangladesh's exposure to floods and cyclones (deltaic geography, monsoon, cyclones from the Bay of Bengal); population exposed to flood risk: 94.42 million, 57.5% of the population (Rentschler & Salhab).
- Use the Ganges Delta satellite-monitoring paper for the geographic context.
- Why responders need rapid situational information in the first hours: damage location, people affected, relief activity.
- Limits of manual assessment and of satellite-only monitoring (coarse, no ground-level human context).
PLACEHOLDERS: [FIGURE 1.1: figure_1_1_flood_stats.png — Population of Bangladesh exposed to flood risk (Rentschler & Salhab, 2020)] after the sentence giving the exposure figure; refer to it in the text.
CITATIONS: Include 3–5 [REF] placeholders; put author/year inside when known.
```

#### PROMPT 1.1.2 — Social Media Imagery and Deep Learning for Crisis Response
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 1.1.2 Social Media Imagery and Deep Learning for Crisis Response
TARGET LENGTH: 310–350 words.
CONTENT:
- During disasters people post photographs and short texts; CrisisMMD collected such tweets with humanitarian labels.
- Deep learning (CNNs, then Vision Transformers) can sort these images into categories useful for response; multimodal models also read the text.
- Text analytics for emergency response (AI and NLP for social good paper).
- Two practical requirements that motivate this thesis: (a) the information must be usable in Bengali, the language of local responders; (b) model output must be trustworthy — explanations and reliable confidence, because errors cost time and resources.
Do not present results of this study here.
CITATIONS: Include 4–6 [REF] placeholders.
```

#### PROMPT 1.2 — Problem Statement
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 1.2 Problem Statement
TARGET LENGTH: 380–420 words.
CONTENT — four connected problems, one paragraph each:
1. Language gap: crisis image tools work almost only with English text; Bengali multimodal disaster resources are few (BanglaMM-Disaster is text+image classification; Bengali captioning work uses general-domain images).
2. Evaluation gap: results on tweet-based datasets are often reported on random splits where several images of the same tweet or retweets fall into both training and test sets; such leakage inflates scores (Kapoor & Narayanan). Single-seed results hide variance.
3. Trust gap: explanations are usually shown as pictures without testing whether they reflect the model's decision; confidence calibration is rarely reported.
4. Resource gap: full fine-tuning of large vision and language models is expensive for institutions with one GPU.
End with one sentence stating the problem this thesis addresses.
CITATIONS: Include 3–5 [REF] placeholders.
```

#### PROMPT 1.3 — Research Objectives
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 1.3 Research Objectives
TARGET LENGTH: 260–300 words: one introductory sentence, a numbered list of 5 objectives (one or two sentences each), one closing sentence.
OBJECTIVES (rephrase in academic English; keep the meaning):
1. To build a Bengali-text version of the CrisisMMD humanitarian image subset and a leakage-free, tweet-grouped train/validation/test split.
2. To design a parameter-efficient multimodal classifier that adapts ViT-Base/16 and BanglaBERT with LoRA and runs on a single 16 GB GPU.
3. To measure what the Bengali text contributes to classification and whether a jointly trained decoder can generate Bengali descriptions of the images.
4. To explain the model's decisions with Grad-CAM, Attention Rollout and decoder cross-attention, to test the faithfulness of these explanations, and to assess confidence calibration.
5. To test the classifier on independently labelled disaster images from Bangladesh.
Do not state results.
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 1.4 — Research Questions
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 1.4 Research Questions
TARGET LENGTH: 180–220 words: one introductory sentence, then five research questions (RQ1–RQ5) as a numbered list, each followed by one sentence saying which chapter section answers it.
QUESTIONS (rephrase, keep meaning):
RQ1: How accurately can a LoRA-adapted ViT-Base/16 + BanglaBERT model classify disaster images on a leakage-free split, and how does it compare with baselines that train many more parameters? (4.3, 4.4)
RQ2: Does the machine-translated Bengali text improve classification beyond the image? (4.3.3)
RQ3: Can a jointly trained decoder produce useful Bengali descriptions, according to automatic and human evaluation? (4.6, 4.7)
RQ4: Are the visual explanations faithful to the model's decision, and is the model's confidence calibrated? (4.8, 4.9)
RQ5: How does the classifier perform on disaster images from Bangladesh? (4.10)
CITATIONS: Do not include any [REF].
```

#### PROMPT 1.5 — Scope of the Study
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 1.5 Scope of the Study
TARGET LENGTH: 230–270 words.
CONTENT:
- In scope: three humanitarian classes (Severe Damage, Humanitarian Rescue, Affected People); still images with accompanying Bengali text; CrisisMMD-derived corpus (mostly 2017 hurricanes, some Sri Lanka floods); ViT-Base/16 + BanglaBERT with LoRA; explanation and calibration analysis; a small external test on Bangladeshi images.
- Out of scope: satellite or drone imagery, video, real-time deployment, alert dispatch and integration with responders' systems, collection of a new large Bengali dataset, training with human-written Bengali image descriptions, large generative vision-language models.
- Explain that DRISHTI-Bn is the analysis module of a larger workflow (Figure 1.2).
PLACEHOLDERS: [FIGURE 1.2: figure_1_2_system_pipeline.png — Position of DRISHTI-Bn in a disaster-information workflow; dashed parts are outside the scope of this thesis]
CITATIONS: Include 0–2 [REF] placeholders.
```

#### PROMPT 1.6 — Significance of the Study
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 1.6 Significance of the Study
TARGET LENGTH: 280–320 words.
CONTENT — significance for three groups:
- Researchers: a leakage-free split and a multi-seed, honestly reported Bengali multimodal baseline; evidence of how much leakage inflates scores; a tested explanation method.
- Practitioners in Bangladesh: a model that needs only 1.38 M trainable parameters and one GPU; calibrated confidence and faithful heat maps support human review instead of blind automation.
- Bengali NLP: shows the limits of machine-translated tweets as training text and the need for human-written Bengali image descriptions.
Keep claims modest (preliminary external test; text did not help).
CITATIONS: Include 2–3 [REF] placeholders.
```

#### PROMPT 1.7 — Research Contributions
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 1.7 Research Contributions
TARGET LENGTH: 280–320 words: one introductory sentence and a numbered list of the six contributions in fact sheet §10, each in 2–3 sentences with the key number.
Do not add contributions that are not in §10. Respect the "DO NOT CLAIM" list (§11).
CITATIONS: Do not include any [REF].
```

#### PROMPT 1.8 — Thesis Structure
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 1.8 Thesis Structure
TARGET LENGTH: 150–190 words, one paragraph.
CONTENT: one sentence per chapter, using the structure in "01_NOTEBOOKLM_WRITING_PLAN.md" Part C: Chapter 2 literature; Chapter 3 data, model, training, explanation and evaluation methods; Chapter 4 results and discussion (leakage, classification, baselines, ablation, text generation, human evaluation, explanations, calibration, Bangladesh test); Chapter 5 conclusions, limitations, future work. Mention the appendices.
CITATIONS: Do not include any [REF].
```

### ━━━━━━━━━━ CHAPTER 2: LITERATURE REVIEW ━━━━━━━━━━

#### PROMPT 2.1 — Introduction
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: Begin with "CHAPTER 2 LITERATURE REVIEW", then "2.1 Introduction".
TARGET LENGTH: 160–200 words.
CONTENT: purpose of the chapter; how the literature was selected (peer-reviewed work mostly from 2020 onward, plus foundational methods); the order of the sections (2.2 disaster image analysis, 2.3 transformers and vision-language models, 2.4 Bengali NLP and captioning, 2.5 parameter-efficient fine-tuning, 2.6 explainability and calibration, 2.7 evaluation practice, 2.8 comparison table, 2.9 gap).
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 2.2.1 — Unimodal Crisis Image Classification
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: Write "2.2 Deep Learning for Disaster Image Analysis" with a two-sentence lead-in, then "2.2.1 Unimodal Crisis Image Classification".
TARGET LENGTH: 430–470 words (including the lead-in).
CONTENT — synthesize, do not summarize paper by paper:
- Image-only classification of crisis imagery: CNN baselines on CrisisMMD; CrisisViT (crisis-specific Vision Transformer pretraining, reported accuracy); MEDIC multi-task disaster image dataset (if uploaded).
- Flood-specific work: explainable flood damage assessment with multi-atrous self-attention and vision-language integration; satellite deep learning for flood dynamics in the Ganges Delta.
- Common limitations: English-only or no text, single random splits, explanations shown only qualitatively.
- Link to this study: CrisisViT-style model is re-implemented as a baseline (Section 3.7).
Report other papers' numbers only as given in those papers, with [REF].
CITATIONS: Include 5–7 [REF] placeholders.
```

#### PROMPT 2.2.2 — Multimodal Image–Text Fusion for Crisis Data
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.2.2 Multimodal Image–Text Fusion for Crisis Data
TARGET LENGTH: 430–470 words.
CONTENT:
- CrisisMMD: tweets + images, informativeness and humanitarian labels, events of 2017.
- Multimodal fusion on CrisisMMD (Ofli et al., 2020; Abavisani et al., 2020): reported gains from combining text and image, and how fusion is done (late fusion, cross-attention).
- BanglaMM-Disaster: Bengali text + image classification; ReliefNet: reliable multimodal severity classification; BiMCF: bidirectional multi-task cascaded multimodal fusion for damage analysis.
- Critical point: gains from text depend on the text being informative; few studies test whether the model really uses the text (e.g. by swapping texts) — this thesis does (Section 4.3.3).
CITATIONS: Include 5–7 [REF] placeholders.
```

#### PROMPT 2.3.1 — Transformer Encoders for Vision and Language
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: Write "2.3 Transformers and Vision-Language Models" with a two-sentence lead-in, then "2.3.1 Transformer Encoders for Vision and Language".
TARGET LENGTH: 310–350 words.
CONTENT:
- Transformer self-attention (Vaswani et al.) in brief — no textbook definition, only what is needed later.
- Vision Transformer: images as sequences of 16 × 16 patches with a [CLS] token; needs large-scale pretraining (ImageNet-21k).
- Pretrained text encoders: BERT-style masked language models and ELECTRA's replaced-token detection, which BanglaBERT uses.
- Why encoders pretrained separately can be combined through a small fusion head.
CITATIONS: Include 3–5 [REF] placeholders.
```

#### PROMPT 2.3.2 — Multi-Task Vision-Language Learning
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.3.2 Multi-Task Vision-Language Learning
TARGET LENGTH: 310–350 words.
CONTENT:
- Joint classification and text generation with shared encoders: PetroMind (rock image classification + lithological description), co-training vision-language models for remote-sensing multi-task learning, SARChat-Bench-2M (multi-task SAR vision-language benchmark), BiMCF (multi-task cascaded fusion).
- Expected benefits (shared representation, one model for two outputs) and reported risks (task interference, need for good descriptive references).
- Note that these works use expert-written or curated descriptions, unlike tweet text.
CITATIONS: Include 3–5 [REF] placeholders.
```

#### PROMPT 2.4.1 — Bengali Language Models and Machine Translation
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: Write "2.4 Bengali Language Processing" with a two-sentence lead-in, then "2.4.1 Bengali Language Models and Machine Translation".
TARGET LENGTH: 300–340 words.
CONTENT:
- Bengali as a widely spoken but low-resource language for NLP.
- BanglaBERT: ELECTRA-based pretraining on a large Bengali corpus; benchmark results as reported.
- Bengali–English machine translation resources (Hasan et al., 2020) and typical errors in translating informal social-media text (hashtags, URLs, named entities).
- NLP for social good / emergency text analytics in low-resource settings.
CITATIONS: Include 3–4 [REF] placeholders.
```

#### PROMPT 2.4.2 — Bengali Image Captioning
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.4.2 Bengali Image Captioning
TARGET LENGTH: 330–370 words.
CONTENT:
- Encoder–decoder captioning in Bengali: BiCap (attention-based encoder–decoder), image captioning in Bengali with visual attention (Masud et al.), transfer learning for low-resource Bengali captioning.
- Datasets used by these works (general-domain images with human-written Bengali captions) and the metrics they report (BLEU etc., as given in the papers).
- Gap: no Bengali captioning work on disaster images; disaster datasets contain tweets, not descriptions.
CITATIONS: Include 3–5 [REF] placeholders.
```

#### PROMPT 2.5 — Parameter-Efficient Fine-Tuning
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.5 Parameter-Efficient Fine-Tuning
TARGET LENGTH: 360–400 words.
CONTENT:
- Cost of full fine-tuning for large pretrained models; families of parameter-efficient methods (adapters, prompt tuning, low-rank updates) — from the PEFT survey if uploaded.
- LoRA: frozen weight W₀ plus a trainable low-rank product BA scaled by α/r; B starts at zero so training begins from the pretrained model; no extra inference latency after merging.
- Use of LoRA beyond language models (vision and multimodal models).
- Bridge sentence: this study applies LoRA to both encoders (details in Section 3.5).
PLACEHOLDERS: [FIGURE 2.1: figure_2_1_lora.png — Low-Rank Adaptation: a frozen pretrained weight with a trainable low-rank update]
CITATIONS: Include 3–4 [REF] placeholders.
```

#### PROMPT 2.6.1 — Visual Explanation Methods for Vision Transformers
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: Write "2.6 Explainability and Reliability of Deep Models" with a two-sentence lead-in, then "2.6.1 Visual Explanation Methods for Vision Transformers".
TARGET LENGTH: 330–370 words.
CONTENT:
- Why explanations matter in humanitarian decisions.
- Grad-CAM: class-specific gradients weighting feature maps; originally for CNNs; adaptation to Vision Transformers by treating patch tokens as a spatial map.
- Attention Rollout: propagating attention through layers; raw attention is not an explanation of the decision (Chefer et al. if uploaded).
- Cross-attention maps in captioning models as a view of what the decoder looks at.
- Use in disaster work: explainable flood damage assessment paper.
CITATIONS: Include 4–5 [REF] placeholders.
```

#### PROMPT 2.6.2 — Evaluating the Faithfulness of Explanations
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.6.2 Evaluating the Faithfulness of Explanations
TARGET LENGTH: 180–220 words.
CONTENT: plausibility (looks right to a human) vs faithfulness (reflects the model's reasoning); deletion/insertion metrics (RISE): remove the most important pixels or patches and measure the drop in the class probability; comparison with random removal as a lower bound. Note that most disaster XAI papers show heat maps without such a test.
CITATIONS: Include 2–3 [REF] placeholders.
```

#### PROMPT 2.6.3 — Confidence Calibration
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.6.3 Confidence Calibration
TARGET LENGTH: 210–250 words.
CONTENT: definition of a calibrated classifier (confidence equals accuracy); modern deep networks tend to be over-confident (Guo et al.); Expected Calibration Error and reliability diagrams; temperature scaling as a one-parameter post-hoc fix fitted on validation data; why calibration matters when responders must decide which predictions to verify by hand.
CITATIONS: Include 2–3 [REF] placeholders.
```

#### PROMPT 2.7.1 — Data Leakage and Reproducibility
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: Write "2.7 Evaluation Practice" with a one-sentence lead-in, then "2.7.1 Data Leakage and Reproducibility".
TARGET LENGTH: 180–220 words.
CONTENT: types of leakage described by Kapoor & Narayanan (e.g. duplicates or dependent samples across train and test); why social-media datasets are prone to it (several images per tweet, retweets, copied text); group-aware splitting as the remedy; the value of reporting several random seeds.
CITATIONS: Include 2–3 [REF] placeholders.
```

#### PROMPT 2.7.2 — Automatic and Human Evaluation of Generated Text
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.7.2 Automatic and Human Evaluation of Generated Text
TARGET LENGTH: 260–300 words.
CONTENT: n-gram metrics (BLEU) and the reporting problems addressed by sacreBLEU (Post); character-level chrF, better suited to morphologically rich languages such as Bengali; embedding-based BERTScore; limits of all reference-based metrics when the reference itself is not a description; human evaluation practice (van der Lee et al.): criteria such as fluency and adequacy, blind presentation, several annotators, agreement statistics (Krippendorff's α).
CITATIONS: Include 3–5 [REF] placeholders.
```

#### PROMPT 2.8 — Comparative Analysis of Reviewed Studies
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.8 Comparative Analysis of Reviewed Studies
TARGET LENGTH: 330–370 words of prose plus Table 2.1.
CONTENT:
- Table 2.1 with 8–10 rows (CrisisViT, BanglaMM-Disaster, BiMCF, ReliefNet, explainable flood damage assessment, Ofli et al. 2020, Abavisani et al. 2020, BiCap, Masud et al., PetroMind) and columns: Study (with [REF]) | Year | Modality | Task | Language | Explanation evaluated? | Leakage-aware split? | Main limitation. Fill only with information found in the papers; write "not reported" otherwise.
- Last row: "This study" — image + Bengali text; classification + Bengali generation; Bengali; Grad-CAM with deletion test; tweet-grouped split, 3 seeds.
- Prose: discuss patterns in the table (few Bengali multimodal works, explanations rarely tested, splits rarely leakage-aware, captioning never on disaster images).
PLACEHOLDERS: [TABLE 2.1: Comparison of reviewed studies with the present study] (caption above the table).
CITATIONS: Include 6–10 [REF] placeholders (one per study in the table).
```

#### PROMPT 2.9 — Research Gap and Summary
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 2.9 Research Gap and Summary
TARGET LENGTH: 280–320 words.
CONTENT: four gaps derived from the review — (1) no Bengali multimodal disaster framework that combines classification, text generation and explanation; (2) leakage and seed variance rarely controlled on tweet-based data; (3) explanations not tested for faithfulness, calibration rarely reported; (4) almost no evaluation on Bangladeshi disaster images. Map each gap to the objective in Section 1.3 that addresses it. Close with a two-sentence summary of the chapter.
CITATIONS: Include 1–3 [REF] placeholders.
```

### ━━━━━━━━━━ CHAPTER 3: METHODOLOGY ━━━━━━━━━━

#### PROMPT 3.1 — Introduction
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Begin with "CHAPTER 3 METHODOLOGY", then "3.1 Introduction".
TARGET LENGTH: 130–170 words.
CONTENT: what the chapter describes and in which order (framework overview, dataset construction, architecture, LoRA, training, baselines and ablations, explanation and calibration methods, evaluation protocol, implementation).
CITATIONS: Do not include any [REF].
```

#### PROMPT 3.2 — Overview of the Proposed Framework
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.2 Overview of the Proposed Framework
TARGET LENGTH: 240–280 words.
CONTENT: walk through Figure 3.1: image → ViT-Base/16; Bengali text → BanglaBERT; both frozen with LoRA adapters; [CLS] vectors concatenated → MLP head → three classes. The multi-task variant adds a 3-layer Transformer decoder that cross-attends to the ViT patch tokens and generates Bengali tokens. State clearly that the CLS-Only model is the main model and why (equal or better classification with 1.38 M instead of 74.79 M trainable parameters). Mention the explanation and calibration analysis applied after training (Section 3.8).
PLACEHOLDERS: [FIGURE 3.1: figure_3_1_architecture.png — Architecture of DRISHTI-Bn. Solid: CLS-Only model; dashed: caption branch of the multi-task variant]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 3.3.1 — Source Data and Class Definition
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Write "3.3 Dataset Construction" with a two-sentence lead-in, then "3.3.1 Source Data and Class Definition".
TARGET LENGTH: 310–350 words (including the lead-in).
CONTENT: CrisisMMD v2.0 and its humanitarian categories; mapping of the CrisisMMD categories into three classes (fact sheet §2.1); excluded categories; resulting 3,755 images and class shares; event composition (95.2% hurricanes, 4.8% Sri Lanka floods, no Bangladeshi images) — explain why the thesis says "disaster images" rather than "flood images"; class imbalance (Affected People 8.8%).
PLACEHOLDERS: [TABLE 3.1: Mapping of CrisisMMD humanitarian categories to the three classes of this study, with image counts]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.3.2 — Bengali Text through Machine Translation
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.3.2 Bengali Text through Machine Translation
TARGET LENGTH: 240–280 words.
CONTENT: English tweet text translated automatically to Bengali with the open-source model monirbishal/en-bn-nmt; no manual verification; observed properties (8.9% partly untranslated, 97.3% with URL fragments, hashtags and mentions); consequence: the Bengali text is a translated tweet that often does not describe the image — important for Sections 4.3.3 and 4.6. Be explicit that the translation was not human-verified.
CITATIONS: Include 1–2 [REF] placeholders (Bengali–English MT literature).
```

#### PROMPT 3.3.3 — Leakage-Free Grouped Data Split
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.3.3 Leakage-Free Grouped Data Split
TARGET LENGTH: 280–320 words.
CONTENT: problem found in a random split (80 of 376 test samples, 21.3%, share their exact text with a training sample); grouping rule (same tweet ID or identical text → same group, union-find, 3,189 groups); group-aware 80/10/10 split, most class-balanced of 200 random group splits; zero overlap; class distribution per split (Table 3.2). All results in Chapter 4 use this split unless marked otherwise.
PLACEHOLDERS: [TABLE 3.2: Class distribution of the tweet-grouped train, validation and test sets]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.3.4 — Image and Text Preprocessing
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.3.4 Image and Text Preprocessing
TARGET LENGTH: 130–170 words.
CONTENT: resize to 224 × 224, ImageNet mean/std normalization, no augmentation; BanglaBERT tokenizer (32,000 sub-words), padding/truncation to 128 tokens, attention mask; the same tokenizer provides the vocabulary of the caption decoder.
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.4.1 — Vision Encoder: ViT-Base/16
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Write "3.4 Model Architecture" with a two-sentence lead-in, then "3.4.1 Vision Encoder: ViT-Base/16".
TARGET LENGTH: 260–300 words.
CONTENT: google/vit-base-patch16-224-in21k; image split into 196 patches of 16 × 16 pixels, linearly projected to d = 768, plus a [CLS] token and position embeddings; 12 layers, 12 heads; pooled [CLS] vector v_CLS ∈ ℝ^768 used by the classifier; the 196 patch tokens used by the caption decoder and by Grad-CAM; pretrained weights frozen.
PLACEHOLDERS: [EQUATION 3.1: z₀ = [x_cls ; x_p¹E ; … ; x_p¹⁹⁶E] + E_pos, E ∈ ℝ^((16·16·3)×768)]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.4.2 — Text Encoder: BanglaBERT
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.4.2 Text Encoder: BanglaBERT
TARGET LENGTH: 200–240 words.
CONTENT: csebuetnlp/banglabert, ELECTRA discriminator, 12 layers, hidden size 768; input = tokenized Bengali text (≤ 128 tokens); output [CLS] vector t_CLS ∈ ℝ^768; pretrained weights frozen, adapted only through LoRA; why a Bengali-specific encoder rather than a multilingual one.
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.4.3 — Feature Fusion and Classification Head
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.4.3 Feature Fusion and Classification Head
TARGET LENGTH: 230–270 words.
CONTENT: concatenation f = [v_CLS ; t_CLS] ∈ ℝ^1536; head Linear(1536→512) → BatchNorm → ReLU → Dropout 0.3 → Linear(512→3) → softmax; 789,507 head parameters; reason for simple late fusion (small data, parameter budget, makes the text ablation in 4.3.3 straightforward).
PLACEHOLDERS: [EQUATION 3.2: ŷ = softmax(W₂ · Dropout(ReLU(BN(W₁ f + b₁))) + b₂), W₁ ∈ ℝ^(512×1536), W₂ ∈ ℝ^(3×512)]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 3.4.4 — Caption Decoder of the Multi-Task Variant
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.4.4 Caption Decoder of the Multi-Task Variant
TARGET LENGTH: 240–280 words.
CONTENT: 3 Transformer decoder layers, 8 heads, feed-forward 2048, token embedding 32,000 × 768 with sinusoidal positions; masked self-attention over generated tokens and cross-attention to the 196 ViT patch tokens; right-shifted teacher forcing in training; inference by beam search (width 3, max 40 tokens, length normalization 0.7, repetition penalty, bigram blocking). The decoder is not used by the classification path.
PLACEHOLDERS: [EQUATION 3.3: CrossAttn(Q, K, V) = softmax(Q Kᵀ / √d_k) V, with Q from decoder states and K, V from ViT patch tokens]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.5 — Parameter-Efficient Adaptation with LoRA
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.5 Parameter-Efficient Adaptation with LoRA
TARGET LENGTH: 280–320 words.
CONTENT: LoRA on the query and value projections of all attention layers in both encoders; r = 8, α = 16, dropout 0.1; B initialized to zero; parameter accounting in Table 3.3 (LoRA 589,824 = 294,912 per encoder; head 789,507; total trainable 1,379,331 ≈ 0.70%; multi-task 74.79 M because of the decoder). Explain that 0.30% refers to LoRA relative to the encoders only.
PLACEHOLDERS:
[EQUATION 3.4: h = W₀x + (α / r) · B A x, A ∈ ℝ^(r×d), B ∈ ℝ^(d×r), r = 8, α = 16]
[TABLE 3.3: Trainable parameters of the CLS-Only and Multi-Task models]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.6.1 — Loss Functions
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Write "3.6 Training Procedure" with a one-sentence lead-in, then "3.6.1 Loss Functions".
TARGET LENGTH: 200–240 words.
CONTENT: cross-entropy for classification; token-level negative log-likelihood for captioning (padding ignored); multi-task loss with λ_cls = 1.0, λ_cap = 0.5; CLS-Only uses only the classification loss.
PLACEHOLDERS:
[EQUATION 3.5: L_cls = − Σ_c y_c log ŷ_c]
[EQUATION 3.6: L = λ_cls L_cls + λ_cap L_cap, L_cap = − (1/T) Σ_t log p(w_t | w_<t, V), λ_cls = 1.0, λ_cap = 0.5]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 3.6.2 — Optimization and Hyperparameters
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.6.2 Optimization and Hyperparameters
TARGET LENGTH: 230–270 words.
CONTENT: AdamW (lr 1 × 10⁻⁴, weight decay 0.01), cosine annealing, batch 16, mixed precision; CLS-Only 10 epochs, checkpoint with best validation Macro F1, seeds 42/123/2024; Multi-Task 15 epochs, checkpoint with lowest validation total loss (epoch 4), seed 42; the test set is used only once per final model.
PLACEHOLDERS: [TABLE 3.4: Training hyperparameters of the CLS-Only and Multi-Task models]
CITATIONS: Include 1–2 [REF] placeholders (AdamW / cosine schedule if available; otherwise [REF]).
```

#### PROMPT 3.7 — Baselines and Ablation Design
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.7 Baselines and Ablation Design
TARGET LENGTH: 280–320 words.
CONTENT:
- Baselines on the grouped split: ViT-Only (ViT-Base, last 2 blocks + pooler fine-tuned, 15.16 M trainable, 3 seeds); CrisisViT-style re-implementation (ViT-Large, last 4 blocks + pooler, 51.96 M, 1 seed) — say clearly it is a re-implementation, not the original weights.
- Text ablation at test time (original tweet, generated caption, fixed placeholder, another sample's tweet, empty text).
- Ablation on the original split (single seed): no LoRA, LoRA rank 4/8/16, decoder without cross-attention; also random, BanglaBERT-only and full fine-tuning baselines on that split — explain they were not re-run after the split correction and are reported as such.
PLACEHOLDERS: [TABLE 3.5: Baselines and ablation configurations, split used and number of seeds]
CITATIONS: Include 1–3 [REF] placeholders.
```

#### PROMPT 3.8.1 — Grad-CAM for the Vision Transformer
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Write "3.8 Explainability and Reliability Analysis" with a two-sentence lead-in referring to Figure 3.2, then "3.8.1 Grad-CAM for the Vision Transformer".
TARGET LENGTH: 230–270 words.
CONTENT: Grad-CAM computed on the 196 patch tokens entering the last ViT block; weights = mean gradient of the predicted-class logit over tokens; map = ReLU of the weighted sum, reshaped to 14 × 14 and upsampled; explain why the input of the last block is used (patch outputs of the last block receive no gradient because only [CLS] reaches the classifier; an earlier implementation produced blank maps for this reason).
PLACEHOLDERS:
[FIGURE 3.2: figure_3_3_xai_pipeline.png — Explanation and reliability analysis applied to the trained models]
[EQUATION 3.7: α_k = (1/N) Σ_i ∂y^c / ∂A_ik ;  M^c = ReLU(Σ_k α_k A_k), A ∈ ℝ^(196×768)]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.8.2 — Attention Rollout and Decoder Cross-Attention
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.8.2 Attention Rollout and Decoder Cross-Attention
TARGET LENGTH: 200–240 words.
CONTENT: rollout over 12 layers — head-averaged attention, lowest 90% of weights discarded, identity added, row-normalized, multiplied layer by layer, [CLS]-to-patch row used; decoder cross-attention averaged over 3 layers, 8 heads and generated tokens, which explains caption generation and not the classification.
PLACEHOLDERS: [EQUATION 3.8: R = Ã_12 · Ã_11 · … · Ã_1, Ã_l = rownorm(D_0.9(Ā_l) + I), where Ā_l is the head-averaged attention of layer l and D_0.9 sets its lowest 90% of weights to zero]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.8.3 — Deletion-Based Faithfulness Test
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.8.3 Deletion-Based Faithfulness Test
TARGET LENGTH: 150–190 words.
CONTENT: for every test image (n = 375) the top 10%, 25% and 50% of patches ranked by each map are set to the mean color; the drop in the predicted-class probability and the remaining accuracy are recorded; random patch order is the reference; agreement between methods measured by IoU of their top-10% patches.
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.8.4 — Temperature Scaling and Calibration Metrics
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.8.4 Temperature Scaling and Calibration Metrics
TARGET LENGTH: 180–220 words.
CONTENT: one temperature T fitted on the validation set (n = 373) by minimizing NLL; metrics on the test set: ECE and MCE with 10 equal-width bins, NLL, Brier score, reliability diagram.
PLACEHOLDERS:
[EQUATION 3.9: p̂_i = softmax(z_i / T)]
[EQUATION 3.10: ECE = Σ_{m=1}^{10} (|B_m| / n) · |acc(B_m) − conf(B_m)|]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.9.1 — Classification Metrics and Seed Protocol
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Write "3.9 Evaluation Protocol" with a one-sentence lead-in, then "3.9.1 Classification Metrics and Seed Protocol".
TARGET LENGTH: 150–190 words.
CONTENT: Macro F1 as the main metric because of class imbalance; also macro precision, recall, accuracy, per-class F1 and confusion matrices; mean ± sample standard deviation over seeds 42, 123, 2024; differences smaller than about 2.5 percentage points treated as within seed variance.
PLACEHOLDERS: [EQUATION 3.11: Macro-F1 = (1/C) Σ_c 2·P_c·R_c / (P_c + R_c), C = 3]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 3.9.2 — Caption Metrics and Reference Baselines
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.9.2 Caption Metrics and Reference Baselines
TARGET LENGTH: 180–220 words.
CONTENT: corpus BLEU and chrF with sacreBLEU 2.6.0, BERTScore-F1 with multilingual BERT; computed on raw references and on cleaned references (URLs, RT, @mentions and hashtags removed); two trivial reference baselines — a random training tweet and one constant tweet for all images — to show what a model that ignores the image would score.
CITATIONS: Include 2–3 [REF] placeholders.
```

#### PROMPT 3.9.3 — Human Evaluation Design
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.9.3 Human Evaluation Design
TARGET LENGTH: 200–240 words.
CONTENT: 100 test samples stratified by class; 3 native Bengali-speaking annotators; blind A/B presentation of model output and reference tweet in random order; 1–5 ratings for Fluency, Adequacy, Faithfulness (define each in one clause) plus preference; analysis with Wilcoxon signed-rank tests and Krippendorff's α / quadratic-weighted κ; the guideline is in Appendix A. State that the ratings were collected for the earlier v2 checkpoint and that an un-blinding error in the first analysis was corrected.
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 3.9.4 — External Test Set from Bangladesh
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.9.4 External Test Set from Bangladesh
TARGET LENGTH: 180–220 words.
CONTENT: 419 candidate images from six Wikimedia Commons categories (floods in Bangladesh, cyclones Sidr, Amphan, Aila, Remal, 2017 South Asian floods), open licenses recorded; three annotators labelled independently with a written guideline (SD / HR / AP / exclude), one of them the researcher; majority vote, 10 ties decided jointly; 94 excluded, 325 used; perceptual-hash check found no near-duplicates of CrisisMMD images; model used without any tuning, with a fixed placeholder text because the images have no text; 2,000-sample bootstrap for 95% confidence intervals. Guideline and sources in Appendix C.
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 3.10 — Implementation Environment
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 3.10 Implementation Environment
TARGET LENGTH: 100–140 words.
CONTENT: one NVIDIA T4 (16 GB) on Google Colab Pro; PyTorch 2.14.1, Transformers 5.22.1, PEFT 0.21.2, scikit-learn, sacreBLEU 2.6.0; Bangladesh evaluation and figure generation on a laptop CPU; code and split files in a public GitHub repository.
CITATIONS: Do not include any [REF].
```

#### PROMPT 3.11 — Summary
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 3.11 Summary
TARGET LENGTH: 110–150 words, one paragraph.
CONTENT: recap of data, model, training, explanation and evaluation methods; one sentence leading to Chapter 4.
CITATIONS: Do not include any [REF].
```

### ━━━━━━━━━━ CHAPTER 4: RESULTS AND DISCUSSION ━━━━━━━━━━

#### PROMPT 4.1 — Introduction
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Begin with "CHAPTER 4 RESULTS AND DISCUSSION", then "4.1 Introduction".
TARGET LENGTH: 110–150 words.
CONTENT: order of the chapter; all results are on the grouped test set (n = 375) unless stated; Section 4.5 is the only one on the original split.
CITATIONS: Do not include any [REF].
```

#### PROMPT 4.2 — Effect of Data Leakage on Reported Performance
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.2 Effect of Data Leakage on Reported Performance
TARGET LENGTH: 350–410 words.
CONTENT: use fact sheet §4.6 and §2.3: the random split shared text for 21.3% of test samples; under the grouped split CLS-Only falls from 83.21% (1 seed) to 75.69 ± 1.33% (−7.5 pp), ViT-Only and CrisisViT-style fall less; interpret why text-using models gain most from leakage (exact text match); consequence for published CrisisMMD results that use random splits (cautious wording); from here on only grouped-split numbers are used. This is the ONLY section where 83.21% may appear.
PLACEHOLDERS: [TABLE 4.1: Macro F1 on the original random split and on the tweet-grouped split]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 4.3.1 — Main Results Across Seeds
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Write "4.3 Classification Performance" with a one-sentence lead-in, then "4.3.1 Main Results Across Seeds".
TARGET LENGTH: 280–320 words.
CONTENT: CLS-Only Macro F1 75.69 ± 1.33%, precision 77.09 ± 1.31%, recall 74.93 ± 1.59%, accuracy 88.00 ± 0.27%; per-seed values 74.83 / 75.02 / 77.22; the ~2.4 pp spread means single-seed differences of this size are not meaningful; gap between accuracy and Macro F1 explained by the weak minority class.
PLACEHOLDERS: [TABLE 4.2: CLS-Only results per seed and mean ± standard deviation]
CITATIONS: Do not include any [REF].
```

#### PROMPT 4.3.2 — Per-Class Performance and Error Patterns
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.3.2 Per-Class Performance and Error Patterns
TARGET LENGTH: 350–410 words.
CONTENT: per-class F1 (SD 93.28, HR 88.31, AP 45.48 ± 4.63); confusion matrix of seed 42 (Figure 4.1): SD 94% and HR 92% recall, AP only 11 of 32 correct, errors split between SD (11) and HR (10); reasons — few AP examples (8.8%), visual overlap (people standing in flood water also show damage; people receiving relief look like rescue scenes), label ambiguity in CrisisMMD. Mention that AP is also the weakest class in every baseline.
PLACEHOLDERS:
[FIGURE 4.1: figure_4_1_confusion_matrix_grouped.png — Confusion matrix of the CLS-Only model (seed 42) on the grouped test set (n = 375)]
[TABLE 4.3: Per-class F1 of CLS-Only and the baselines]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 4.3.3 — Contribution of the Bengali Text
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.3.3 Contribution of the Bengali Text
TARGET LENGTH: 310–350 words.
CONTENT: text ablation (fact sheet §4.4): original tweet 74.83%, generated caption 76.41%, placeholder 76.52%, another sample's tweet 76.63% with 96.5% of predictions unchanged, empty text 61.95% (out-of-distribution, not a "no text" condition). Conclusion: the classifier is effectively image-driven; the text encoder adds a near-constant input. Discuss why (translated tweets are noisy and often unrelated to image content; the image alone separates the classes well). State clearly that the study cannot claim a benefit from multimodal fusion, and that this also means the model works with image input only (relevant for deployment and for the Bangladesh test).
PLACEHOLDERS: [TABLE 4.4: Macro F1 of CLS-Only (seed 42) with different text inputs at test time]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 4.4 — Comparison with Baselines
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.4 Comparison with Baselines
TARGET LENGTH: 420–480 words.
CONTENT: Table 4.5 with CLS-Only, ViT-Only, CrisisViT-style, Multi-Task v3 (tweet) and Multi-Task v3 (generated caption), including trainable parameters. Points: CLS-Only beats ViT-Only in all three seeds (+2.42 pp mean) with 11× fewer trainable parameters; CLS-Only and CrisisViT-style are tied within one standard deviation while CLS-Only trains 38× fewer parameters on a smaller backbone → the claim is parameter efficiency; Multi-Task 77.29% is one seed inside the CLS-Only range → no evidence that multi-task learning helps classification; Multi-Task with its own caption (73.63%) ≈ ViT-Only. Figure 4.3 shows the multi-task confusion matrices. Mention the CrisisViT result reported in its own paper only as context, not as a direct comparison (different data and split).
PLACEHOLDERS:
[TABLE 4.5: Comparison with baselines on the grouped test set]
[FIGURE 4.2: figure_4_2_baseline_comparison_grouped.png — Macro F1 of DRISHTI-Bn and the baselines (points: seeds; bars: mean ± std)]
[FIGURE 4.3: phase3b_confusion_matrix.png — Confusion matrices of the Multi-Task v3 model with the original tweet and with its own generated caption]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 4.5 — Ablation Study
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.5 Ablation Study
TARGET LENGTH: 320–380 words.
CONTENT: state first that these runs use the ORIGINAL random split with a single seed and were not repeated after the correction, so absolute values are inflated. Results (fact sheet §4.5): no LoRA 77.74%; LoRA r = 4 79.41%, r = 8 81.47% and 83.21% (two runs of the same configuration), r = 16 79.14%; decoder without cross-attention 81.88%. Interpretation: LoRA adaptation helps over a frozen encoder; the rank choice is not resolved because two identical runs differ by 1.7 pp; the cross-attention ablation does not affect classification because the decoder is not on the classification path. Caption-only (16.90%) is excluded because its classification head was never trained (one footnote sentence).
PLACEHOLDERS:
[TABLE 4.6: Ablation results on the original split (single seed)]
[FIGURE 4.4: figure_4_3_ablation_original_split.png — Ablation study on the original split]
CITATIONS: Do not include any [REF].
```

#### PROMPT 4.6.1 — Automatic Evaluation
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Write "4.6 Bengali Text Generation" with a one-sentence lead-in, then "4.6.1 Automatic Evaluation".
TARGET LENGTH: 310–350 words.
CONTENT: BLEU 14.38 / chrF 14.19 / BERTScore 69.66% on raw tweet references; 0.11 / 8.46 / 62.27% on cleaned references; trivial baselines score higher (random tweet chrF 21.58 / 11.93; constant tweet 20.46 / 15.19). Interpretation: the raw BLEU comes from shared tweet patterns (URLs, hashtags, event names), not from describing the image; the model performs below a system that ignores the image.
PLACEHOLDERS: [TABLE 4.7: Automatic evaluation of the generated Bengali text, with trivial reference baselines]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 4.6.2 — Qualitative Analysis of Generated Text
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.6.2 Qualitative Analysis of Generated Text
TARGET LENGTH: 260–300 words.
CONTENT: Table 4.8 with the three reference/generated examples from fact sheet §5 (copy the Bengali text exactly); 96.8% of outputs contain URL fragments, 99.2% hashtags; outputs mention hurricane names regardless of the image; root cause: the references are translated tweets, so the decoder learned the style of tweets; implication: Bengali image description needs human-written descriptive references (future work, Section 5.6).
PLACEHOLDERS: [TABLE 4.8: Examples of reference tweets and generated Bengali text]
CITATIONS: Do not include any [REF].
```

#### PROMPT 4.7 — Human Evaluation
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.7 Human Evaluation
TARGET LENGTH: 390–450 words.
CONTENT (fact sheet §6): model vs reference — Fluency 1.58 vs 3.42, Adequacy 1.24 vs 3.08, Faithfulness 1.23 vs 3.23, all p < 10⁻¹⁶ (Wilcoxon); per class (AP lowest); reference preferred in 225 of 237 judgements (94.9%); agreement: pooled α = 0.38 (fair), within caption type ≈ 0 because model scores cluster at the bottom of the scale. Interpretation consistent with 4.6. Limitations: captions from the earlier v2 checkpoint, three annotators without a calibration session. Mention that a first analysis had ignored the A/B order and was corrected.
PLACEHOLDERS: [TABLE 4.9: Human evaluation scores (1–5) for generated and reference text]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 4.8.1 — Qualitative Analysis of Explanation Maps
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: Write "4.8 Explainability Analysis" with a one-sentence lead-in, then "4.8.1 Qualitative Analysis of Explanation Maps".
TARGET LENGTH: 370–430 words.
CONTENT: 21 random test images (7 per class, seed 42, not cherry-picked) — 15 correct (SD 6/7, HR 7/7, AP 2/7), errors have lower confidence (0.58 vs 0.89); never call 15/21 an accuracy of the explanation. Figures 4.5–4.8 show by a fixed rule the most confident correct prediction per class (#1 SD aerial flood, #10 HR relief boxes, #18 AP people in flood water) and the most confident error (#21, AP predicted as SD). Describe what each method highlights: Grad-CAM on flooded roads/vegetation (#1), relief boxes (#10), flood water rather than the people (#18) — a sign the model uses scene context; Rollout more diffuse; cross-attention spread across the image. All 21 samples in Appendix B.
PLACEHOLDERS:
[FIGURE 4.5: figure_4_4_gradcam_grouped.png — Grad-CAM maps of the CLS-Only model]
[FIGURE 4.6: figure_4_6_rollout_grouped.png — Attention Rollout maps of the CLS-Only model]
[FIGURE 4.7: figure_4_5_crossattn_grouped.png — Cross-attention of the Multi-Task v3 caption decoder]
[FIGURE 4.8: figure_4_7_integrated_xai_grouped.png — Side-by-side comparison of the three explanation methods]
(Each caption must also state: "Rows: most confident correct prediction per class and most confident error among 21 randomly drawn test images.")
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 4.8.2 — Faithfulness of the Explanations
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.8.2 Faithfulness of the Explanations
TARGET LENGTH: 280–320 words.
CONTENT (fact sheet §7.3): removing Grad-CAM's top patches lowers the predicted-class probability by 6.92 / 14.27 / 21.27 pp at 10 / 25 / 50%, against 0.34 / 1.57 / 4.16 pp for random patches and 0.85 / 3.26 / 7.35 pp for Rollout; accuracy after 50% removal 73.87% (Grad-CAM) vs 81.60% (Rollout) vs 86.93% (random), from 88.27%. Conclusion: Grad-CAM identifies the evidence the classifier uses; Rollout is weakly faithful.
PLACEHOLDERS:
[TABLE 4.10: Deletion test on the grouped test set (n = 375)]
[FIGURE 4.9: figure_4_9_faithfulness_deletion.png — Drop in predicted-class probability as top-ranked patches are removed]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 4.8.3 — Agreement Between Explanation Methods
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.8.3 Agreement Between Explanation Methods
TARGET LENGTH: 160–200 words.
CONTENT: top-10% patch IoU — Grad-CAM vs Rollout 0.148 (2.7× chance), Grad-CAM vs decoder cross-attention 0.059 (chance 0.054); so the caption decoder looks at different regions than the classifier uses; cross-attention is therefore reported only as an explanation of text generation.
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 4.9 — Confidence Calibration
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.9 Confidence Calibration
TARGET LENGTH: 280–320 words.
CONTENT (fact sheet §8): ECE 2.57% before scaling (mean confidence 88.84% vs accuracy 88.27%), MCE 19.06% from sparsely filled low-confidence bins; T* = 1.1349 gives ECE 2.99%, NLL and Brier essentially unchanged → the model is already well calibrated and temperature scaling is not needed; relate to the LoRA/frozen-backbone setting cautiously ("may"); practical meaning: confidence can be used to route uncertain predictions to human review.
PLACEHOLDERS:
[TABLE 4.11: Calibration metrics before and after temperature scaling]
[FIGURE 4.10: figure_4_8_calibration_diagram_grouped.png — Reliability diagrams before and after temperature scaling]
CITATIONS: Include 1–2 [REF] placeholders.
```

#### PROMPT 4.10 — External Validation on Bangladeshi Images
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.10 External Validation on Bangladeshi Images
TARGET LENGTH: 400–460 words.
CONTENT (fact sheet §9): n = 325, Fleiss' κ 0.613; Macro F1 81.98% (95% CI 76.1–87.2), accuracy 89.85%, per-class F1 SD 93.9 / HR 93.2 / AP 58.8, confusion matrix; sensitivity — against single annotators 81.65 / 81.36 / 55.82%, unanimous images 86.93%; interpretation: no performance drop observed, but (a) the estimate depends heavily on who labels the images, mainly on the HR vs AP boundary; (b) Wikimedia images are more professional than social-media photos; (c) the label source differs from CrisisMMD — so the difference from 74.83% is NOT a measure of domain shift; AP remains the weakest class. Wording: "preliminary external validation".
PLACEHOLDERS:
[TABLE 4.12: Results on the Bangladeshi external test set (n = 325)]
[TABLE 4.13: Macro F1 against each annotator's labels]
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 4.11 — Discussion
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.11 Discussion
TARGET LENGTH: 420–480 words.
CONTENT — connect the findings and relate them to the literature of Chapter 2:
1. Evaluation choices change conclusions: leakage (−7.5 pp), seed variance, the text ablation.
2. Parameter efficiency is the supported strength; multimodality and multi-task learning are not.
3. Why the text did not help, compared with multimodal CrisisMMD studies that report gains (translated tweets, small data, late fusion).
4. Trustworthiness: faithful Grad-CAM and good calibration support a human-in-the-loop use; scene-context reliance (#18) shows why explanations are needed.
5. Bengali generation needs descriptive references, not tweets.
6. Bangladesh test: encouraging but preliminary.
CITATIONS: Include 3–5 [REF] placeholders.
```

#### PROMPT 4.12 — Summary
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 4.12 Summary
TARGET LENGTH: 110–150 words, one paragraph with the key numbers.
CITATIONS: Do not include any [REF].
```

### ━━━━━━━━━━ CHAPTER 5: CONCLUSION ━━━━━━━━━━

#### PROMPT 5.1 — Introduction
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: Begin with "CHAPTER 5 CONCLUSION", then "5.1 Introduction".
TARGET LENGTH: 80–120 words.
CONTENT: purpose of the chapter and its sections.
CITATIONS: Do not include any [REF].
```

#### PROMPT 5.2 — Summary of Findings
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 5.2 Summary of Findings
TARGET LENGTH: 420–480 words.
CONTENT: one paragraph per objective of Section 1.3, each with the key result: dataset + grouped split (21.3% overlap removed); classifier 75.69 ± 1.33% with 1.38 M trainable parameters, tied with ViT-Large baseline; text ablation → image-driven; generation imitates tweets (human rating ≤ 1.58 / 5); Grad-CAM faithful, calibration ECE 2.57%; Bangladesh 81.98% (CI 76.1–87.2), annotator-dependent.
CITATIONS: Do not include any [REF].
```

#### PROMPT 5.3 — Answers to the Research Questions
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 5.3 Answers to the Research Questions
TARGET LENGTH: 280–320 words: RQ1–RQ5 from Section 1.4, each restated in short form and answered in 2–3 sentences with evidence.
CITATIONS: Do not include any [REF].
```

#### PROMPT 5.4 — Limitations of the Study
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 5.4 Limitations of the Study
TARGET LENGTH: 380–420 words.
CONTENT: the ten limitations of fact sheet §12, grouped into prose paragraphs (data and labels; experimental design; text generation and human evaluation; external validity). State each plainly with its consequence; do not minimize.
CITATIONS: Do not include any [REF].
```

#### PROMPT 5.5 — Recommendations
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 5.5 Recommendations
TARGET LENGTH: 180–220 words.
CONTENT: for researchers — use group-aware splits and several seeds on tweet data, test explanations quantitatively, include trivial baselines for generation metrics; for practitioners — use the classifier as a triage aid with Grad-CAM and confidence shown to a human reviewer, not as an autonomous decision system; for Bengali NLP — invest in human-written Bengali image descriptions of local disasters.
CITATIONS: Include 0–1 [REF] placeholders.
```

#### PROMPT 5.6 — Future Work
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule. Take every number about this study only from the fact sheet.

SECTION: 5.6 Future Work
TARGET LENGTH: 280–320 words.
CONTENT: (1) a larger Bangladeshi image set with several annotators and human-written Bengali descriptions, collected from local social media and news; (2) training the caption decoder on descriptive references and evaluating it with human judges; (3) re-running the ablation study on the grouped split with several seeds; (4) better handling of the Affected People class (class weighting, more data, clearer label definitions); (5) comparison with large vision-language models fine-tuned with LoRA; (6) a field study with responders on whether explanations and calibrated confidence help decisions.
CITATIONS: Include 1–3 [REF] placeholders.
```

#### PROMPT 5.7 — Concluding Remarks
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md" completely before writing and follow every rule.

SECTION: 5.7 Concluding Remarks
TARGET LENGTH: 130–170 words, one paragraph.
CONTENT: the main message — a small, parameter-efficient and well-calibrated model with faithful explanations is a sound starting point for Bengali disaster image analysis, and careful evaluation revealed what still needs to be built (descriptive Bengali references, local data). No new numbers beyond those already given in 5.2.
CITATIONS: Do not include any [REF].
```

### ━━━━━━━━━━ EXTRA PROMPTS ━━━━━━━━━━

#### PROMPT X.1 — List of Abbreviations (after all chapters)
```
Read "00_MASTER_WRITING_GUIDELINES.txt" and "02_RESULTS_FACT_SHEET.md".

TASK: Produce the LIST OF ABBREVIATIONS for this thesis as a two-column table (Abbreviation | Full form), sorted alphabetically. Include at least: AI, AMP, AP, BLEU, CE, chrF, CI, CLS, CNN, ECE, GPU, HR, IoU, LoRA, MCE, MLP, MT, NLL, NLP, PEFT, SD, ViT, XAI, and any other abbreviation used in the fact sheet. No prose.
```

#### PROMPT X.2 — Consistency check (after each chapter)
```
Read "02_RESULTS_FACT_SHEET.md". I will paste a chapter of the thesis below. List, as a table, every number or claim about this study in the text that differs from the fact sheet or appears in its "DO NOT CLAIM" list (§11), with the correct value. If everything matches, answer "No inconsistencies found."

[PASTE CHAPTER TEXT HERE]
```

---

## Part G — শেষ ধাপ (জমা দেওয়ার আগে)

1. **Front matter:** template-এর Declaration, Supervisor's Declaration, Student's Declaration পূরণ করো। Acknowledgements নিজের ভাষায় লেখো (Bangladesh test set আর human evaluation-এর annotator-দের নাম চাইলে এখানে দিতে পারো, তাঁদের অনুমতি নিয়ে)।
2. **Zotero:** সব paper Zotero-তে আনো → Word plugin-এ style "APA 7th" → প্রতিটা `[REF: …]` বদলে citation insert করো → শেষে Bibliography।
3. **Lists:** Word-এ References → Insert Table of Figures দিয়ে List of Figures / List of Tables বানাও। Table of Contents update করো।
4. **Word count:** Abstract + Chapter 1–5 ≈ 20,000 (ছবি, টেবিল, reference বাদে)।
5. **Turnitin:** DIU library-তে জমা দেওয়ার আগে নিজে একবার পড়ে বাক্যগুলো নিজের ভাষায় ঠিক করো। Similarity 40%-এর নিচে, কোনো একক source 3%-এর নিচে থাকতে হবে। Table, reference list আর equation থেকে কিছু মিল আসা স্বাভাবিক।
6. **AI ব্যবহার:** NotebookLM দিয়ে খসড়া লেখা হয়েছে। Department বা supervisor-এর AI-ব্যবহার নীতি থাকলে সেটা মেনে চলো (দরকার হলে acknowledgement-এ উল্লেখ করো)। চূড়ান্ত লেখার প্রতিটা দাবি তুমি defense-এ ব্যাখ্যা করতে পারো কি না, সেটা নিশ্চিত করো।
7. **Appendix:** A — human evaluation guideline, B — ২১টা XAI sample, C — Bangladesh test set-এর guideline আর image-এর উৎস/লাইসেন্স তালিকা।
