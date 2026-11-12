# DRISHTI-Bn Thesis — Claude Code Context

## Project Identity
- **Thesis:** "DRISHTI-Bn: An Explainable Multimodal Deep Learning Framework for Bengali Disaster Image Captioning and Classification"
- **Student:** Jahid Hasan | ID: 0242 | DIU SWE
- **Supervisor:** Md. Fahad Hossain

## Response Rules
- Explain in **Bengali**, code/technical content in **English**
- Be concise — minimize token usage
- Max 1–3 `[REF]` per paragraph (APA format, NOT after every sentence)
- Figure captions: **BELOW** images | Table captions: **ABOVE** tables

## Formatting (DIU Guidelines)
- Font: Times New Roman 12pt, 1.5 line spacing, 4cm left margin
- Writing tool: Google Docs (DIU Template)
- Citations: APA `(Author, Year)` — NOT IEEE `[1]` style

---

## Actual Experiment Results (NEVER change these numbers)

### Classification (Test set, n=376, `best_cls_only.pt`)
| Metric | CLS-Only (Best) | Multi-Task |
|--------|----------------|------------|
| Macro F1 | **83.21%** | 77.95% |
| Precision | 85.33% | 93.28% |
| Recall | 81.84% | 73.03% |

**Per-class (multi-task):** Severe_Damage=92.53%, Humanitarian_Rescue=88.00%, Affected_People=53.33%

**Confusion matrix:** SD [223,1,0] | HR [20,99,0] | AP [15,6,12]

### Caption Generation (Test set, n=376, Beam Width=3)
| Metric | Score |
|--------|-------|
| BLEU-1 | 24.85 |
| BLEU-2 | 14.20 |
| BLEU-3 | 8.95 |
| BLEU-4 | 6.13 |
| METEOR | 0.32 |
| ROUGE-L | 0.56 |
| BERTScore-F1 | **60.35%** ← (NOT 60.65%) |

### Baselines
| Model | Macro F1 |
|-------|----------|
| Stratified Random | 39.95% |
| BanglaBERT-Only | 43.02% |
| Full Fine-Tune | 74.19% |
| CrisisViT (SOTA, ViT-Large) | 78.15% |
| ViT-Only | 78.64% |
| **DRISHTI-Bn (Ours)** | **83.21%** |

### Ablation Study
| Config | Macro F1 |
|--------|----------|
| Caption-Only | 16.90% |
| No-LoRA | 77.74% |
| LoRA r=4 | 79.41% |
| LoRA r=16 | 79.14% |
| r=8 Multi-Task | 81.47% |
| No-CrossAttn | 81.88% |
| **CLS-Only (Best)** | **83.21%** |

### Calibration
- T* = 1.1113 | ECE before = 2.16%
- Mean confidence: 88.25% → 86.65%

### XAI (21 samples, 7/class)
- Overall: 17/21 = **81%** | SD: 7/7 | HR: 7/7 | AP: 3/7

### Human Evaluation (100 samples, 3 annotators, blind A/B)
| Dimension | Model | Ground Truth |
|-----------|-------|--------------|
| Fluency | 2.49/5 | 2.51/5 |
| Adequacy | 2.11/5 | 2.21/5 |
| Faithfulness | 2.17/5 | 2.30/5 |

**Per-class (model):** SD(2.46, 2.12, 2.16) | HR(2.49, 2.12, 2.13) | AP(2.71, 2.08, 2.33)

**IAA:** Mean κ ≈ −0.000 | Krippendorff's α ≈ −0.052

---

## Key Files
| File | Purpose |
|------|---------|
| `DEFENSE_FIX_PLAN.md` | **Active:** phase-by-phase pre-defense fix plan (update after each phase, then commit + push) |
| `PHASE4_WRITING_PLAN.md` | Writing plan & section prompts (writing starts after fix plan is done) |
| `Thesis Writing/` | DIU guidelines, SWE thesis template, defense slide templates |
| `DRISHTI_XAI_CHECKLIST.md` | XAI task checklist |
| `test_set_indices.csv` | Test set sample indices |
| `human_eval_outputs/human_eval_report.md` | Human eval full report |
| `human_eval_outputs/iaa_scores.csv` | IAA scores |

## Generated Figures (all in project root)
| File | Content |
|------|---------|
| `figure_3_1_architecture.png` | DRISHTI-Bn architecture |
| `figure_2_1_lora.png` | LoRA injection diagram |
| `figure_3_3_xai_pipeline.png` | XAI pipeline flowchart |
| `figure_4_1_confusion_matrix.png` | Confusion matrix (n=376) |
| `figure_4_2_baseline_comparison.png` | Baseline bar chart |
| `figure_4_3_ablation_study.png` | Ablation bar chart |
| `figure_4_4_gradcam_examples.png` | Grad-CAM (samples 003,012,016) |
| `figure_4_5_crossattn_examples.png` | Cross-Attention examples |
| `figure_4_6_rollout_examples.png` | Attention Rollout examples |
| `figure_4_7_integrated_xai.png` | Integrated XAI comparison |
| `figure_4_8_calibration_diagram.png` | Calibration reliability diagram |

---

## Model & Dataset Info
- **Model:** ViT-Base/16 + BanglaBERT + LoRA (r=8, α=16, 0.30% trainable params)
- **Dataset:** CrisisMMD-derived Bengali flood corpus, 3755 samples, 80/10/10 split
- **Hardware:** Single NVIDIA T4 GPU (16GB VRAM), Colab Pro
- **Google Drive folder:** `My Drive > DisasterNet` (src/ NOT uploaded to Drive)
- **Thesis draft:** old drafts (`DRISHTI_Bn_Thesis_Final_Working_Draft.docx`, `..._v1.txt`) were intentionally deleted (pre-results work). A fresh draft will be written in the DIU template after `DEFENSE_FIX_PLAN.md` is complete.
- **Env note:** local venv uses transformers 5.x, where ViT attention modules are `q_proj/v_proj` (v4: `query/value`). `src/model.py` auto-detects, so checkpoint keys stay compatible.

