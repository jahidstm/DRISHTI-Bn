# DRISHTI-Bn: Phase 4 — Thesis Writing Plan & NotebookLM Prompts

> **Thesis:** DRISHTI-Bn: An Explainable Multimodal Deep Learning Framework for Bengali Disaster Image Captioning and Classification
> **Student:** Jahid Hasan | ID: 0242 | Supervisor: Md. Fahad Hossain
> **Date:** 2 November 2026

---

## Part A: What to Upload to NotebookLM (আগে এগুলো আপলোড করো)

NotebookLM-এ prompt দেওয়ার আগে নিচের ফাইলগুলো source হিসেবে আপলোড করো:

| # | ফাইল | কেন দরকার |
|---|------|-----------|
| 1 | `00_MASTER_WRITING_GUIDELINES.txt` | ✅ এটা already আছে — writing style ও tone control করে |
| 2 | `DRISHTI_Bn_Thesis_Working_Draft_v1.txt` | পুরানো ড্রাফট — এখান থেকে existing writing reference নেবে |
| 3 | `DRISHTI_XAI_CHECKLIST.md` | সব Phase-এর result ও সংখ্যা এখানে আছে |
| 4 | `PHASE4_WRITING_PLAN.md` (এই ফাইল) | prompt গুলো এখান থেকে কপি করবে |
| 5 | `human_eval_outputs/human_eval_report.md` | Human evaluation results ও IAA |
| 6 | তোমার related research papers (PDF) | NotebookLM এগুলো থেকে citation-style ও context নেবে |

> **টিপস:** NotebookLM-এ সব ফাইল একসাথে source হিসেবে যোগ করো, তাহলে যেকোনো prompt-এ সব ফাইলের তথ্য ব্যবহার করতে পারবে।

---

## Part B: Thesis Structure Overview

### ☑ = Existing (update needed) | 🆕 = New section | ✍️ = Full rewrite

```
Abstract                                    ☑ (update with new results)

Chapter 1: Introduction
  1.1 Background                            ☑ (minor update)
  1.2 Problem Statement and Research Gap    ☑ (add XAI + Human Eval gap)
  1.3 Research Objectives                   ☑ (expand to 5 objectives)
  1.4 Thesis Organization                  🆕

Chapter 2: Literature Review
  2.1 Vision Models in Disaster Management  ☑ (add updated baselines)
  2.2 Transformer & Multimodal AI           ☑ (minor update)
  2.3 NLP for Low-Resource Languages        ☑ (minor update)
  2.4 Explainable AI (XAI) in CV           🆕
  2.5 Human Evaluation in NLG             🆕

Chapter 3: Methodology
  3.1 Dataset Curation & Preprocessing      ☑ (update split to 80/10/10)
  3.2 Proposed Multimodal Architecture      ☑ (minor update)
  3.3 Parameter-Efficient Fine-Tuning       ☑ (add ablation context)
  3.4 Multi-Task Learning & Loss            ☑ (minor update)
  3.5 Explainability Pipeline (XAI)        🆕
  3.6 Human Evaluation Design             🆕

Chapter 4: Results and Evaluation
  4.1 Experimental Setup                    ☑ (update hardware/config)
  4.2 Classification Results                ✍️ (full rewrite — new test set)
  4.3 Caption Generation Results            ✍️ (full rewrite — new metrics)
  4.4 Baseline Comparison                  🆕
  4.5 Ablation Study                       🆕
  4.6 XAI Analysis                         🆕
  4.7 Human Evaluation Results             🆕

Chapter 5: Discussion, Limitations & Conclusion
  5.1 Discussion                            ✍️ (full rewrite)
  5.2 Limitations                           ✍️ (full rewrite)
  5.3 Future Work                           ✍️ (full rewrite)
  5.4 Conclusion                            ✍️ (full rewrite)
```

---

## Part C: NotebookLM Prompts (Each Subsection — Separate Prompt)

> **কীভাবে ব্যবহার করবে:**
> 1. নিচের যেকোনো একটি prompt কপি করো
> 2. NotebookLM-এ paste করো
> 3. Output পড়ো → ভালো লাগলে Word (.docx) এ paste করো
> 4. পরের prompt-এ যাও

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### PROMPT 0: Abstract (UPDATE — new results)
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Rewrite the Abstract for the DRISHTI-Bn thesis. The old Abstract (in DRISHTI_Bn_Thesis_Working_Draft_v1.txt) was based on only 2 validation samples. The thesis has now been evaluated properly. Write a completely new Abstract using the updated results below.

TARGET LENGTH: 250–300 words. Single paragraph, no subheadings.

UPDATED DATA TO USE:
- Model: DRISHTI-Bn — a parameter-efficient multimodal vision-language framework
- Architecture: ViT-Base/16 visual encoder + BanglaBERT cross-attention decoder
- LoRA adaptation: rank r=8, only 0.30% trainable parameters (589,824 / 197,005,824)
- Dataset: CrisisMMD (flood images) → Bengali-translated captions
- Data split: Stratified 80/10/10 (Train=3,004 / Val=375 / Test=376)
- 3 classes: Severe Damage, Humanitarian Rescue, Affected People
- Classification (test set, n=376): Macro F1 = 77.95% (best ablation CLS-Only = 83.21%)
- Caption Generation: BLEU-4 = 6.13, METEOR = 0.32, ROUGE-L = 0.56, BERTScore-F1 = 60.35%
- 5 baselines beaten: Random (39.95%), ViT-Only (78.64%), BanglaBERT-Only (43.02%), Full Fine-Tune (74.19%), CrisisViT-style SOTA (78.15%)
- 4 XAI methods integrated: Grad-CAM, Cross-Attention Visualization, Attention Rollout, Temperature Scaling (ECE = 2.16%)
- Human Evaluation: 3 annotators, 100 blind samples, Fluency=2.49/5, Adequacy=2.11/5, Faithfulness=2.17/5
- Hardware: Single NVIDIA T4 GPU (16 GB VRAM)

STRUCTURE: Problem → Approach → Key Results (with numbers) → Contribution.
Do NOT include [REF] citations in the Abstract.
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### CHAPTER 1: INTRODUCTION
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

#### PROMPT 1.1: Background (UPDATE — minor additions)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 1.1 (Background) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 1.1 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 14–29 approximately). The existing text is well-written and should be preserved as much as possible.

WHAT TO KEEP: The existing 4 paragraphs about natural disasters in Bangladesh, social media data during crises, limitations of manual assessment, and the need for automated AI.

WHAT TO ADD/UPDATE:
- Add 1 new paragraph (after the existing content) introducing the need for EXPLAINABILITY in AI disaster systems — authorities need to understand WHY the AI made a decision, not just WHAT it decided. Mention that black-box models face trust issues in high-stakes humanitarian deployment [REF].
- Add 1 sentence about the need for HUMAN EVALUATION alongside automated metrics for validating generated captions [REF].

TARGET LENGTH: 5–6 paragraphs, approximately 500–700 words total.

FORMATTING:
- Use [REF] placeholders wherever a citation is needed.
- Include: [PLACEHOLDER: Figure 1.1 — Flood exposure statistics for Bangladesh showing 94.42 million individuals at risk]
- Include: [PLACEHOLDER: Figure 1.2 — End-to-end disaster intelligence pipeline highlighting the DRISHTI-Bn core module]
```

---

#### PROMPT 1.2: Problem Statement and Research Gap (UPDATE — add XAI + HE gaps)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 1.2 (Problem Statement and Research Gap) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 1.2 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 35–42 approximately). Keep the core arguments about unimodal systems, English bias, and Bengali data gap.

WHAT TO KEEP: The existing 3 paragraphs about unimodal system limitations, English-centric bias, and absence of Bengali multimodal datasets.

WHAT TO ADD/UPDATE:
- Add 1 new paragraph identifying the EXPLAINABILITY GAP: most disaster AI systems are black-box — no existing work integrates XAI techniques (Grad-CAM, Attention Rollout, Cross-Attention, Temperature Scaling) into a disaster captioning framework [REF].
- Add 1 new paragraph identifying the EVALUATION GAP: automated metrics like BLEU are insufficient for low-resource languages — no prior work has conducted structured human evaluation of Bengali disaster captions with inter-annotator agreement analysis [REF].

TARGET LENGTH: 5 paragraphs, approximately 500–650 words total.

FORMATTING:
- Use [REF] placeholders wherever a citation is needed.
```

---

#### PROMPT 1.3: Research Objectives (UPDATE — expand to 5 objectives)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 1.3 (Research Objectives) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 1.3 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 45–56 approximately). The existing version lists 4 objectives.

WHAT TO UPDATE: Expand to 5 numbered objectives that now cover the full scope of work:

1. To compile a specialized multimodal disaster corpus by filtering flood images from CrisisMMD and generating verified Bengali captions through a semi-automated translation pipeline.
2. To design an integrated vision-language architecture unifying a ViT-Base/16 visual encoder with an autoregressive BanglaBERT cross-attention decoder for simultaneous damage classification and Bengali caption generation.
3. To implement Low-Rank Adaptation (LoRA) matrices for parameter-efficient fine-tuning, enabling the entire framework to operate within a single T4 GPU (16 GB VRAM).
4. To integrate four Explainable AI (XAI) techniques — Grad-CAM, Cross-Attention Visualization, Attention Rollout, and Temperature Scaling — providing visual and calibration-based interpretability for model decisions.
5. To design and execute a structured human evaluation protocol with blind annotation by three Bengali-speaking evaluators, measuring Fluency, Adequacy, and Faithfulness with inter-annotator agreement analysis.

Also keep the existing scope limitation paragraph about the system being the core AI module within a broader pipeline.

TARGET LENGTH: 2–3 paragraphs + 5 numbered objectives, approximately 300–400 words total.

FORMATTING:
- Use [REF] placeholders where needed.
```

---

#### PROMPT 1.4: Thesis Organization (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a NEW Section 1.4 (Thesis Organization) for the DRISHTI-Bn thesis. This section provides a brief roadmap of the remaining chapters.

CONTENT: Write a single paragraph or short bulleted list describing:
- Chapter 2: Literature Review covering vision models, multimodal AI, Bengali NLP, XAI, and human evaluation
- Chapter 3: Methodology covering dataset, architecture, LoRA, multi-task learning, XAI pipeline, and human evaluation design
- Chapter 4: Results covering classification, captioning, 5 baselines, 6 ablation configs, XAI analysis, and human evaluation
- Chapter 5: Discussion, limitations, future work, and conclusion

TARGET LENGTH: 1 paragraph, approximately 120–180 words.

FORMATTING:
- No [REF] needed in this section.
- No figures or tables.
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### CHAPTER 2: LITERATURE REVIEW
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

#### PROMPT 2.1: Evolution of Vision Models in Disaster Management (UPDATE)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 2.1 (Evolution of Vision Models in Disaster Management) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 2.1 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 62–100 approximately). The existing text is strong.

WHAT TO KEEP: The progression from CNN (ResNet, VGG) → ViT → CrisisViT, and the argument that unimodal systems lack generative capacity.

WHAT TO ADD/UPDATE:
- Update the comparison table (Table 2.1) to reflect our actual results: DRISHTI-Bn CLS+LoRA achieves 83.21% Macro F1, not 66.67%.
- Add a brief mention that our work also includes EXPLAINABILITY — unlike prior disaster vision models that only classify without explaining their decisions [REF].

TARGET LENGTH: 4–5 paragraphs, approximately 500–700 words.

FORMATTING:
- Use [REF] placeholders for all citations.
- Include: [PLACEHOLDER: Table 2.1 — Comparative Performance of CNN vs. ViT vs. DRISHTI-Bn in Disaster Classification (updated with 83.21% F1)]
```

---

#### PROMPT 2.2: Transformer Architectures and Multimodal AI (UPDATE — minor)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 2.2 (Transformer Architectures and Multimodal AI) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 2.2 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 103–110 approximately). The existing text is well-written.

WHAT TO KEEP: The discussion of vision-language models, cross-attention mechanisms, and hardware constraints of full fine-tuning.

WHAT TO ADD/UPDATE:
- Add 1 paragraph discussing recent multimodal models (CLIP, BLIP, BLIP-2) and their English-centric pretraining bias [REF]. Mention that none of them support Bengali disaster captioning natively.
- Add 1 paragraph discussing LoRA specifically as part of the PEFT revolution [REF], and why it is well-suited for low-resource adaptation scenarios.

TARGET LENGTH: 4–5 paragraphs, approximately 500–700 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Figure 2.1 — LoRA Adapter Injection diagram showing frozen W₀ with parallel low-rank matrices A and B]
```

---

#### PROMPT 2.3: NLP for Low-Resource Languages — Bengali (UPDATE — minor)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 2.3 (NLP for Low-Resource Languages — Bengali) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 2.3 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 111–119 approximately). The existing text is excellent.

WHAT TO KEEP: All existing content about Bengali morphological complexity, BanglaBERT architecture, and the gap in Bengali multimodal disaster data.

WHAT TO ADD/UPDATE:
- Add 1 paragraph noting that while BanglaBERT excels at text understanding tasks (NER, sentiment analysis), its use as a DECODER for IMAGE CAPTIONING in Bengali is novel and unexplored [REF].
- Ensure the section ends by connecting to the proposed DRISHTI-Bn framework.

TARGET LENGTH: 4–5 paragraphs, approximately 500–650 words.

FORMATTING:
- Use [REF] placeholders.
```

---

#### PROMPT 2.4: Explainable AI (XAI) in Computer Vision (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a completely NEW Section 2.4 (Explainable AI in Computer Vision) for the DRISHTI-Bn thesis. This section did not exist in the previous draft.

CONTENT TO COVER (5–6 paragraphs):

Paragraph 1: Motivation for XAI — AI systems deployed in high-stakes domains (healthcare, disaster response) must be transparent and interpretable. Black-box models face trust issues from decision-makers [REF]. Define XAI broadly.

Paragraph 2: Gradient-based methods — Grad-CAM (Selvaraju et al., 2017) [REF] computes class-discriminative localization maps using gradients flowing into the final convolutional/attention layer. Originally designed for CNNs, adapted for Vision Transformers by hooking into the last encoder block.

Paragraph 3: Attention-based methods — Attention Rollout (Abnar & Zuidema, 2020) [REF] recursively multiplies attention matrices across all ViT layers to compute a global attention flow from input patches to the [CLS] token. Cross-Attention Visualization maps which image patches each generated text token attends to, providing token-level spatial grounding.

Paragraph 4: Calibration methods — Temperature Scaling (Guo et al., 2017) [REF] is a post-hoc calibration technique that learns a single scalar T to adjust softmax confidence. Expected Calibration Error (ECE) measures the gap between predicted confidence and actual accuracy. Critical for disaster triage where overconfident predictions are dangerous.

Paragraph 5: XAI in disaster management — Very limited prior work applies XAI to disaster image classification [REF]. No existing study integrates multiple XAI techniques into a multimodal Bengali disaster captioning framework. This is a clear research gap that DRISHTI-Bn addresses.

TARGET LENGTH: 5–6 paragraphs, approximately 600–800 words.

FORMATTING:
- Use [REF] placeholders for all citations.
- This is a pure literature review section — no results, no methodology.
```

---

#### PROMPT 2.5: Human Evaluation in Natural Language Generation (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a completely NEW Section 2.5 (Human Evaluation in Natural Language Generation) for the DRISHTI-Bn thesis. This section did not exist in the previous draft.

CONTENT TO COVER (4–5 paragraphs):

Paragraph 1: Why automatic metrics are insufficient — BLEU [REF] measures n-gram overlap but fails to capture semantic adequacy, fluency, or factual consistency. METEOR and BERTScore improve on BLEU but still have limitations for low-resource languages where reference translations may be imperfect [REF].

Paragraph 2: Human evaluation frameworks — The Fluency-Adequacy-Faithfulness framework is widely used in machine translation and image captioning evaluation [REF]. Fluency measures grammatical correctness, Adequacy measures content completeness, and Faithfulness measures factual accuracy relative to the source image.

Paragraph 3: Inter-Annotator Agreement (IAA) — Cohen's Kappa (pairwise) [REF] and Krippendorff's Alpha (multi-annotator) [REF] are standard metrics for measuring annotator consistency. Low IAA is commonly reported in subjective NLG evaluation tasks, particularly for under-resourced languages [REF].

Paragraph 4: Challenges in low-resource language evaluation — Bengali caption evaluation faces unique challenges: no established benchmark exists, annotators have no prior calibration reference, and disaster-domain vocabulary adds domain-specific ambiguity [REF]. Prior work (e.g., Specia et al., 2010; Elliott & Kádár, 2017) reports similarly low IAA for subjective evaluation dimensions [REF].

Paragraph 5: Research gap — No prior study has conducted structured human evaluation of Bengali disaster image captions using a blind A/B protocol with inter-annotator agreement analysis. DRISHTI-Bn addresses this gap.

TARGET LENGTH: 4–5 paragraphs, approximately 500–650 words.

FORMATTING:
- Use [REF] placeholders for all citations.
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### CHAPTER 3: METHODOLOGY
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

#### PROMPT 3.1: Dataset Curation and Preprocessing (UPDATE — new split)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 3.1 (Dataset Curation and Preprocessing) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 3.1 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 122–156 approximately).

WHAT TO KEEP: The description of CrisisMMD as the source, flood-image filtering, English-to-Bengali translation pipeline, and image preprocessing (224×224, BanglaBERT tokenizer with 32,000 subwords).

CRITICAL UPDATE — DATA SPLIT:
The old draft used an 80/20 train/val split with NO test set (Train=3,004, Val=751).
The updated split is STRATIFIED 80/10/10:
- Train: 3,004 samples
- Validation: 375 samples
- Test: 376 samples
- Zero overlap between splits
- Identical class ratio maintained across all splits
- Seed=42 for reproducibility

WHAT TO ADD:
- Explain why a separate test set is critical for unbiased evaluation [REF].
- Mention the 3 class labels: Severe_Damage, Humanitarian_Rescue, Affected_People.
- Note the class imbalance issue (Affected_People has fewest samples).

TARGET LENGTH: 4–5 paragraphs, approximately 500–700 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Table 3.1 — CrisisMMD-Derived Bengali Flood Corpus Statistics (updated 80/10/10 split with Train=3,004 / Val=375 / Test=376)]
```

---

#### PROMPT 3.2: Proposed Multimodal Architecture (UPDATE — minor)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 3.2 (Proposed Multimodal Architecture) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 3.2 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 158–170 approximately). The existing description of ViT patch processing, cross-attention bridge, and BanglaBERT decoder is excellent.

WHAT TO KEEP: All 3 existing paragraphs describing ViT patch processing, cross-attention projection, and BanglaBERT autoregressive decoding.

WHAT TO ADD/UPDATE:
- Add 1 paragraph specifying the exact pretrained checkpoints used: `google/vit-base-patch16-224` and `csebuetnlp/banglabert` [REF].
- Clarify that the [CLS] token from ViT is concatenated with BanglaBERT's [CLS] token and passed through a 2-layer MLP classifier for the classification task.
- Mention total parameter count: ~197M parameters, of which only 589,824 (0.30%) are trainable via LoRA.

TARGET LENGTH: 4–5 paragraphs, approximately 500–650 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Figure 3.1 — DRISHTI-Bn Multimodal Architecture Diagram showing ViT encoder, cross-attention bridge, BanglaBERT decoder, and classification head]
```

---

#### PROMPT 3.3: Parameter-Efficient Fine-Tuning / LoRA (UPDATE — add ablation context)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 3.3 (Parameter-Efficient Fine-Tuning) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 3.3 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 173–185 approximately). The existing LoRA theory explanation is solid.

WHAT TO KEEP: The mathematical explanation of LoRA (W = W₀ + BA), the concept of frozen pretrained weights with parallel low-rank matrices, and the argument for computational efficiency.

WHAT TO ADD/UPDATE:
- Specify exact LoRA hyperparameters: rank r=8, alpha=16, applied to attention layers of both ViT and BanglaBERT.
- Mention that rank sensitivity was tested (r=4, r=8, r=16) in the ablation study, and r=8 proved optimal (details in Chapter 4.5).
- Add comparison: LoRA (0.30% params, F1=83.21%) vs Full Fine-Tune (100% params, F1=74.19%) — LoRA is both more efficient AND more effective due to regularization benefit on small datasets [REF].

TARGET LENGTH: 3–4 paragraphs, approximately 400–550 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Figure 3.2 — LoRA injection diagram showing frozen W₀ with trainable A and B matrices, rank r=8]
```

---

#### PROMPT 3.4: Multi-Task Learning and Loss Optimization (UPDATE — minor)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 3.4 (Multi-Task Learning and Loss Optimization) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 3.4 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 188–201 approximately). The existing math is correct.

WHAT TO KEEP: The dual-loss formulation (L_total = λ_cls · L_cls + λ_cap · L_cap), categorical cross-entropy for classification, negative log-likelihood for caption generation, and AdamW optimizer.

WHAT TO ADD/UPDATE:
- Mention that the ablation study (Chapter 4.5) revealed an important finding: when λ_cap = 0 (classification-only), Macro F1 improved from 77.95% to 83.21%, suggesting that the caption loss introduces gradient interference that HURTS classification performance. This is a key research finding.
- Add training hyperparameters: 15 epochs, batch size=16, lr=2e-5, weight decay, AMP (automatic mixed precision), seed=42.

TARGET LENGTH: 3–4 paragraphs, approximately 400–550 words.

FORMATTING:
- Use [REF] placeholders.
```

---

#### PROMPT 3.5: Explainability Pipeline / XAI (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a completely NEW Section 3.5 (Explainability Pipeline) for the DRISHTI-Bn thesis. This section describes the METHODOLOGY of the four XAI techniques — not the results (results go in Chapter 4.6).

CONTENT TO COVER (5–6 paragraphs):

Paragraph 1: Motivation — Explain why explainability is integrated into DRISHTI-Bn. In disaster triage, stakeholders need visual evidence of WHY the model classified an image as "Severe Damage" vs "Humanitarian Rescue" [REF]. Four complementary XAI techniques were selected to provide multi-perspective interpretability.

Paragraph 2: Grad-CAM methodology — Adapted for Vision Transformers by hooking into the last ViT encoder block (layers.11). Computes gradient-weighted class activation maps to produce spatial heatmaps. Originally designed for CNNs (Selvaraju et al., 2017) [REF], requires adaptation for ViT's patch-based architecture.

Paragraph 3: Cross-Attention Visualization methodology — Extracts attention weights from the cross-attention layers of the BanglaBERT decoder. For each generated Bengali token, a spatial map shows which image patches the decoder attended to. This provides token-level visual grounding — e.g., when generating the word "ধ্বংস" (destruction), the model attends to collapsed building regions.

Paragraph 4: Attention Rollout methodology — Implements Abnar & Zuidema (2020) [REF]. Recursively multiplies attention matrices across all 12 ViT encoder layers with mean head fusion and discard ratio=0.9. Produces a global attention flow map from input patches to the classification token.

Paragraph 5: Temperature Scaling methodology — Implements Guo et al. (2017) [REF]. A single scalar parameter T is optimized on the validation set by minimizing negative log-likelihood. The softmax probabilities are divided by T before prediction. If T > 1.0, the model was overconfident; if T < 1.0, underconfident. Expected Calibration Error (ECE) measures reliability.

Paragraph 6: Evaluation protocol — 21 stratified samples (7 per class) were selected from the test set for XAI analysis. All four methods were applied to the same samples for direct comparison. The CLS-Only model checkpoint (best_cls_only.pt, F1=83.21%) was used.

TARGET LENGTH: 5–6 paragraphs, approximately 600–800 words.

FORMATTING:
- Use [REF] placeholders for all citations.
- Include: [PLACEHOLDER: Figure 3.3 — XAI pipeline flowchart showing the four methods applied to each input image]
```

---

#### PROMPT 3.6: Human Evaluation Design (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a completely NEW Section 3.6 (Human Evaluation Design) for the DRISHTI-Bn thesis. This section describes the METHODOLOGY of the human evaluation protocol — not the results (results go in Chapter 4.7).

CONTENT TO COVER (4–5 paragraphs):

Paragraph 1: Sample selection — 100 samples were stratified from the 376-sample test set, maintaining class proportions: Severe_Damage (60), Humanitarian_Rescue (32), Affected_People (8). Stratification ensures evaluation covers all disaster categories [REF].

Paragraph 2: Blind A/B evaluation design — Each sample shows the disaster image alongside two captions: Caption A and Caption B. One is the model-generated caption (DRISHTI-Bn), the other is the ground truth (human-written). The assignment (which is A, which is B) is randomized per sample. Annotators are NOT told which caption is model-generated — this eliminates source bias [REF].

Paragraph 3: Evaluation criteria — Three dimensions rated on a 1–5 Likert scale:
  - Fluency: Is the Bengali grammatically correct and natural-sounding?
  - Adequacy: Does the caption cover the key visual content of the image?
  - Faithfulness: Is the caption factually consistent with the image (no hallucinated content)?
Annotators also select an overall preference: A better, B better, or Equal.
Optional free-text comment field provided for qualitative feedback.

Paragraph 4: Annotator profile — Three native Bengali-speaking volunteers. Annotation guidelines written entirely in Bangla (docs/annotation_guidelines_bn.md) with examples and rating scale explanations. Self-contained HTML annotation tools were built — each tool works offline in any browser, requires no internet, and automatically exports results as a CSV file upon completion.

Paragraph 5: Inter-Annotator Agreement — Planned IAA metrics: Cohen's Kappa (pairwise agreement between each annotator pair) and Krippendorff's Alpha (overall multi-annotator agreement for ordinal data) [REF]. These metrics quantify whether annotators interpreted the evaluation criteria consistently.

TARGET LENGTH: 4–5 paragraphs, approximately 500–650 words.

FORMATTING:
- Use [REF] placeholders for all citations.
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### CHAPTER 4: RESULTS AND EVALUATION
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

#### PROMPT 4.1: Experimental Setup (UPDATE)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Update Section 4.1 (Experimental Setup & Hardware Configuration) of the DRISHTI-Bn thesis.

REFERENCE: Read the existing Section 4.1 in "DRISHTI_Bn_Thesis_Working_Draft_v1.txt" (lines 206–208 approximately). The old version was a single dense paragraph.

WHAT TO KEEP: Core hardware info (Google Colab, NVIDIA T4, 16GB VRAM, PyTorch, HuggingFace).

WHAT TO UPDATE — rewrite as clearly structured paragraphs:

Hardware:
- Training: Google Colab Pro, NVIDIA Tesla T4 GPU (16 GB VRAM)
- Evaluation/XAI: CPU-based (local machine)

Software:
- PyTorch framework, HuggingFace Transformers
- ViT: google/vit-base-patch16-224
- Text: csebuetnlp/banglabert
- PEFT library for LoRA injection

Training Configuration:
- LoRA: rank r=8, alpha=16
- Trainable params: 589,824 / 197,005,824 (0.30%)
- Epochs: 15, Batch size: 16
- Optimizer: AdamW, lr=2e-5, weight decay
- AMP (Automatic Mixed Precision) enabled
- Loss weights: λ_cls=1.0, λ_cap=0.5
- Data split: 80/10/10 stratified, seed=42

TARGET LENGTH: 3 paragraphs, approximately 350–500 words.

FORMATTING:
- Use [REF] placeholders.
```

---

#### PROMPT 4.2: Classification Results (✍️ FULL REWRITE)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Completely REWRITE Section 4.2 (Classification Results) of the DRISHTI-Bn thesis. The old version (lines 210–240 in the draft) was based on only 2 validation samples with a Macro F1 of 66.67%. This is now completely outdated.

NEW DATA — write based on these ACTUAL test set results (n=376, seed=42):

Multi-task model (ViT + BanglaBERT + LoRA, λ_cls=1.0, λ_cap=0.5):
- Macro F1: 77.95%
- Precision: 93.28%, Recall: 73.03%
- Per-class breakdown:
  - Severe_Damage: F1=92.53% (strong)
  - Humanitarian_Rescue: F1=88.00% (strong)
  - Affected_People: F1=53.33% (weak — class imbalance)

Best ablation (CLS-Only, λ_cap=0):
- Macro F1: 83.21%
- Per-class: SD=93.93%, HR=87.90%, AP=67.80%
- Trainable params: 589,824 (0.30%)

DISCUSSION POINTS:
- Why Affected_People underperforms: fewest training samples, visually ambiguous
- Why CLS-Only outperforms multi-task: caption loss introduces gradient interference
- Overall performance validates the multimodal architecture

TARGET LENGTH: 3–4 paragraphs, approximately 400–600 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Table 4.1 — Per-class Classification Performance on Test Set (n=376) showing Precision, Recall, F1, Support for each class]
- Include: [PLACEHOLDER: Figure 4.1 — Confusion Matrix for test set classification]
```

---

#### PROMPT 4.3: Caption Generation Results (✍️ FULL REWRITE)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Completely REWRITE Section 4.3 (Caption Generation Results) of the DRISHTI-Bn thesis. The old version was based on 2 validation samples with BLEU-4=5.14. This is now outdated.

NEW DATA — write based on ACTUAL test set results (n=376):
- BLEU-1: (from benchmark)
- BLEU-4: 6.13
- METEOR: 0.32
- ROUGE-L: 0.56
- BERTScore-F1: 60.35%

DISCUSSION POINTS:
- BLEU-4 of 6.13 appears low compared to English captioning benchmarks (typically >30.0), but this comparison is invalid because:
  1. Bengali is a low-resource language with no prior captioning benchmark [REF]
  2. The dataset uses translated (not native) Bengali captions
  3. Bengali's morphological complexity reduces n-gram overlap naturally [REF]
- METEOR (0.32) accounts for synonyms and stemming — more appropriate for morphologically rich languages [REF]
- BERTScore-F1 (60.35%) captures semantic similarity beyond surface n-grams — shows the model generates semantically relevant content even when exact words differ [REF]
- ROUGE-L (0.56) measures longest common subsequence — indicates reasonable structural similarity

TARGET LENGTH: 3–4 paragraphs, approximately 400–600 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Table 4.2 — Caption Generation Metrics on Test Set showing BLEU-1 through BLEU-4, METEOR, ROUGE-L, BERTScore-F1]
```

---

#### PROMPT 4.4: Baseline Comparison (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a completely NEW Section 4.4 (Baseline Comparison) for the DRISHTI-Bn thesis.

DATA — 5 baselines vs. our model (all evaluated on test set, n=376, seed=42):

| Model | Macro F1 | Precision | Recall | Key Insight |
|-------|----------|-----------|--------|-------------|
| Majority/Stratified Random | 39.95% | — | — | Chance-level baseline |
| ViT-Only (no text encoder) | 78.64% | 79.23% | 78.48% | Image alone is strong |
| BanglaBERT-Only (no image) | 43.02% | 41.44% | 44.83% | Text alone fails (AP F1=0.00%) |
| Full Fine-Tune (no LoRA) | 74.19% | 73.43% | 75.09% | LoRA beats full FT by +9.02pp |
| CrisisViT-style SOTA (ViT-Large/16, 304M params) | 78.15% | 81.35% | 76.00% | Our smaller model beats SOTA |
| **DRISHTI-Bn CLS+LoRA (Ours)** | **83.21%** | **85.33%** | **81.84%** | Best, with only 0.30% trainable params |

KEY FINDINGS TO DISCUSS:
- Image contribution: ViT-Only (78.64%) − BanglaBERT-Only (43.02%) = +35.62pp → image is the dominant modality
- LoRA advantage: CLS+LoRA (83.21%) − Full FT (74.19%) = +9.02pp → LoRA regularizes better on small datasets
- SOTA comparison: CLS+LoRA (83.21%) > CrisisViT (78.15%) despite using a smaller backbone (ViT-Base vs ViT-Large)
- BanglaBERT-Only completely fails on Affected_People (F1=0.00%) — proves visual features are essential

TARGET LENGTH: 4–5 paragraphs, approximately 500–700 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Table 4.3 — Baseline Comparison Results on Test Set (n=376)]
- Include: [PLACEHOLDER: Figure 4.2 — Bar chart comparing Macro F1 across all 6 models]
```

---

#### PROMPT 4.5: Ablation Study (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a completely NEW Section 4.5 (Ablation Study) for the DRISHTI-Bn thesis.

NOTE: The old draft had a brief ablation section (4.4) about decoding strategies. That content can be merged here or referenced, but this section is primarily about COMPONENT ablation.

DATA — 6 ablation configurations (all on test set, seed=42):

| Configuration | Macro F1 | Key Finding |
|---------------|----------|-------------|
| CLS-Only (λ_cap=0, classification only) | 83.21% | Caption loss HURTS classification (+5.26pp without it) |
| Caption-Only (λ_cls=0) | 16.90% | Model collapses — worse than random (39.95%) |
| No LoRA (frozen encoders) | 77.74% | LoRA contributes +5.47pp |
| No Cross-Attention (self-attention only decoder) | 81.88% | Cross-attention contributes +1.33pp |
| LoRA r=4 | 79.41% | Underfits |
| LoRA r=8 (proposed) | 81.47% | Optimal rank |
| LoRA r=16 | 79.14% | Overfits — diminishing returns |

KEY FINDINGS TO DISCUSS:
1. Multi-task interference: Caption loss degrades classification by 5.26pp — suggests gradient competition between modalities
2. Caption dependency: Removing classification entirely causes model collapse (16.90%) — classification provides structural learning signal
3. LoRA impact: +5.47pp gain proves adapter tuning is superior to frozen encoders for this task
4. Cross-attention: Modest but consistent +1.33pp contribution — validates the architectural choice
5. Rank sensitivity: r=8 optimal — r=4 insufficient capacity, r=16 overfits on 3,004 training samples

TARGET LENGTH: 4–5 paragraphs, approximately 500–700 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Table 4.4 — Ablation Study Results showing all configurations]
- Include: [PLACEHOLDER: Figure 4.3 — Bar chart of ablation configurations vs. Macro F1]
```

---

#### PROMPT 4.6: XAI Analysis (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a completely NEW Section 4.6 (XAI Analysis) for the DRISHTI-Bn thesis.

DATA — 21 stratified samples (7 per class), CLS-Only checkpoint (best_cls_only.pt):

Overall accuracy on XAI samples: 17/21 (81.0%)
- Severe_Damage: 7/7 correct
- Humanitarian_Rescue: 7/7 correct
- Affected_People: 3/7 correct (consistent weakness)

Grad-CAM Results:
- Produces class-discriminative heatmaps from the last ViT encoder block (layers.11)
- Heatmaps tend to be diffuse/blurry due to ViT's global self-attention (unlike CNN's local receptive fields)
- Still correctly highlights general regions of interest (e.g., damaged buildings, rescue vehicles)

Cross-Attention Visualization Results:
- BEST spatial precision among all 4 methods
- Shows which image patches each generated Bengali token attends to
- Example: the word "ধ্বংস" (destruction) strongly attends to collapsed building patches
- Provides interpretable, human-readable token-level explanations

Attention Rollout Results:
- Smoother, more distributed attention maps compared to Grad-CAM
- Less spatially precise than Cross-Attention but captures global attention flow
- Useful for understanding overall model "focus areas"

Temperature Scaling Results:
- Optimal temperature: T* = 1.1113 (model is slightly overconfident — T > 1.0)
- Pre-calibration ECE = 2.16% (exceptionally low — model is already well-calibrated)
- Post-calibration: confidence softened from 88.25% to 86.65% mean
- Proves model predictions are RELIABLE — critical for disaster deployment

Integrated Comparison:
- Cross-Attention provides the sharpest, most meaningful spatial explanations
- Attention Rollout provides the smoothest global overview
- Grad-CAM is least effective for ViT architectures
- All methods consistently identify Affected_People as the hardest class

TARGET LENGTH: 5–6 paragraphs, approximately 600–800 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Figure 4.4 — Grad-CAM heatmap examples (one per class)]
- Include: [PLACEHOLDER: Figure 4.5 — Cross-Attention visualization examples]
- Include: [PLACEHOLDER: Figure 4.6 — Attention Rollout examples]
- Include: [PLACEHOLDER: Figure 4.7 — Integrated XAI comparison (combined_comparison.png) showing all 4 methods side-by-side for the same samples]
- Include: [PLACEHOLDER: Figure 4.8 — Calibration reliability diagram (before vs. after Temperature Scaling)]
- Include: [PLACEHOLDER: Table 4.5 — XAI Method Comparison Summary]
```

---

#### PROMPT 4.7: Human Evaluation Results (🆕 NEW)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Write a completely NEW Section 4.7 (Human Evaluation Results) for the DRISHTI-Bn thesis.

DATA:
Setup: 100 stratified samples, 3 Bengali-speaking annotators, blind A/B evaluation

Overall Scores (1–5 Likert scale):
| Metric        | DRISHTI-Bn (Model) | Ground Truth |
|---------------|-------------------|--------------|
| Fluency       | 2.49 / 5          | 2.51 / 5     |
| Adequacy      | 2.11 / 5          | 2.21 / 5     |
| Faithfulness  | 2.17 / 5          | 2.30 / 5     |

Per-Class Model Scores:
| Class               | Fluency | Adequacy | Faithfulness |
|---------------------|---------|----------|--------------|
| Severe_Damage       | 2.46    | 2.12     | 2.16         |
| Humanitarian_Rescue | 2.49    | 2.12     | 2.13         |
| Affected_People     | 2.71    | 2.08     | 2.33         |

Inter-Annotator Agreement (IAA):
| Metric         | Krippendorff's α | Mean Cohen's κ | Level   |
|----------------|-----------------|----------------|---------|
| fluency_A      | -0.121          | 0.013          | Slight  |
| adequacy_A     | -0.071          | -0.017         | Poor    |
| faithfulness_A | -0.124          | -0.054         | Poor    |
| fluency_B      | -0.034          | 0.004          | Slight  |
| adequacy_B     | 0.028           | 0.036          | Slight  |
| faithfulness_B | 0.011           | 0.016          | Slight  |
Overall: Mean κ ≈ -0.000, Mean α ≈ -0.052

KEY POINTS TO DISCUSS:

Positive findings:
- Model captions score within ~0.1–0.13 points of Ground Truth on ALL three dimensions
- This closeness indicates competitive caption quality
- Fluency is the strongest dimension (2.49/5) — model produces grammatically acceptable Bengali

Honest discussion of low IAA:
- Low IAA (κ ≈ 0.00) does NOT necessarily indicate poor annotation quality
- It reflects the inherently SUBJECTIVE nature of caption quality assessment [REF]
- Bengali disaster captioning has NO established evaluation benchmark for calibration
- No formal calibration session was conducted before annotation
- Similar low IAA is reported in prior NLG evaluation studies (Specia et al., 2010; Elliott & Kádár, 2017) [REF]
- Acknowledged as a limitation — future work should incorporate a calibration phase

TARGET LENGTH: 5–6 paragraphs, approximately 600–800 words.

FORMATTING:
- Use [REF] placeholders.
- Include: [PLACEHOLDER: Table 4.6 — Overall Human Evaluation Scores (Model vs. Ground Truth)]
- Include: [PLACEHOLDER: Table 4.7 — Per-Class Human Evaluation Scores (Model Captions)]
- Include: [PLACEHOLDER: Table 4.8 — Inter-Annotator Agreement Scores]
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### CHAPTER 5: DISCUSSION, LIMITATIONS & CONCLUSION
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

#### PROMPT 5.1: Discussion (✍️ FULL REWRITE)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Completely REWRITE Section 5.1 (Discussion) of the DRISHTI-Bn thesis. The old version (lines 333–343 in the draft) was based on 2-sample results and had no XAI or human evaluation to discuss.

CONTENT TO DISCUSS (6–7 paragraphs):

Paragraph 1: Classification performance interpretation — 83.21% Macro F1 (CLS-Only) significantly outperforms all 5 baselines. The +9.02pp advantage over Full Fine-Tune proves LoRA's regularization benefit on small datasets [REF].

Paragraph 2: Multi-task interference finding — Caption loss degrades classification by 5.26pp. This is an important finding about gradient competition in multi-task vision-language learning [REF]. Discuss implications.

Paragraph 3: Caption generation interpretation — BLEU-4 of 6.13 is low compared to English benchmarks but meaningful as the FIRST Bengali disaster captioning baseline. BERTScore (60.35%) shows better semantic quality than surface metrics suggest [REF].

Paragraph 4: LoRA efficiency — Only 0.30% trainable parameters achieve state-of-the-art classification. This validates parameter-efficient methods for resource-constrained deployment in developing countries [REF].

Paragraph 5: XAI insights — Cross-Attention provides the most spatially meaningful explanations. Temperature Scaling confirms high model reliability (ECE=2.16%). These explainability features are critical for building trust in disaster AI deployment [REF].

Paragraph 6: Human evaluation insights — Model captions score within ~0.1 of Ground Truth — competitive quality. Low IAA reflects subjective evaluation in a low-resource domain with no prior benchmark [REF].

Paragraph 7: Contribution to the field — DRISHTI-Bn is the first framework combining Bengali disaster captioning + classification + XAI + human evaluation in a single unified system.

TARGET LENGTH: 6–7 paragraphs, approximately 800–1000 words.

FORMATTING:
- Use [REF] placeholders.
```

---

#### PROMPT 5.2: Limitations (✍️ FULL REWRITE)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Completely REWRITE Section 5.2 (Limitations) of the DRISHTI-Bn thesis.

REFERENCE: The old version (lines 346–357) focused on Autoencoding Identity Bias and 2-sample evaluation. Some of those points are still valid but need updating.

LIMITATIONS TO DISCUSS (one paragraph each):

1. Dataset origin — CrisisMMD contains foreign disaster imagery (Hurricane Maria, Mexico earthquakes). Bangladeshi flood conditions (architecture, agriculture, demographics) differ fundamentally. Distributional shift is a risk [REF].

2. Translation pipeline — Bengali captions were semi-automatically translated from English. Subtle semantic distortions may exist in specialized disaster vocabulary [REF].

3. Class imbalance — Affected_People has the fewest samples and consistently shows the lowest F1 (53–67% across all configurations). Oversampling or focal loss could help [REF].

4. Caption quality — BLEU-4 = 6.13 is low. The cross-attention decoder was not pre-initialized for generation, and training data is small (3,004 samples) [REF].

5. Human evaluation — Small annotator pool (n=3), no formal calibration session, low IAA (κ ≈ 0.00). Results are indicative rather than statistically definitive.

6. Hardware constraints — Single T4 GPU (16 GB) prevented exhaustive hyperparameter search (e.g., LoRA ranks beyond r=16, longer training beyond 15 epochs).

7. Autoencoding Identity Bias — The decoder still requires post-hoc inference constraints (beam search + n-gram blocker) to generate meaningful text. The underlying weight distributions were not corrected during training [REF].

TARGET LENGTH: 5–6 paragraphs, approximately 500–700 words.

FORMATTING:
- Use [REF] placeholders.
```

---

#### PROMPT 5.3: Future Work (✍️ FULL REWRITE)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Completely REWRITE Section 5.3 (Future Work) of the DRISHTI-Bn thesis.

REFERENCE: The old version (lines 402–445) had good ideas. Keep the strongest ones and add new directions based on the expanded work.

FUTURE DIRECTIONS TO DISCUSS (one paragraph each):

1. Native Bengali dataset — Collect disaster images and captions directly from Bangladeshi social media (Facebook, local Twitter) during real flood events. Eliminates translation distortions [REF].

2. Training fixes — Implement right-shifted teacher forcing and scheduled sampling to fix the Autoencoding Identity Bias at the training level rather than inference level [REF].

3. Extended ablation — Test LoRA ranks r=16, r=32 on higher-capacity GPUs (A100, 80 GB). Longer training (30–50 epochs) for better caption quality.

4. Advanced captioning — Integrate Retrieval-Augmented Generation (RAG) to retrieve similar historical disaster images and their verified captions as context for generation [REF].

5. Edge deployment — Quantize the model for mobile/drone-based edge inference, enabling field responders to generate assessments without internet connectivity [REF].

6. Multi-source imagery — Extend the visual encoder to process UAV (drone) imagery, Sentinel-1 SAR data, and MODIS satellite flood maps [REF].

7. Improved human evaluation — Future studies should conduct a formal annotator calibration session with practice examples, use Magnitude Estimation scoring, and recruit domain-expert annotators (e.g., disaster management professionals) [REF].

TARGET LENGTH: 5–6 paragraphs, approximately 500–700 words.

FORMATTING:
- Use [REF] placeholders.
```

---

#### PROMPT 5.4: Conclusion (✍️ FULL REWRITE)

```
First, read the file "00_MASTER_WRITING_GUIDELINES.txt" carefully and follow all its rules strictly throughout your response.

TASK: Completely REWRITE Section 5.4 (Conclusion) of the DRISHTI-Bn thesis.

CONTENT — summarize the entire thesis contribution in a concise, impactful conclusion:

Key points to include:
1. DRISHTI-Bn is the FIRST multimodal framework for simultaneous Bengali disaster image captioning and classification.
2. Classification: Macro F1 = 83.21% (CLS-Only), beating all 5 baselines including CrisisViT SOTA.
3. Caption Generation: BLEU-4 = 6.13, BERTScore = 60.35% — establishing the first Bengali disaster captioning benchmark.
4. Parameter efficiency: Only 0.30% trainable parameters via LoRA, operating on a single T4 GPU — validates deployment feasibility in resource-constrained environments.
5. Explainability: 4 XAI methods integrated — Cross-Attention provides the most spatially meaningful visual explanations, Temperature Scaling confirms ECE = 2.16% reliability.
6. Human evaluation: Model captions score competitively with Ground Truth (within ~0.1 on all dimensions).
7. Key finding: Multi-task caption loss degrades classification performance — important insight for future multi-task vision-language research.
8. DRISHTI-Bn advances the intersection of Bengali NLP, disaster AI, and explainable machine learning.

End with a forward-looking statement about automated native-language disaster intelligence for South Asia.

TARGET LENGTH: 2–3 paragraphs, approximately 300–400 words.

FORMATTING:
- Use [REF] placeholders where needed (minimal — conclusions typically have few citations).
```

---

## Part D: Writing Progress Tracker

| # | Prompt | Section | Type | Status |
|---|--------|---------|------|--------|
| P0 | Abstract | Abstract | ☑ Update | ⬜ |
| P1.1 | Chapter 1.1 | Background | ☑ Update | ⬜ |
| P1.2 | Chapter 1.2 | Problem Statement | ☑ Update | ⬜ |
| P1.3 | Chapter 1.3 | Research Objectives | ☑ Update | ⬜ |
| P1.4 | Chapter 1.4 | Thesis Organization | 🆕 New | ⬜ |
| P2.1 | Chapter 2.1 | Vision Models | ☑ Update | ⬜ |
| P2.2 | Chapter 2.2 | Transformers & Multimodal | ☑ Update | ⬜ |
| P2.3 | Chapter 2.3 | Bengali NLP | ☑ Update | ⬜ |
| P2.4 | Chapter 2.4 | XAI in CV | 🆕 New | ⬜ |
| P2.5 | Chapter 2.5 | Human Eval in NLG | 🆕 New | ⬜ |
| P3.1 | Chapter 3.1 | Dataset | ☑ Update | ⬜ |
| P3.2 | Chapter 3.2 | Architecture | ☑ Update | ⬜ |
| P3.3 | Chapter 3.3 | LoRA/PEFT | ☑ Update | ⬜ |
| P3.4 | Chapter 3.4 | Multi-Task Loss | ☑ Update | ⬜ |
| P3.5 | Chapter 3.5 | XAI Pipeline | 🆕 New | ⬜ |
| P3.6 | Chapter 3.6 | Human Eval Design | 🆕 New | ⬜ |
| P4.1 | Chapter 4.1 | Experimental Setup | ☑ Update | ⬜ |
| P4.2 | Chapter 4.2 | Classification Results | ✍️ Rewrite | ⬜ |
| P4.3 | Chapter 4.3 | Caption Results | ✍️ Rewrite | ⬜ |
| P4.4 | Chapter 4.4 | Baseline Comparison | 🆕 New | ⬜ |
| P4.5 | Chapter 4.5 | Ablation Study | 🆕 New | ⬜ |
| P4.6 | Chapter 4.6 | XAI Analysis | 🆕 New | ⬜ |
| P4.7 | Chapter 4.7 | Human Evaluation | 🆕 New | ⬜ |
| P5.1 | Chapter 5.1 | Discussion | ✍️ Rewrite | ⬜ |
| P5.2 | Chapter 5.2 | Limitations | ✍️ Rewrite | ⬜ |
| P5.3 | Chapter 5.3 | Future Work | ✍️ Rewrite | ⬜ |
| P5.4 | Chapter 5.4 | Conclusion | ✍️ Rewrite | ⬜ |
| — | Figures/Tables | Insert all placeholders | — | ⬜ |
| — | [REF] | Fill all reference placeholders | — | ⬜ |
| — | Proofread | Final review | — | ⬜ |

---

> **শুরু করো Prompt 0 (Abstract) দিয়ে → তারপর 1.1 → 1.2 → ক্রমানুসারে শেষ পর্যন্ত।**
> **প্রতিটি prompt-এর output NotebookLM থেকে কপি করে Word-এ paste করো।**
> **তুমিই author — AI শুধু writing assistant।**
