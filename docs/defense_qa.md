# DRISHTI-Bn — Defense প্রস্তুতি: সম্ভাব্য প্রশ্ন ও উত্তর

> **কীভাবে ব্যবহার করবে:** প্রতিটা প্রশ্নের উত্তর ইংরেজিতে লেখা, যাতে defense-এ সরাসরি বলা যায়। "মূল কথা" লাইনটা মনে রাখার জন্য।
> সব সংখ্যা `CLAUDE.md` আর `docs/corrected_claims.md`-এর সঙ্গে মেলানো (2026-11-12)।
> **নিয়ম:** examiner দুর্বলতা ধরলে অস্বীকার নয় — আগে স্বীকার, তারপর "কীভাবে ধরা পড়েছে, কী ঠিক করেছি, কী অর্থ"।

---

## ক. Data ও evaluation

### Q1. Your earlier results were 83.21%. Why is the final number 75.69%?
"In the original split, images of the same tweet were divided across train and test; 80 of the 376 test images shared their exact text with a training image. I rebuilt the split so that no tweet or identical text crosses splits, and re-ran the model with three seeds. The leakage-free result is 75.69 ± 1.33% Macro F1. I report this as the main number; 83.21% is mentioned only as an inflated, superseded result."
**মূল কথা:** নিজেই leakage ধরেছি, ঠিক করেছি, সৎ সংখ্যা দিয়েছি।

### Q2. Why three seeds? Is 75.69% stable?
"A single run is not reliable: two runs of the identical configuration on the old split differed by 1.7 points. Over seeds 42, 123 and 2024 the model gives 74.83, 75.02 and 77.22%. Most of the variance comes from Affected People, which has only 32 test images."

### Q3. Your dataset is called a 'flood corpus' but it is mostly hurricanes.
"That is correct: 95.2% of the images come from Hurricanes Harvey, Maria and Irma, and 4.8% from the 2017 Sri Lanka floods. Harvey and Maria caused extensive flooding, so the images are flood- and storm-related, but the corpus is not flood-only, and it contains no images from Bangladesh. Generalisation to Bangladeshi floods is untested and is listed as a limitation."

### Q4. How were the Bengali texts produced? Were they verified?
"They are machine translations of the CrisisMMD tweets using the `monirbishal/en-bn-nmt` model. They were not manually verified; about 8.9% were left untranslated and most contain URLs. I state this explicitly."

### Q5. Why is Affected People so poor (F1 ≈ 45%)?
"It is the smallest class: 330 of 3,755 images, 32 in the test set. It is also visually close to the other two classes: people in flood water versus rescue scenes. In the confusion matrix most errors go to Severe Damage. Class-weighted loss or more data are natural next steps."

---

## খ. মডেল ও multimodality

### Q6. If the model is 'multimodal', how much does the text help?
"Very little, and I tested this directly. If I give each image the tweet of a different test sample, Macro F1 does not drop: 76.63% versus 74.83% with the correct tweet, and 96.5% of predictions stay the same. So the classifier is effectively image-driven. I do not claim complementary text–image fusion. The advantage over the ViT-Only baseline comes from the LoRA-adapted image encoder, not from tweet semantics."
**মূল কথা:** দুর্বলতা আমি নিজেই পরীক্ষা করে বের করেছি — এটা গবেষণার সততা।

### Q7. Then why keep BanglaBERT at all?
"In this dataset tweets are noisy and often off-topic, so they add little. The architecture remains useful when text is informative. A practical benefit is that the model works without any tweet: with a fixed placeholder text it scores 76.52%, which is how the demo runs."

### Q8. Do you beat the state of the art?
"No. The CrisisViT-style baseline is my own re-implementation with ViT-Large, not the published result. On the same leakage-free split it scores 75.87% and my model scores 75.69 ± 1.33%, which is a tie. My claim is efficiency: comparable accuracy while training 1.38M parameters instead of 51.96M, about 38 times fewer, on a smaller backbone."

### Q9. Your old baselines (ViT-Only 78.64%) were higher than your new ones. Why?
"Two reasons. First, they were measured on the leaky split. Second, there was a bug: the code selected layers by the name `encoder.layer.N`, but in transformers 5.x the layers are called `layers.N`, so only the pooler was trained. I fixed the selection rule and re-ran both baselines on the grouped split."

### Q10. What exactly is trainable? You said 0.30%.
"LoRA adds 589,824 parameters, which is 0.30% of the two encoders. With the classification head, 1.38M parameters, or about 0.70%, are trained for classification. The multi-task model also trains a 72.8M-parameter decoder."

### Q11. Why LoRA rank 8? Your rank ablation shows r = 4 and r = 16 lower.
"The rank ablation was run once, on the original split. Two identical r = 8 runs differed by 1.7 points, so the differences between ranks, about 2 to 4 points, are not conclusive. I chose r = 8 as the common default from the LoRA paper and report the ablation with that caveat."

---

## গ. Bengali text generation

### Q12. Your BLEU is 14.38. Is the caption model good?
"No, and I say so. Almost all of that overlap is tweet boilerplate such as URLs and hashtags; after removing it, BLEU is 0.11. On chrF, the model scores below two trivial references: a random training tweet and a single constant tweet. The reason is that the targets are translated tweets, not image descriptions, so the decoder learned tweet style. I present the generation branch as a negative result and a limitation."

### Q13. Your earlier BERTScore was 60.35% and BLEU-1 24.85. What happened?
"Those numbers came from an older checkpoint. It was trained with unshifted teacher forcing, meaning the decoder could see the token it had to predict. It also used a random split that overlapped with most of the test set. In addition, the BLEU implementation was non-standard. I retrained the model with right-shifted teacher forcing on the grouped split and evaluated with sacreBLEU."

### Q14. What about the human evaluation?
"Three native speakers rated 100 pairs blind. The analysis script originally ignored the random A/B swap, so the first reported scores mixed model and reference captions. After mapping each rating to its true source, generated text scored 1.58 / 1.24 / 1.23 out of 5 against 3.42 / 3.08 / 3.23 for the references (p < 10⁻¹⁶), and annotators preferred the reference 94.9% of the time. Agreement is fair (α = 0.38) when both caption types are pooled. One limitation is that this evaluation used the earlier checkpoint; the final model was assessed with automatic metrics, which show the same weakness."

### Q15. Is the demo caption real?
"Yes. The demo shows the decoder's output unedited, with a warning that it imitates tweet style. An earlier version of the demo code contained a hard-coded example sentence for a screenshot. I removed it, and that screenshot is not used anywhere."
**মূল কথা:** এই প্রশ্ন এলে সরাসরি স্বীকার করবে; লুকানোর চেষ্টা করবে না।

---

## ঘ. Explainability ও calibration

### Q16. How do you know Grad-CAM shows what the model uses?
"I ran a deletion test on all 375 test images. Removing the 10% of patches that Grad-CAM ranks highest reduces the predicted-class probability by 6.92 points, against 0.34 for random patches, about 20 times more. Removing half of them drops accuracy from 88.27% to 73.87%, against 86.93% for random. That is quantitative evidence of faithfulness."

### Q17. Earlier you wrote that Grad-CAM gave 'dark heatmaps'. Why the change?
"That was a bug in my first implementation. It hooked the output of the last ViT block, but the classifier reads only the [CLS] token, so the patch tokens at that point receive exactly zero gradient and every map was blank. I verified the zero gradient numerically, moved the hook to the input of the last block, and the maps became informative and faithful."

### Q18. Why is Attention Rollout weaker? Why the isolated bright spots?
"Rollout is class-agnostic: it shows how information flows through the ViT in general, not the evidence for a specific class. Its deletion effect is only about twice random. The isolated bright spots are a known property of ViTs, which store global information in a few high-norm background tokens."

### Q19. What does the cross-attention map show?
"It comes from the trained text decoder and explains text generation only. Its most-attended patches overlap with Grad-CAM no more than chance (IoU 0.059 against 0.054), so I present it as an auxiliary analysis, not as an explanation of the classification."

### Q20. Temperature scaling made calibration worse. Why include it?
"The model is already well calibrated without it: ECE is 2.57%, and mean confidence of 88.84% is close to accuracy of 88.27%. Temperature scaling fitted on 373 validation images (T* = 1.1349) gives 2.99%, which is within noise. I report it as a check showing no correction is needed, not as an improvement."

### Q21. You reported '81% XAI accuracy' before.
"That was a misnomer. It was the classification accuracy on the 21 images chosen for visualisation. On the corrected setup, 15 of 21 random test images are classified correctly. The real XAI metric is the faithfulness test in Q16."

---

## ঙ. সাধারণ প্রশ্ন

### Q22. What is your actual contribution, given all these corrections?
"First, a Bengali multimodal disaster benchmark on CrisisMMD with a leakage-free, multi-seed evaluation protocol. Second, a parameter-efficient LoRA model that matches a ViT-Large baseline with 38 times fewer trainable parameters. Third, a quantitatively validated explanation pipeline with a deletion test. Fourth, honest negative findings: the classifier barely uses the tweet text, and tweet-translated references cannot train an image captioner. These findings are useful for anyone building Bengali crisis datasets."

### Q23. What would you do with more time?
"Collect real Bangladeshi flood images with human-written Bengali descriptions. Clean tweets of URLs and mentions before training. Use class-balanced training for Affected People. Run every ablation over several seeds on the grouped split."

### Q24. Why should we trust your numbers now?
"Every experiment has a script in the `colab/` folder and saved outputs in `results/defense_fixes/`. Checkpoints are fixed, and `DEFENSE_FIX_PLAN.md` records each correction with its commit. For example, the Phase 2 evaluation re-loaded the saved checkpoint and reproduced the Phase 3 result exactly, 74.83% with an identical confusion matrix."
