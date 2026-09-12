import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# ── DRISHTI-XAI modules ──────────────────────────────────────────────────
from data_loader import DisasterNetMultimodalDataset, get_split_indices
from evaluate import DisasterNetPredictor
from metrics import MetricsSuite

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']


def run_benchmark(
    csv_path      = '../data/processed/master_dataset_translated.csv',
    img_dir       = '../data/processed/',
    model_path    = '../models/disasternet_multitask_v2.pth',
    split         = 'test',       # 'test' | 'val' | 'train'
    max_samples   = None,         # None → evaluate full split
    beam_width    = 3,
    use_bertscore = False,        # set True on Colab (needs GPU + time)
    save_csv      = None,         # optional path to save per-sample results
):
    """
    DRISHTI-XAI Quantitative Benchmark Suite.

    Evaluates the trained checkpoint on a stratified held-out split and
    reports BLEU-1/2/3/4, METEOR, ROUGE-L, BERTScore, Macro F1,
    per-class F1, and confusion matrix.
    """
    print(f"\n{'='*62}")
    print(f"📊 DRISHTI-XAI QUANTITATIVE BENCHMARK SUITE")
    print(f"{'='*62}")
    print(f"  Split        : {split.upper()} (stratified 80/10/10)")
    print(f"  Model        : {model_path}")
    print(f"  Decoding     : {'Beam Search (w=' + str(beam_width) + ')' if beam_width > 1 else 'Greedy'}")
    print(f"  BERTScore    : {'ON' if use_bertscore else 'OFF (use --bertscore to enable)'}")
    print(f"{'='*62}\n")

    # ── Load predictor ────────────────────────────────────────────────────
    predictor = DisasterNetPredictor(model_path=model_path)

    # ── Stratified split → pick the right indices ─────────────────────────
    full_dataset = DisasterNetMultimodalDataset(csv_file=csv_path, root_dir=img_dir)
    train_idx, val_idx, test_idx = get_split_indices(csv_path, random_state=42)
    split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    split_idx = split_map[split]

    if max_samples is not None:
        split_idx = split_idx[:max_samples]

    print(f"[+] Evaluating {len(split_idx)} samples from {split.upper()} split...\n")

    # ── Inference loop ────────────────────────────────────────────────────
    true_cls, pred_cls = [], []
    references, hypotheses = [], []
    per_sample_rows = []

    for i in tqdm(split_idx, desc=f"Benchmarking ({split})"):
        row = full_dataset.df.iloc[i]

        img_full      = os.path.join(img_dir, row['image_path'])
        ref_caption   = str(row['bengali_caption']) if pd.notna(row['bengali_caption']) else ""
        true_label_id = full_dataset.label_map[row['macro_label']]

        if not os.path.exists(img_full):
            continue

        image        = Image.open(img_full).convert('RGB')
        pixel_values = predictor.transform(image).unsqueeze(0).to(predictor.device)

        # Task 2: caption generation
        gen_caption = predictor.generate_caption(pixel_values, beam_width=beam_width)

        # Task 1: multimodal classification
        pred_cat_str, conf, _ = predictor.classify_damage(pixel_values, gen_caption)

        # Label ID from prediction string
        if   'Severe'       in pred_cat_str: pred_label_id = 0
        elif 'Humanitarian' in pred_cat_str: pred_label_id = 1
        else:                                 pred_label_id = 2

        true_cls.append(true_label_id)
        pred_cls.append(pred_label_id)
        references.append(ref_caption)
        hypotheses.append(gen_caption)

        per_sample_rows.append({
            'image_path': row['image_path'],
            'true_label': row['macro_label'],
            'pred_label': LABEL_NAMES[pred_label_id],
            'confidence': round(conf, 2),
            'reference':  ref_caption,
            'hypothesis': gen_caption,
        })

    # ── Compute all metrics via MetricsSuite ──────────────────────────────
    suite   = MetricsSuite(use_bertscore=use_bertscore, bertscore_lang='bn')
    cap_res = suite.evaluate_captions(references, hypotheses)
    cls_res = suite.evaluate_classification(true_cls, pred_cls)

    suite.print_report(cap_res, cls_res)

    # ── Optional: save per-sample CSV for error analysis ──────────────────
    if save_csv:
        df_out = pd.DataFrame(per_sample_rows)
        df_out.to_csv(save_csv, index=False, encoding='utf-8-sig')
        print(f"[+] Per-sample results saved to: {save_csv}")

    return {**cap_res, **cls_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DRISHTI-XAI Quantitative Benchmark Suite"
    )
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which data split to evaluate (default: test)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of samples (None = full split)")
    parser.add_argument("--beam-width",  type=int, default=3,
                        help="Beam width (1=Greedy, >1=Beam Search)")
    parser.add_argument("--bertscore",   action="store_true",
                        help="Enable BERTScore (requires GPU + bert-score)")
    parser.add_argument("--save-csv",    type=str, default=None,
                        help="Path to save per-sample results CSV")
    args = parser.parse_args()

    run_benchmark(
        split         = args.split,
        max_samples   = args.max_samples,
        beam_width    = args.beam_width,
        use_bertscore = args.bertscore,
        save_csv      = args.save_csv,
    )
