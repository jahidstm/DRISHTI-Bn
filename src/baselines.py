import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_PATH    = '../data/processed/master_dataset_translated.csv'
LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']
LABEL_MAP   = {name: i for i, name in enumerate(LABEL_NAMES)}
RANDOM_SEED = 42


def get_split_indices(csv_path, random_state=42):
    """Mirrors the same 80/10/10 stratified split used in data_loader.py."""
    df = pd.read_csv(csv_path)
    labels = df['macro_label'].values
    all_indices = list(range(len(df)))
    train_idx, temp_idx = train_test_split(
        all_indices, test_size=0.2, stratify=labels, random_state=random_state
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=random_state
    )
    return train_idx, val_idx, test_idx



# ──────────────────────────────────────────────────────────────────────────────
# Helper: print a metrics block (mirrors benchmark.py output format)
# ──────────────────────────────────────────────────────────────────────────────
def print_cls_report(title: str, y_true, y_pred):
    """Print classification metrics in DRISHTI-XAI benchmark format."""
    macro_f1  = f1_score(y_true, y_pred, average='macro',  zero_division=0) * 100
    macro_p   = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    macro_r   = recall_score(y_true, y_pred, average='macro',  zero_division=0) * 100

    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_p  = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_r  = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    w = 62
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")
    print(f"  Macro F1        : {macro_f1:.2f}%")
    print(f"  Macro Precision : {macro_p:.2f}%")
    print(f"  Macro Recall    : {macro_r:.2f}%")
    print()
    print("  Per-class breakdown:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"    [{name:<22}] P={per_p[i]:.4f}  R={per_r[i]:.4f}  F1={per_f1[i]:.4f}")
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    header = f"{'':>16}" + "".join(f"  {n[:8]:>8}" for n in LABEL_NAMES)
    print(f"  {header}")
    row_labels = ["Severe_Damag", "Humanitarian", "Affected_Peo"]
    for i, row_label in enumerate(row_labels):
        row_str = "  ".join(f"{cm[i][j]:>8}" for j in range(3))
        print(f"  {row_label:<14}  {row_str}")
    print(f"{'='*w}")

    return macro_f1


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load CSV & compute stratified splits ──────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df['label_id'] = df['macro_label'].map(LABEL_MAP)

    train_idx, val_idx, test_idx = get_split_indices(CSV_PATH, random_state=RANDOM_SEED)

    # Ground-truth labels for TEST split
    y_true = df.iloc[test_idx]['label_id'].values
    n_test = len(y_true)

    # Training class frequencies (needed for stratified random)
    train_labels = df.iloc[train_idx]['label_id'].values
    class_counts = np.bincount(train_labels, minlength=3)
    class_probs  = class_counts / class_counts.sum()
    majority_cls = int(np.argmax(class_counts))

    print("\n" + "=" * 62)
    print("  DRISHTI-XAI -- Task 0.4: Statistical Baseline Classifiers")
    print("=" * 62)
    print(f"  Test samples      : {n_test}")
    print(f"  Training counts   : " + " | ".join(
        f"{LABEL_NAMES[i]}={class_counts[i]}" for i in range(3)))
    print(f"  Majority class    : {LABEL_NAMES[majority_cls]} ({class_counts[majority_cls]} samples)")
    print(f"  Class probs (train): {class_probs.round(4).tolist()}")
    print("=" * 62)

    results = {}

    # ── Baseline 1: ZeroR / Majority Classifier ───────────────────────────────
    y_pred_majority = np.full(n_test, majority_cls, dtype=int)
    f1 = print_cls_report(
        "BASELINE 1 -- ZeroR / Majority Classifier  (always predicts most frequent class)",
        y_true, y_pred_majority
    )
    results['Majority'] = f1

    # ── Baseline 2: Uniform Random Classifier ─────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    y_pred_uniform = rng.integers(0, 3, size=n_test)
    f1 = print_cls_report(
        "BASELINE 2 -- Uniform Random  (each class predicted with prob 1/3)",
        y_true, y_pred_uniform
    )
    results['Uniform Random'] = f1

    # ── Baseline 3: Stratified Random Classifier ──────────────────────────────
    rng2 = np.random.default_rng(RANDOM_SEED)
    y_pred_stratified = rng2.choice([0, 1, 2], size=n_test, p=class_probs)
    f1 = print_cls_report(
        "BASELINE 3 -- Stratified Random  (prob proportional to training class freq)",
        y_true, y_pred_stratified
    )
    results['Stratified Random'] = f1

    # ── Summary Comparison Table ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SUMMARY: Baseline vs. DisasterNet-v2")
    print("=" * 62)
    print(f"  {'Method':<28} {'Macro F1':>10}")
    print(f"  {'-'*28} {'-'*10}")
    for name, f1 in results.items():
        print(f"  {name:<28} {f1:>9.2f}%")
    # Our model result from Task 0.3
    print(f"  {'DisasterNet-v2 (ours)':<28} {'77.95':>9}%  <- Task 0.3")
    print("=" * 62)
    print()


if __name__ == '__main__':
    main()
