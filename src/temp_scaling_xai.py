"""
DRISHTI-XAI -- Task 2.4: Temperature Scaling (Post-hoc Confidence Calibration)
================================================================================
Method: Guo et al. (2017) - "On Calibration of Modern Neural Networks"
https://arxiv.org/abs/1706.04599

What this does:
  - মডেল প্রায়ই predictions-এ over-confident বা under-confident হয়।
  - Temperature Scaling একটি single scalar T শিখে যা logits-কে rescale করে:
        calibrated_prob = softmax(logits / T)
  - T > 1 → softens distribution (reduces over-confidence)
  - T < 1 → sharpens distribution (reduces under-confidence)
  - T = 1 → no change (already calibrated)

Algorithm:
  1. Load best checkpoint (Task 1.1 CLS-Only)
  2. Collect raw logits on Validation set (375 samples)
  3. Learn T* = argmin NLL(softmax(logits/T), true_labels) using L-BFGS-B
  4. Evaluate BEFORE vs AFTER calibration on Test set (376 samples)
  5. Generate: Reliability Diagram, Confidence Histogram, per-sample CSV

Metrics:
  - ECE  (Expected Calibration Error) — weighted avg |confidence - accuracy| per bin
  - MCE  (Maximum Calibration Error)  — worst-case bin error
  - NLL  (Negative Log-Likelihood)
  - Brier Score

Colab Usage (run cells 1,2,3 first, then):
    !git pull origin main
    !cp /content/drive/MyDrive/DisasterNet/models/best_cls_only.pt /content/DRISHTI-Bn/src/
    %cd /content/DRISHTI-Bn/src
    !python temp_scaling_xai.py --checkpoint best_cls_only.pt --output_dir calibration_outputs
"""

import os, random, argparse, math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel, ViTModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar

# ─── Config ──────────────────────────────────────────────────────────────────
CSV_PATH    = "../data/processed/master_dataset_translated.csv"
IMG_DIR     = "../data/processed/"
LABEL_NAMES = ["Severe_Damage", "Humanitarian_Rescue", "Affected_People"]
LABEL_MAP   = {n: i for i, n in enumerate(LABEL_NAMES)}
RANDOM_SEED = 42
MAX_LEN     = 128
VOCAB_SIZE  = 32000
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_ID     = "csebuetnlp/banglabert"
N_BINS      = 10    # bins for ECE/reliability diagram


# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[+] Random seed fixed: {seed}")


# ─── Dataset ─────────────────────────────────────────────────────────────────
class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, max_len=MAX_LEN):
        self.df        = pd.read_csv(csv_file)
        self.root_dir  = root_dir
        self.tokenizer = tokenizer
        self.max_len   = max_len

        for col in ["bangla_caption", "bengali_caption", "caption_bn", "caption", "text"]:
            if col in self.df.columns:
                self.text_col = col; break
        else:
            raise ValueError("No caption column found.")

        self.df[self.text_col] = self.df[self.text_col].fillna("").astype(str)
        self.df["label_id"]    = self.df["macro_label"].map(LABEL_MAP)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root_dir, row["image_path"])).convert("RGB")
        pv  = self.transform(img)
        enc = self.tokenizer(
            row[self.text_col], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values":   pv,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(row["label_id"], dtype=torch.long),
            "macro_label":    row["macro_label"],
        }


def get_split_indices(csv_path, rs=42):
    df   = pd.read_csv(csv_path)
    lbls = df["macro_label"].values
    idx  = list(range(len(df)))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=lbls, random_state=rs)
    vl, ts  = train_test_split(tmp, test_size=0.5, stratify=lbls[tmp], random_state=rs)
    return tr, vl, ts


def _find_lora_targets(model, candidates):
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    found = [c for c in candidates if c in names]
    if not found:
        raise ValueError(f"None of {candidates} found.")
    return found


# ─── Model (same as Task 1.1 / 2.1 / 2.2 / 2.3) ─────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DisasterNetAblation(nn.Module):
    def __init__(self, num_classes=3, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            attn_implementation="eager"
        )
        self.text_encoder = AutoModel.from_pretrained(BERT_ID)

        vit_t  = _find_lora_targets(self.vision_encoder, ["query", "value", "q_proj", "v_proj"])
        bert_t = _find_lora_targets(self.text_encoder,   ["query", "value", "q_proj", "v_proj"])
        self.vision_encoder = get_peft_model(
            self.vision_encoder,
            LoraConfig(r=8, lora_alpha=16, target_modules=vit_t, lora_dropout=0.1, bias="none"))
        self.text_encoder = get_peft_model(
            self.text_encoder,
            LoraConfig(r=8, lora_alpha=16, target_modules=bert_t, lora_dropout=0.1, bias="none"))

        self.classifier = nn.Sequential(
            nn.Linear(768+768, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
        self.d_model           = 768
        self.decoder_embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder       = PositionalEncoding(self.d_model)
        self.caption_decoder   = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8,
                                       dim_feedforward=2048, dropout=0.1, batch_first=True),
            num_layers=3)
        self.vocab_projector = nn.Linear(self.d_model, vocab_size)

    def forward_cls(self, pixel_values, input_ids, attention_mask):
        vis = self.vision_encoder(pixel_values=pixel_values).pooler_output
        txt = self.text_encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([vis, txt], dim=1))


# ─── Collect Logits ──────────────────────────────────────────────────────────
def collect_logits(model, dataset, indices, batch_size=16):
    from torch.utils.data import DataLoader, Subset
    subset     = Subset(dataset, indices)
    loader     = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pv  = batch["pixel_values"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            msk = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"]
            logits = model.forward_cls(pv, ids, msk)
            all_logits.append(logits.cpu())
            all_labels.append(lbl)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ─── Temperature Scaling ─────────────────────────────────────────────────────
def find_optimal_temperature(logits, labels, T_range=(0.1, 10.0)):
    logits_np = logits.numpy()
    labels_np = labels.numpy()

    def nll_loss(T):
        scaled = logits_np / T
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True) + 1e-10)
        return -np.mean(log_probs[np.arange(len(labels_np)), labels_np])

    nll_before = nll_loss(1.0)
    result = minimize_scalar(nll_loss, bounds=T_range, method='bounded')
    T_star = result.x
    nll_after = result.fun
    return float(T_star), float(nll_before), float(nll_after)


# ─── Calibration Metrics ─────────────────────────────────────────────────────
def compute_calibration_metrics(probs, labels, n_bins=N_BINS):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels)

    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_accs   = np.zeros(n_bins)
    bin_confs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask   = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_accs[i]   = correct[mask].mean()
            bin_confs[i]  = confidences[mask].mean()
            bin_counts[i] = mask.sum()

    N   = len(labels)
    ece = np.sum((bin_counts / N) * np.abs(bin_accs - bin_confs))
    mce = np.max(np.abs(bin_accs - bin_confs))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_accs":   bin_accs,
        "bin_confs":  bin_confs,
        "bin_counts": bin_counts,
    }


def compute_nll(probs, labels):
    eps = 1e-10
    log_probs = np.log(probs[np.arange(len(labels)), labels] + eps)
    return float(-log_probs.mean())


def compute_brier_score(probs, labels, n_classes=3):
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ─── Visualization ───────────────────────────────────────────────────────────
def save_calibration_figure(
    cal_before, cal_after, T_star,
    probs_before, probs_after, labels,
    save_path, split_name="Test"
):
    fig = plt.figure(figsize=(22, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    bin_edges   = np.linspace(0, 1, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width   = 0.08

    # (a) Reliability Diagram — Before
    ax0 = fig.add_subplot(gs[0])
    ax0.bar(bin_centers, cal_before["bin_accs"], width=bar_width,
            color='#3498db', alpha=0.75, label='Accuracy', zorder=3)
    ax0.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
    ax0.plot(bin_centers, cal_before["bin_confs"], 'r--o', ms=4, lw=1, label='Confidence', zorder=4)
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.set_xlabel("Confidence", fontsize=9); ax0.set_ylabel("Accuracy", fontsize=9)
    ax0.set_title(f"Reliability Diagram\nBEFORE (T=1.0)\nECE={cal_before['ece']:.4f}",
                  fontsize=9, fontweight='bold')
    ax0.legend(fontsize=7); ax0.grid(True, alpha=0.3)

    # (b) Reliability Diagram — After
    ax1 = fig.add_subplot(gs[1])
    ax1.bar(bin_centers, cal_after["bin_accs"], width=bar_width,
            color='#2ecc71', alpha=0.75, label='Accuracy', zorder=3)
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
    ax1.plot(bin_centers, cal_after["bin_confs"], 'r--o', ms=4, lw=1, label='Confidence', zorder=4)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence", fontsize=9); ax1.set_ylabel("Accuracy", fontsize=9)
    ax1.set_title(f"Reliability Diagram\nAFTER (T={T_star:.4f})\nECE={cal_after['ece']:.4f}",
                  fontsize=9, fontweight='bold', color='#27ae60')
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # (c) Confidence Histogram
    ax2 = fig.add_subplot(gs[2])
    conf_before = probs_before.max(axis=1)
    conf_after  = probs_after.max(axis=1)
    ax2.hist(conf_before, bins=20, alpha=0.55, color='#3498db',
             label=f'Before (T=1.0)', density=True)
    ax2.hist(conf_after,  bins=20, alpha=0.55, color='#2ecc71',
             label=f'After  (T={T_star:.2f})', density=True)
    ax2.axvline(conf_before.mean(), color='#2980b9', linestyle='--', lw=1.5,
                label=f'Mean before: {conf_before.mean():.3f}')
    ax2.axvline(conf_after.mean(),  color='#27ae60', linestyle='--', lw=1.5,
                label=f'Mean after:  {conf_after.mean():.3f}')
    ax2.set_xlabel("Max Confidence", fontsize=9); ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title(f"Confidence Distribution\n({split_name} set, n={len(labels)})",
                  fontsize=9, fontweight='bold')
    ax2.legend(fontsize=6.5); ax2.grid(True, alpha=0.3)

    # (d) Metrics Bar Chart
    ax3 = fig.add_subplot(gs[3])
    metrics     = ['ECE', 'MCE']
    vals_before = [cal_before['ece'], cal_before['mce']]
    vals_after  = [cal_after['ece'],  cal_after['mce']]
    x = np.arange(len(metrics)); w = 0.3
    bars_b = ax3.bar(x - w/2, vals_before, width=w, color='#3498db', alpha=0.8, label='Before (T=1.0)')
    bars_a = ax3.bar(x + w/2, vals_after,  width=w, color='#2ecc71', alpha=0.8, label=f'After (T={T_star:.2f})')
    for bar in bars_b:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)
    for bar in bars_a:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7,
                 color='#27ae60', fontweight='bold')
    ax3.set_xticks(x); ax3.set_xticklabels(metrics, fontsize=9)
    ax3.set_ylabel("Error (lower is better)", fontsize=9)
    ax3.set_title(f"Calibration Metrics\nBefore vs After (T={T_star:.4f})",
                  fontsize=9, fontweight='bold')
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, max(max(vals_before), max(vals_after)) * 1.35)

    fig.suptitle(
        f"DRISHTI-XAI Task 2.4: Temperature Scaling | T*={T_star:.4f} | {split_name} set",
        fontsize=10, fontweight='bold', y=1.02
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved calibration figure: {save_path}")


def save_per_class_reliability(probs_before, probs_after, labels, T_star, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    labels_np = labels if isinstance(labels, np.ndarray) else np.array(labels)

    for c, (cls_name, ax) in enumerate(zip(LABEL_NAMES, axes)):
        mask = (labels_np == c)
        if mask.sum() == 0:
            ax.set_visible(False); continue

        conf_b    = probs_before[mask].max(axis=1)
        conf_a    = probs_after[mask].max(axis=1)
        correct_b = (probs_before[mask].argmax(axis=1) == c)
        correct_a = (probs_after[mask].argmax(axis=1)  == c)

        ax.hist(conf_b, bins=10, alpha=0.5, color='#3498db',
                label=f'Before  acc={correct_b.mean():.2f}', density=True)
        ax.hist(conf_a, bins=10, alpha=0.5, color='#2ecc71',
                label=f'After   acc={correct_a.mean():.2f}', density=True)
        ax.axvline(conf_b.mean(), color='#2980b9', linestyle='--', lw=1.5,
                   label=f'mu_before={conf_b.mean():.3f}')
        ax.axvline(conf_a.mean(), color='#27ae60', linestyle='--', lw=1.5,
                   label=f'mu_after={conf_a.mean():.3f}')
        ax.set_title(f"{cls_name}\n(n={mask.sum()}, T={T_star:.2f})", fontsize=9, fontweight='bold')
        ax.set_xlabel("Confidence", fontsize=8); ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Class Confidence Distribution: Before vs After Temperature Scaling",
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved per-class figure: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Task 2.4: Temperature Scaling Calibration")
    ap.add_argument("--checkpoint",  type=str,   default="best_cls_only.pt")
    ap.add_argument("--output_dir",  type=str,   default="calibration_outputs")
    ap.add_argument("--batch_size",  type=int,   default=16)
    ap.add_argument("--n_bins",      type=int,   default=10)
    ap.add_argument("--T_min",       type=float, default=0.1)
    ap.add_argument("--T_max",       type=float, default=10.0)
    args = ap.parse_args()

    set_seed(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  DRISHTI-XAI -- Task 2.4: Temperature Scaling")
    print(f"  Method: Guo et al. (2017) Post-hoc Calibration")
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*65}\n")

    # Tokenizer & Dataset
    print("[+] Loading tokenizer...")
    tokenizer  = AutoTokenizer.from_pretrained(BERT_ID)
    dataset    = MultimodalDataset(CSV_PATH, IMG_DIR, tokenizer)
    tr_i, vl_i, ts_i = get_split_indices(CSV_PATH, RANDOM_SEED)
    print(f"[+] Splits -- Train: {len(tr_i)} | Val: {len(vl_i)} | Test: {len(ts_i)}")

    # Build & Load Model
    print("[+] Building model (Task 1.1 architecture)...")
    model = DisasterNetAblation(num_classes=3).to(DEVICE)

    if os.path.isfile(args.checkpoint):
        print(f"[+] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt)
        print("[+] Checkpoint loaded successfully")
    else:
        print(f"[!] WARNING: '{args.checkpoint}' not found -- using random weights!")

    model.eval()

    # Step 1: Collect Logits on Validation set
    print(f"\n[+] Collecting logits on Validation set ({len(vl_i)} samples)...")
    val_logits, val_labels = collect_logits(model, dataset, vl_i, args.batch_size)
    print(f"    Logits shape: {val_logits.shape} | Labels shape: {val_labels.shape}")

    # Step 2: Learn optimal temperature T*
    print("\n[+] Optimizing temperature T on Validation set...")
    T_star, nll_before_val, nll_after_val = find_optimal_temperature(
        val_logits, val_labels, T_range=(args.T_min, args.T_max)
    )
    print(f"    T* = {T_star:.6f}")
    print(f"    NLL before (T=1.0): {nll_before_val:.6f}")
    print(f"    NLL after  (T={T_star:.4f}): {nll_after_val:.6f}")
    improvement = (nll_before_val - nll_after_val) / nll_before_val * 100
    print(f"    NLL improvement: {improvement:.2f}%")

    # Step 3: Collect Logits on Test set
    print(f"\n[+] Collecting logits on Test set ({len(ts_i)} samples)...")
    test_logits, test_labels = collect_logits(model, dataset, ts_i, args.batch_size)
    test_labels_np = test_labels.numpy()

    probs_before = F.softmax(test_logits, dim=1).numpy()
    probs_after  = F.softmax(test_logits / T_star, dim=1).numpy()

    # Step 4: Compute Calibration Metrics
    print("\n[+] Computing calibration metrics...")
    cal_before = compute_calibration_metrics(probs_before, test_labels_np, n_bins=args.n_bins)
    cal_after  = compute_calibration_metrics(probs_after,  test_labels_np, n_bins=args.n_bins)

    nll_test_before = compute_nll(probs_before, test_labels_np)
    nll_test_after  = compute_nll(probs_after,  test_labels_np)
    brier_before    = compute_brier_score(probs_before, test_labels_np)
    brier_after     = compute_brier_score(probs_after,  test_labels_np)

    acc_before = (probs_before.argmax(axis=1) == test_labels_np).mean()
    acc_after  = (probs_after.argmax(axis=1)  == test_labels_np).mean()

    # Step 5: Print Summary
    w = 65
    print(f"\n{'='*w}")
    print(f"  TEMPERATURE SCALING SUMMARY")
    print(f"{'='*w}")
    print(f"  Optimal Temperature   : T* = {T_star:.6f}")
    print(f"  Interpretation        : {'Over-confident (T>1 softens)' if T_star > 1 else 'Under-confident (T<1 sharpens)'}")
    print(f"\n  {'Metric':<28} {'Before (T=1.0)':>15} {'After (T=T*)':>15} {'Delta':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Accuracy':<28} {acc_before*100:>14.2f}% {acc_after*100:>14.2f}% {'(unchanged)':>10}")
    print(f"  {'ECE (lower better)':<28} {cal_before['ece']:>15.6f} {cal_after['ece']:>15.6f} {cal_after['ece']-cal_before['ece']:>+10.6f}")
    print(f"  {'MCE (lower better)':<28} {cal_before['mce']:>15.6f} {cal_after['mce']:>15.6f} {cal_after['mce']-cal_before['mce']:>+10.6f}")
    print(f"  {'NLL (lower better)':<28} {nll_test_before:>15.6f} {nll_test_after:>15.6f} {nll_test_after-nll_test_before:>+10.6f}")
    print(f"  {'Brier Score (lower better)':<28} {brier_before:>15.6f} {brier_after:>15.6f} {brier_after-brier_before:>+10.6f}")
    print(f"\n  Mean confidence before: {probs_before.max(axis=1).mean():.4f}")
    print(f"  Mean confidence after : {probs_after.max(axis=1).mean():.4f}")
    print(f"{'='*w}\n")

    # Step 6: Per-class breakdown
    print("  Per-class breakdown (Test set):")
    for c, cls_name in enumerate(LABEL_NAMES):
        mask = (test_labels_np == c)
        if mask.sum() == 0: continue
        acc_c_b = (probs_before[mask].argmax(axis=1) == c).mean()
        conf_b  = probs_before[mask].max(axis=1).mean()
        conf_a  = probs_after[mask].max(axis=1).mean()
        print(f"    {cls_name:<25}: Acc={acc_c_b*100:.1f}% | "
              f"Conf_before={conf_b:.3f} -> Conf_after={conf_a:.3f}")
    print()

    # Step 7: Save per-sample CSV
    rows = []
    for i, global_idx in enumerate(ts_i):
        true_c = test_labels_np[i]
        pred_b = probs_before[i].argmax()
        pred_a = probs_after[i].argmax()
        rows.append({
            "dataset_idx":        global_idx,
            "true_label":         LABEL_NAMES[true_c],
            "pred_before":        LABEL_NAMES[pred_b],
            "pred_after":         LABEL_NAMES[pred_a],
            "correct_before":     bool(pred_b == true_c),
            "correct_after":      bool(pred_a == true_c),
            "conf_before":        float(probs_before[i].max()),
            "conf_after":         float(probs_after[i].max()),
            "prob_severe_before": float(probs_before[i][0]),
            "prob_human_before":  float(probs_before[i][1]),
            "prob_affect_before": float(probs_before[i][2]),
            "prob_severe_after":  float(probs_after[i][0]),
            "prob_human_after":   float(probs_after[i][1]),
            "prob_affect_after":  float(probs_after[i][2]),
            "T_star":             float(T_star),
        })

    df_results = pd.DataFrame(rows)
    csv_path   = os.path.join(args.output_dir, "calibration_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[+] Per-sample CSV saved: {csv_path}")

    # Save aggregate metrics CSV
    metrics_path = os.path.join(args.output_dir, "calibration_metrics.csv")
    pd.DataFrame([{
        "T_star":           T_star,
        "ece_before":       cal_before["ece"],
        "ece_after":        cal_after["ece"],
        "mce_before":       cal_before["mce"],
        "mce_after":        cal_after["mce"],
        "nll_before":       nll_test_before,
        "nll_after":        nll_test_after,
        "brier_before":     brier_before,
        "brier_after":      brier_after,
        "acc_before":       acc_before,
        "acc_after":        acc_after,
        "mean_conf_before": float(probs_before.max(axis=1).mean()),
        "mean_conf_after":  float(probs_after.max(axis=1).mean()),
        "n_val_samples":    len(vl_i),
        "n_test_samples":   len(ts_i),
        "nll_val_before":   nll_before_val,
        "nll_val_after":    nll_after_val,
    }]).to_csv(metrics_path, index=False)
    print(f"[+] Metrics CSV saved: {metrics_path}")

    # Step 8: Generate Figures
    print("\n[+] Generating calibration figures...")
    save_calibration_figure(
        cal_before, cal_after, T_star,
        probs_before, probs_after, test_labels_np,
        os.path.join(args.output_dir, "calibration_main.png"), split_name="Test"
    )
    save_per_class_reliability(
        probs_before, probs_after, test_labels_np,
        T_star, os.path.join(args.output_dir, "calibration_per_class.png")
    )

    print(f"\n[+] Done! Output files in '{args.output_dir}/':")
    print(f"    - calibration_main.png       (4-panel main figure)")
    print(f"    - calibration_per_class.png  (per-class confidence distribution)")
    print(f"    - calibration_summary.csv    (per-sample predictions)")
    print(f"    - calibration_metrics.csv    (aggregate metrics)")

    print(f"\n[+] To view in Colab:")
    print(f"    from IPython.display import Image, display")
    print(f"    display(Image('{args.output_dir}/calibration_main.png'))")
    print(f"    display(Image('{args.output_dir}/calibration_per_class.png'))")

    print(f"\n[+] To backup & download:")
    print(f"    import shutil, os")
    print(f"    from google.colab import files")
    print(f"    shutil.make_archive('calibration_results', 'zip', '{args.output_dir}')")
    print(f"    os.makedirs('/content/drive/MyDrive/DisasterNet/results', exist_ok=True)")
    print(f"    shutil.copy('calibration_results.zip',")
    print(f"        '/content/drive/MyDrive/DisasterNet/results/calibration_results.zip')")
    print(f"    files.download('calibration_results.zip')")


if __name__ == "__main__":
    main()
