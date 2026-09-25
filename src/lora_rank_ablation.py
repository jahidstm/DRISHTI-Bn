"""
DRISHTI-XAI -- Task 1.5: Ablation -- LoRA Rank Sensitivity
===========================================================
This ablation sweeps LoRA rank r ∈ {4, 8, 16} to measure sensitivity.

Research Question:
    Is performance sensitive to LoRA rank?
    Which rank gives the best accuracy / parameter tradeoff?

Configuration:
    r=4:  trainable params ≈ 295K  (0.15%)
    r=8:  trainable params ≈ 590K  (0.30%)  <- Task 1.1 baseline
    r=16: trainable params ≈ 1.18M (0.60%)

Loss equation (same as Task 1.1 — CLS only):
    L = 1.0 * L_cls + 0.0 * L_cap

Expected runtime: ~90-120 min on T4 GPU (3 x 10 epochs).

Usage:
    python lora_rank_ablation.py --epochs 10 --batch_size 16
"""

import os, random, argparse, math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from transformers import AutoModel, ViTModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score,
                              recall_score, confusion_matrix)

# ---------- Config -----------------------------------------------------------
CSV_PATH    = "../data/processed/master_dataset_translated.csv"
IMG_DIR     = "../data/processed/"
LABEL_NAMES = ["Severe_Damage", "Humanitarian_Rescue", "Affected_People"]
LABEL_MAP   = {n: i for i, n in enumerate(LABEL_NAMES)}
RANDOM_SEED = 42
EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 1e-4
MAX_LEN     = 128
VOCAB_SIZE  = 32000
LORA_RANKS  = [4, 8, 16]

LAMBDA_CLS  = 1.0
LAMBDA_CAP  = 0.0

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_ID     = "csebuetnlp/banglabert"


# ---------- Reproducibility --------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[+] Random seed fixed: {seed} (deterministic mode ON)")

def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws); random.seed(ws)


# ---------- Dataset ----------------------------------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, max_len=MAX_LEN):
        self.df        = pd.read_csv(csv_file)
        self.root_dir  = root_dir
        self.tokenizer = tokenizer
        self.max_len   = max_len

        for col in ["bangla_caption", "bengali_caption", "caption_bn",
                    "caption", "text"]:
            if col in self.df.columns:
                self.text_col = col
                break
        else:
            raise ValueError(f"No caption column found. Available: "
                             f"{list(self.df.columns)}")
        print(f"[+] Using text column: '{self.text_col}'")
        self.df[self.text_col] = self.df[self.text_col].fillna("").astype(str)
        self.df["label_id"]    = self.df["macro_label"].map(LABEL_MAP)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(
            os.path.join(self.root_dir, row["image_path"])
        ).convert("RGB")
        pixel_values = self.transform(img)
        enc = self.tokenizer(
            row[self.text_col],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pixel_values":  pixel_values,
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "labels":        torch.tensor(row["label_id"], dtype=torch.long)
        }


def get_split_indices(csv_path, rs=42):
    df   = pd.read_csv(csv_path)
    lbls = df["macro_label"].values
    idx  = list(range(len(df)))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=lbls,
                               random_state=rs)
    vl, ts  = train_test_split(tmp, test_size=0.5, stratify=lbls[tmp],
                               random_state=rs)
    return tr, vl, ts


# ---------- LoRA target detection --------------------------------------------
def _find_lora_targets(model, candidates):
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    found = [c for c in candidates if c in names]
    if not found:
        raise ValueError(
            f"None of {candidates} found in model. "
            f"Available leaf names (sample): {list(names)[:20]}"
        )
    return found


# ---------- Positional Encoding ----------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ---------- Model ------------------------------------------------------------
class DisasterNetLoRA(nn.Module):
    """DisasterNet-v2 CLS-only with configurable LoRA rank."""

    def __init__(self, lora_rank, num_classes=3, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k")
        self.text_encoder   = AutoModel.from_pretrained(BERT_ID)

        vit_target  = _find_lora_targets(self.vision_encoder,
                                         ["query", "value", "q_proj", "v_proj"])
        bert_target = _find_lora_targets(self.text_encoder,
                                         ["query", "value", "q_proj", "v_proj"])

        vit_lora  = LoraConfig(r=lora_rank, lora_alpha=lora_rank*2,
                               target_modules=vit_target,
                               lora_dropout=0.1, bias="none")
        bert_lora = LoraConfig(r=lora_rank, lora_alpha=lora_rank*2,
                               target_modules=bert_target,
                               lora_dropout=0.1, bias="none")
        self.vision_encoder = get_peft_model(self.vision_encoder, vit_lora)
        self.text_encoder   = get_peft_model(self.text_encoder,   bert_lora)

        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Decoder (present but unused in CLS-only training)
        self.d_model          = 768
        self.decoder_embedding= nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder      = PositionalEncoding(self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.caption_decoder  = nn.TransformerDecoder(decoder_layer,
                                                       num_layers=3)
        self.vocab_projector  = nn.Linear(self.d_model, vocab_size)

    def forward_cls(self, pixel_values, input_ids, attention_mask):
        vis  = self.vision_encoder(pixel_values=pixel_values).pooler_output
        txt  = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([vis, txt], dim=1))


# ---------- Metrics ----------------------------------------------------------
def report(y_true, y_pred, rank):
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    mp  = precision_score(y_true, y_pred, average="macro", zero_division=0)*100
    mr  = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    pf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    pp  = precision_score(y_true, y_pred, average=None, zero_division=0)
    pr  = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    w = 62
    print(f"\n{'='*w}")
    print(f"  Task 1.5 -- LoRA r={rank} (TEST SET)")
    print(f"{'='*w}")
    print(f"  Macro F1        : {mf1:.2f}%")
    print(f"  Macro Precision : {mp:.2f}%")
    print(f"  Macro Recall    : {mr:.2f}%")
    print("\n  Per-class breakdown:")
    for i, n in enumerate(LABEL_NAMES):
        print(f"    [{n:<22}] P={pp[i]:.4f}  R={pr[i]:.4f}  F1={pf1[i]:.4f}")
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print("  " + " "*16 + "".join(f"  {n[:8]:>8}" for n in LABEL_NAMES))
    rl = ["Severe_Damag", "Humanitarian", "Affected_Peo"]
    for i, r in enumerate(rl):
        print(f"  {r:<14}  " + "  ".join(f"{cm[i][j]:>8}" for j in range(3)))
    print(f"{'='*w}")
    return mf1


# ---------- Train / Eval -----------------------------------------------------
def train_epoch(model, loader, opt, crit_cls, scaler):
    model.train()
    tl, nc, nt = 0.0, 0, 0
    for batch in loader:
        pv  = batch["pixel_values"].to(DEVICE)
        ids = batch["input_ids"].to(DEVICE)
        msk = batch["attention_mask"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)
        opt.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model.forward_cls(pv, ids, msk)
                loss   = crit_cls(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            logits = model.forward_cls(pv, ids, msk)
            loss   = crit_cls(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tl += loss.item() * pv.size(0)
        nc += (logits.argmax(1) == lbl).sum().item()
        nt += pv.size(0)
    return tl / nt, nc / nt


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        pv  = batch["pixel_values"].to(DEVICE)
        ids = batch["input_ids"].to(DEVICE)
        msk = batch["attention_mask"].to(DEVICE)
        logits = model.forward_cls(pv, ids, msk)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
    return np.array(trues), np.array(preds)


# ---------- Run one rank experiment ------------------------------------------
def run_rank(rank, epochs, batch_size, lr, use_amp, tr_i, vl_i, ts_i,
             dataset, tokenizer):
    set_seed(RANDOM_SEED)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\n{'*'*62}")
    print(f"  LoRA rank r={rank}  (lora_alpha={rank*2})")
    print(f"{'*'*62}")

    g = torch.Generator(); g.manual_seed(RANDOM_SEED)
    tr_ld = DataLoader(Subset(dataset, tr_i), batch_size=batch_size,
                       shuffle=True, num_workers=2, pin_memory=True,
                       worker_init_fn=seed_worker, generator=g)
    vl_ld = DataLoader(Subset(dataset, vl_i), batch_size=batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)
    ts_ld = DataLoader(Subset(dataset, ts_i), batch_size=batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)

    model = DisasterNetLoRA(lora_rank=rank, num_classes=3).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[+] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    crit_cls = nn.CrossEntropyLoss()
    opt      = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, best_ep, best_st = 0.0, 0, None
    for ep in range(1, epochs + 1):
        loss, acc = train_epoch(model, tr_ld, opt, crit_cls, scaler)
        sch.step()
        vt, vp = evaluate(model, vl_ld)
        vf1 = f1_score(vt, vp, average="macro", zero_division=0) * 100
        print(f"  Ep {ep:02d}/{epochs} | Loss={loss:.4f} | "
              f"Acc={acc*100:.2f}% | Val F1={vf1:.2f}%")
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[+] Best Epoch={best_ep} | Val F1={best_f1:.2f}%")
    model.load_state_dict(best_st); model.to(DEVICE)
    tt, tp = evaluate(model, ts_ld)
    test_f1 = report(tt, tp, rank)
    return test_f1, trainable


# ---------- Main -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=EPOCHS)
    ap.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--no_amp",     action="store_true")
    ap.add_argument("--ranks",      type=int, nargs="+", default=LORA_RANKS)
    args = ap.parse_args()

    set_seed(RANDOM_SEED)
    use_amp = (not args.no_amp) and (DEVICE.type == "cuda")

    print(f"\n{'='*62}")
    print(f"  DRISHTI-XAI -- Task 1.5: LoRA Rank Sensitivity")
    print(f"  Ranks: {args.ranks}  |  Epochs: {args.epochs}")
    print(f"  Device: {DEVICE}  |  Seed: {RANDOM_SEED}  |  AMP: {use_amp}")
    print(f"{'='*62}\n")

    print("[+] Loading tokenizer (csebuetnlp/banglabert)...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)
    dataset   = MultimodalDataset(CSV_PATH, IMG_DIR, tokenizer)
    tr_i, vl_i, ts_i = get_split_indices(CSV_PATH, RANDOM_SEED)
    print(f"[+] Split: Train={len(tr_i)} | Val={len(vl_i)} | Test={len(ts_i)}")

    results = {}
    for rank in args.ranks:
        f1, n_params = run_rank(rank, args.epochs, args.batch_size, args.lr,
                                use_amp, tr_i, vl_i, ts_i, dataset, tokenizer)
        results[rank] = (f1, n_params)

    # Final summary
    w = 66
    print(f"\n{'='*w}")
    print(f"  TASK 1.5 SUMMARY -- LoRA Rank Sensitivity")
    print(f"{'='*w}")
    print(f"  {'Rank':<8} {'Trainable Params':<20} {'% of Total':<14} {'Macro F1':>10}")
    print(f"  {'-'*8} {'-'*20} {'-'*14} {'-'*10}")
    total_params = 197_005_824
    for rank, (f1, n_params) in results.items():
        pct = 100 * n_params / total_params
        marker = " <- Task 1.1" if rank == 8 else ""
        print(f"  r={rank:<6} {n_params:<20,} {pct:<14.2f} {f1:>9.2f}%{marker}")
    print(f"{'='*w}")

    best_rank = max(results, key=lambda r: results[r][0])
    best_f1   = results[best_rank][0]
    print(f"\n[!] Best rank: r={best_rank} (Macro F1={best_f1:.2f}%)")

    if all(abs(results[r][0] - results[8][0]) < 1.0 for r in results if r != 8):
        print(f"[!] FINDING: Performance is ROBUST to LoRA rank — "
              f"all ranks within ±1pp of r=8.")
    else:
        print(f"[!] FINDING: Performance IS sensitive to LoRA rank — "
              f"r={best_rank} significantly outperforms others.")
    print()


if __name__ == "__main__":
    main()
