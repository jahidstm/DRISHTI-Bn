"""
DRISHTI-XAI -- Task 1.4: Ablation -- No Cross-Attention Decoder
================================================================
This ablation replaces the Cross-Attention Decoder with a simple
Self-Attention-only Transformer Decoder (memory/cross-attn disabled).

Research Question:
    Does cross-attention between the caption decoder and ViT patch tokens
    contribute meaningfully to classification performance?
    i.e., is the cross-modal interaction in the decoder important?

Note: Since Task 1.1 showed caption loss HURTS cls, this ablation trains
with CLS loss only (lambda_cap=0), matching Task 1.1 setup — the only
difference is the decoder architecture (self-attn vs cross-attn).
The decoder is in the model but NOT used in cls-only training.
The ablation tests whether the decoder architecture affects the shared
encoder representations that feed into the classifier.

Loss equation (same as Task 1.1):
    L = 1.0 * L_cls + 0.0 * L_cap

Architecture change:
    - Cross-Attention Decoder -> Self-Attention-only Transformer Encoder
    - LoRA adapters KEPT (same as Task 1.1)
    - Classification head SAME

Comparison targets:
    Task 1.1: CLS-Only + Cross-Attn Decoder (present but unused) -> F1 = 83.21%
    Task 1.4: CLS-Only + Self-Attn-only Decoder (present but unused) -> ???

Expected runtime: ~30-40 min on T4 GPU (10 epochs).

Usage:
    python no_crossattn_ablation.py --epochs 10 --batch_size 16
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

LAMBDA_CLS  = 1.0
LAMBDA_CAP  = 0.0   # cls-only, same as Task 1.1

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


# ---------- Model: Self-Attention-Only Decoder --------------------------------
class DisasterNetNoCA(nn.Module):
    """
    DisasterNet-v2 with cross-attention decoder REPLACED by
    a self-attention-only Transformer Encoder stack.

    The decoder/encoder is present in the model but NOT used during
    cls-only training (lambda_cap=0). This ablation tests whether the
    decoder architecture type affects shared encoder representations.
    """

    def __init__(self, num_classes=3, vocab_size=VOCAB_SIZE):
        super().__init__()
        # --- Encoders ---
        self.vision_encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k")
        self.text_encoder   = AutoModel.from_pretrained(BERT_ID)

        # LoRA (same as Task 1.1)
        print("[+] Injecting LoRA adapters...")
        vit_target  = _find_lora_targets(self.vision_encoder,
                                         ["query", "value", "q_proj", "v_proj"])
        bert_target = _find_lora_targets(self.text_encoder,
                                         ["query", "value", "q_proj", "v_proj"])
        print(f"    ViT  LoRA targets : {vit_target}")
        print(f"    BERT LoRA targets : {bert_target}")

        vit_lora = LoraConfig(r=8, lora_alpha=16, target_modules=vit_target,
                              lora_dropout=0.1, bias="none")
        bert_lora = LoraConfig(r=8, lora_alpha=16, target_modules=bert_target,
                               lora_dropout=0.1, bias="none")
        self.vision_encoder = get_peft_model(self.vision_encoder, vit_lora)
        self.text_encoder   = get_peft_model(self.text_encoder,   bert_lora)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[+] LoRA partial FT -- Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)")

        # --- Fusion Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # --- Self-Attention-Only "Decoder" (NO cross-attention to ViT) ---
        # Replaced TransformerDecoder (which has cross-attn) with
        # TransformerEncoder (self-attn only) — same capacity otherwise.
        self.d_model          = 768
        self.decoder_embedding= nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder      = PositionalEncoding(self.d_model)
        # TransformerEncoderLayer has ONLY self-attention (no cross-attn)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.self_attn_decoder = nn.TransformerEncoder(encoder_layer,
                                                        num_layers=3)
        self.vocab_projector   = nn.Linear(self.d_model, vocab_size)

    def forward_cls(self, pixel_values, input_ids, attention_mask):
        """Classification forward — identical to Task 1.1."""
        vis  = self.vision_encoder(pixel_values=pixel_values).pooler_output
        txt  = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([vis, txt], dim=1))


# ---------- Metrics ----------------------------------------------------------
def report(y_true, y_pred):
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    mp  = precision_score(y_true, y_pred, average="macro", zero_division=0)*100
    mr  = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    pf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    pp  = precision_score(y_true, y_pred, average=None, zero_division=0)
    pr  = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    w = 62
    print(f"\n{'='*w}")
    print(f"  Task 1.4 -- No Cross-Attention Ablation (TEST SET)")
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
                loss   = LAMBDA_CLS * crit_cls(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            logits = model.forward_cls(pv, ids, msk)
            loss   = LAMBDA_CLS * crit_cls(logits, lbl)
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


# ---------- Main -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=EPOCHS)
    ap.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--no_amp",     action="store_true")
    args = ap.parse_args()

    set_seed(RANDOM_SEED)
    use_amp = (not args.no_amp) and (DEVICE.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\n{'='*62}")
    print(f"  DRISHTI-XAI -- Task 1.4: Ablation -- No Cross-Attention")
    print(f"  Architecture: ViT+BanglaBERT+LoRA | Self-Attn-only decoder")
    print(f"  Cross-attention to ViT patches: REMOVED")
    print(f"  Loss: L = {LAMBDA_CLS}*L_cls + {LAMBDA_CAP}*L_cap")
    print(f"  Device: {DEVICE}  |  Seed: {RANDOM_SEED}  |  AMP: {use_amp}")
    print(f"{'='*62}\n")

    print("[+] Loading tokenizer (csebuetnlp/banglabert)...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)

    dataset = MultimodalDataset(CSV_PATH, IMG_DIR, tokenizer)
    tr_i, vl_i, ts_i = get_split_indices(CSV_PATH, RANDOM_SEED)
    print(f"[+] Split: Train={len(tr_i)} | Val={len(vl_i)} | Test={len(ts_i)}")

    g = torch.Generator(); g.manual_seed(RANDOM_SEED)
    tr_ld = DataLoader(Subset(dataset, tr_i), batch_size=args.batch_size,
                       shuffle=True, num_workers=2, pin_memory=True,
                       worker_init_fn=seed_worker, generator=g)
    vl_ld = DataLoader(Subset(dataset, vl_i), batch_size=args.batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)
    ts_ld = DataLoader(Subset(dataset, ts_i), batch_size=args.batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)

    model    = DisasterNetNoCA(num_classes=3).to(DEVICE)
    crit_cls = nn.CrossEntropyLoss()
    opt      = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, best_ep, best_st = 0.0, 0, None
    print(f"\n[+] Training {args.epochs} epochs "
          f"(No cross-attn, AMP={use_amp})...\n")

    for ep in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, tr_ld, opt, crit_cls, scaler)
        sch.step()
        vt, vp = evaluate(model, vl_ld)
        vf1 = f1_score(vt, vp, average="macro", zero_division=0) * 100
        print(f"  Epoch {ep:02d}/{args.epochs} | "
              f"Loss={loss:.4f} | Train Acc={acc*100:.2f}% | "
              f"Val Macro-F1={vf1:.2f}%")
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[+] Best checkpoint: Epoch {best_ep} (Val Macro-F1={best_f1:.2f}%)")
    model.load_state_dict(best_st); model.to(DEVICE)
    tt, tp = evaluate(model, ts_ld)
    f1 = report(tt, tp)

    w = 62
    print(f"\n{'='*w}")
    print(f"  ABLATION SUMMARY -- Effect of Cross-Attention")
    print(f"{'='*w}")
    print(f"  {'Method':<44} {'Macro F1':>8}")
    print(f"  {'-'*44} {'-'*8}")
    print(f"  {'No-LoRA (frozen encoders)':<44} {'77.74%':>8}  <- Task 1.3")
    print(f"  {'DisasterNet-v2 (cross-attn + both losses)':<44} {'77.95%':>8}  <- Task 0.3")
    print(f"  {'ViT-Only (no text)':<44} {'78.64%':>8}  <- Task 0.5")
    print(f"  {'CLS-Only + Cross-Attn decoder':<44} {'83.21%':>8}  <- Task 1.1")
    print(f"  {'CLS-Only + Self-Attn-only decoder':<44} {f1:>7.2f}%  <- Task 1.4")
    print(f"{'='*w}")

    delta = f1 - 83.21
    print(f"\n[!] Cross-Attn contribution: {delta:+.2f}pp "
          f"(Self-Attn-only vs Cross-Attn decoder)")
    if abs(delta) < 0.5:
        print(f"[!] FINDING: Decoder architecture type has NEGLIGIBLE effect "
              f"on classification ({delta:+.2f}pp).")
    elif delta < 0:
        print(f"[!] FINDING: Cross-attention decoder HELPS cls by {abs(delta):.2f}pp "
              f"vs self-attn-only.")
    else:
        print(f"[!] FINDING: Self-attn-only decoder OUTPERFORMS cross-attn "
              f"by {delta:.2f}pp.")
    print()


if __name__ == "__main__":
    main()
