"""
DRISHTI-XAI -- Task 1.2: Ablation -- Caption Only (lambda_cls = 0)
===================================================================
This ablation study removes the classification loss entirely (lambda_cls = 0),
training the FULL DisasterNet-v2 multimodal architecture (ViT + BanglaBERT +
Cross-Attention Decoder) using ONLY the captioning loss.

After training, we still evaluate classification performance by running
a forward pass through the classification head — even though the head was
never trained. This reveals whether caption-guided representation learning
incidentally captures class-discriminative features.

Research Question:
    Can a model trained ONLY with caption loss still classify disasters?
    i.e., do caption representations encode class information implicitly?

Loss equation:
    Full  model: L = lambda_cls * L_cls + lambda_cap * L_cap  (lambda_cap=0.5)
    THIS script: L = 0 * L_cls + 1.0 * L_cap                 (lambda_cls=0.0)

Architecture: IDENTICAL to DisasterNet-v2 (ViT + BanglaBERT + LoRA + Decoder)
              Only the training objective changes.

Comparison targets:
    Task 0.3: DisasterNet-v2  (lambda_cap=0.5)  -> Macro F1 = 77.95%
    Task 1.1: CLS-Only ablation (lambda_cap=0)  -> Macro F1 = 83.21%
    Task 1.2: CAP-Only ablation (lambda_cls=0)  -> ???

Expected runtime: ~35-45 min on T4 GPU (10 epochs).

Usage:
    python caption_only_ablation.py --epochs 10 --batch_size 8
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
BATCH_SIZE  = 8   # smaller: decoder needs more memory
LR          = 1e-4
MAX_LEN     = 128
VOCAB_SIZE  = 32000

# THE KEY ABLATION PARAMETER
LAMBDA_CLS  = 0.0  # <-- This is the ablation: classification loss = OFF
LAMBDA_CAP  = 1.0

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


# ---------- Dataset (Multimodal: image + text) --------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, max_len=MAX_LEN):
        self.df       = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer= tokenizer
        self.max_len  = max_len

        # Detect caption column
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
        self.df["label_id"] = self.df["macro_label"].map(LABEL_MAP)

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
        input_ids     = enc["input_ids"].squeeze(0)
        attention_mask= enc["attention_mask"].squeeze(0)

        # Decoder targets: shift right (standard teacher-forcing)
        # input to decoder: [BOS, tok1, tok2, ..., tok_{N-1}]
        # target:           [tok1, tok2, ..., tok_{N-1}, EOS]
        # We approximate by using input_ids directly as both src and tgt
        # (standard for seq2seq cross-entropy with teacher forcing)
        tgt_input = input_ids[:-1].clone()   # decoder input  (shift right)
        tgt_label = input_ids[1:].clone()    # decoder target (shift left)

        return {
            "pixel_values":  pixel_values,
            "input_ids":     input_ids,
            "attention_mask":attention_mask,
            "tgt_input":     tgt_input,
            "tgt_label":     tgt_label,
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


# ---------- LoRA target detection (works across HF model versions) ------------
def _find_lora_targets(model, candidates):
    """Return the subset of candidate names that actually exist as
    named modules in *model*.  Raises if none are found."""
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


# ---------- Model (DisasterNet-v2 full architecture) -------------------------
class DisasterNetAblation(nn.Module):
    """
    Full DisasterNet-v2 architecture.
    Task 1.2: ONLY caption loss is used (lambda_cls=0).
    Classification head is still present but NOT trained.
    At eval time, we test if caption-trained representations can classify.
    """

    def __init__(self, num_classes=3, vocab_size=VOCAB_SIZE):
        super().__init__()
        # --- Encoders ---
        self.vision_encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k")
        self.text_encoder   = AutoModel.from_pretrained(BERT_ID)

        # LoRA (same config as DisasterNet-v2)
        print("[+] Injecting LoRA adapters...")

        # Detect actual attention module names in ViT/BERT
        vit_target  = _find_lora_targets(self.vision_encoder,
                                         ["query", "value", "q_proj", "v_proj"])
        bert_target = _find_lora_targets(self.text_encoder,
                                         ["query", "value", "q_proj", "v_proj"])
        print(f"    ViT  LoRA targets : {vit_target}")
        print(f"    BERT LoRA targets : {bert_target}")

        vit_lora = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=vit_target,
            lora_dropout=0.1, bias="none"
        )
        bert_lora = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=bert_target,
            lora_dropout=0.1, bias="none"
        )
        self.vision_encoder = get_peft_model(self.vision_encoder, vit_lora)
        self.text_encoder   = get_peft_model(self.text_encoder,   bert_lora)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[+] LoRA partial FT -- Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.2f}%)")

        # --- Fusion Classifier (identical to model.py) ---
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # --- Caption Decoder ---
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

    def _encode(self, pixel_values, input_ids, attention_mask):
        """Shared encoder forward — used by both cls and cap heads."""
        vis_out = self.vision_encoder(pixel_values=pixel_values)
        vis_seq = vis_out.last_hidden_state          # (B, 197, 768) patch tokens
        vis_cls = vis_out.pooler_output              # (B, 768)

        txt_out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        txt_cls = txt_out[:, 0, :]                  # (B, 768) [CLS]
        return vis_seq, vis_cls, txt_cls

    def forward_cls(self, pixel_values, input_ids, attention_mask):
        """Classification head only."""
        _, vis_cls, txt_cls = self._encode(pixel_values, input_ids, attention_mask)
        return self.classifier(torch.cat([vis_cls, txt_cls], dim=1))

    def forward_cap(self, pixel_values, input_ids, attention_mask, tgt_input):
        """
        Caption decoder forward.
        Uses ViT patch tokens (vis_seq) as memory for cross-attention.
        """
        vis_seq, _, _ = self._encode(pixel_values, input_ids, attention_mask)

        tgt_emb = self.pos_encoder(
            self.decoder_embedding(tgt_input)   # (B, T-1, 768)
        )
        T = tgt_emb.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=pixel_values.device
        )
        dec_out = self.caption_decoder(
            tgt_emb, vis_seq, tgt_mask=causal_mask, tgt_is_causal=True
        )
        return self.vocab_projector(dec_out)   # (B, T-1, vocab_size)


# ---------- Metrics ----------------------------------------------------------
def report(y_true, y_pred, f1_task11=83.21):
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    mp  = precision_score(y_true, y_pred, average="macro", zero_division=0)*100
    mr  = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    pf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    pp  = precision_score(y_true, y_pred, average=None, zero_division=0)
    pr  = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    w = 62
    print(f"\n{'='*w}")
    print(f"  Task 1.2 -- CAP-Only Ablation (TEST SET)")
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
def train_epoch(model, loader, opt, crit_cap, scaler, vocab_size):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        pv  = batch["pixel_values"].to(DEVICE)
        ids = batch["input_ids"].to(DEVICE)
        msk = batch["attention_mask"].to(DEVICE)
        tgt_in  = batch["tgt_input"].to(DEVICE)
        tgt_lbl = batch["tgt_label"].to(DEVICE)

        opt.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_cap = model.forward_cap(pv, ids, msk, tgt_in)
                # logits_cap: (B, T-1, vocab_size)
                loss = LAMBDA_CAP * crit_cap(
                    logits_cap.reshape(-1, vocab_size),
                    tgt_lbl.reshape(-1)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            logits_cap = model.forward_cap(pv, ids, msk, tgt_in)
            loss = LAMBDA_CAP * crit_cap(
                logits_cap.reshape(-1, vocab_size),
                tgt_lbl.reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total_loss += loss.item() * pv.size(0)
        n          += pv.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate_cls(model, loader):
    """Evaluate classification performance WITHOUT having trained cls head."""
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


@torch.no_grad()
def compute_val_cap_loss(model, loader, crit_cap, vocab_size):
    """Compute validation captioning loss for checkpoint selection."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        pv      = batch["pixel_values"].to(DEVICE)
        ids     = batch["input_ids"].to(DEVICE)
        msk     = batch["attention_mask"].to(DEVICE)
        tgt_in  = batch["tgt_input"].to(DEVICE)
        tgt_lbl = batch["tgt_label"].to(DEVICE)
        logits_cap = model.forward_cap(pv, ids, msk, tgt_in)
        loss = crit_cap(logits_cap.reshape(-1, vocab_size),
                        tgt_lbl.reshape(-1))
        total += loss.item() * pv.size(0)
        n     += pv.size(0)
    return total / n


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
    print(f"  DRISHTI-XAI -- Task 1.2: Ablation -- CAP-Only (λ_cls=0)")
    print(f"  Architecture: DisasterNet-v2 (ViT+BanglaBERT+LoRA)")
    print(f"  Loss: L = {LAMBDA_CLS}*L_cls + {LAMBDA_CAP}*L_cap  (cls DISABLED)")
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

    model    = DisasterNetAblation(num_classes=3).to(DEVICE)
    crit_cap = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    opt      = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss, best_ep, best_st = float("inf"), 0, None
    print(f"\n[+] Training {args.epochs} epochs (λ_cls=0, cap ONLY, AMP={use_amp})...\n")

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tr_ld, opt, crit_cap, scaler, VOCAB_SIZE)
        sch.step()
        vl_loss = compute_val_cap_loss(model, vl_ld, crit_cap, VOCAB_SIZE)
        print(f"  Epoch {ep:02d}/{args.epochs} | "
              f"Train Cap-Loss={tr_loss:.4f} | Val Cap-Loss={vl_loss:.4f}")
        if vl_loss < best_loss:
            best_loss, best_ep = vl_loss, ep
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[+] Best checkpoint: Epoch {best_ep} (Val Cap-Loss={best_loss:.4f})")
    model.load_state_dict(best_st); model.to(DEVICE)

    # Evaluate CLS performance of caption-only trained model
    tt, tp = evaluate_cls(model, ts_ld)
    f1 = report(tt, tp)

    w = 62
    print(f"\n{'='*w}")
    print(f"  ABLATION SUMMARY -- λ_cls vs λ_cap")
    print(f"{'='*w}")
    print(f"  {'Method':<44} {'Macro F1':>8}")
    print(f"  {'-'*44} {'-'*8}")
    print(f"  {'BanglaBERT-Only (no image)':<44} {'43.02%':>8}  <- Task 0.6")
    print(f"  {'DisasterNet-v2 (λ_cls=1.0, λ_cap=0.5)':<44} {'77.95%':>8}  <- Task 0.3")
    print(f"  {'ViT-Only (no text)':<44} {'78.64%':>8}  <- Task 0.5")
    print(f"  {'CLS-Only Ablation (λ_cls=1.0, λ_cap=0)':<44} {'83.21%':>8}  <- Task 1.1")
    print(f"  {'CAP-Only Ablation (λ_cls=0, λ_cap=1.0)':<44} {f1:>7.2f}%  <- Task 1.2")
    print(f"{'='*w}")

    if f1 > 50.0:
        print(f"\n[!] FINDING: Caption representations DO encode class info "
              f"(CLS F1={f1:.2f}% > 39.95% stratified baseline)")
    else:
        print(f"\n[!] FINDING: Caption-only training does NOT produce "
              f"class-discriminative features (CLS F1={f1:.2f}%)")
    print()


if __name__ == "__main__":
    main()
