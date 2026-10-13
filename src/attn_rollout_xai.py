"""
DRISHTI-XAI -- Task 2.3: Attention Rollout Visualization
=========================================================
Method: Abnar & Zuidema (2020) - "Quantifying Attention Flow in Transformers"
https://arxiv.org/abs/2005.00928

What this does:
  - ViT-এর 12টি self-attention layer-এর attention matrix গুলো recursively
    multiply করে CLS token থেকে প্রতিটি image patch-এ information flow বের করে।

  Rollout Algorithm:
    A_hat[l] = 0.5 * A[l] + 0.5 * I        (residual connection account)
    Rollout  = A_hat[0] @ A_hat[1] @ ... @ A_hat[L-1]
    CAM      = Rollout[CLS_row, 1:]          (CLS-to-patch row, skip CLS col)
    Reshape  = 14 x 14 → upsample → 224 x 224

  Why better than Grad-CAM:
    - Grad-CAM: only looks at last layer gradients (dark for ViT)
    - Attention Rollout: traces full information flow across all 12 layers

Output per sample (4-panel):
  (a) Original Image
  (b) Attention Rollout Heatmap (14x14 -> 224x224)
  (c) Overlay (image + heatmap)
  (d) Class probability bar chart

Colab Usage (new session - run cells 1,2,3 first):
    !git pull origin main
    !cp /content/drive/MyDrive/DisasterNet/models/best_cls_only.pt /content/DRISHTI-Bn/src/
    %cd /content/DRISHTI-Bn/src
    !python attn_rollout_xai.py --num_samples 21 --per_class 7 \
        --checkpoint best_cls_only.pt --output_dir rollout_outputs
"""

import os, random, argparse, math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModel, ViTModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

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
VIT_GRID    = 14     # ViT-Base/16: 14x14 = 196 patches
VIT_LAYERS  = 12     # ViT-Base has 12 transformer layers
VIT_HEADS   = 12     # 12 attention heads per layer


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

        self.transform_vis = transforms.Compose([transforms.Resize((224, 224))])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img     = Image.open(os.path.join(self.root_dir, row["image_path"])).convert("RGB")
        img_vis = self.transform_vis(img)
        pv      = self.transform(img)
        enc     = self.tokenizer(
            row[self.text_col], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values":  pv,
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "labels":        torch.tensor(row["label_id"], dtype=torch.long),
            "img_vis":       transforms.ToTensor()(img_vis),
            "img_path":      os.path.join(self.root_dir, row["image_path"]),
            "true_label":    LABEL_NAMES[row["label_id"]],
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


# ─── Model (same as Task 1.1 / 2.1 / 2.2) ───────────────────────────────────
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
        # attn_implementation="eager" is REQUIRED so that output_attentions=True works.
        # Default "sdpa" (Flash Attention) does NOT return attention weights.
        self.vision_encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            attn_implementation="eager"
        )
        self.text_encoder   = AutoModel.from_pretrained(BERT_ID)

        vit_t  = _find_lora_targets(self.vision_encoder, ["query","value","q_proj","v_proj"])
        bert_t = _find_lora_targets(self.text_encoder,   ["query","value","q_proj","v_proj"])
        self.vision_encoder = get_peft_model(
            self.vision_encoder,
            LoraConfig(r=8, lora_alpha=16, target_modules=vit_t, lora_dropout=0.1, bias="none"))
        self.text_encoder   = get_peft_model(
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
        self.vocab_projector   = nn.Linear(self.d_model, vocab_size)

    def forward_cls(self, pixel_values, input_ids, attention_mask):
        vis = self.vision_encoder(pixel_values=pixel_values).pooler_output
        txt = self.text_encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([vis, txt], dim=1))

    def get_vit_all_attentions(self, pixel_values):
        """
        ViT-এর 12টি layer-এর সব self-attention matrix return করে।
        Output: list of 12 tensors, each [B, 12_heads, 197, 197]
        (197 = 1 CLS + 196 patches)

        HuggingFace ViT Model-এ output_attentions=True দিলে
        encoder.layer[i].attention.attention-এর output_weights পাওয়া যায়।
        """
        # LoRA-wrapped model: access base_model.model
        base_vit = self.vision_encoder.base_model.model

        outputs = base_vit(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True
        )
        # outputs.attentions: tuple of 12 tensors [B, num_heads, seq_len, seq_len]
        return outputs.attentions   # tuple of 12 x [B, 12, 197, 197]


# ─── Attention Rollout Algorithm ─────────────────────────────────────────────
def compute_attention_rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    """
    Abnar & Zuidema (2020) Attention Rollout implementation.

    Args:
        attentions   : tuple/list of L tensors, each [1, H, N, N]
                       L = num layers (12), H = num heads (12), N = seq_len (197)
        discard_ratio: low attention values fraction to zero out (noise reduction)
        head_fusion  : how to combine heads — 'mean', 'max', or 'min'

    Returns:
        rollout_cam  : np.ndarray [224, 224] — normalized heatmap
    """
    # All matrices in one stack: [L, H, N, N]
    attn_stack = torch.stack([a.squeeze(0) for a in attentions], dim=0)
    L, H, N, _ = attn_stack.shape   # 12, 12, 197, 197

    # Head fusion
    if head_fusion == "mean":
        attn_fused = attn_stack.mean(dim=1)      # [L, N, N]
    elif head_fusion == "max":
        attn_fused = attn_stack.max(dim=1).values
    elif head_fusion == "min":
        attn_fused = attn_stack.min(dim=1).values
    else:
        raise ValueError(f"Unknown head_fusion: {head_fusion}")

    # Discard ratio: set bottom `discard_ratio` attention values to 0
    flat       = attn_fused.view(L, -1)
    threshold  = flat.quantile(discard_ratio, dim=1, keepdim=True).unsqueeze(-1)
    attn_fused = torch.where(attn_fused < threshold, torch.zeros_like(attn_fused), attn_fused)

    # Re-normalize each row to sum to 1
    attn_fused = attn_fused / (attn_fused.sum(dim=-1, keepdim=True) + 1e-8)

    # Add residual connection (identity): A_hat = 0.5*A + 0.5*I
    eye     = torch.eye(N, device=attn_fused.device).unsqueeze(0)  # [1, N, N]
    A_hat   = 0.5 * attn_fused + 0.5 * eye                         # [L, N, N]

    # Rollout: multiply all layer matrices recursively
    rollout = A_hat[0]
    for layer_idx in range(1, L):
        rollout = A_hat[layer_idx] @ rollout          # [N, N]

    # CLS token row (index 0) attends to all patches (indices 1:)
    cls_attn = rollout[0, 1:]           # [196]

    # Normalize to [0, 1]
    c_min, c_max = cls_attn.min(), cls_attn.max()
    if (c_max - c_min) > 1e-8:
        cls_attn = (cls_attn - c_min) / (c_max - c_min)
    else:
        cls_attn = torch.zeros_like(cls_attn)

    # Reshape 196 → 14×14, then upsample to 224×224
    cam_2d  = cls_attn.reshape(VIT_GRID, VIT_GRID).cpu().numpy()
    cam_pil = Image.fromarray((cam_2d * 255).astype(np.uint8))
    cam_up  = np.array(cam_pil.resize((224, 224), Image.BILINEAR)) / 255.0

    return cam_up


# ─── Visualization ───────────────────────────────────────────────────────────
def save_rollout_figure(img_vis_tensor, cam, pred_class, true_class,
                        probs, save_path, sample_idx, head_fusion="mean"):
    img_np = img_vis_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    try:
        cmap = matplotlib.colormaps['hot']
    except AttributeError:
        cmap = cm.get_cmap('hot')
    heatmap = cmap(cam)[:, :, :3]
    overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    correct   = (pred_class == true_class)
    status    = "CORRECT" if correct else "WRONG"
    color     = "green" if correct else "red"

    title = (f"Sample #{sample_idx:03d} | "
             f"True: {LABEL_NAMES[true_class]} | "
             f"Pred: {LABEL_NAMES[pred_class]} ({status}) | "
             f"Head fusion: {head_fusion}")
    fig.suptitle(title, fontsize=9, fontweight='bold', color=color, y=1.02)

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=9)
    axes[0].axis('off')

    im = axes[1].imshow(cam, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f"Attention Rollout\n(CLS->patch flow, {VIT_LAYERS} layers)", fontsize=9)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Image + Rollout)", fontsize=9)
    axes[2].axis('off')

    colors = ['#2ecc71' if i == pred_class else
              '#3498db' if i == true_class else '#bdc3c7'
              for i in range(len(LABEL_NAMES))]
    bars = axes[3].barh(LABEL_NAMES, probs * 100, color=colors)
    axes[3].set_xlim(0, 100)
    axes[3].set_xlabel("Confidence (%)", fontsize=8)
    axes[3].set_title("Class Probabilities", fontsize=9)
    for bar, p in zip(bars, probs):
        axes[3].text(bar.get_width() + 0.5,
                     bar.get_y() + bar.get_height() / 2,
                     f'{p*100:.1f}%', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Stratified Sampling ─────────────────────────────────────────────────────
def get_stratified_samples(dataset, ts_indices, n_per_class=7, seed=42):
    rng       = random.Random(seed)
    class_idx = {c: [] for c in range(len(LABEL_NAMES))}
    for i in ts_indices:
        lbl = int(dataset[i]["labels"].item())
        class_idx[lbl].append(i)
    samples = []
    for c in range(len(LABEL_NAMES)):
        avail = class_idx[c]; rng.shuffle(avail); samples.extend(avail[:n_per_class])
    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Task 2.3: Attention Rollout Visualization")
    ap.add_argument("--num_samples",   type=int,   default=21)
    ap.add_argument("--per_class",     type=int,   default=7)
    ap.add_argument("--output_dir",    type=str,   default="rollout_outputs")
    ap.add_argument("--checkpoint",    type=str,   default="best_cls_only.pt")
    ap.add_argument("--discard_ratio", type=float, default=0.9,
                    help="Bottom fraction of attn weights to zero out (noise reduction). Default: 0.9")
    ap.add_argument("--head_fusion",   type=str,   default="mean",
                    choices=["mean", "max", "min"],
                    help="How to combine attention heads. Default: mean")
    ap.add_argument("--split",         type=str,   default="test",
                    choices=["train", "val", "test"])
    args = ap.parse_args()

    set_seed(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  DRISHTI-XAI -- Task 2.3: Attention Rollout Visualization")
    print(f"  Method: Abnar & Zuidema (2020)")
    print(f"  Device: {DEVICE} | Split: {args.split}")
    print(f"  Head fusion: {args.head_fusion} | Discard ratio: {args.discard_ratio}")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*65}\n")

    # Tokenizer & Dataset
    print("[+] Loading tokenizer...")
    tokenizer  = AutoTokenizer.from_pretrained(BERT_ID)
    dataset    = MultimodalDataset(CSV_PATH, IMG_DIR, tokenizer)
    _, _, ts_i = get_split_indices(CSV_PATH, RANDOM_SEED)
    print(f"[+] Split: TEST = {len(ts_i)} samples")

    selected = get_stratified_samples(dataset, ts_i, n_per_class=args.per_class)[:args.num_samples]
    print(f"[+] Selected {len(selected)} samples ({args.per_class} per class, stratified)")

    # Build model
    print("[+] Building model...")
    model = DisasterNetAblation(num_classes=3).to(DEVICE)

    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"[+] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt)
        print("[+] Checkpoint loaded successfully")
    else:
        print(f"[!] WARNING: '{args.checkpoint}' not found - using random weights!")

    model.eval()

    # Run
    results = []; correct = 0
    print(f"\n[+] Generating Attention Rollout maps for {len(selected)} samples...\n")

    for rank, idx in enumerate(selected, 1):
        sample   = dataset[idx]
        pv       = sample["pixel_values"].unsqueeze(0).to(DEVICE)
        ids      = sample["input_ids"].unsqueeze(0).to(DEVICE)
        msk      = sample["attention_mask"].unsqueeze(0).to(DEVICE)
        true_c   = sample["labels"].item()
        img_vis  = sample["img_vis"]
        img_path = sample["img_path"]

        try:
            with torch.no_grad():
                # Classification prediction + probs
                logits     = model.forward_cls(pv, ids, msk)
                probs      = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred_c     = int(logits.argmax(dim=1).item())

                # ViT self-attention weights from all 12 layers
                attentions = model.get_vit_all_attentions(pv)

            # Compute rollout map
            cam = compute_attention_rollout(
                attentions,
                discard_ratio=args.discard_ratio,
                head_fusion=args.head_fusion
            )

        except Exception as e:
            print(f"  [!] Sample {idx} SKIPPED -- {e}")
            import traceback; traceback.print_exc()
            continue

        is_correct = (pred_c == true_c)
        if is_correct: correct += 1

        fname = (f"sample_{rank:03d}_"
                 f"true_{LABEL_NAMES[true_c][:8]}_"
                 f"pred_{LABEL_NAMES[pred_c][:8]}"
                 f"{'_OK' if is_correct else '_ERR'}.png")
        save_rollout_figure(img_vis, cam, pred_c, true_c, probs,
                            os.path.join(args.output_dir, fname),
                            rank, args.head_fusion)

        results.append({
            "sample_rank":   rank,
            "dataset_idx":   idx,
            "image_path":    img_path,
            "true_label":    LABEL_NAMES[true_c],
            "pred_label":    LABEL_NAMES[pred_c],
            "correct":       is_correct,
            "conf_severe":   float(probs[0]),
            "conf_human":    float(probs[1]),
            "conf_affected": float(probs[2]),
            "output_file":   fname,
        })

        status = "+" if is_correct else "x"
        print(f"  [{status}] #{rank:03d} | True: {LABEL_NAMES[true_c]:<22} "
              f"| Pred: {LABEL_NAMES[pred_c]:<22} "
              f"| Conf: {probs[pred_c]*100:.1f}% -> {fname}")

    # Save summary
    df_res   = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "rollout_summary.csv")
    df_res.to_csv(csv_path, index=False)

    acc = correct / len(results) * 100 if results else 0
    w   = 65
    print(f"\n{'='*w}")
    print(f"  ATTENTION ROLLOUT SUMMARY")
    print(f"{'='*w}")
    print(f"  Method          : Abnar & Zuidema (2020) Attention Rollout")
    print(f"  Head fusion     : {args.head_fusion} | Discard ratio: {args.discard_ratio}")
    print(f"  Total samples   : {len(results)}")
    print(f"  Correct preds   : {correct} / {len(results)} ({acc:.1f}%)")
    print(f"  Output dir      : {args.output_dir}/")
    if not df_res.empty:
        print(f"\n  Per-class correct predictions:")
        for cls in LABEL_NAMES:
            sub  = df_res[df_res["true_label"] == cls]
            n_ok = sub["correct"].sum()
            print(f"    {cls:<25}: {n_ok}/{len(sub)} correct")
    print(f"{'='*w}\n")

    print("[+] Done! To view in Colab:")
    print("    from IPython.display import Image, display")
    print("    import glob")
    print(f"    for p in sorted(glob.glob('{args.output_dir}/*.png'))[:5]:")
    print("        display(Image(p))")

    print(f"\n[+] To backup to Drive:")
    print(f"    import shutil, os")
    print(f"    shutil.make_archive('rollout_results', 'zip', '{args.output_dir}')")
    print(f"    os.makedirs('/content/drive/MyDrive/DisasterNet/results', exist_ok=True)")
    print(f"    shutil.copy('rollout_results.zip',")
    print(f"        '/content/drive/MyDrive/DisasterNet/results/rollout_results.zip')")


if __name__ == "__main__":
    main()
