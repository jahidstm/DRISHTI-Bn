"""
DRISHTI-XAI -- Task 2.1: Grad-CAM Visual Explanation
=====================================================
ViT attention layer থেকে Gradient-weighted Class Activation Maps (Grad-CAM)
তৈরি করা — মডেল কোন patch দেখে সিদ্ধান্ত নিচ্ছে তা হিটম্যাপে দেখানো।

Technique: Grad-CAM for Vision Transformers (Chefer et al., 2021 adaptation)
  - ViT-এর last attention layer-এর gradient ও attention weight ব্যবহার করা হয়
  - cls_token → patch attention weights → reshape → overlay on image

Key Design:
  - মডেল হিসেবে Task 1.1-এর CLS-Only architecture (best model) ব্যবহার
  - LoRA weights সহ ViT এর শেষ Transformer block-এর self-attention target করা হবে
  - output: প্রতিটি image-এর জন্য:
      (a) Original image
      (b) Grad-CAM heatmap
      (c) Overlay (image + heatmap)
      (d) Predicted class + confidence

Usage (Colab):
    python gradcam_xai.py --num_samples 20 --output_dir gradcam_outputs

Output folder structure:
    gradcam_outputs/
    ├── sample_001_Severe_Damage_pred_Severe_Damage.png
    ├── sample_002_Humanitarian_Rescue_pred_Humanitarian_Rescue.png
    └── ...
    └── gradcam_summary.csv
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
from torch.utils.data import Dataset, DataLoader, Subset
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
BATCH_SIZE  = 1          # Grad-CAM은 sample별 처리
MAX_LEN     = 128
VOCAB_SIZE  = 32000
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_ID     = "csebuetnlp/banglabert"

# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[+] Random seed fixed: {seed}")

# ─── Dataset ─────────────────────────────────────────────────────────────────
class MultimodalDataset(Dataset):
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

        # Grad-CAM visualization-এর জন্য unnormalized image সংরক্ষণ
        self.transform_vis = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        img      = Image.open(img_path).convert("RGB")
        img_vis  = self.transform_vis(img)   # PIL for visualization
        pixel_values = self.transform(img)

        enc = self.tokenizer(
            row[self.text_col],
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values":  pixel_values,
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "labels":        torch.tensor(row["label_id"], dtype=torch.long),
            "img_vis":       transforms.ToTensor()(img_vis),
            "img_path":      img_path,
            "true_label":    LABEL_NAMES[row["label_id"]],
        }

def get_split_indices(csv_path, rs=42):
    df   = pd.read_csv(csv_path)
    lbls = df["macro_label"].values
    idx  = list(range(len(df)))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=lbls, random_state=rs)
    vl, ts  = train_test_split(tmp, test_size=0.5, stratify=lbls[tmp], random_state=rs)
    return tr, vl, ts

# ─── LoRA target detection ────────────────────────────────────────────────────
def _find_lora_targets(model, candidates):
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    found = [c for c in candidates if c in names]
    if not found:
        raise ValueError(f"None of {candidates} found.")
    return found

# ─── Model (Task 1.1 architecture) ───────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DisasterNetAblation(nn.Module):
    """Task 1.1 CLS-Only architecture — best performing model (F1=83.21%)."""
    def __init__(self, num_classes=3, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k")
        self.text_encoder   = AutoModel.from_pretrained(BERT_ID)

        vit_t  = _find_lora_targets(self.vision_encoder, ["query","value","q_proj","v_proj"])
        bert_t = _find_lora_targets(self.text_encoder,   ["query","value","q_proj","v_proj"])
        self.vision_encoder = get_peft_model(
            self.vision_encoder,
            LoraConfig(r=8, lora_alpha=16, target_modules=vit_t,
                       lora_dropout=0.1, bias="none"))
        self.text_encoder   = get_peft_model(
            self.text_encoder,
            LoraConfig(r=8, lora_alpha=16, target_modules=bert_t,
                       lora_dropout=0.1, bias="none"))

        self.classifier = nn.Sequential(
            nn.Linear(768+768, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
        self.d_model           = 768
        self.decoder_embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder       = PositionalEncoding(self.d_model)
        self.caption_decoder   = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8,
                                       dim_feedforward=2048,
                                       dropout=0.1, batch_first=True),
            num_layers=3)
        self.vocab_projector   = nn.Linear(self.d_model, vocab_size)

    def forward_cls(self, pixel_values, input_ids, attention_mask):
        vis = self.vision_encoder(pixel_values=pixel_values).pooler_output
        txt = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([vis, txt], dim=1))


# ─── Grad-CAM for ViT ────────────────────────────────────────────────────────
class ViTGradCAM:
    """
    Gradient-weighted Class Activation Map for Vision Transformer.

    Strategy:
      1. Forward pass-এ last ViT attention layer-এর output feature hook করা
      2. Target class score-এর gradient সেই feature-এর সাপেক্ষে নেওয়া
      3. Gradient × Feature → channel-wise average → 14×14 heatmap
      4. Upsample to 224×224 → image-এ overlay

    ViT-Base/16-এ: 197 tokens (1 CLS + 196 patches), patch_size=14x14
    """

    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        """
        ViT encoder-এর শেষ block খোঁজা — HuggingFace ViTModel-এ
        path হলো: .encoder.layer[-1]  (NOT .vit.encoder.layer)
        PEFT wrapping-এর কারণে কয়েকটি path try করা হচ্ছে।
        """
        last_block = None

        # Path 1: PEFT → base_model → model → encoder (HF ViTModel সরাসরি)
        try:
            last_block = (self.model.vision_encoder
                          .base_model.model.encoder.layer[-1])
            print("[+] Grad-CAM hook path: base_model.model.encoder.layer[-1]")
        except AttributeError:
            pass

        # Path 2: PEFT passthrough → encoder
        if last_block is None:
            try:
                last_block = self.model.vision_encoder.encoder.layer[-1]
                print("[+] Grad-CAM hook path: vision_encoder.encoder.layer[-1]")
            except AttributeError:
                pass

        # Path 3: Generic scan — named_modules() থেকে encoder blocks খোঁজা
        if last_block is None:
            encoder_blocks = [
                m for name, m in self.model.vision_encoder.named_modules()
                if "encoder.layer" in name and not any(
                    sub in name for sub in
                    ["attention", "intermediate", "output", "layernorm"]
                )
            ]
            if encoder_blocks:
                last_block = encoder_blocks[-1]
                print(f"[+] Grad-CAM hook path: generic scan → found {len(encoder_blocks)} blocks, using last")

        if last_block is None:
            # Debug: print available top-level module names to help diagnose
            top_names = [n for n, _ in self.model.vision_encoder.named_children()]
            raise AttributeError(
                f"Cannot locate ViT encoder layers. "
                f"Top-level children of vision_encoder: {top_names}"
            )

        def save_activation(module, input, output):
            # output: [B, num_tokens, hidden_dim] = [B, 197, 768]
            self.activations = (output[0].detach()
                                if isinstance(output, tuple)
                                else output.detach())

        def save_gradient(module, grad_input, grad_output):
            self.gradients = (grad_output[0].detach()
                              if isinstance(grad_output, tuple)
                              else grad_output.detach())

        h1 = last_block.register_forward_hook(save_activation)
        h2 = last_block.register_full_backward_hook(save_gradient)
        self._hooks = [h1, h2]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, pixel_values, input_ids, attention_mask, target_class=None):
        """
        Returns:
            cam: np.ndarray of shape (224, 224), range [0, 1]
            pred_class: int
            probs: np.ndarray of shape (num_classes,)
        """
        self.model.eval()
        pixel_values = pixel_values.to(DEVICE)
        input_ids    = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        # ─ Forward pass ──────────────────────────────────────────────────────
        logits = self.model.forward_cls(pixel_values, input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().detach().numpy()

        pred_class = logits.argmax(dim=1).item()
        if target_class is None:
            target_class = pred_class

        # ─ Backward pass ─────────────────────────────────────────────────────
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # ─ Compute CAM ───────────────────────────────────────────────────────
        # gradients / activations: [num_tokens, hidden_dim] = [197, 768]
        grads = self.gradients   # [197, 768]
        acts  = self.activations # [197, 768]

        if grads is None or acts is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture gradients/activations. "
                "Ensure model is not in torch.no_grad() context."
            )

        # Patch tokens only (exclude CLS token at index 0)
        patch_grads = grads[1:, :]   # [196, 768]
        patch_acts  = acts[1:, :]    # [196, 768]

        # Global average pooling over hidden dim → importance weights
        weights = patch_grads.mean(dim=1)              # [196]

        # Weighted sum of activations (channel dim)
        cam = (patch_acts * weights.unsqueeze(1)).sum(dim=1)  # [196]

        # ReLU: focus only on positive contributions
        cam = torch.relu(cam)

        # Reshape to 2D patch grid: 14×14 (for ViT-Base/16 with 224×224 input)
        num_patches_side = int(math.sqrt(cam.shape[0]))  # 14
        cam = cam.reshape(num_patches_side, num_patches_side)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.cpu().numpy()

        # Upsample to 224×224 using PIL
        cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
            (224, 224), Image.BILINEAR)
        cam_upsampled = np.array(cam_img) / 255.0

        return cam_upsampled, pred_class, probs


# ─── Visualization ───────────────────────────────────────────────────────────
def save_gradcam_figure(img_vis_tensor, cam, pred_class, true_class,
                        probs, save_path, sample_idx):
    """
    4-panel figure সংরক্ষণ:
    [Original] [Heatmap] [Overlay] [Class Probabilities]
    """
    # img_vis_tensor: [3, 224, 224] → [224, 224, 3]
    img_np = img_vis_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    # Colormap
    heatmap = cm.get_cmap('jet')(cam)[:, :, :3]  # [224, 224, 3]

    # Overlay (alpha blend)
    alpha   = 0.45
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Sample #{sample_idx:03d} | "
        f"True: {LABEL_NAMES[true_class]} | "
        f"Predicted: {LABEL_NAMES[pred_class]} "
        f"({'✓ CORRECT' if pred_class == true_class else '✗ WRONG'})",
        fontsize=11, fontweight='bold',
        color='green' if pred_class == true_class else 'red'
    )

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=9)
    axes[1].axis('off')
    plt.colorbar(
        plt.cm.ScalarMappable(cmap='jet',
                              norm=plt.Normalize(vmin=0, vmax=1)),
        ax=axes[1], fraction=0.046, pad=0.04
    )

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Image + CAM)", fontsize=9)
    axes[2].axis('off')

    # Class probability bar chart
    colors = ['#2ecc71' if i == pred_class else
              '#3498db' if i == true_class else '#bdc3c7'
              for i in range(len(LABEL_NAMES))]
    bars   = axes[3].barh(LABEL_NAMES, probs * 100, color=colors)
    axes[3].set_xlim(0, 100)
    axes[3].set_xlabel("Confidence (%)", fontsize=8)
    axes[3].set_title("Class Probabilities", fontsize=9)
    for bar, p in zip(bars, probs):
        axes[3].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{p*100:.1f}%', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Per-class sampling ───────────────────────────────────────────────────────
def get_stratified_samples(dataset, ts_indices, n_per_class=5, seed=42):
    """Test set-এ প্রতিটি class থেকে n_per_class samples নেওয়া।"""
    rng   = random.Random(seed)
    class_idx = {c: [] for c in range(len(LABEL_NAMES))}
    for i in ts_indices:
        lbl = int(dataset[i]["labels"].item())
        class_idx[lbl].append(i)
    samples = []
    for c in range(len(LABEL_NAMES)):
        avail = class_idx[c]
        rng.shuffle(avail)
        samples.extend(avail[:n_per_class])
    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples",  type=int, default=20,
                    help="Total samples to visualize (default: 20, ~7 per class)")
    ap.add_argument("--per_class",    type=int, default=7,
                    help="Samples per class for stratified selection")
    ap.add_argument("--output_dir",   type=str, default="gradcam_outputs")
    ap.add_argument("--target_class", type=int, default=None,
                    help="Force Grad-CAM for a specific class (0/1/2). Default: predicted class.")
    ap.add_argument("--split",        type=str, default="test",
                    choices=["train", "val", "test"],
                    help="Which split to visualize (default: test)")
    args = ap.parse_args()

    set_seed(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DRISHTI-XAI -- Task 2.1: Grad-CAM Visualization")
    print(f"  Model: DisasterNet-v2 CLS-Only (Task 1.1 best, F1=83.21%)")
    print(f"  Device: {DEVICE} | Split: {args.split}")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*60}\n")

    # ── Load tokenizer & dataset
    print("[+] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)
    dataset   = MultimodalDataset(CSV_PATH, IMG_DIR, tokenizer)
    tr_i, vl_i, ts_i = get_split_indices(CSV_PATH, RANDOM_SEED)
    split_map  = {"train": tr_i, "val": vl_i, "test": ts_i}
    split_idx  = split_map[args.split]
    print(f"[+] Split: {args.split.upper()} = {len(split_idx)} samples")

    # Stratified selection
    selected = get_stratified_samples(dataset, split_idx,
                                      n_per_class=args.per_class)[:args.num_samples]
    print(f"[+] Selected {len(selected)} samples "
          f"({args.per_class} per class, stratified)")

    # ── Build model
    print("[+] Building model (Task 1.1 architecture)...")
    model = DisasterNetAblation(num_classes=3).to(DEVICE)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[+] Model parameters: {total_p:,}")

    # ── Init Grad-CAM engine
    gradcam = ViTGradCAM(model)
    print("[+] Grad-CAM hooks registered on last ViT encoder block.")

    # ── Run Grad-CAM
    results = []
    correct = 0

    print(f"\n[+] Generating Grad-CAM for {len(selected)} samples...\n")
    for rank, idx in enumerate(selected, 1):
        sample = dataset[idx]
        pv     = sample["pixel_values"].unsqueeze(0)
        ids    = sample["input_ids"].unsqueeze(0)
        msk    = sample["attention_mask"].unsqueeze(0)
        true_c = sample["labels"].item()
        img_vis= sample["img_vis"]
        img_path = sample["img_path"]

        try:
            cam, pred_c, probs = gradcam.generate(
                pv, ids, msk, target_class=args.target_class
            )
        except Exception as e:
            print(f"  [!] Sample {idx} SKIPPED — {e}")
            continue

        is_correct = (pred_c == true_c)
        if is_correct:
            correct += 1

        # Save figure
        fname = (f"sample_{rank:03d}_"
                 f"true_{LABEL_NAMES[true_c][:8]}_"
                 f"pred_{LABEL_NAMES[pred_c][:8]}"
                 f"{'_OK' if is_correct else '_ERR'}.png")
        save_path = os.path.join(args.output_dir, fname)
        save_gradcam_figure(img_vis, cam, pred_c, true_c, probs,
                            save_path, rank)

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

        status = "✓" if is_correct else "✗"
        print(f"  [{status}] #{rank:03d} | True: {LABEL_NAMES[true_c]:<22} "
              f"| Pred: {LABEL_NAMES[pred_c]:<22} "
              f"| Conf: {probs[pred_c]*100:.1f}% → {fname}")

    # ── Summary CSV
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "gradcam_summary.csv")
    df_res.to_csv(csv_path, index=False)

    acc = correct / len(results) * 100 if results else 0
    w = 60
    print(f"\n{'='*w}")
    print(f"  GRAD-CAM SUMMARY")
    print(f"{'='*w}")
    print(f"  Total samples processed : {len(results)}")
    print(f"  Correct predictions     : {correct} / {len(results)} ({acc:.1f}%)")
    print(f"  Output directory        : {args.output_dir}/")
    print(f"  Summary CSV             : {csv_path}")
    print(f"\n  Per-class distribution of CORRECT predictions:")
    if not df_res.empty:
        for cls in LABEL_NAMES:
            subset   = df_res[df_res["true_label"] == cls]
            n_total  = len(subset)
            n_correct= subset["correct"].sum()
            print(f"    {cls:<25}: {n_correct}/{n_total} correct")
    print(f"{'='*w}\n")
    print(f"[+] All done! Open '{args.output_dir}/' to view heatmaps.")
    print(f"[+] For Google Colab, run:")
    print(f"    from IPython.display import Image, display")
    print(f"    import glob")
    print(f"    for p in sorted(glob.glob('{args.output_dir}/*.png'))[:5]:")
    print(f"        display(Image(p))")

    gradcam.remove_hooks()


if __name__ == "__main__":
    main()
