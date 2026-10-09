"""
DRISHTI-XAI -- Task 2.2: Cross-Attention Visualization
=======================================================
Caption Decoder-এর Cross-Attention layer থেকে Text Token → Image Patch
alignment map তৈরি করা।

কী দেখানো হবে:
  - মডেলের ভেতরে caption_decoder (TransformerDecoder) cross-attention করে:
    queries  = text token embeddings
    keys     = ViT image patch features (memory)
    → এই attention weight দেখালে বোঝা যায় কোন text token কোন image patch দেখে।

Output per sample:
  (a) Original image (224x224)
  (b) Cross-Attention heatmap - averaged over all heads & decoder layers
  (c) Overlay (image + heatmap)
  (d) Class probabilities bar chart

Usage (Colab):
    python crossattn_xai.py --num_samples 21 --per_class 7 \
        --checkpoint best_cls_only.pt --output_dir crossattn_outputs

Colab setup (new session):
    !git pull origin main
    !cp /content/drive/MyDrive/DisasterNet/models/best_cls_only.pt /content/DRISHTI-Bn/src/
    !python crossattn_xai.py --num_samples 21 --per_class 7 \
        --checkpoint best_cls_only.pt --output_dir crossattn_outputs
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

# Config
CSV_PATH    = "../data/processed/master_dataset_translated.csv"
IMG_DIR     = "../data/processed/"
LABEL_NAMES = ["Severe_Damage", "Humanitarian_Rescue", "Affected_People"]
LABEL_MAP   = {n: i for i, n in enumerate(LABEL_NAMES)}
RANDOM_SEED = 42
MAX_LEN     = 128
VOCAB_SIZE  = 32000
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_ID     = "csebuetnlp/banglabert"
VIT_GRID    = 14    # 14x14 = 196 patches for ViT-Base/16


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[+] Random seed fixed: {seed}")


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
            row[self.text_col],
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values":  pv,
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "labels":        torch.tensor(row["label_id"], dtype=torch.long),
            "img_vis":       transforms.ToTensor()(img_vis),
            "img_path":      os.path.join(self.root_dir, row["image_path"]),
            "true_label":    LABEL_NAMES[row["label_id"]],
            "caption_text":  row[self.text_col],
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
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
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
            nn.TransformerDecoderLayer(d_model=768, nhead=8, dim_feedforward=2048,
                                       dropout=0.1, batch_first=True), num_layers=3)
        self.vocab_projector   = nn.Linear(self.d_model, vocab_size)

    def forward_cls(self, pixel_values, input_ids, attention_mask):
        vis = self.vision_encoder(pixel_values=pixel_values).pooler_output
        txt = self.text_encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([vis, txt], dim=1))

    def get_vit_patch_features(self, pixel_values):
        # Returns [B, 196, 768] - patch features without CLS token
        out = self.vision_encoder(pixel_values=pixel_values)
        return out.last_hidden_state[:, 1:, :]


class CrossAttentionExtractor:
    """
    caption_decoder-এর cross-attention weights extract করে।
    Query = text token embeddings, Key/Value = ViT patch features
    """
    def __init__(self, model):
        self.model        = model
        self.hooks        = []
        self.attn_weights = []

    def register_hooks(self):
        for i, layer in enumerate(self.model.caption_decoder.layers):
            # Need_weights=True for MHA to return attention weights
            original_forward = layer.multihead_attn.forward

            def make_patched(orig_fwd, layer_idx):
                def patched_forward(*args, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = True
                    result = orig_fwd(*args, **kwargs)
                    if isinstance(result, tuple) and len(result) == 2 and result[1] is not None:
                        self.attn_weights.append(result[1].detach().cpu())
                    return result
                return patched_forward

            layer.multihead_attn.forward = make_patched(original_forward, i)
        print(f"[+] Cross-attention hooks registered on {len(list(self.model.caption_decoder.layers))} decoder layers.")

    def remove_hooks(self):
        # Restore original forward methods
        for layer in self.model.caption_decoder.layers:
            if hasattr(layer.multihead_attn, '_orig_forward'):
                layer.multihead_attn.forward = layer.multihead_attn._orig_forward
        self.attn_weights = []

    def extract(self, pixel_values, input_ids, attention_mask):
        self.attn_weights = []
        self.model.eval()

        pixel_values   = pixel_values.to(DEVICE)
        input_ids      = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        with torch.no_grad():
            # Classification forward (for pred + probs)
            logits     = self.model.forward_cls(pixel_values, input_ids, attention_mask)
            probs      = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_class = int(logits.argmax(dim=1).item())

            # ViT patch features (memory for cross-attention)
            patch_features = self.model.get_vit_patch_features(pixel_values)

            # Text token embeddings -> decoder queries
            tok_emb = self.model.decoder_embedding(input_ids)
            tok_emb = self.model.pos_encoder(tok_emb)

            # Causal mask for decoder self-attention
            T            = tok_emb.size(1)
            causal_mask  = nn.Transformer.generate_square_subsequent_mask(T, device=DEVICE)

            # Run decoder - hooks capture cross-attention weights
            _ = self.model.caption_decoder(
                tgt=tok_emb,
                memory=patch_features,
                tgt_mask=causal_mask,
                tgt_is_causal=True,
            )

        if not self.attn_weights:
            raise RuntimeError("No cross-attention weights captured.")

        # Stack layers: [num_layers, B, T, 196]
        stacked = torch.stack(self.attn_weights, dim=0)

        # Average over layers -> [B, T, 196]
        avg_attn = stacked.mean(dim=0).squeeze(0)  # [T, 196]

        # Average over valid (non-padding) tokens -> [196]
        mask_cpu     = attention_mask.squeeze(0).cpu()
        valid_tokens = int(mask_cpu.sum().item())
        avg_attn     = avg_attn[:valid_tokens, :].mean(dim=0).numpy()  # [196]

        # Reshape to 14x14, normalize, upsample
        cam = avg_attn.reshape(VIT_GRID, VIT_GRID)
        c_min, c_max = cam.min(), cam.max()
        if (c_max - c_min) > 1e-8:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_up  = np.array(cam_pil.resize((224, 224), Image.BILINEAR)) / 255.0

        return cam_up, pred_class, probs


def save_crossattn_figure(img_vis_tensor, cam, pred_class, true_class,
                          probs, save_path, sample_idx, caption_text=""):
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
             f"Pred: {LABEL_NAMES[pred_class]} ({status})")
    # Bengali font warning avoid korar jonno caption remove kora holo
    fig.suptitle(title, fontsize=10, fontweight='bold', color=color, y=1.02)

    axes[0].imshow(img_np);        axes[0].set_title("Original Image", fontsize=9);   axes[0].axis('off')
    im = axes[1].imshow(cam, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Cross-Attn Heatmap\n(text->patch alignment)", fontsize=9); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay);       axes[2].set_title("Overlay (Image + CAM)", fontsize=9); axes[2].axis('off')

    colors = ['#2ecc71' if i == pred_class else '#3498db' if i == true_class else '#bdc3c7'
              for i in range(len(LABEL_NAMES))]
    bars = axes[3].barh(LABEL_NAMES, probs * 100, color=colors)
    axes[3].set_xlim(0, 100); axes[3].set_xlabel("Confidence (%)", fontsize=8)
    axes[3].set_title("Class Probabilities", fontsize=9)
    for bar, p in zip(bars, probs):
        axes[3].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{p*100:.1f}%', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=21)
    ap.add_argument("--per_class",   type=int, default=7)
    ap.add_argument("--output_dir",  type=str, default="crossattn_outputs")
    ap.add_argument("--checkpoint",  type=str, default="best_cls_only.pt")
    ap.add_argument("--split",       type=str, default="test",
                    choices=["train", "val", "test"])
    args = ap.parse_args()

    set_seed(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DRISHTI-XAI -- Task 2.2: Cross-Attention Visualization")
    print(f"  Model: DisasterNet-v2 CLS-Only (Task 1.1 best)")
    print(f"  Device: {DEVICE} | Split: {args.split}")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*60}\n")

    print("[+] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_ID)
    dataset   = MultimodalDataset(CSV_PATH, IMG_DIR, tokenizer)
    _, _, ts_i = get_split_indices(CSV_PATH, RANDOM_SEED)
    print(f"[+] Split: TEST = {len(ts_i)} samples")

    selected = get_stratified_samples(dataset, ts_i, n_per_class=args.per_class)[:args.num_samples]
    print(f"[+] Selected {len(selected)} samples ({args.per_class} per class, stratified)")

    print("[+] Building model (Task 1.1 architecture)...")
    model = DisasterNetAblation(num_classes=3).to(DEVICE)

    if os.path.isfile(args.checkpoint):
        print(f"[+] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt)
        print("[+] Checkpoint loaded successfully")
    else:
        print(f"[!] WARNING: '{args.checkpoint}' not found - using random weights!")

    extractor = CrossAttentionExtractor(model)
    extractor.register_hooks()

    results = []; correct = 0
    print(f"\n[+] Generating Cross-Attention maps for {len(selected)} samples...\n")

    for rank, idx in enumerate(selected, 1):
        sample   = dataset[idx]
        pv       = sample["pixel_values"].unsqueeze(0)
        ids      = sample["input_ids"].unsqueeze(0)
        msk      = sample["attention_mask"].unsqueeze(0)
        true_c   = sample["labels"].item()
        img_vis  = sample["img_vis"]
        caption  = sample["caption_text"]
        img_path = sample["img_path"]

        try:
            cam, pred_c, probs = extractor.extract(pv, ids, msk)
        except Exception as e:
            print(f"  [!] Sample {idx} SKIPPED -- {e}"); continue

        is_correct = (pred_c == true_c)
        if is_correct: correct += 1

        fname = (f"sample_{rank:03d}_true_{LABEL_NAMES[true_c][:8]}_"
                 f"pred_{LABEL_NAMES[pred_c][:8]}{'_OK' if is_correct else '_ERR'}.png")
        save_crossattn_figure(img_vis, cam, pred_c, true_c, probs,
                              os.path.join(args.output_dir, fname), rank, caption)

        results.append({
            "sample_rank": rank, "dataset_idx": idx, "image_path": img_path,
            "caption": caption, "true_label": LABEL_NAMES[true_c],
            "pred_label": LABEL_NAMES[pred_c], "correct": is_correct,
            "conf_severe": float(probs[0]), "conf_human": float(probs[1]),
            "conf_affected": float(probs[2]), "output_file": fname,
        })

        status = "+" if is_correct else "x"
        print(f"  [{status}] #{rank:03d} | True: {LABEL_NAMES[true_c]:<22} "
              f"| Pred: {LABEL_NAMES[pred_c]:<22} | Conf: {probs[pred_c]*100:.1f}% -> {fname}")

    extractor.remove_hooks()

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(args.output_dir, "crossattn_summary.csv"), index=False)

    acc = correct / len(results) * 100 if results else 0
    w   = 60
    print(f"\n{'='*w}")
    print(f"  CROSS-ATTENTION SUMMARY")
    print(f"{'='*w}")
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
    print(f"    shutil.make_archive('crossattn_results', 'zip', '{args.output_dir}')")
    print(f"    os.makedirs('/content/drive/MyDrive/DisasterNet/results', exist_ok=True)")
    print(f"    shutil.copy('crossattn_results.zip', '/content/drive/MyDrive/DisasterNet/results/crossattn_results.zip')")


if __name__ == "__main__":
    main()
