"""
=============================================================================
  DRISHTI-XAI -- Task 3.1: Human Evaluation Protocol
  Evaluation Protocol Design — 100 stratified samples from test set
  Generates:
    1. human_eval_samples.csv  — 100 samples with GT + Model captions
    2. annotation_sheet.csv    — Blank sheet for 3 annotators
  =============================================================================
  Usage (Google Colab):
    !python human_eval_prep.py \\
        --checkpoint best_cls_only.pt \\
        --output_dir human_eval_outputs
  =============================================================================
"""

import os
import sys
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_NAMES  = {0: "Severe_Damage", 1: "Humanitarian_Rescue", 2: "Affected_People"}
LABEL_MAP    = {"Severe_Damage": 0, "Humanitarian_Rescue": 1, "Affected_People": 2}

# Proportional stratified counts: Test set = 224 SD / 119 HR / 33 AP (total 376)
# Proportional to test set class distribution → exactly 100
SAMPLE_PER_CLASS = {
    "Severe_Damage":       60,
    "Humanitarian_Rescue": 32,
    "Affected_People":      8,
}

# ── Paths (Auto-detect Local vs Colab) ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = "/content/DRISHTI-Bn" if os.path.exists("/content/DRISHTI-Bn") else PROJECT_ROOT
CSV_PATH  = os.path.join(REPO_ROOT, "data/processed/master_dataset_translated.csv")
IMG_DIR   = os.path.join(REPO_ROOT, "data/processed")
CKPT_DIR  = os.path.join(REPO_ROOT, "src")



# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model — DisasterNetAblation (Task 1.1 CLS-Only architecture)
# ═══════════════════════════════════════════════════════════════════════════════
def build_model(device):
    """Load Task 1.1 CLS-Only model (DisasterNetAblation)."""
    try:
        sys.path.insert(0, CKPT_DIR)
        from cls_only_ablation import DisasterNetAblation
        model = DisasterNetAblation(num_classes=3, lora_r=8)
        print("[+] DisasterNetAblation imported from cls_only_ablation.py")
    except ImportError:
        raise RuntimeError(
            "cls_only_ablation.py not found in src/. "
            "Make sure you ran: git pull origin main"
        )
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    """Load best_cls_only.pt checkpoint."""
    from safetensors.torch import load_file as st_load
    import glob

    # Try loading .pt first, then .safetensors
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[+] Checkpoint loaded: {checkpoint_path}")
        if missing:    print(f"    Missing keys  : {len(missing)}")
        if unexpected: print(f"    Unexpected    : {len(unexpected)}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Caption generation — greedy decode from BanglaBERT encoder features
# ═══════════════════════════════════════════════════════════════════════════════
def generate_caption_greedy(model, tokenizer, image_tensor, text_input_ids,
                             text_attention_mask, device, max_new_tokens=60):
    """
    Greedy caption generation using the model's caption decoder.
    Returns decoded Bangla string.
    """
    model.eval()
    BOS_ID = tokenizer.cls_token_id or tokenizer.bos_token_id or 1
    EOS_ID = tokenizer.sep_token_id or tokenizer.eos_token_id or 2
    PAD_ID = tokenizer.pad_token_id or 0

    pixel_values     = image_tensor.unsqueeze(0).to(device)
    input_ids        = text_input_ids.unsqueeze(0).to(device)
    attention_mask   = text_attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode visual + text features
        try:
            vis_feats  = model.get_patch_features(pixel_values)        # (1, 197, 768)
            txt_hidden = model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state                                          # (1, L, 768)

            # Greedy decoding
            dec_ids = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
            for _ in range(max_new_tokens):
                dec_emb = model.decoder_embedding(dec_ids)              # (1, t, 768)
                dec_out = model.caption_decoder(
                    tgt=dec_emb,
                    memory=vis_feats,
                )                                                        # (1, t, 768)
                logits  = model.caption_head(dec_out[:, -1, :])         # (1, vocab)
                next_id = logits.argmax(dim=-1, keepdim=True)           # (1,1)
                dec_ids = torch.cat([dec_ids, next_id], dim=1)
                if next_id.item() == EOS_ID:
                    break

            token_ids = dec_ids[0, 1:].tolist()                         # strip BOS
            caption   = tokenizer.decode(token_ids,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)
            return caption.strip()

        except (AttributeError, RuntimeError):
            # Fallback: model may not expose caption decoder in this checkpoint
            return "[Caption decoder not available in cls_only checkpoint]"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════════
def get_test_df(csv_path):
    """Return test-split DataFrame with original indices."""
    df     = pd.read_csv(csv_path)
    labels = df["macro_label"].values
    all_idx = list(range(len(df)))

    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.2, stratify=labels, random_state=SEED)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED)

    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    test_df["original_df_index"] = test_idx
    print(f"[+] Test set: {len(test_df)} samples")
    for cls, n in test_df["macro_label"].value_counts().items():
        print(f"    {cls:<25}: {n}")
    return test_df


def stratified_sample_100(test_df, sample_per_class=SAMPLE_PER_CLASS, seed=SEED):
    """Stratified sample: exactly SAMPLE_PER_CLASS per class."""
    frames = []
    for cls, n in sample_per_class.items():
        cls_df = test_df[test_df["macro_label"] == cls]
        if len(cls_df) < n:
            raise ValueError(f"Class {cls} has only {len(cls_df)} samples, need {n}")
        sampled = cls_df.sample(n=n, random_state=seed)
        frames.append(sampled)

    selected = pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    selected.insert(0, "sample_id", [f"S{i+1:03d}" for i in range(len(selected))])
    print(f"\n[+] Stratified sample: {len(selected)} total")
    for cls, n in sample_per_class.items():
        print(f"    {cls:<25}: {n}")
    return selected


def build_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  default="best_cls_only.pt")
    ap.add_argument("--output_dir",  default="human_eval_outputs")
    ap.add_argument("--caption_gen", action="store_true",
                    help="Generate model captions (requires full model; slow on CPU)")
    ap.add_argument("--max_samples", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  DRISHTI-XAI -- Task 3.1: Human Evaluation Protocol")
    print(f"  Device: {device} | Output: {args.output_dir}/")
    print("=" * 70)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    test_df   = get_test_df(CSV_PATH)
    selected  = stratified_sample_100(test_df)

    # ── 2. (Optional) Generate model captions ─────────────────────────────────
    model_captions = []
    if args.caption_gen:
        tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
        transform = build_image_transform()

        print("\n[+] Building model for caption generation...")
        model = build_model(device)
        ckpt_path = os.path.join(CKPT_DIR, args.checkpoint)
        model = load_checkpoint(model, ckpt_path, device)
        model.eval()

        print("[+] Generating captions...")
        for _, row in selected.iterrows():
            img_path = os.path.join(IMG_DIR, row["image_path"])
            image    = Image.open(img_path).convert("RGB")
            img_t    = transform(image)

            text = str(row["bengali_caption"]) if pd.notna(row["bengali_caption"]) else ""
            enc  = tokenizer(text, padding="max_length", truncation=True,
                             max_length=128, return_tensors="pt")
            cap  = generate_caption_greedy(
                model, tokenizer,
                img_t, enc["input_ids"].squeeze(), enc["attention_mask"].squeeze(),
                device)
            model_captions.append(cap)
            print(f"  {row['sample_id']}: {cap[:60]}...")
    else:
        # Placeholder — captions to be filled by running the model on Colab
        model_captions = ["[Run with --caption_gen to fill]"] * len(selected)
        print("\n[!] Caption generation skipped (run with --caption_gen on GPU)")

    # ── 3. Build human_eval_samples.csv ───────────────────────────────────────
    samples_df = pd.DataFrame({
        "sample_id":         selected["sample_id"],
        "class_label":       selected["macro_label"],
        "image_path":        selected["image_path"],
        "ground_truth_caption": selected["bengali_caption"],
        "model_caption":     model_captions,
    })

    samples_path = os.path.join(args.output_dir, "human_eval_samples.csv")
    samples_df.to_csv(samples_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Saved: {samples_path}")

    # ── 4. Build annotation_sheet.csv (one row per sample per annotator) ──────
    rows = []
    for _, row in samples_df.iterrows():
        for ann_id in ["Annotator_1", "Annotator_2", "Annotator_3"]:
            rows.append({
                "annotator_id":         ann_id,
                "sample_id":            row["sample_id"],
                "class_label":          row["class_label"],
                "image_path":           row["image_path"],
                # --- Blind: shuffle which caption appears first ---------------
                "caption_A":            row["ground_truth_caption"],
                "caption_B":            row["model_caption"],
                # --- Scoring columns (to be filled by annotator) -------------
                "fluency_A":            "",   # 1-5
                "adequacy_A":           "",   # 1-5
                "faithfulness_A":       "",   # 1-5
                "fluency_B":            "",   # 1-5
                "adequacy_B":           "",   # 1-5
                "faithfulness_B":       "",   # 1-5
                "preference":           "",   # A / B / Equal
                "comments":             "",
                # --- Ground truth label (hidden from annotators) -------------
                "HIDDEN_A_is":          "ground_truth",
                "HIDDEN_B_is":          "model",
            })

    ann_df = pd.DataFrame(rows)
    ann_path = os.path.join(args.output_dir, "annotation_sheet.csv")
    ann_df.to_csv(ann_path, index=False, encoding="utf-8-sig")
    print(f"[+] Saved: {ann_path}")

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    print(f"""
[+] Task 3.1 DONE! Files in '{args.output_dir}/':
    - human_eval_samples.csv     ({len(samples_df)} samples — GT + Model captions)
    - annotation_sheet.csv       ({len(ann_df)} rows — 3 annotators × 100 samples)

Next steps:
    Task 3.2: Share annotation_sheet.csv with 3 Bengali-speaking annotators
    Task 3.3: Annotators fill Fluency/Adequacy/Faithfulness (1-5) for A and B
    Task 3.4: Run human_eval_iaa.py to compute Cohen's Kappa + final report
""")

    # ── 6. Download ZIP (Colab) ────────────────────────────────────────────────
    try:
        import shutil
        from google.colab import files
        zip_path = os.path.abspath(args.output_dir)
        archive_name = os.path.join(os.path.dirname(zip_path), "human_eval_task31")
        shutil.make_archive(archive_name, "zip", zip_path)
        files.download(f"{archive_name}.zip")
        print(f"[+] Downloaded: {archive_name}.zip")
    except Exception as e:
        print(f"[!] Auto-download skipped or failed ({e}). Output saved in: {args.output_dir}/")



if __name__ == "__main__":
    main()
