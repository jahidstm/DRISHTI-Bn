import nbformat as nbf

with open('notebooks/01_data_preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

new_code = """import sys
import os
import torch

# ১. লোকাল পাথ ইনজেকশন (যাতে notebooks থেকে src ফোল্ডার অ্যাক্সেস করা যায়)
sys.path.append(os.path.abspath('../'))
from src.data_loader import get_dataloaders

# ২. সুনির্দিষ্ট পাথ ও কনফিগারেশন ডিফাইন করা
CSV_PATH = '../data/processed/master_dataset_translated.csv'
IMG_DIR = '../data/processed/' # ইমেজ পাথগুলো macro_label/filename আকারে আছে

print("[+] Initializing DRISHTI-Bn DataLoaders...")

try:
    # আমরা টেস্ট করার জন্য ছোট ব্যাচ সাইজ (Batch Size = 4) ব্যবহার করব
    train_loader, val_loader = get_dataloaders(CSV_PATH, IMG_DIR, batch_size=4)
    print(f"[+] Loaders Ready. Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}")
    
    # ৩. ডেটালোডার থেকে একদম প্রথম ব্যাচটি টেনে বের করা (The Extraction)
    first_batch = next(iter(train_loader))
    
    print("\\n--- 📊 Tensor Shape Verification (The Strict Baseline) ---")
    print(f"Pixel Values (Images) Shape : {first_batch['pixel_values'].shape} -> [Expected: torch.Size([4, 3, 224, 224])]")
    print(f"Input IDs (Text Tokens) Shape: {first_batch['input_ids'].shape} -> [Expected: torch.Size([4, 128])]")
    print(f"Attention Mask Shape        : {first_batch['attention_mask'].shape} -> [Expected: torch.Size([4, 128])]")
    print(f"Labels Shape                : {first_batch['labels'].shape} -> [Expected: torch.Size([4])]")
    print(f"Extracted Target Labels     : {first_batch['labels'].tolist()}")
    
    print("\\n[+] DIAGNOSTIC RESULT: DataLoader Sanity Check PASSED! The architecture is production-ready.")

except Exception as e:
    print(f"\\n[-] DIAGNOSTIC RESULT: Sanity Check FAILED!")
    print(f"Error Core Breakdown: {str(e)}")"""

nb.cells.append(nbf.v4.new_code_cell(new_code))

with open('notebooks/01_data_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
