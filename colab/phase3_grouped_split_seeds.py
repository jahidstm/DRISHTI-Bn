# =====================================================================
# DEFENSE_FIX_PLAN — Phase 3: leakage-free grouped split + 3 seeds
# Runtime: Colab GPU (T4).  Total ~3–3.5 h.  Safe to re-run: finished
# runs are read back from Drive and skipped (resume after disconnect).
# ---------------------------------------------------------------------
# What it does
#   1. Builds an 80/10/10 split where no tweet (and no identical caption)
#      is shared between train / val / test, stratified by class.
#   2. Trains CLS-only (ViT + BanglaBERT + LoRA r=8, lambda_cap=0) with
#      seeds 42, 123, 2024  — same hyper-parameters as Task 1.1.
#   3. Trains the ViT-Only baseline with the same 3 seeds — same
#      hyper-parameters as Task 0.5, with a version-robust layer unfreeze.
#   4. Re-runs the CrisisViT-style baseline (ViT-Large, last 4 blocks +
#      pooler, Task 0.8 setup) with the same robust unfreeze — seed 42.
#      (The original Task 0.5 / 0.8 name rule "encoder.layer.N" matches
#      nothing in transformers 5.x, so only the pooler was trained.)
#   5. Prints mean ± std and saves everything to Drive.
# Paste back to Claude: everything from "PHASE 3 FINAL SUMMARY" down.
# =====================================================================

from google.colab import drive
drive.mount('/content/drive')

import os, re, sys, glob, json, time, shutil, zipfile, subprocess
import numpy as np, pandas as pd

DRIVE_ROOT = '/content/drive/MyDrive/DisasterNet'
REPO_DIR   = '/content/DRISHTI-Bn'
DATA_DIR   = '/content/data/processed'
CSV_NAME   = 'master_dataset_translated.csv'
OUT_DIR    = f'{DRIVE_ROOT}/defense_fixes/phase3'
CKPT_DIR   = f'{DRIVE_ROOT}/models'
SEEDS      = [42, 123, 2024]
CRISISVIT_SEEDS = [42]   # ViT-Large is slow (~30-40 min/run); add 123, 2024 if time allows
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. Code: clone / update the repo --------------------------------
if os.path.isdir(REPO_DIR):
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '-q'], check=True)
else:
    subprocess.run(['git', 'clone', '-q',
                    'https://github.com/jahidstm/DRISHTI-Bn.git', REPO_DIR], check=True)
sys.path.insert(0, f'{REPO_DIR}/src')
print('[+] Repo ready:', subprocess.run(['git', '-C', REPO_DIR, 'log', '--oneline', '-1'],
                                        capture_output=True, text=True).stdout.strip())

# ---- 2. Data: copy to local disk (fast I/O) --------------------------
if not os.path.exists(f'{DATA_DIR}/{CSV_NAME}'):
    os.makedirs('/content/data', exist_ok=True)
    zips = glob.glob(f'{DRIVE_ROOT}/**/processed*.zip', recursive=True)
    csvs = glob.glob(f'{DRIVE_ROOT}/**/processed/{CSV_NAME}', recursive=True)
    if zips:
        print('[+] Unzipping', zips[0])
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall('/content/data')          # zip contains processed/...
    elif csvs:
        print('[+] Copying', os.path.dirname(csvs[0]))
        shutil.copytree(os.path.dirname(csvs[0]), DATA_DIR, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(
            "Dataset not found in Drive. Upload the laptop file data/processed.zip "
            "to  My Drive > DisasterNet  and run this cell again.")

df = pd.read_csv(f'{DATA_DIR}/{CSV_NAME}')
missing = [p for p in df.image_path if not os.path.exists(os.path.join(DATA_DIR, p))]
assert not missing, f'{len(missing)} images missing, e.g. {missing[:3]}'
print(f'[+] Dataset: {len(df)} rows, all images present')

# ---- 3. Grouped stratified split -------------------------------------
from sklearn.model_selection import StratifiedGroupKFold

LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']
y        = df.macro_label.map({n: i for i, n in enumerate(LABEL_NAMES)}).values
tweet_id = df.image_path.str.split('/').str[-1].str.split('_').str[0].values
caption  = df.bengali_caption.fillna('').astype(str).str.strip().values

# Union-find: samples sharing a tweet OR an identical caption form one group
parent = list(range(len(df)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra
first = {}
for i in range(len(df)):
    for key in (('t', tweet_id[i]), ('c', caption[i]) if caption[i] else None):
        if key is None: continue
        if key in first: union(i, first[key])
        else: first[key] = i
groups = np.array([find(i) for i in range(len(df))])
print(f'[+] {len(df)} samples -> {len(set(groups))} leakage groups')

idx = np.arange(len(df))
outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
rest, test_idx = next(outer.split(idx, y, groups))
inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
tr_rel, va_rel = next(inner.split(rest, y[rest], groups[rest]))
train_idx, val_idx = rest[tr_rel], rest[va_rel]

def overlap(a, b, arr): return len(set(arr[a]) & set(arr[b]))
for name, arr in [('group', groups), ('tweet', tweet_id), ('caption', caption)]:
    o = [overlap(train_idx, test_idx, arr), overlap(train_idx, val_idx, arr), overlap(val_idx, test_idx, arr)]
    assert sum(o) == 0, f'{name} overlap {o}'
print('[+] Zero tweet / caption overlap between train, val and test  ✔')

split_col = np.empty(len(df), dtype=object)
split_col[train_idx], split_col[val_idx], split_col[test_idx] = 'train', 'val', 'test'
split_df = pd.DataFrame({'index': idx, 'image_path': df.image_path, 'macro_label': df.macro_label,
                         'tweet_id': tweet_id, 'group': groups, 'split': split_col})
split_df.to_csv(f'{OUT_DIR}/grouped_split.csv', index=False)
split_df[split_df.split == 'test'].to_csv(f'{OUT_DIR}/test_set_indices_grouped.csv', index=False)

print(f'\n  {"split":<6}{"n":>6}  ' + '  '.join(f'{n[:12]:>12}' for n in LABEL_NAMES))
for s, ii in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
    c = np.bincount(y[ii], minlength=3)
    print(f'  {s:<6}{len(ii):>6}  ' + '  '.join(f'{c[k]:>6} ({c[k]/len(ii)*100:4.1f}%)' for k in range(3)))

# ---- 4. Training helpers ---------------------------------------------
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, ViTModel
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import cls_only_ablation as C          # Task 1.1 model / dataset / loops
import vit_only_baseline as V          # Task 0.5 dataset / loops

DEVICE = C.DEVICE
assert DEVICE.type == 'cuda', 'Switch the Colab runtime to GPU (T4)'
RESULTS_CSV = f'{OUT_DIR}/phase3_seed_results.csv'
CSV_PATH = f'{DATA_DIR}/{CSV_NAME}'

def done_runs():
    if not os.path.exists(RESULTS_CSV): return set()
    r = pd.read_csv(RESULTS_CSV)
    return set(zip(r.model, r.seed))

def save_row(model_name, seed, yt, yp, best_ep, best_vf1, trainable, minutes):
    pf1 = f1_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0) * 100
    row = dict(model=model_name, seed=seed,
               macro_f1=f1_score(yt, yp, average='macro', zero_division=0) * 100,
               macro_p=precision_score(yt, yp, average='macro', zero_division=0) * 100,
               macro_r=recall_score(yt, yp, average='macro', zero_division=0) * 100,
               accuracy=(yt == yp).mean() * 100,
               f1_severe=pf1[0], f1_human=pf1[1], f1_affected=pf1[2],
               best_epoch=best_ep, val_macro_f1=best_vf1, trainable_params=trainable,
               minutes=round(minutes, 1),
               confusion=json.dumps(confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist()))
    r = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        r = pd.concat([pd.read_csv(RESULTS_CSV), r], ignore_index=True)
    r.to_csv(RESULTS_CSV, index=False)
    print(f'  ==> {model_name} seed={seed}: TEST Macro F1 = {row["macro_f1"]:.2f}% '
          f'(SD {pf1[0]:.1f} | HR {pf1[1]:.1f} | AP {pf1[2]:.1f})  [{minutes:.0f} min]')

def loaders(ds, seed, bs):
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(num_workers=2, pin_memory=True)
    return (DataLoader(Subset(ds, train_idx), batch_size=bs, shuffle=True,
                       worker_init_fn=C.seed_worker, generator=g, **kw),
            DataLoader(Subset(ds, val_idx), batch_size=bs, shuffle=False, **kw),
            DataLoader(Subset(ds, test_idx), batch_size=bs, shuffle=False, **kw))

# ---- 4a. CLS-only (Task 1.1 setup: 10 ep, bs 16, lr 1e-4, AdamW wd .01, cosine, AMP)
def run_cls(seed, epochs=10):
    t0 = time.time(); C.set_seed(seed)
    ds = C.MultimodalDataset(CSV_PATH, DATA_DIR, AutoTokenizer.from_pretrained(C.BERT_ID))
    tr, va, te = loaders(ds, seed, 16)
    model = C.DisasterNetAblation(num_classes=3).to(DEVICE)
    trainable = sum(p.numel() for m in (model.vision_encoder, model.text_encoder, model.classifier)
                    for p in m.parameters() if p.requires_grad)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler, crit = torch.amp.GradScaler('cuda'), nn.CrossEntropyLoss()
    best_vf1, best_ep, best_st = -1, 0, None
    for ep in range(1, epochs + 1):
        loss, acc = C.train_epoch(model, tr, opt, crit, scaler); sch.step()
        vt, vp = C.evaluate(model, va)
        vf1 = f1_score(vt, vp, average='macro', zero_division=0) * 100
        print(f'  [CLS s{seed}] ep {ep:02d} loss={loss:.4f} train_acc={acc*100:.1f}% val_F1={vf1:.2f}%')
        if vf1 > best_vf1:
            best_vf1, best_ep = vf1, ep
            best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_st)
    if seed == 42:   # kept for Phase 2 / 4
        torch.save(best_st, f'{CKPT_DIR}/cls_only_grouped_s42.pt')
        print(f'  [+] Saved {CKPT_DIR}/cls_only_grouped_s42.pt')
    yt, yp = C.evaluate(model, te)
    save_row('CLS-Only (ViT+BanglaBERT+LoRA r=8)', seed, yt, yp, best_ep, best_vf1,
             trainable, (time.time() - t0) / 60)
    del model, best_st; torch.cuda.empty_cache()

# ---- 4b. ViT-only (Task 0.5 setup: 10 ep, bs 32, lr 2e-4, AdamW wd 1e-4, cosine)
LAST2 = re.compile(r'(encoder\.layer|layers)\.(10|11)\.')
class ViTOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for n, p in self.vit.named_parameters():
            p.requires_grad = bool(LAST2.search(n)) or 'pooler' in n
        self.classifier = nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(512, 3))
    def forward(self, x):
        return self.classifier(self.vit(pixel_values=x).pooler_output)

def run_vit(seed, epochs=10):
    t0 = time.time(); V.set_seed(seed)
    ds = V.ImageOnlyDataset(CSV_PATH, DATA_DIR)
    tr, va, te = loaders(ds, seed, 32)
    model = ViTOnly().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_vf1, best_ep, best_st = -1, 0, None
    for ep in range(1, epochs + 1):
        loss, acc = V.train(model, tr, opt, crit)
        vt, vp = V.evaluate(model, va); sch.step()
        vf1 = f1_score(vt, vp, average='macro', zero_division=0) * 100
        print(f'  [ViT s{seed}] ep {ep:02d} loss={loss:.4f} train_acc={acc*100:.1f}% val_F1={vf1:.2f}%')
        if vf1 > best_vf1:
            best_vf1, best_ep = vf1, ep
            best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_st)
    yt, yp = V.evaluate(model, te)
    save_row('ViT-Only (last 2 blocks + pooler)', seed, yt, yp, best_ep, best_vf1,
             trainable, (time.time() - t0) / 60)
    del model, best_st; torch.cuda.empty_cache()

# ---- 4c. CrisisViT-style (Task 0.8 setup: ViT-Large, 10 ep, bs 8, lr 2e-5,
#          AdamW wd 1e-4, cosine, label smoothing 0.1, augmentation, AMP)
import crisisvit_baseline as K
LAST4_L = re.compile(r'(encoder\.layer|layers)\.(20|21|22|23)\.')
class CrisisViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained(K.VIT_MODEL_ID)
        for n, p in self.vit.named_parameters():
            p.requires_grad = bool(LAST4_L.search(n)) or 'pooler' in n
        self.cls = nn.Sequential(nn.Linear(K.VIT_HIDDEN, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(512, 3))
    def forward(self, x):
        return self.cls(self.vit(pixel_values=x).pooler_output)

def run_crisisvit(seed, epochs=10):
    t0 = time.time(); K.set_seed(seed)
    tr_ds = K.ImageOnlyDataset(CSV_PATH, DATA_DIR, augment=True)
    ev_ds = K.ImageOnlyDataset(CSV_PATH, DATA_DIR, augment=False)
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(num_workers=2, pin_memory=True)
    tr = DataLoader(Subset(tr_ds, train_idx), batch_size=8, shuffle=True,
                    worker_init_fn=C.seed_worker, generator=g, **kw)
    va = DataLoader(Subset(ev_ds, val_idx), batch_size=8, shuffle=False, **kw)
    te = DataLoader(Subset(ev_ds, test_idx), batch_size=8, shuffle=False, **kw)
    model = CrisisViT().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vit_total = sum(p.numel() for p in model.vit.parameters())
    print(f'  [CrisisViT] trainable {trainable:,} (ViT-Large total {vit_total:,})')
    opt = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit, scaler = nn.CrossEntropyLoss(label_smoothing=0.1), torch.amp.GradScaler('cuda')
    best_vf1, best_ep, best_st = -1, 0, None
    for ep in range(1, epochs + 1):
        loss, acc = K.train_epoch(model, tr, opt, crit, scaler); sch.step()
        vt, vp = K.evaluate(model, va)
        vf1 = f1_score(vt, vp, average='macro', zero_division=0) * 100
        print(f'  [CrisisViT s{seed}] ep {ep:02d} loss={loss:.4f} train_acc={acc*100:.1f}% val_F1={vf1:.2f}%')
        if vf1 > best_vf1:
            best_vf1, best_ep = vf1, ep
            best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_st)
    yt, yp = K.evaluate(model, te)
    save_row('CrisisViT-style (ViT-L, last 4 blocks + pooler)', seed, yt, yp, best_ep, best_vf1,
             trainable, (time.time() - t0) / 60)
    del model, best_st; torch.cuda.empty_cache()

# ---- 4d. Diagnostic: did the ORIGINAL ViT-only rule unfreeze anything? --
_vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
orig = sum(p.numel() for n, p in _vit.named_parameters()
           if 'encoder.layer.10' in n or 'encoder.layer.11' in n or 'pooler' in n)
fixed = sum(p.numel() for n, p in _vit.named_parameters() if LAST2.search(n) or 'pooler' in n)
print(f'\n[diag] ViT param names look like: {[n for n, _ in _vit.named_parameters()][5]}')
print(f'[diag] Original Task 0.5 rule unfreezes {orig:,} params | robust rule unfreezes {fixed:,}')
del _vit

# ---- 5. Run everything (resumable) -----------------------------------
for seed in SEEDS:
    if ('CLS-Only (ViT+BanglaBERT+LoRA r=8)', seed) not in done_runs():
        print(f'\n===== CLS-Only | seed {seed} =====');  run_cls(seed)
    if ('ViT-Only (last 2 blocks + pooler)', seed) not in done_runs():
        print(f'\n===== ViT-Only | seed {seed} =====');  run_vit(seed)
for seed in CRISISVIT_SEEDS:
    if ('CrisisViT-style (ViT-L, last 4 blocks + pooler)', seed) not in done_runs():
        print(f'\n===== CrisisViT-style | seed {seed} =====');  run_crisisvit(seed)

# ---- 6. Summary -------------------------------------------------------
r = pd.read_csv(RESULTS_CSV)
print('\n' + '=' * 70)
print('PHASE 3 FINAL SUMMARY  (grouped split, test n = %d)' % len(test_idx))
print('=' * 70)
for m, g in r.groupby('model'):
    print(f'\n{m}  (seeds: {list(g.seed)})')
    for col in ['macro_f1', 'macro_p', 'macro_r', 'accuracy', 'f1_severe', 'f1_human', 'f1_affected']:
        print(f'  {col:<12} {g[col].mean():6.2f} ± {g[col].std(ddof=1):5.2f}   runs: '
              + ', '.join(f'{v:.2f}' for v in g[col]))
print(f'\n[diag] original ViT rule unfrozen params = {orig:,} | robust rule = {fixed:,}')
print('\n--- phase3_seed_results.csv ---')
print(r.to_csv(index=False))
print(f'Files saved in: {OUT_DIR}')
