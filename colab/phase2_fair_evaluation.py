# =====================================================================
# DEFENSE_FIX_PLAN — Phase 2: fair evaluation of the CLS-Only model
# Runtime: Colab GPU (T4).  ~10 min.  No training.
# ---------------------------------------------------------------------
# Loads models/cls_only_grouped_s42.pt (Phase 3, seed 42) and evaluates it
# on the 375-sample grouped test set with five kinds of text input:
#   A  original tweet text              (sanity: should give 74.83%)
#   B  v3 generated caption             (image-only pipeline; from Phase 3B)
#   C  empty text                       (image-only, no text at all)
#   D  demo placeholder "বন্যা পরিস্থিতি"  (what app.py feeds when no caption)
#   E  shuffled tweets                  (another test sample's tweet: how much
#                                        does the model rely on the text?)
# Saves summary, per-sample predictions and the new Figure 4.1 (condition A).
# Paste back to Claude: everything from "PHASE 2 FINAL SUMMARY" down.
# =====================================================================

import os, sys, json, glob, zipfile, subprocess
import importlib.metadata as md

# ---- 0. Colab env fix (same as Phase 3) -------------------------------
try:
    _tv = md.version('torchao')
    if tuple(int(x) for x in _tv.split('.')[:2]) < (0, 16):
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=True)
        print(f'[+] Removed incompatible torchao {_tv}')
        if 'peft' in sys.modules:
            raise SystemExit('>>> peft was already imported in this session. '
                             'Runtime -> Restart session, then run this cell again. <<<')
except md.PackageNotFoundError:
    pass

from google.colab import drive
drive.mount('/content/drive')

import numpy as np, pandas as pd

DRIVE_ROOT = '/content/drive/MyDrive/DisasterNet'
REPO_DIR   = '/content/DRISHTI-Bn'
DATA_DIR   = '/content/data/processed'
CSV_NAME   = 'master_dataset_translated.csv'
SPLIT_CSV  = f'{DRIVE_ROOT}/defense_fixes/phase3/grouped_split.csv'
GEN_CSV    = f'{DRIVE_ROOT}/defense_fixes/phase3b/phase3b_per_sample.csv'
CKPT_PATH  = f'{DRIVE_ROOT}/models/cls_only_grouped_s42.pt'
OUT_DIR    = f'{DRIVE_ROOT}/defense_fixes/phase2'
os.makedirs(OUT_DIR, exist_ok=True)
PLACEHOLDER = 'বন্যা পরিস্থিতি'
LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']

for p in (SPLIT_CSV, GEN_CSV, CKPT_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} missing (Phase 3 / 3B output)')

# ---- 1. Code + data ----------------------------------------------------
if os.path.isdir(REPO_DIR):
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '-q'], check=True)
else:
    subprocess.run(['git', 'clone', '-q', 'https://github.com/jahidstm/DRISHTI-Bn.git', REPO_DIR], check=True)
sys.path.insert(0, f'{REPO_DIR}/src')
print('[+] Repo ready:', subprocess.run(['git', '-C', REPO_DIR, 'log', '--oneline', '-1'],
                                        capture_output=True, text=True).stdout.strip())

if not os.path.exists(f'{DATA_DIR}/{CSV_NAME}'):
    os.makedirs('/content/data', exist_ok=True)
    zips = glob.glob(f'{DRIVE_ROOT}/**/processed*.zip', recursive=True)
    if not zips:
        raise FileNotFoundError('processed.zip not found in My Drive > DisasterNet')
    print('[+] Unzipping', zips[0])
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall('/content/data')
CSV_PATH = f'{DATA_DIR}/{CSV_NAME}'

split = pd.read_csv(SPLIT_CSV)
test_idx = np.where(split.split.values == 'test')[0]
assert len(test_idx) == 375

# ---- 2. Model (exact Phase 3 CLS-Only class and evaluation loop) ------
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import cls_only_ablation as C

assert torch.cuda.is_available(), 'Switch the Colab runtime to GPU (T4)'
C.set_seed(42)
ds = C.MultimodalDataset(CSV_PATH, DATA_DIR, AutoTokenizer.from_pretrained(C.BERT_ID))
assert (split.image_path.values == ds.df.image_path.values).all(), 'dataset order changed'
original = ds.df[ds.text_col].copy()

model = C.DisasterNetAblation(num_classes=3).to(C.DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=C.DEVICE), strict=True)
model.eval()
print(f'[+] Loaded {CKPT_PATH}')

gen = pd.read_csv(GEN_CSV, keep_default_na=False)
gen_by_idx = dict(zip(gen['index'], gen.generated))
assert set(gen_by_idx) == set(test_idx), 'Phase 3B captions do not match the grouped test set'

rng = np.random.default_rng(42)
perm = test_idx[rng.permutation(len(test_idx))]
while (perm == test_idx).any():                     # nobody keeps their own tweet
    perm = test_idx[rng.permutation(len(test_idx))]

conditions = {
    'A_original_tweet':   {i: original[i] for i in test_idx},
    'B_v3_generated':     gen_by_idx,
    'C_empty_text':       {i: '' for i in test_idx},
    'D_placeholder':      {i: PLACEHOLDER for i in test_idx},
    'E_shuffled_tweet':   {i: original[j] for i, j in zip(test_idx, perm)},
}

def metrics(yt, yp):
    pf1 = f1_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0) * 100
    return dict(macro_f1=f1_score(yt, yp, average='macro', zero_division=0) * 100,
                macro_p=precision_score(yt, yp, average='macro', zero_division=0) * 100,
                macro_r=recall_score(yt, yp, average='macro', zero_division=0) * 100,
                accuracy=(yt == yp).mean() * 100,
                f1_severe=pf1[0], f1_human=pf1[1], f1_affected=pf1[2],
                confusion=json.dumps(confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist()))

rows, per = [], pd.DataFrame({'index': test_idx, 'image_path': ds.df.image_path.values[test_idx],
                              'true_label': ds.df.macro_label.values[test_idx]})
for name, texts in conditions.items():
    ds.df[ds.text_col] = original
    ds.df.loc[test_idx, ds.text_col] = [texts[i] for i in test_idx]
    loader = DataLoader(Subset(ds, test_idx), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    yt, yp = C.evaluate(model, loader)                      # fp32, same as Phase 3
    m = metrics(yt, yp)
    rows.append(dict(condition=name, **m))
    per[f'pred_{name}'] = [LABEL_NAMES[p] for p in yp]
    print(f'  {name:<20} Macro F1 = {m["macro_f1"]:.2f}%  (SD {m["f1_severe"]:.1f} | '
          f'HR {m["f1_human"]:.1f} | AP {m["f1_affected"]:.1f})')
ds.df[ds.text_col] = original

res = pd.DataFrame(rows)
res.to_csv(f'{OUT_DIR}/phase2_eval_summary.csv', index=False)
per.to_csv(f'{OUT_DIR}/phase2_per_sample.csv', index=False, encoding='utf-8-sig')

# ---- 3. Figure 4.1 (new): CLS-Only, condition A -----------------------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
cm = np.array(json.loads(res.loc[res.condition == 'A_original_tweet', 'confusion'].iloc[0]))
f1A = res.loc[res.condition == 'A_original_tweet', 'macro_f1'].iloc[0]
fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(cm, cmap='Blues')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cm[i, j]}\n({cm[i, j] / cm[i].sum() * 100:.0f}%)', ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11)
short = ['Severe\nDamage', 'Humanitarian\nRescue', 'Affected\nPeople']
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(short); ax.set_yticklabels(short)
ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
ax.set_title(f'DRISHTI-Bn (CLS-Only, seed 42) — grouped test set, n = 375\nMacro F1 = {f1A:.2f}%', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figure_4_1_confusion_matrix_grouped.png', dpi=300, bbox_inches='tight')
plt.close()

# ---- 4. Report ---------------------------------------------------------
agree = lambda a, b: (per[f'pred_{a}'] == per[f'pred_{b}']).mean() * 100
print('\n' + '=' * 70)
print('PHASE 2 FINAL SUMMARY  (CLS-Only seed 42, grouped test n = 375)')
print('=' * 70)
print(res.drop(columns='confusion').round(2).to_string(index=False))
print('\nConfusion matrices (rows SD, HR, AP):')
for _, r in res.iterrows():
    print(f'  {r.condition:<20} {r.confusion}')
print('\nPrediction agreement with condition A (original tweet):')
for c in list(conditions)[1:]:
    print(f'  {c:<20} {agree("A_original_tweet", c):.1f}%')
print('\nReference (Phase 3/3B): CLS-Only s42 = 74.83 | ViT-Only = 73.27 ± 0.39 | '
      'Multi-task v3 tweet = 77.29, generated = 73.63')
print(f'\nFiles saved in: {OUT_DIR}')
