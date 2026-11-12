# =====================================================================
# DEFENSE_FIX_PLAN — Phase 4: XAI + calibration on the final models
# Runtime: Colab GPU (T4).  ~15-20 min.  No training.
# ---------------------------------------------------------------------
# Models (both on Drive):
#   CLS-Only seed 42  models/cls_only_grouped_s42.pt   -> classification XAI
#   Multi-task v3     models/drishti_multitask_v3.pt   -> cross-attention
# Data: Phase 3 grouped split (val 373, test 375), original tweet text.
#
#  4.2 Calibration: fit T* on the grouped VAL set, report ECE / MCE / NLL /
#      Brier before vs after on the grouped TEST set + reliability diagram.
#  4.1 XAI on 21 test samples (7/class, random.Random(42), not cherry-picked):
#      * Grad-CAM  — FIXED: hooks the input of the last ViT block. The old
#        code hooked the last block's OUTPUT patch tokens, whose gradient is
#        exactly zero (the pooler reads only the CLS token) -> blank maps.
#      * Attention Rollout (Abnar & Zuidema 2020), discard ratio 0.9.
#      * Cross-attention of the TRAINED v3 decoder while it produces its
#        own generated caption (auxiliary: explains generation, not class).
#  4.5 Faithfulness (deletion test) on all 375 test images: remove the
#      top 10/25/50% patches ranked by Grad-CAM, Rollout or at random and
#      measure the drop in the predicted-class probability.
# Paste back to Claude: everything from "PHASE 4 FINAL SUMMARY" down.
# =====================================================================

import os, sys, json, glob, random, zipfile, subprocess
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
CLS_CKPT   = f'{DRIVE_ROOT}/models/cls_only_grouped_s42.pt'
MT_CKPT    = f'{DRIVE_ROOT}/models/drishti_multitask_v3.pt'
OUT_DIR    = f'{DRIVE_ROOT}/defense_fixes/phase4'
os.makedirs(OUT_DIR, exist_ok=True)
LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']
SHORT = ['Severe Damage', 'Humanitarian Rescue', 'Affected People']
PER_CLASS, N_BINS, DISCARD = 7, 10, 0.9
FRACS = [0.10, 0.25, 0.50]

for p in (SPLIT_CSV, GEN_CSV, CLS_CKPT, MT_CKPT):
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
val_idx  = np.where(split.split.values == 'val')[0]
test_idx = np.where(split.split.values == 'test')[0]
assert (len(val_idx), len(test_idx)) == (373, 375)

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, ViTModel
from transformers.models.vit.modeling_vit import ViTLayer
from scipy.optimize import minimize_scalar
import cls_only_ablation as C

DEVICE = C.DEVICE
assert DEVICE.type == 'cuda', 'Switch the Colab runtime to GPU (T4)'
C.set_seed(42)

# ViT with eager attention so that output_attentions works (needed for rollout).
_vit_from_pretrained = ViTModel.from_pretrained
def _eager_vit(*a, **k):
    k.setdefault('attn_implementation', 'eager')
    return _vit_from_pretrained(*a, **k)
C.ViTModel = type('EagerViT', (), {'from_pretrained': staticmethod(_eager_vit)})

tok = AutoTokenizer.from_pretrained(C.BERT_ID)
ds  = C.MultimodalDataset(CSV_PATH, DATA_DIR, tok)
assert (split.image_path.values == ds.df.image_path.values).all(), 'dataset order changed'

cls_model = C.DisasterNetAblation(num_classes=3).to(DEVICE)
cls_model.load_state_dict(torch.load(CLS_CKPT, map_location=DEVICE), strict=True)
cls_model.eval()
print(f'[+] Loaded {CLS_CKPT}')

def loader(idx, bs=16):
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

@torch.no_grad()
def collect_logits(idx):
    L, Y = [], []
    for b in loader(idx):
        L.append(cls_model.forward_cls(b['pixel_values'].to(DEVICE), b['input_ids'].to(DEVICE),
                                       b['attention_mask'].to(DEVICE)).float().cpu())
        Y.append(b['labels'])
    return torch.cat(L), torch.cat(Y)

# ---- 2. Calibration (4.2) ----------------------------------------------
def calib(probs, y, n_bins=N_BINS):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce, bins = 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            gap = abs(acc[m].mean() - conf[m].mean())
            ece += m.mean() * gap; mce = max(mce, gap)
            bins.append((lo, hi, acc[m].mean(), conf[m].mean(), int(m.sum())))
        else:
            bins.append((lo, hi, np.nan, np.nan, 0))
    onehot = np.eye(probs.shape[1])[y]
    return dict(ece=ece * 100, mce=mce * 100,
                nll=float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()),
                brier=float(((probs - onehot) ** 2).sum(1).mean()),
                mean_conf=conf.mean() * 100, accuracy=acc.mean() * 100), bins

vl, vy = collect_logits(val_idx)
tl, ty = collect_logits(test_idx)
T_star = float(minimize_scalar(lambda T: F.cross_entropy(vl / T, vy).item(),
                               bounds=(0.05, 10.0), method='bounded').x)
before, bins_b = calib(F.softmax(tl, 1).numpy(), ty.numpy())
after,  bins_a = calib(F.softmax(tl / T_star, 1).numpy(), ty.numpy())
val_before, _ = calib(F.softmax(vl, 1).numpy(), vy.numpy())
cal_df = pd.DataFrame([dict(metric=k, before=before[k], after=after[k]) for k in before])
cal_df.to_csv(f'{OUT_DIR}/phase4_calibration.csv', index=False)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, (title, bins, m) in zip(axes, [('Before temperature scaling (T = 1)', bins_b, before),
                                       (f'After temperature scaling (T* = {T_star:.3f})', bins_a, after)]):
    centers = [(lo + hi) / 2 for lo, hi, *_ in bins]
    accs = [a if n else 0 for _, _, a, _, n in bins]
    ax.bar(centers, accs, width=1 / N_BINS, edgecolor='black', color='#4C78A8', alpha=0.85, label='Accuracy')
    for c, (_, _, a, cf, n) in zip(centers, bins):
        if n:
            ax.bar(c, cf - a, bottom=a, width=1 / N_BINS, color='#E45756', alpha=0.35,
                   edgecolor='#E45756', hatch='//')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\nECE = {m["ece"]:.2f}%  |  test n = {len(test_idx)}', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figure_4_8_calibration_diagram_grouped.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'[+] Calibration: T* = {T_star:.4f} | ECE {before["ece"]:.2f}% -> {after["ece"]:.2f}%')

# ---- 3. XAI methods ----------------------------------------------------
vit_layers = [m for m in cls_model.vision_encoder.modules() if isinstance(m, ViTLayer)]
assert len(vit_layers) == 12

def gradcam(pv, ids, am, target):
    """Grad-CAM on the tokens ENTERING the last ViT block (= output of block 11).
    weights = channel-wise mean gradient over patch tokens; cam = ReLU(sum_c w_c A_c)."""
    cap = {}
    def hook(mod, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        a.retain_grad(); cap['a'] = a
    h = vit_layers[-2].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            cls_model.zero_grad(set_to_none=True)
            logit = cls_model.forward_cls(pv, ids, am)
            logit[0, target].backward()
    finally:
        h.remove()
    A, G = cap['a'][0, 1:].detach(), cap['a'].grad[0, 1:]
    cam = torch.relu((A * G.mean(0)).sum(1)).cpu().numpy()
    return cam  # (196,)

@torch.no_grad()
def rollout(pv):
    att = cls_model.vision_encoder(pixel_values=pv, output_attentions=True).attentions
    assert att is not None and len(att) == 12, 'attentions not returned (eager attention needed)'
    res = torch.eye(197, device=DEVICE)
    for a in att:
        f = a[0].mean(0).clone()
        thr = f.flatten().kthvalue(int(DISCARD * f.numel())).values
        f[f < thr] = 0
        f = f + torch.eye(197, device=DEVICE)
        f = f / f.sum(-1, keepdim=True)
        res = f @ res
    return res[0, 1:].cpu().numpy()  # (196,)

def norm01(c):
    c = c - c.min()
    return c / c.max() if c.max() > 0 else c

# Multi-task v3 for cross-attention (auxiliary)
from model import DisasterNetMultimodal
mt = DisasterNetMultimodal(num_classes=3, vocab_size=32000, use_lora=True).to(DEVICE)
mt.load_state_dict(torch.load(MT_CKPT, map_location=DEVICE), strict=True)
mt.eval()
print(f'[+] Loaded {MT_CKPT}')
gen = pd.read_csv(GEN_CSV, keep_default_na=False)
gen_by_idx = dict(zip(gen['index'], gen.generated))

@torch.no_grad()
def cross_attention(pv, caption):
    """Mean cross-attention (3 layers x 8 heads x caption tokens) of the v3
    decoder over the 196 ViT patches while teacher-forcing its own caption."""
    ids = tok(caption if caption else ' ', return_tensors='pt').input_ids.to(DEVICE)[:, :-1]
    memory = mt.vision_encoder(pixel_values=pv).last_hidden_state          # (1,197,768)
    captured, originals = [], []
    for layer in mt.caption_decoder.layers:
        o = layer.multihead_attn.forward; originals.append(o)
        def patched(*a, _o=o, **k):
            k['need_weights'] = True; k['average_attn_weights'] = True
            out, w = _o(*a, **k); captured.append(w); return out, w
        layer.multihead_attn.forward = patched
    try:
        tgt = mt.pos_encoder(mt.decoder_embedding(ids))
        mt.caption_decoder(tgt=tgt, memory=memory,
                           tgt_mask=mt.generate_causal_mask(ids.size(1), DEVICE))
    finally:
        for layer, o in zip(mt.caption_decoder.layers, originals):
            layer.multihead_attn.forward = o
    w = torch.stack(captured).mean(0)[0]          # (T, 197)
    return w[:, 1:].mean(0).cpu().numpy()          # drop ViT CLS -> (196,)

# ---- 4. 21 XAI samples (7 per class, not cherry-picked) ---------------
rng = random.Random(42)
selected = []
for c in range(3):
    pool = [i for i in test_idx if ds.df.label_id.values[i] == c]
    selected += rng.sample(pool, PER_CLASS)

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
def to_img(pv):
    return np.clip(pv[0].permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1)
def up(cam):
    t = torch.tensor(norm01(cam), dtype=torch.float32).view(1, 1, 14, 14)
    return F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)[0, 0].numpy()

recs = []
for k, gi in enumerate(selected):
    b = ds[gi]
    pv, ids, am = (b[x].unsqueeze(0).to(DEVICE) for x in ('pixel_values', 'input_ids', 'attention_mask'))
    with torch.no_grad():
        logit = cls_model.forward_cls(pv, ids, am)[0].float()
    p_raw, p_cal = F.softmax(logit, 0).cpu().numpy(), F.softmax(logit / T_star, 0).cpu().numpy()
    pred = int(p_raw.argmax()); y = int(b['labels'])
    g, r = gradcam(pv, ids, am, pred), rollout(pv)
    ca = cross_attention(pv, gen_by_idx[gi])
    top = lambda c: set(np.argsort(c)[-20:])                     # top-10% patches
    recs.append(dict(n=k + 1, index=gi, image_path=ds.df.image_path.values[gi], true=LABEL_NAMES[y],
                     pred=LABEL_NAMES[pred], correct=pred == y, conf_raw=p_raw.max(), conf_cal=p_cal.max(),
                     iou_gc_ro=len(top(g) & top(r)) / len(top(g) | top(r)),
                     iou_gc_ca=len(top(g) & top(ca)) / len(top(g) | top(ca)),
                     gradcam_zero=bool(g.max() == 0), caption=gen_by_idx[gi],
                     _img=to_img(pv), _g=up(g), _r=up(r), _c=up(ca)))
xai = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in recs])
xai.to_csv(f'{OUT_DIR}/phase4_xai_samples.csv', index=False, encoding='utf-8-sig')

def grid(rows, cols, path, title):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.75 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        for j, (name, key) in enumerate(cols):
            ax = axes[i, j]; ax.imshow(r['_img'])
            if key: ax.imshow(r[key], cmap='jet', alpha=0.45)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(name, fontsize=10)
            if j == 0:
                ok = '✓' if r['correct'] else '✗'
                ax.set_ylabel(f"#{r['n']} GT: {SHORT[LABEL_NAMES.index(r['true'])]}\n"
                              f"Pred: {SHORT[LABEL_NAMES.index(r['pred'])]} {ok}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()

ALL = [('Image', None), ('Grad-CAM', '_g'), ('Attention Rollout', '_r'), ('Cross-attn (v3 decoder)', '_c')]
grid(recs, ALL, f'{OUT_DIR}/xai_all_21_samples.png',
     'All 21 XAI samples (7 per class, random seed 42) — CLS-Only seed 42 / v3 decoder')
first = [next(r for r in recs if r['true'] == c) for c in LABEL_NAMES]   # first sample of each class
grid(first, ALL[:2], f'{OUT_DIR}/figure_4_4_gradcam_grouped.png', 'Grad-CAM (CLS-Only, predicted class)')
grid(first, [ALL[0], ALL[3]], f'{OUT_DIR}/figure_4_5_crossattn_grouped.png',
     'Cross-attention of the v3 caption decoder (auxiliary)')
grid(first, [ALL[0], ALL[2]], f'{OUT_DIR}/figure_4_6_rollout_grouped.png', 'Attention Rollout (CLS-Only)')
grid(first, ALL, f'{OUT_DIR}/figure_4_7_integrated_xai_grouped.png', 'Integrated XAI comparison')

# ---- 5. Faithfulness: deletion test on all 375 test images -----------
def patch_mask(order, frac):
    m = torch.ones(196); m[order[-int(round(frac * 196)):]] = 0
    return m.view(1, 1, 14, 14).repeat_interleave(16, 2).repeat_interleave(16, 3)

gen_rng = np.random.default_rng(42)
rows = []
for b in loader(test_idx, bs=1):
    pv, ids, am = (b[x].to(DEVICE) for x in ('pixel_values', 'input_ids', 'attention_mask'))
    y = int(b['labels'])
    with torch.no_grad():
        p0 = F.softmax(cls_model.forward_cls(pv, ids, am)[0].float(), 0)
    pred = int(p0.argmax())
    orders = {'Grad-CAM': np.argsort(gradcam(pv, ids, am, pred)),
              'Rollout': np.argsort(rollout(pv)),
              'Random': gen_rng.permutation(196)}
    batch, keys = [], []
    for meth, o in orders.items():
        for f in FRACS:
            batch.append(pv * patch_mask(o, f).to(DEVICE)); keys.append((meth, f))
    with torch.no_grad():
        P = F.softmax(cls_model.forward_cls(torch.cat(batch), ids.repeat(len(batch), 1),
                                            am.repeat(len(batch), 1)).float(), 1)
    for (meth, f), p in zip(keys, P):
        rows.append(dict(method=meth, frac=f, p_orig=p0[pred].item(), p_masked=p[pred].item(),
                         still_correct=int(p.argmax() == y), was_correct=int(pred == y)))
fa = pd.DataFrame(rows)
fa['drop'] = fa.p_orig - fa.p_masked
faith = fa.groupby(['method', 'frac']).agg(mean_prob_drop=('drop', 'mean'),
                                            accuracy_after=('still_correct', 'mean')).reset_index()
faith['mean_prob_drop'] *= 100; faith['accuracy_after'] *= 100
faith.to_csv(f'{OUT_DIR}/phase4_faithfulness.csv', index=False)
base_acc = fa.was_correct.mean() * 100

# ---- 6. Report ---------------------------------------------------------
pd.set_option('display.width', 160)
print('\n' + '=' * 70)
print('PHASE 4 FINAL SUMMARY  (CLS-Only seed 42 + multi-task v3, grouped split)')
print('=' * 70)
print(f'\n[Calibration] T* = {T_star:.4f} (fitted on grouped val, n = {len(val_idx)}; '
      f'val ECE before = {val_before["ece"]:.2f}%)')
print(cal_df.round(4).to_string(index=False))
print(f'\n[XAI samples] {int(xai.correct.sum())}/{len(xai)} correctly classified | per class: ' +
      ' | '.join(f'{c}: {int(xai[xai.true == c].correct.sum())}/{PER_CLASS}' for c in LABEL_NAMES))
print(f'  Grad-CAM all-zero maps: {int(xai.gradcam_zero.sum())}/{len(xai)}')
print(f'  Top-10% patch IoU  Grad-CAM vs Rollout: {xai.iou_gc_ro.mean():.3f} | '
      f'Grad-CAM vs Cross-attn: {xai.iou_gc_ca.mean():.3f}  (random ≈ 0.054)')
print(xai[['n', 'true', 'pred', 'correct', 'conf_raw', 'conf_cal', 'iou_gc_ro', 'iou_gc_ca']]
      .round(3).to_string(index=False))
print(f'\n[Faithfulness] deletion test, test n = {len(test_idx)}, accuracy before masking = {base_acc:.2f}%')
print('  (larger probability drop than Random = the map points to evidence the model uses)')
print(faith.round(2).to_string(index=False))
print(f'\nFiles saved in: {OUT_DIR}')
