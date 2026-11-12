# =====================================================================
# DEFENSE_FIX_PLAN — Phase 3B: retrain the multi-task model (fixed)
# Runtime: Colab GPU (T4).  Total ~45-60 min.  Safe to re-run: if the
# trained checkpoint + train log already exist, training is skipped.
# ---------------------------------------------------------------------
# What it does
#   1. Loads the leakage-free split saved by Phase 3 (grouped_split.csv).
#   2. Trains DisasterNetMultimodal (src/model.py, ViT + BanglaBERT +
#      LoRA r=8 + 3-layer caption decoder) with
#        L = 1.0 * L_cls + 0.5 * L_cap,  RIGHT-SHIFTED teacher forcing,
#      seed 42, 15 epochs, AdamW 1e-4 (wd .01), cosine, AMP, bs 16.
#      Best epoch = lowest validation total loss.
#   3. Saves it to Drive: models/drishti_multitask_v3.pt (v2 untouched).
#   4. Benchmarks on the 375-sample grouped test set:
#        classification with (a) the original tweet text and
#        (b) the model's own generated caption (image-only, as in the demo)
#        captions: beam 3, BLEU-1..4, METEOR, ROUGE-L, BERTScore
#        + caption diversity check (mode collapse).
# Paste back to Claude: the [diag] lines, the epoch lines, and
# everything from "PHASE 3B FINAL SUMMARY" down.
# =====================================================================

import os, re, sys, json, time, glob, shutil, zipfile, subprocess
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
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rouge-score', 'bert-score'], check=True)

from google.colab import drive
drive.mount('/content/drive')

import numpy as np, pandas as pd

DRIVE_ROOT = '/content/drive/MyDrive/DisasterNet'
REPO_DIR   = '/content/DRISHTI-Bn'
DATA_DIR   = '/content/data/processed'
CSV_NAME   = 'master_dataset_translated.csv'
SPLIT_CSV  = f'{DRIVE_ROOT}/defense_fixes/phase3/grouped_split.csv'
OUT_DIR    = f'{DRIVE_ROOT}/defense_fixes/phase3b'
CKPT_PATH  = f'{DRIVE_ROOT}/models/drishti_multitask_v3.pt'
LOG_CSV    = f'{OUT_DIR}/phase3b_train_log.csv'
os.makedirs(OUT_DIR, exist_ok=True)

SEED, EPOCHS, BS, LR = 42, 15, 16, 1e-4
LAMBDA_CLS, LAMBDA_CAP = 1.0, 0.5
BEAM, MAX_GEN = 3, 40
LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']

# ---- 1. Code -----------------------------------------------------------
if os.path.isdir(REPO_DIR):
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '-q'], check=True)
else:
    subprocess.run(['git', 'clone', '-q', 'https://github.com/jahidstm/DRISHTI-Bn.git', REPO_DIR], check=True)
sys.path.insert(0, f'{REPO_DIR}/src')
print('[+] Repo ready:', subprocess.run(['git', '-C', REPO_DIR, 'log', '--oneline', '-1'],
                                        capture_output=True, text=True).stdout.strip())

# ---- 2. Data -----------------------------------------------------------
if not os.path.exists(f'{DATA_DIR}/{CSV_NAME}'):
    os.makedirs('/content/data', exist_ok=True)
    zips = glob.glob(f'{DRIVE_ROOT}/**/processed*.zip', recursive=True)
    if not zips:
        raise FileNotFoundError('processed.zip not found in My Drive > DisasterNet')
    print('[+] Unzipping', zips[0])
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall('/content/data')
CSV_PATH = f'{DATA_DIR}/{CSV_NAME}'
df = pd.read_csv(CSV_PATH)

if not os.path.exists(SPLIT_CSV):
    raise FileNotFoundError(f'{SPLIT_CSV} missing - Phase 3 must be run first.')
split = pd.read_csv(SPLIT_CSV)
assert (split.image_path.values == df.image_path.values).all(), 'dataset order changed since Phase 3'
train_idx, val_idx, test_idx = (np.where(split.split.values == s)[0] for s in ('train', 'val', 'test'))
print(f'[+] Phase 3 grouped split: train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}')
assert (len(train_idx), len(val_idx), len(test_idx)) == (3007, 373, 375), 'unexpected split sizes'

# ---- 3. Model / data ---------------------------------------------------
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import cls_only_ablation as C                      # set_seed / seed_worker (same as Phase 3)
from data_loader import DisasterNetMultimodalDataset
from model import DisasterNetMultimodal

DEVICE = torch.device('cuda')
assert torch.cuda.is_available(), 'Switch the Colab runtime to GPU (T4)'
C.set_seed(SEED)

ds  = DisasterNetMultimodalDataset(CSV_PATH, DATA_DIR)          # max_length 128, as train.py
tok = ds.tokenizer
PAD, CLS_ID, SEP = tok.pad_token_id, tok.cls_token_id, tok.sep_token_id
VOCAB = 32000
print(f'[diag] tokenizer: vocab {tok.vocab_size} | PAD {PAD} | CLS {CLS_ID} | SEP {SEP}')
assert tok.vocab_size <= VOCAB

g = torch.Generator(); g.manual_seed(SEED)
kw = dict(num_workers=2, pin_memory=True)
tr_dl = DataLoader(Subset(ds, train_idx), batch_size=BS, shuffle=True,
                   worker_init_fn=C.seed_worker, generator=g, **kw)
va_dl = DataLoader(Subset(ds, val_idx), batch_size=BS, shuffle=False, **kw)
te_dl = DataLoader(Subset(ds, test_idx), batch_size=BS, shuffle=False, **kw)

model = DisasterNetMultimodal(num_classes=3, vocab_size=VOCAB, use_lora=True).to(DEVICE)
n_tr  = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'[diag] trainable: ViT-LoRA {n_tr(model.vision_encoder):,} | BERT-LoRA {n_tr(model.text_encoder):,} | '
      f'cls head {n_tr(model.classifier):,} | decoder '
      f'{n_tr(model.decoder_embedding) + n_tr(model.caption_decoder) + n_tr(model.vocab_projector):,} | '
      f'total {n_tr(model):,} of {sum(p.numel() for p in model.parameters()):,}')

def fused_forward(model, pv, ids, mask):
    """One encoder pass for both tasks (same weights and maths as model.forward).

    Classification: [ViT pooler ; BanglaBERT CLS] -> classifier.
    Captioning:     right-shifted teacher forcing - decoder sees [CLS, w1..wN-1]
                    and predicts [w1..wN, SEP], cross-attending to ViT patches.
    """
    vo = model.vision_encoder(pixel_values=pv)
    to = model.text_encoder(input_ids=ids, attention_mask=mask)
    logits_cls = model.classifier(torch.cat((vo.pooler_output, to.last_hidden_state[:, 0, :]), dim=1))
    dec_in = ids[:, :-1]
    tgt = model.pos_encoder(model.decoder_embedding(dec_in))
    out = model.caption_decoder(tgt=tgt, memory=vo.last_hidden_state,
                                tgt_mask=model.generate_causal_mask(dec_in.size(1), ids.device))
    return logits_cls, model.vocab_projector(out), ids[:, 1:]

crit_cls = nn.CrossEntropyLoss()
crit_cap = nn.CrossEntropyLoss(ignore_index=PAD)

def run_epoch(loader, train, opt=None, scaler=None):
    model.train(train)
    tot = cls_l = cap_l = 0.0; yt, yp = [], []
    with torch.set_grad_enabled(train):
        for b in loader:
            pv, ids, am, y = (b[k].to(DEVICE, non_blocking=True)
                              for k in ('pixel_values', 'input_ids', 'attention_mask', 'labels'))
            with torch.amp.autocast('cuda'):
                lc, lp, tgt = fused_forward(model, pv, ids, am)
                l1 = crit_cls(lc, y)
                l2 = crit_cap(lp.reshape(-1, VOCAB).float(), tgt.reshape(-1))
                loss = LAMBDA_CLS * l1 + LAMBDA_CAP * l2
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); cls_l += l1.item(); cap_l += l2.item()
            yt += y.tolist(); yp += lc.argmax(1).tolist()
    n = len(loader)
    return tot / n, cls_l / n, cap_l / n, f1_score(yt, yp, average='macro', zero_division=0) * 100

# ---- 4. Train (skipped if already done) --------------------------------
if os.path.exists(CKPT_PATH) and os.path.exists(LOG_CSV):
    print(f'[+] Found {CKPT_PATH} + train log - skipping training')
    log = pd.read_csv(LOG_CSV)
else:
    t0 = time.time()
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    rows, best = [], (float('inf'), 0, None)
    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(tr_dl, True, opt, scaler); sch.step()
        va = run_epoch(va_dl, False)
        rows.append(dict(epoch=ep, train_total=tr[0], train_cls=tr[1], train_cap=tr[2], train_f1=tr[3],
                         val_total=va[0], val_cls=va[1], val_cap=va[2], val_f1=va[3]))
        flag = ''
        if va[0] < best[0]:
            best = (va[0], ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            flag = '  *best'
        print(f'  [MT ep {ep:02d}] train L={tr[0]:.4f} (cls {tr[1]:.4f}, cap {tr[2]:.4f}) | '
              f'val L={va[0]:.4f} (cls {va[1]:.4f}, cap {va[2]:.4f}) val_F1={va[3]:.2f}%{flag}  '
              f'[{(time.time() - t0) / 60:.0f} min]')
    log = pd.DataFrame(rows); log['best'] = log.epoch == best[1]
    torch.save(best[2], CKPT_PATH)
    log.to_csv(LOG_CSV, index=False)
    print(f'[+] Best epoch {best[1]} (val total loss {best[0]:.4f}) -> saved {CKPT_PATH}')
    del best

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=True)
model.eval()
best_ep = int(log.loc[log.best, 'epoch'].iloc[0])

# ---- 5. Benchmark ------------------------------------------------------
def cls_metrics(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    pf1 = f1_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0) * 100
    return dict(macro_f1=f1_score(yt, yp, average='macro', zero_division=0) * 100,
                macro_p=precision_score(yt, yp, average='macro', zero_division=0) * 100,
                macro_r=recall_score(yt, yp, average='macro', zero_division=0) * 100,
                accuracy=(yt == yp).mean() * 100,
                f1_severe=pf1[0], f1_human=pf1[1], f1_affected=pf1[2],
                confusion=confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist())

@torch.no_grad()
def classify(pv, text_ids, text_mask):
    vo = model.vision_encoder(pixel_values=pv)
    to = model.text_encoder(input_ids=text_ids, attention_mask=text_mask)
    return model.classifier(torch.cat((vo.pooler_output, to.last_hidden_state[:, 0, :]), 1))

@torch.no_grad()
def beam_caption(pv1):
    """Same algorithm as evaluate.DisasterNetPredictor.generate_caption
    (repetition penalty, immediate/bigram blocking, length penalty 0.7),
    but the ViT memory is computed once instead of at every step."""
    memory = model.vision_encoder(pixel_values=pv1).last_hidden_state
    def next_logits(seq):
        ids = torch.tensor([seq], device=DEVICE)
        tgt = model.pos_encoder(model.decoder_embedding(ids))
        out = model.caption_decoder(tgt=tgt, memory=memory,
                                    tgt_mask=model.generate_causal_mask(len(seq), DEVICE))
        return model.vocab_projector(out)[:, -1, :].float()
    beams = [([CLS_ID], 0.0)]
    for _ in range(MAX_GEN):
        cand, finished = [], True
        for seq, sc in beams:
            if seq[-1] == SEP:
                cand.append((seq, sc)); continue
            finished = False
            lg = next_logits(seq)
            lg[:, CLS_ID] = -float('inf'); lg[:, PAD] = -float('inf')
            for t in set(seq):
                lg[:, t] = lg[:, t] * 2.0 if lg[:, t] < 0 else lg[:, t] / 2.0
            lg[:, seq[-1]] = -float('inf')
            if len(seq) >= 2:
                for i in range(len(seq) - 1):
                    if seq[i] == seq[-1]:
                        lg[:, seq[i + 1]] = -float('inf')
            lp = torch.log_softmax(lg[0], -1)
            tp, ti = torch.topk(lp, BEAM)
            cand += [(seq + [ti[k].item()], sc + tp[k].item()) for k in range(BEAM)]
        if finished: break
        beams = sorted(cand, key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)[:BEAM]
    return tok.decode(beams[0][0], skip_special_tokens=True).strip()

t0 = time.time()
yt, yp_text, yp_gen, rows = [], [], [], []
for k, b in enumerate(te_dl):
    pv, ids, am, y = (b[x].to(DEVICE) for x in ('pixel_values', 'input_ids', 'attention_mask', 'labels'))
    with torch.amp.autocast('cuda'):
        lt = classify(pv, ids, am)                                  # (a) original tweet text
    gens = [beam_caption(pv[i:i + 1]) for i in range(len(pv))]      # caption (fp32)
    enc = tok([c if c else 'বন্যা পরিস্থিতি' for c in gens], padding='max_length',
              max_length=128, truncation=True, return_tensors='pt').to(DEVICE)
    with torch.amp.autocast('cuda'):
        lg = classify(pv, enc.input_ids, enc.attention_mask)        # (b) generated caption
    yt += y.tolist(); yp_text += lt.argmax(1).tolist(); yp_gen += lg.argmax(1).tolist()
    for i in range(len(pv)):
        j = test_idx[k * BS + i]
        ref = df.bengali_caption.iloc[j]
        rows.append(dict(index=j, image_path=df.image_path.iloc[j], true_label=df.macro_label.iloc[j],
                         pred_with_text=LABEL_NAMES[yp_text[-len(pv) + i]],
                         pred_with_generated=LABEL_NAMES[yp_gen[-len(pv) + i]],
                         reference='' if pd.isna(ref) else str(ref), generated=gens[i]))
    if k % 5 == 0:
        print(f'  [bench] {len(yt)}/{len(test_idx)}  e.g. "{gens[0][:70]}"  [{(time.time() - t0) / 60:.1f} min]')

per = pd.DataFrame(rows)
per.to_csv(f'{OUT_DIR}/phase3b_per_sample.csv', index=False, encoding='utf-8-sig')

from metrics import MetricsSuite
suite = MetricsSuite(use_bertscore=True, bertscore_lang='bn')
refs, hyps = per.reference.tolist(), per.generated.tolist()
cap_raw = suite.evaluate_captions(refs, hyps)
strip = lambda s: re.sub(r'\s+', ' ', re.sub(r'https?\s*:\s*/\s*/\s*\S+|t\.co\s*/\s*\S+', ' ', s)).strip()
cap_nourl = suite.evaluate_captions([strip(r) for r in refs], [strip(h) for h in hyps])

m_text, m_gen = cls_metrics(yt, yp_text), cls_metrics(yt, yp_gen)
vc = per.generated.value_counts()
diversity = dict(n_unique=int(per.generated.nunique()), n=len(per),
                 top1_share=float(vc.iloc[0] / len(per) * 100), top1=vc.index[0],
                 mean_words=float(per.generated.str.split().str.len().mean()),
                 empty=int((per.generated == '').sum()))

summary = [dict(metric=f'cls_text_{k}', value=v) for k, v in m_text.items() if k != 'confusion'] + \
          [dict(metric=f'cls_generated_{k}', value=v) for k, v in m_gen.items() if k != 'confusion'] + \
          [dict(metric=f'cap_{k}', value=v) for k, v in cap_raw.items()] + \
          [dict(metric=f'cap_nourl_{k}', value=v) for k, v in cap_nourl.items()] + \
          [dict(metric=f'div_{k}', value=v) for k, v in diversity.items()] + \
          [dict(metric='best_epoch', value=best_ep)]
pd.DataFrame(summary).to_csv(f'{OUT_DIR}/phase3b_benchmark_summary.csv', index=False, encoding='utf-8-sig')

# Confusion matrices figure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
short = ['Severe\nDamage', 'Humanitarian\nRescue', 'Affected\nPeople']
for ax, (title, m) in zip(axes, [('(a) image + original tweet text', m_text),
                                 ('(b) image + generated caption', m_gen)]):
    cm = np.array(m['confusion'])
    ax.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(short); ax.set_yticklabels(short)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{title}\nMacro F1 = {m["macro_f1"]:.2f}%', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/phase3b_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ---- 6. Report ---------------------------------------------------------
print('\n' + '=' * 70)
print(f'PHASE 3B FINAL SUMMARY  (multi-task v3, grouped split, test n = {len(test_idx)})')
print('=' * 70)
print(f'Best epoch {best_ep} of {EPOCHS} (lowest val total loss) | lambda_cls {LAMBDA_CLS} | lambda_cap {LAMBDA_CAP}')
print('\n--- train log ---')
print(log.round(4).to_string(index=False))
for title, m in [('Classification (a): image + ORIGINAL tweet text', m_text),
                 ('Classification (b): image + GENERATED caption (image-only)', m_gen)]:
    print(f'\n{title}')
    print('  ' + ' | '.join(f'{k} {v:.2f}' for k, v in m.items() if k != 'confusion'))
    print(f'  confusion (rows SD, HR, AP): {m["confusion"]}')
print('\nCaptions (beam 3), scores x100:')
print('  raw refs   : ' + ' | '.join(f'{k} {v:.2f}' for k, v in cap_raw.items()))
print('  URLs strip : ' + ' | '.join(f'{k} {v:.2f}' for k, v in cap_nourl.items()))
print(f'\nDiversity: {diversity["n_unique"]}/{diversity["n"]} unique captions | most common '
      f'{diversity["top1_share"]:.1f}% | mean {diversity["mean_words"]:.1f} words | empty {diversity["empty"]}')
print(f'  most common: "{diversity["top1"]}"')
print('\n--- 8 example captions (2-3 per class) ---')
for _, r in per.groupby('true_label', group_keys=False).head(3).head(8).iterrows():
    print(f'  [{r.true_label}]\n    REF: {r.reference[:120]}\n    GEN: {r.generated[:120]}')
print(f'\nFiles saved in: {OUT_DIR}\nCheckpoint: {CKPT_PATH}')
