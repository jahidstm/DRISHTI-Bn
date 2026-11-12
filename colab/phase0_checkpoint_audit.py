# =====================================================================
# DEFENSE_FIX_PLAN — Phase 0.7: Checkpoint audit (Colab, CPU runtime is enough)
# ---------------------------------------------------------------------
# Goal: find out which checkpoint the reported numbers came from, and
# whether the multi-task checkpoint was trained AFTER
#   - the teacher-forcing fix   (2026-08-07)
#   - the 80/10/10 split         (2026-09-12)
# Paste the FULL printed output back to Claude.
# =====================================================================

from google.colab import drive
drive.mount('/content/drive')

import os, hashlib, datetime, torch, transformers, peft

ROOT = '/content/drive/MyDrive/DisasterNet'
LOCAL_V2_MD5 = '16bb292e5d62115cc5c966dc70c120d4'  # models/disasternet_multitask_v2.pth on the laptop (2026-06-25)

print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}\n")

def md5(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()

def ts(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')

# ---- 1. Every checkpoint under the Drive folder -----------------------
ckpts = []
for dirpath, _, files in os.walk(ROOT):
    for fn in files:
        if fn.endswith(('.pt', '.pth', '.bin', '.safetensors')):
            ckpts.append(os.path.join(dirpath, fn))

print(f"=== {len(ckpts)} checkpoint file(s) found ===")
for p in sorted(ckpts, key=os.path.getmtime):
    size_mb = os.path.getsize(p) / 1e6
    h = md5(p)
    tag = '  <-- SAME as laptop v2' if h == LOCAL_V2_MD5 else ''
    print(f"\n{p.replace(ROOT, '')}")
    print(f"  modified={ts(p)} | size={size_mb:.1f} MB | md5={h}{tag}")
    try:
        sd = torch.load(p, map_location='cpu', weights_only=True)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        keys = list(sd.keys())
        vit_lora  = sum(k.startswith('vision_encoder') and 'lora' in k for k in keys)
        bert_lora = sum(k.startswith('text_encoder') and 'lora' in k for k in keys)
        dec       = [k for k in keys if k.startswith('caption_decoder')]
        vit_names = sorted({k.split('.')[-4] for k in keys if k.startswith('vision_encoder') and 'lora_A' in k})
        print(f"  keys={len(keys)} | vit_lora={vit_lora} {vit_names} | bert_lora={bert_lora} | decoder_keys={len(dec)}")
        # Decoder vocab projector stats: an untrained (lambda_cap=0) decoder keeps its
        # default init, whose std stays close to 1/sqrt(768*3) ~= 0.0208
        if 'vocab_projector.weight' in sd:
            w = sd['vocab_projector.weight'].float()
            print(f"  vocab_projector.weight std={w.std():.5f} (default init ~0.0208)")
        del sd
    except Exception as e:
        print(f"  [could not load: {e}]")

# ---- 2. Evaluation outputs that might record which checkpoint was used ----
print("\n=== Result / log files (csv, txt, json, md) ===")
for dirpath, _, files in os.walk(ROOT):
    for fn in files:
        if fn.endswith(('.csv', '.txt', '.json', '.md', '.log')):
            p = os.path.join(dirpath, fn)
            print(f"  {ts(p)}  {p.replace(ROOT, '')}")
