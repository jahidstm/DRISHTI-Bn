"""
Main-text XAI figures 4.4-4.7 (local, CPU)
==========================================
Phase 4 drew 21 random grouped-test samples (7 per class, random.Random(42)) and showed
the FIRST sample of each class in Figures 4.4-4.7. Those happened to be a news-TV screenshot
(CNN banner) and a photo of children whose disaster context is not visible.

This script keeps the same 21 random samples and selects the rows by a fixed, stated rule:
  * per class: the correctly classified sample with the highest confidence
  * plus the misclassified sample with the highest confidence (an honest failure case)
All 21 samples remain in the appendix figure (results/defense_fixes/phase4/xai_all_21_samples.png).

Needs: models/cls_only_grouped_s42.pt, models/drishti_multitask_v3.pt, data/processed/.
Usage: python src/make_xai_main_figures.py        (~2 min on CPU)
"""
import os, sys
import numpy as np, pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA_DIR = os.path.join(ROOT, "data", "processed")
CSV_PATH = os.path.join(DATA_DIR, "master_dataset_translated.csv")
SAMPLES = os.path.join(ROOT, "results", "defense_fixes", "phase4", "phase4_xai_samples.csv")
GEN_CSV = os.path.join(ROOT, "results", "defense_fixes", "phase3b", "phase3b_per_sample.csv")
SPLIT_CSV = os.path.join(ROOT, "results", "defense_fixes", "phase3", "grouped_split.csv")
CLS_CKPT = os.path.join(ROOT, "models", "cls_only_grouped_s42.pt")
MT_CKPT = os.path.join(ROOT, "models", "drishti_multitask_v3.pt")
LABEL_NAMES = ["Severe_Damage", "Humanitarian_Rescue", "Affected_People"]
SHORT = ["Severe Damage", "Humanitarian Rescue", "Affected People"]
DISCARD = 0.9

# ---- 1. rows by rule ---------------------------------------------------------
xs = pd.read_csv(SAMPLES, encoding="utf-8-sig")
xs["correct"] = xs.correct.astype(str) == "True"
rows = [xs[(xs.true == c) & xs.correct].sort_values("conf_raw").iloc[-1] for c in LABEL_NAMES]
rows.append(xs[~xs.correct].sort_values("conf_raw").iloc[-1])
print("Selected:", ", ".join(f"#{r.n} ({r.true} -> {r.pred}, {r.conf_raw:.3f})" for r in rows))

import torch, torch.nn.functional as F
from transformers import AutoTokenizer, ViTModel
from transformers.models.vit.modeling_vit import ViTLayer
import cls_only_ablation as C

DEVICE = torch.device("cpu"); C.DEVICE = DEVICE
_vit = ViTModel.from_pretrained
C.ViTModel = type("EagerViT", (), {"from_pretrained": staticmethod(
    lambda *a, **k: _vit(*a, **{**k, "attn_implementation": "eager"}))})

tok = AutoTokenizer.from_pretrained(C.BERT_ID)
ds = C.MultimodalDataset(CSV_PATH, DATA_DIR, tok)
split = pd.read_csv(SPLIT_CSV)
assert (split.image_path.values == ds.df.image_path.values).all(), "dataset order changed"

cls_model = C.DisasterNetAblation(num_classes=3).to(DEVICE)
cls_model.load_state_dict(torch.load(CLS_CKPT, map_location=DEVICE), strict=True); cls_model.eval()
vit_layers = [m for m in cls_model.vision_encoder.modules() if isinstance(m, ViTLayer)]
assert len(vit_layers) == 12

from model import DisasterNetMultimodal
mt = DisasterNetMultimodal(num_classes=3, vocab_size=32000, use_lora=True).to(DEVICE)
mt.load_state_dict(torch.load(MT_CKPT, map_location=DEVICE), strict=True); mt.eval()
gen = pd.read_csv(GEN_CSV, keep_default_na=False)
gen_by_idx = dict(zip(gen["index"], gen.generated))


# ---- 2. XAI methods (identical to colab/phase4_xai_calibration.py) ---------------
def gradcam(pv, ids, am, target):
    cap = {}
    def hook(mod, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        a.retain_grad(); cap["a"] = a
    h = vit_layers[-2].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            cls_model.zero_grad(set_to_none=True)
            cls_model.forward_cls(pv, ids, am)[0, target].backward()
    finally:
        h.remove()
    A, G = cap["a"][0, 1:].detach(), cap["a"].grad[0, 1:]
    return torch.relu((A * G.mean(0)).sum(1)).numpy()


@torch.no_grad()
def rollout(pv):
    att = cls_model.vision_encoder(pixel_values=pv, output_attentions=True).attentions
    assert att is not None and len(att) == 12, "eager attention needed"
    res = torch.eye(197)
    for a in att:
        f = a[0].mean(0).clone()
        f[f < f.flatten().kthvalue(int(DISCARD * f.numel())).values] = 0
        f = f + torch.eye(197); f = f / f.sum(-1, keepdim=True); res = f @ res
    return res[0, 1:].numpy()


@torch.no_grad()
def cross_attention(pv, caption):
    ids = tok(caption if caption else " ", return_tensors="pt").input_ids[:, :-1]
    memory = mt.vision_encoder(pixel_values=pv).last_hidden_state
    captured, originals = [], []
    for layer in mt.caption_decoder.layers:
        o = layer.multihead_attn.forward; originals.append(o)
        def patched(*a, _o=o, **k):
            k["need_weights"] = True; k["average_attn_weights"] = True
            out, w = _o(*a, **k); captured.append(w); return out, w
        layer.multihead_attn.forward = patched
    try:
        tgt = mt.pos_encoder(mt.decoder_embedding(ids))
        mt.caption_decoder(tgt=tgt, memory=memory, tgt_mask=mt.generate_causal_mask(ids.size(1), DEVICE))
    finally:
        for layer, o in zip(mt.caption_decoder.layers, originals):
            layer.multihead_attn.forward = o
    return torch.stack(captured).mean(0)[0][:, 1:].mean(0).numpy()


def norm01(c):
    c = c - c.min(); return c / c.max() if c.max() > 0 else c
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
def up(cam):
    t = torch.tensor(norm01(cam), dtype=torch.float32).view(1, 1, 14, 14)
    return F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()


recs = []
for r in rows:
    b = ds[int(r["index"])]
    pv, ids, am = (b[x].unsqueeze(0) for x in ("pixel_values", "input_ids", "attention_mask"))
    with torch.no_grad():
        p = F.softmax(cls_model.forward_cls(pv, ids, am)[0].float(), 0).numpy()
    pred = int(p.argmax())
    assert LABEL_NAMES[pred] == r.pred and abs(p.max() - r.conf_raw) < 1e-3, f"#{r.n}: does not reproduce Phase 4"
    g = gradcam(pv, ids, am, pred)
    assert g.max() > 0, f"#{r.n}: blank Grad-CAM"
    recs.append(dict(n=int(r.n), true=r.true, pred=r.pred, correct=bool(r.correct), conf=p.max(),
                     _img=np.clip(pv[0].permute(1, 2, 0).numpy() * STD + MEAN, 0, 1),
                     _g=up(g), _r=up(rollout(pv)), _c=up(cross_attention(pv, gen_by_idx[int(r["index"])]))))
    print(f"  #{r.n} reproduced (pred {r.pred}, conf {p.max():.3f})")

# ---- 3. figures ------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def grid(cols, path, title):
    fig, axes = plt.subplots(len(recs), len(cols), figsize=(2.6 * len(cols), 2.75 * len(recs)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(recs):
        for j, (name, key) in enumerate(cols):
            ax = axes[i, j]; ax.imshow(r["_img"])
            if key: ax.imshow(r[key], cmap="jet", alpha=0.45)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(name, fontsize=10)
            if j == 0:
                ok = "✓" if r["correct"] else "✗ (error)"
                ax.set_ylabel(f"#{r['n']} GT: {SHORT[LABEL_NAMES.index(r['true'])]}\n"
                              f"Pred: {SHORT[LABEL_NAMES.index(r['pred'])]} {ok}\nconf {r['conf']:.2f}",
                              fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print("[+]", os.path.relpath(path, ROOT))


ALL = [("Image", None), ("Grad-CAM", "_g"), ("Attention Rollout", "_r"), ("Cross-attn (v3 decoder)", "_c")]
grid(ALL[:2], os.path.join(ROOT, "figure_4_4_gradcam_grouped.png"), "Grad-CAM (CLS-Only, predicted class)")
grid([ALL[0], ALL[3]], os.path.join(ROOT, "figure_4_5_crossattn_grouped.png"),
     "Cross-attention of the v3 caption decoder (auxiliary)")
grid([ALL[0], ALL[2]], os.path.join(ROOT, "figure_4_6_rollout_grouped.png"), "Attention Rollout (CLS-Only)")
grid(ALL, os.path.join(ROOT, "figure_4_7_integrated_xai_grouped.png"), "Integrated XAI comparison")
