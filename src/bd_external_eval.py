"""
BD external test set — step 2: evaluate the final model on Bangladeshi images
============================================================================
Inference only (no training). Runs on CPU in ~5-20 min.

Inputs
  bd_external/bd_labels.csv              labels from two annotators (see docs/bd_external_labelling_guide.md)
  data/bd_external/images/               the images
  models/cls_only_grouped_s42.pt         main classifier            (from Drive)
  models/drishti_multitask_v3.pt         text decoder (optional)    (from Drive)
What it does
  1. Checks the labels; inter-annotator agreement (Cohen's kappa, A vs B).
  2. Near-duplicate check against all 3,755 CrisisMMD training-pool images (dHash).
  3. Classifies each image with the fixed placeholder text (as in the demo; the
     model is image-driven, Phase 2) -> Macro F1 with a 95% bootstrap CI.
  4. Grad-CAM figure (images marked "identifiable face" in notes are skipped).
  5. Optional: generates text with the v3 decoder and scores it against the
     human-written descriptions (sacreBLEU, chrF) with a shuffled-reference baseline.
Outputs -> results/bd_external/

Usage
  python src/bd_external_eval.py                 # everything
  python src/bd_external_eval.py --no-captions   # classification only (faster)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
P = lambda *a: os.path.join(ROOT, *a)
LABELS_CSV = P("bd_external", "bd_labels.csv")
IMG_DIR    = P("data", "bd_external", "images")
CRISIS_DIR = P("data", "processed")
OUT_DIR    = P("results", "bd_external")
CLS_CKPT   = P("models", "cls_only_grouped_s42.pt")
CAP_CKPT   = P("models", "drishti_multitask_v3.pt")
CODES      = {"SD": 0, "HR": 1, "AP": 2}
NAMES      = ["Severe_Damage", "Humanitarian_Rescue", "Affected_People"]
PLACEHOLDER = "বন্যা পরিস্থিতি"
CRISISMMD_REF = 74.83          # same checkpoint on the CrisisMMD grouped test set (Phase 2, condition A)

ap = argparse.ArgumentParser()
ap.add_argument("--no-captions", action="store_true")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--labels", default=LABELS_CSV, help="labels CSV (default bd_external/bd_labels.csv)")
ap.add_argument("--out", default=OUT_DIR)
args = ap.parse_args()
LABELS_CSV, OUT_DIR = args.labels, args.out
os.makedirs(OUT_DIR, exist_ok=True)
report = []
def say(s=""):
    print(s); report.append(s)

# ---- 1. labels + agreement ---------------------------------------------------
df = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False)
for c in ("label_a", "label_b", "final_label"):
    df[c] = df[c].str.strip().str.upper()
bad = df[~df.label_a.isin(["", *CODES, "EX"]) | ~df.label_b.isin(["", *CODES, "EX"])
         | ~df.final_label.isin(["", *CODES, "EX"])]
if len(bad):
    sys.exit(f"Invalid label codes (use SD / HR / AP / EX) in: {', '.join(bad.image_id)}")

both = df[(df.label_a != "") & (df.label_b != "")]
auto = (df.final_label == "") & (df.label_a != "") & (df.label_a == df.label_b)
df.loc[auto, "final_label"] = df.loc[auto, "label_a"]
unresolved = df[(df.final_label == "") & (df.label_a != "") & (df.label_b != "")]
if len(unresolved):
    sys.exit(f"{len(unresolved)} disagreements without final_label: {', '.join(unresolved.image_id)}")

from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, confusion_matrix
say("# BD external test set — results\n")
say(f"Candidates in CSV: {len(df)} | labelled by both A and B: {len(both)}")
if len(both) >= 10:
    k = cohen_kappa_score(both.label_a, both.label_b)
    say(f"Inter-annotator agreement (A vs B, incl. EX): Cohen's kappa = {k:.3f} | "
        f"raw agreement = {(both.label_a == both.label_b).mean() * 100:.1f}% (n = {len(both)})")
test = df[df.final_label.isin(CODES)].reset_index(drop=True)
say(f"Excluded (EX): {(df.final_label == 'EX').sum()} | unlabelled: {(df.final_label == '').sum()}")
say("Test images per class: " + " | ".join(f"{n} {int((test.final_label == c).sum())}"
                                             for c, n in zip(CODES, NAMES)))
if len(test) < 15:
    sys.exit("Fewer than 15 labelled test images — label more before evaluating.")
missing = [f for f in test.file if not os.path.exists(os.path.join(IMG_DIR, f))]
if missing:
    sys.exit(f"Image files missing in {IMG_DIR}: {missing[:5]}")

# ---- 2. near-duplicate check vs CrisisMMD -------------------------------------
def dhash(path, size=8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())

cache = P("data", "bd_external", "crisismmd_dhash.npy")
if os.path.exists(cache):
    ch = np.load(cache)
else:
    print("[..] hashing 3,755 CrisisMMD images once (cached afterwards)")
    paths = pd.read_csv(P("data", "processed", "master_dataset_translated.csv")).image_path
    ch = np.stack([dhash(os.path.join(CRISIS_DIR, p)) for p in paths]); np.save(cache, ch)
bh = np.stack([dhash(os.path.join(IMG_DIR, f)) for f in test.file])
ham = np.unpackbits(bh[:, None, :] ^ ch[None, :, :], axis=2).sum(2).min(1)
test["min_hamming_to_crisismmd"] = ham
dups = test[ham <= 5]
say(f"Near-duplicates of CrisisMMD images (dHash Hamming <= 5): {len(dups)}"
    + (f" -> {', '.join(dups.image_id)} (check and exclude)" if len(dups) else " ✔"))

# ---- 3. classification ---------------------------------------------------------
import torch, torch.nn.functional as F
from transformers.models.vit.modeling_vit import ViTLayer
from transformers import AutoTokenizer
from torchvision import transforms
import cls_only_ablation as C

if not os.path.exists(CLS_CKPT) and not os.environ.get("DRISHTI_ALLOW_RANDOM"):
    sys.exit(f"{CLS_CKPT} not found — download it from Drive (My Drive > DisasterNet > models/).")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); C.DEVICE = DEVICE
model = C.DisasterNetAblation(num_classes=3).to(DEVICE)
if os.path.exists(CLS_CKPT):
    model.load_state_dict(torch.load(CLS_CKPT, map_location=DEVICE), strict=True)
model.eval()
layers = [m for m in model.vision_encoder.modules() if isinstance(m, ViTLayer)]
tok = AutoTokenizer.from_pretrained(C.BERT_ID)
enc = tok(PLACEHOLDER, padding="max_length", max_length=C.MAX_LEN, truncation=True, return_tensors="pt")
ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def predict_gradcam(pv):
    cap = {}
    def hook(m, i, o):
        a = o[0] if isinstance(o, tuple) else o; a.retain_grad(); cap["a"] = a
    h = layers[-2].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            lg = model.forward_cls(pv, ids, am); pred = int(lg[0].argmax()); lg[0, pred].backward()
    finally:
        h.remove()
    A, G = cap["a"][0, 1:].detach(), cap["a"].grad[0, 1:]
    cam = torch.relu((A * G.mean(0)).sum(1)).view(1, 1, 14, 14)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return pred, F.softmax(lg[0].detach().float(), 0).cpu().numpy(), (cam - cam.min()) / (np.ptp(cam) + 1e-8)

preds, probs, cams, pvs = [], [], [], []
for k, f in enumerate(test.file):
    pv = tf(Image.open(os.path.join(IMG_DIR, f)).convert("RGB")).unsqueeze(0).to(DEVICE)
    p, pr, cam = predict_gradcam(pv)
    preds.append(p); probs.append(pr); cams.append(cam); pvs.append(pv)
    if (k + 1) % 20 == 0: print(f"  classified {k + 1}/{len(test)}")
y = test.final_label.map(CODES).values; yp = np.array(preds)
test["pred"] = [list(CODES)[p] for p in yp]
test["confidence"] = [pr.max() for pr in probs]

present = sorted(set(y))
mf1 = lambda a, b: f1_score(a, b, average="macro", labels=present, zero_division=0) * 100
rng = np.random.default_rng(42)
boot = [(mf1(y[s], yp[s]), (y[s] == yp[s]).mean() * 100)
        for s in (rng.integers(0, len(y), len(y)) for _ in range(args.boot))]
lo_f, hi_f = np.percentile([b[0] for b in boot], [2.5, 97.5])
lo_a, hi_a = np.percentile([b[1] for b in boot], [2.5, 97.5])
pf1 = f1_score(y, yp, average=None, labels=[0, 1, 2], zero_division=0) * 100
cm = confusion_matrix(y, yp, labels=[0, 1, 2])
say("\n## Classification (image + placeholder text, CLS-Only seed 42)")
say(f"Macro F1 = {mf1(y, yp):.2f}%  (95% bootstrap CI {lo_f:.1f}–{hi_f:.1f}, {args.boot} resamples, n = {len(y)})")
say(f"Accuracy = {(y == yp).mean() * 100:.2f}%  (95% CI {lo_a:.1f}–{hi_a:.1f})")
say(f"Macro P = {precision_score(y, yp, average='macro', labels=present, zero_division=0) * 100:.2f}% | "
    f"Macro R = {recall_score(y, yp, average='macro', labels=present, zero_division=0) * 100:.2f}%")
say("Per-class F1: " + " | ".join(f"{n} {pf1[i]:.1f}" for i, n in enumerate(NAMES)))
say(f"Confusion (rows true SD, HR, AP; cols pred): {cm.tolist()}")
say(f"Mean confidence: correct {test.confidence[y == yp].mean():.3f} | wrong "
    f"{test.confidence[y != yp].mean() if (y != yp).any() else float('nan'):.3f}")
say(f"Reference — same model on the CrisisMMD grouped test set: {CRISISMMD_REF:.2f}% "
    f"(difference {mf1(y, yp) - CRISISMMD_REF:+.2f} pp = domain shift estimate)")

# ---- 4. Grad-CAM figure ----------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.cm as mcm
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
ok_show = ~test.notes.str.contains("identifiable face", case=False)
show = [i for c in range(3) for i in np.where((y == c) & ok_show.values)[0][:3]]
if show:
    fig, axes = plt.subplots(len(show), 2, figsize=(5.2, 2.6 * len(show)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(show):
        img = np.clip(pvs[i][0].permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1)
        axes[r, 0].imshow(img); axes[r, 1].imshow(0.55 * img + 0.45 * mcm.jet(cams[i])[..., :3])
        for a in axes[r]: a.set_xticks([]); a.set_yticks([])
        mark = "✓" if y[i] == yp[i] else "✗"
        axes[r, 0].set_ylabel(f"{test.image_id[i]}\nGT {list(CODES)[y[i]]} | pred {test.pred[i]} {mark}", fontsize=8)
    axes[0, 0].set_title("Image", fontsize=10); axes[0, 1].set_title("Grad-CAM", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "bd_gradcam_examples.png"), dpi=200, bbox_inches="tight")
    plt.close()

# ---- 5. optional: text generation vs human descriptions --------------------------
desc = test.description_bn.str.strip()
if not args.no_captions and os.path.exists(CAP_CKPT) and (desc != "").sum() >= 10:
    import sacrebleu
    from model import DisasterNetMultimodal
    mt = DisasterNetMultimodal(num_classes=3, vocab_size=32000, use_lora=True).to(DEVICE)
    mt.load_state_dict(torch.load(CAP_CKPT, map_location=DEVICE), strict=True); mt.eval()

    @torch.no_grad()
    def caption(pv, beam=3):          # same beam search as evaluate.py / app.py
        CLS_ID, SEP, PAD = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
        mem = mt.vision_encoder(pixel_values=pv).last_hidden_state
        def nxt(seq):
            t = torch.tensor([seq], device=DEVICE)
            o = mt.caption_decoder(tgt=mt.pos_encoder(mt.decoder_embedding(t)), memory=mem,
                                   tgt_mask=mt.generate_causal_mask(len(seq), DEVICE))
            return mt.vocab_projector(o)[:, -1, :].float()
        beams = [([CLS_ID], 0.0)]
        for _ in range(40):
            cand, done = [], True
            for seq, sc in beams:
                if seq[-1] == SEP: cand.append((seq, sc)); continue
                done = False; lg = nxt(seq)
                lg[:, CLS_ID] = -float("inf"); lg[:, PAD] = -float("inf")
                for t in set(seq): lg[:, t] = lg[:, t] * 2.0 if lg[:, t] < 0 else lg[:, t] / 2.0
                lg[:, seq[-1]] = -float("inf")
                for i in range(len(seq) - 1):
                    if seq[i] == seq[-1]: lg[:, seq[i + 1]] = -float("inf")
                tp, ti = torch.topk(torch.log_softmax(lg[0], -1), beam)
                cand += [(seq + [ti[j].item()], sc + tp[j].item()) for j in range(beam)]
            if done: break
            beams = sorted(cand, key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)[:beam]
        return tok.decode(beams[0][0], skip_special_tokens=True).strip()

    has = np.where(desc != "")[0]
    test["generated"] = ""
    for n, i in enumerate(has):
        test.loc[i, "generated"] = caption(pvs[i])
        if (n + 1) % 10 == 0: print(f"  captioned {n + 1}/{len(has)}")
    refs, hyps = list(desc.iloc[has]), list(test.generated.iloc[has])
    order = np.random.default_rng(42).permutation(len(refs))
    shuf = [None] * len(refs)                      # cyclic shift of a random order:
    for a, b in zip(order, np.roll(order, 1)):     # every image gets a different image's description
        shuf[a] = refs[b]
    say("\n## Text generation vs human-written Bengali descriptions (v3 decoder, beam 3)")
    for name, h in [("DRISHTI-Bn v3", hyps), ("Baseline: another image's description", shuf)]:
        b = sacrebleu.corpus_bleu(h, [refs]); c = sacrebleu.corpus_chrf(h, [refs])
        say(f"{name:<40} BLEU {b.score:6.2f} | chrF {c.score:6.2f}  (n = {len(refs)})")
    say(f"Generated texts containing URL fragments / hashtags: "
        f"{np.mean(['t. co' in h or 'http' in h for h in hyps]) * 100:.0f}% / {np.mean(['#' in h for h in hyps]) * 100:.0f}%")
elif not args.no_captions:
    say("\n(Text generation skipped: needs models/drishti_multitask_v3.pt and >= 10 descriptions.)")

test.drop(columns=[c for c in ("label_a", "label_b") if c in test]).to_csv(
    os.path.join(OUT_DIR, "bd_per_sample.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(OUT_DIR, "bd_eval_summary.md"), "w", encoding="utf-8").write("\n".join(report) + "\n")
print(f"\n[+] Saved to {OUT_DIR}")
