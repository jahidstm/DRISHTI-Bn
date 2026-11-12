import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path so we can import evaluate / cls_only_ablation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as mpl_cm
from transformers.models.vit.modeling_vit import ViTLayer
from evaluate import DisasterNetPredictor
import cls_only_ablation as C

# ─────────────────────────────────────────────────────────────
# 1.  Boot the models (once, at startup)
#     Classification : CLS-Only, seed 42, grouped split. Macro F1 74.83% with the tweet and
#                      76.52% with the placeholder below - the model is image-driven (Phase 2).
#     Text           : multi-task v3 decoder (experimental; imitates tweet style, Phase 3B).
#     Both checkpoints: Google Drive  My Drive > DisasterNet > models/  ->  local models/
# ─────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
CLS_CKPT    = os.path.join(HERE, 'models', 'cls_only_grouped_s42.pt')
CAP_CKPT    = os.path.join(HERE, 'models', 'drishti_multitask_v3.pt')
PLACEHOLDER = "বন্যা পরিস্থিতি"   # fixed text input: no tweet is available in an image-only demo

for _p in (CLS_CKPT, CAP_CKPT):
    if not os.path.exists(_p) and not os.environ.get("DRISHTI_ALLOW_RANDOM"):
        raise FileNotFoundError(f"{_p} not found. Download it from Google Drive "
                                f"(My Drive > DisasterNet > models/) into the local models/ folder.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C.DEVICE = DEVICE

cls_model = C.DisasterNetAblation(num_classes=3).to(DEVICE)
if os.path.exists(CLS_CKPT):
    cls_model.load_state_dict(torch.load(CLS_CKPT, map_location=DEVICE), strict=True)
cls_model.eval()
VIT_LAYERS = [m for m in cls_model.vision_encoder.modules() if isinstance(m, ViTLayer)]

predictor = DisasterNetPredictor(model_path=CAP_CKPT, device=DEVICE)   # v3 decoder + tokenizer + transform
tok = predictor.tokenizer
_ph = tok(PLACEHOLDER, padding="max_length", max_length=C.MAX_LEN, truncation=True, return_tensors="pt")
PH_IDS, PH_MASK = _ph["input_ids"].to(DEVICE), _ph["attention_mask"].to(DEVICE)

# ─────────────────────────────────────────────────────────────
# 2.  Inference functions
# ─────────────────────────────────────────────────────────────
LABELS = [
    "Severe Damage  |  মারাত্মক ক্ষয়ক্ষতি",
    "Humanitarian Rescue  |  ত্রাণ ও উদ্ধারকার্য",
    "Affected People  |  ক্ষতিগ্রস্ত মানুষ",
]
EMOJI = ["🔴", "🟡", "🟠"]
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def classify_with_gradcam(pv):
    """CLS-Only prediction + Grad-CAM on the input of the last ViT block (thesis method,
    Phase 4). The last block's output patch tokens have zero gradient, so they cannot be used."""
    cap = {}
    def hook(mod, inp, out):
        a = out[0] if isinstance(out, tuple) else out
        a.retain_grad(); cap["a"] = a
    h = VIT_LAYERS[-2].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            cls_model.zero_grad(set_to_none=True)
            logits = cls_model.forward_cls(pv, PH_IDS, PH_MASK)
            pred = int(logits[0].argmax())
            logits[0, pred].backward()
    finally:
        h.remove()
    probs = F.softmax(logits[0].detach().float(), 0).cpu().numpy()
    A, G = cap["a"][0, 1:].detach(), cap["a"].grad[0, 1:]
    cam = torch.relu((A * G.mean(0)).sum(1)).view(1, 1, 14, 14)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    img = np.clip(pv[0].permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1)
    overlay = 0.55 * img + 0.45 * mpl_cm.jet(cam)[..., :3]
    return pred, probs, (overlay * 255).astype(np.uint8)


@torch.no_grad()
def generate_caption(pv, beam_width):
    """Same beam search as evaluate.DisasterNetPredictor.generate_caption, but the ViT memory
    is computed once (much faster on CPU). Verified identical output in Phase 3B."""
    m = predictor.model
    CLS_ID, SEP, PAD = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    memory = m.vision_encoder(pixel_values=pv).last_hidden_state
    def next_logits(seq):
        ids = torch.tensor([seq], device=DEVICE)
        out = m.caption_decoder(tgt=m.pos_encoder(m.decoder_embedding(ids)), memory=memory,
                                tgt_mask=m.generate_causal_mask(len(seq), DEVICE))
        return m.vocab_projector(out)[:, -1, :].float()
    beams = [([CLS_ID], 0.0)]
    for _ in range(40):
        cand, finished = [], True
        for seq, sc in beams:
            if seq[-1] == SEP:
                cand.append((seq, sc)); continue
            finished = False
            lg = next_logits(seq)
            lg[:, CLS_ID] = -float("inf"); lg[:, PAD] = -float("inf")
            for t in set(seq):
                lg[:, t] = lg[:, t] * 2.0 if lg[:, t] < 0 else lg[:, t] / 2.0
            lg[:, seq[-1]] = -float("inf")
            for i in range(len(seq) - 1):
                if seq[i] == seq[-1]:
                    lg[:, seq[i + 1]] = -float("inf")
            lp = torch.log_softmax(lg[0], -1)
            tp, ti = torch.topk(lp, beam_width)
            cand += [(seq + [ti[k].item()], sc + tp[k].item()) for k in range(beam_width)]
        if finished:
            break
        beams = sorted(cand, key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)[:beam_width]
    return tok.decode(beams[0][0], skip_special_tokens=True).strip()


def analyze_image(pil_image, beam_width):
    if pil_image is None:
        return "⚠️ Please upload a disaster image.", {}, "—", None, "—"
    pv = predictor.transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    # Task 1: classification (image + fixed placeholder text) + Grad-CAM explanation
    idx, probs, heatmap = classify_with_gradcam(pv)

    # Task 2: text from the v3 decoder - shown exactly as generated, never edited
    caption = generate_caption(pv, int(beam_width))

    prob_dict = {LABELS[i]: float(probs[i]) for i in range(3)}
    return f"{EMOJI[idx]}  {LABELS[idx]}", prob_dict, f"{probs[idx] * 100:.2f}%", heatmap, caption


# ─────────────────────────────────────────────────────────────
# 3.  Gradio UI (custom dark CSS)
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@700;900&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0c14 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0 !important;
}

/* ── Header banner ── */
#header-banner {
    background: linear-gradient(135deg, #1a1f36 0%, #0f1526 50%, #12192f 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 32px 40px 24px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
#header-banner::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
#header-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(251,113,133,0.12) 0%, transparent 70%);
    pointer-events: none;
}

#banner-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #818cf8 0%, #a78bfa 40%, #fb7185 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
#banner-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.6;
}

/* ── Panel cards ── */
.panel-card {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* ── Upload zone ── */
#upload-zone {
    border: 2px dashed rgba(99,102,241,0.45) !important;
    border-radius: 14px !important;
    background: #0d1117 !important;
    transition: border-color 0.3s;
    min-height: 260px !important;
}
#upload-zone:hover {
    border-color: rgba(167,139,250,0.7) !important;
}

/* ── Run button ── */
#run-btn {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 28px !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    cursor: pointer !important;
    width: 100% !important;
}
#run-btn:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
#run-btn:active { transform: translateY(0)   !important; }

/* ── Beam slider ── */
#beam-slider input { accent-color: #818cf8; }

/* ── Result label box ── */
#label-result textarea, #label-result input {
    background: #1e2538 !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    text-align: center !important;
    padding: 16px !important;
}

/* ── Confidence text ── */
#conf-result textarea, #conf-result input {
    background: #1e2538 !important;
    border: 1px solid rgba(251,113,133,0.3) !important;
    color: #fb7185 !important;
    border-radius: 10px !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    padding: 16px !important;
}

/* ── Caption box ── */
#caption-result textarea, #caption-result input {
    background: #111827 !important;
    border: 1px solid rgba(16,185,129,0.25) !important;
    color: #6ee7b7 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    padding: 14px !important;
    line-height: 1.7 !important;
}

/* ── Bar chart ── */
#prob-chart {
    background: #111827 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Section labels ── */
label { color: #94a3b8 !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }

/* ── Examples row ── */
.examples-holder td { border-color: rgba(255,255,255,0.06) !important; }

/* ── Footer ── */
#footer-note {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    margin-top: 20px;
}
"""

BANNER_HTML = """
<div id="header-banner">
  <p id="banner-title">🛰️ DRISHTI-Bn</p>
  <p id="banner-sub">
    Explainable disaster-image severity classification &amp; experimental Bengali text generation<br>
    ViT-Base/16 &nbsp;+&nbsp; BanglaBERT &nbsp;+&nbsp; LoRA (r = 8) &nbsp;|&nbsp; Grad-CAM explanation
  </p>
</div>
"""

with gr.Blocks(title="DRISHTI-Bn Demo") as demo:

    gr.HTML(BANNER_HTML)

    with gr.Row(equal_height=False):

        # ── LEFT COLUMN ─────────────────────────────────
        with gr.Column(scale=4):
            gr.Markdown("### 📷 Upload Disaster Image")
            image_input = gr.Image(
                type="pil",
                label="Drop an image or click to upload",
                elem_id="upload-zone",
                height=300,
            )
            beam_slider = gr.Slider(
                minimum=1, maximum=5, step=1, value=3,
                label="Beam Search Width  (1 = Greedy, 3–5 = Beam Search)",
                elem_id="beam-slider",
            )
            run_btn = gr.Button("⚡  Analyze Disaster Scene", elem_id="run-btn", variant="primary")

        # ── RIGHT COLUMN ────────────────────────────────
        with gr.Column(scale=5):
            gr.Markdown("### 🔎 Analysis Results")

            with gr.Row():
                label_out = gr.Textbox(
                    label="Predicted Severity Class",
                    elem_id="label-result",
                    interactive=False,
                    lines=1,
                )
                conf_out = gr.Textbox(
                    label="Confidence Score",
                    elem_id="conf-result",
                    interactive=False,
                    lines=1,
                )

            prob_out = gr.Label(
                label="Class Probability Distribution",
                num_top_classes=3,
                elem_id="prob-chart",
            )

            gr.Markdown("### 🔥 Grad-CAM — image regions behind the prediction")
            cam_out = gr.Image(label="Grad-CAM overlay (red = strongest evidence)", height=260,
                               interactive=False)

            gr.Markdown("### 🌐 Task 2 — Bengali text generation (experimental)")
            gr.Markdown("⚠️ *The decoder was trained on machine-translated tweets, so it imitates tweet style "
                        "(hashtags, mentions, URL fragments) rather than describing the image. "
                        "Output is shown unedited.*")
            caption_out = gr.Textbox(
                label="Generated Bengali text (unedited model output)",
                elem_id="caption-result",
                interactive=False,
                lines=3,
            )

    # ── How to use ──────────────────────────────────────
    with gr.Accordion("📖  How to Use  |  কীভাবে ব্যবহার করবেন", open=False):
        gr.Markdown("""
**English:**
1. Upload a disaster image (the model was trained on hurricane/flood images from CrisisMMD)
2. Choose the beam width for text generation (1 = greedy)
3. Click **Analyze Disaster Scene**. Classification uses the image plus a fixed placeholder text,
   because no tweet is available here; this does not lower accuracy (the model is image-driven).

**বাংলা:**
1. দুর্যোগের একটি ছবি আপলোড করুন (মডেলটি CrisisMMD-এর ঘূর্ণিঝড়/বন্যার ছবিতে প্রশিক্ষিত)
2. লেখা তৈরির জন্য Beam Width বেছে নিন (1 = greedy)
3. **Analyze** বোতামে ক্লিক করুন। শ্রেণিবিন্যাসে ছবির সঙ্গে একটি নির্দিষ্ট placeholder লেখা ব্যবহৃত হয়।

**Models:** classification — CLS-Only (ViT-Base/16 + BanglaBERT + LoRA r = 8), grouped-split Macro F1 75.69 ± 1.33%; text — multi-task v3 with a 3-layer Transformer decoder.
        """)

    gr.HTML('<div id="footer-note">DRISHTI-Bn &nbsp;|&nbsp; Thesis Research Demo &nbsp;|&nbsp; All rights reserved</div>')

    # ── Wire up ─────────────────────────────────────────
    run_btn.click(
        fn=analyze_image,
        inputs=[image_input, beam_slider],
        outputs=[label_out, prob_out, conf_out, cam_out, caption_out],
    )

# ─────────────────────────────────────────────────────────────
# 4.  Launch
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, show_error=True, css=CUSTOM_CSS)
