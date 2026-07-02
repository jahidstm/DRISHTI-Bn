import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path so we can import evaluate
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
from evaluate import DisasterNetPredictor

# ─────────────────────────────────────────────────────────────
# 1.  Boot the model (once, at startup)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'disasternet_multitask_v2.pth')
predictor  = DisasterNetPredictor(model_path=MODEL_PATH)

# ─────────────────────────────────────────────────────────────
# 2.  Inference function called by Gradio
# ─────────────────────────────────────────────────────────────
LABELS = [
    "Severe Damage  |  মারাত্মক ক্ষয়ক্ষতি",
    "Humanitarian Rescue  |  ত্রাণ ও উদ্ধারকার্য",
    "Affected People  |  ক্ষতিগ্রস্ত মানুষ",
]
EMOJI = ["🔴", "🟡", "🟠"]
COLOR = ["#FF4444", "#FFB800", "#FF7700"]

def analyze_image(pil_image, beam_width):
    if pil_image is None:
        return (
            "⚠️ Please upload a disaster image.",
            {},
            "—",
            "—",
        )

    import torch
    pixel_values = predictor.transform(pil_image.convert("RGB")).unsqueeze(0).to(predictor.device)

    # Task 2: Caption
    caption = predictor.generate_caption(pixel_values, beam_width=int(beam_width))

    # Task 1: Classify
    cat_raw, conf, all_probs = predictor.classify_damage(pixel_values, caption)

    # Map to clean label
    if 'Severe' in cat_raw:
        idx = 0
    elif 'Humanitarian' in cat_raw:
        idx = 1
    else:
        idx = 2

    label_display = f"{EMOJI[idx]}  {LABELS[idx]}"
    conf_display  = f"{conf:.2f}%"

    prob_dict = {
        LABELS[0]: float(all_probs[0]),
        LABELS[1]: float(all_probs[1]),
        LABELS[2]: float(all_probs[2]),
    }

    cap_display = caption if caption.strip() else "[ Caption generation in progress — model is warming up ]"

    return label_display, prob_dict, conf_display, cap_display


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
  <p id="banner-title">🛰️ DisasterNet-Bangla</p>
  <p id="banner-sub">
    <b>Multi-Task Multimodal AI</b> for Disaster Severity Assessment &amp; Bengali Caption Generation<br>
    Vision Transformer (ViT-Base) &nbsp;+&nbsp; BanglaBERT &nbsp;+&nbsp; LoRA &nbsp;+&nbsp; Cross-Attention Decoder
  </p>
</div>
"""

with gr.Blocks(title="DisasterNet-Bangla Demo") as demo:

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

            gr.Markdown("### 🌐 Task 2 — Bengali Caption (বাংলা ক্যাপশন)")
            caption_out = gr.Textbox(
                label="Generated Bengali Caption",
                elem_id="caption-result",
                interactive=False,
                lines=3,
            )

    # ── How to use ──────────────────────────────────────
    with gr.Accordion("📖  How to Use  |  কীভাবে ব্যবহার করবেন", open=False):
        gr.Markdown("""
**English:**
1. Upload any disaster-related image (flood, earthquake, cyclone, rescue operation, etc.)
2. Choose beam width — higher values produce better (but slower) Bengali captions
3. Click **Analyze Disaster Scene** and wait ~10-30 seconds for CPU inference

**বাংলা:**
1. যেকোনো দুর্যোগ-সম্পর্কিত ছবি আপলোড করুন (বন্যা, ভূমিকম্প, ঘূর্ণিঝড়, উদ্ধারকাজ ইত্যাদি)
2. Beam Width বেছে নিন — বেশি মান = আরও নিখুঁত বাংলা ক্যাপশন (তবে একটু ধীর)
3. **Analyze** বোতামে ক্লিক করুন — CPU-তে ১০–৩০ সেকেন্ড অপেক্ষা করুন

**Model Architecture:** ViT-Base/16 + BanglaBERT + LoRA (r=8) + Cross-Attention Decoder (4 layers)
        """)

    gr.HTML('<div id="footer-note">DisasterNet-Bangla &nbsp;|&nbsp; Thesis Research Demo &nbsp;|&nbsp; All rights reserved</div>')

    # ── Wire up ─────────────────────────────────────────
    run_btn.click(
        fn=analyze_image,
        inputs=[image_input, beam_slider],
        outputs=[label_out, prob_out, conf_out, caption_out],
    )

# ─────────────────────────────────────────────────────────────
# 4.  Launch
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, show_error=True, css=CUSTOM_CSS)
