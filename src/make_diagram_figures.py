"""
Thesis diagrams (Chapters 1-3), redrawn after the pre-defense corrections
=========================================================================
  figures/figure_1_1_flood_stats.png      population exposed to flood risk (Rentschler, Salhab & Jafino, 2022)
  figures/figure_1_2_system_pipeline.png  where DRISHTI-Bn sits in a disaster-information workflow
  figures/figure_2_1_lora.png             LoRA update on a frozen weight matrix
  figures/figure_3_1_architecture.png     CLS-Only model (solid) + multi-task caption branch (dashed)
  figures/figure_3_3_xai_pipeline.png     explanation and reliability analysis

The old versions (BanglaBERT drawn as the caption decoder, satellite input, ECE 2.16%,
"fused" XAI output, figure numbers baked into the image) are in figures_superseded/.
No figure number or caption is drawn inside the image: captions go below the figure in Word.
Usage: python src/make_diagram_figures.py
"""
import os
import matplotlib, matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = lambda f: os.path.join(ROOT, "figures", f)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

C = dict(blue="#2B6CB0", blue_l="#E3EEF9", green="#2F855A", green_l="#E3F4EA", orange="#C05621",
         orange_l="#FCEBDD", purple="#6B46C1", purple_l="#EEE8FA", grey="#4A5568", grey_l="#EDF0F3",
         red="#C53030", red_l="#FBE5E5", ink="#1A202C")


def box(ax, x, y, w, h, title, body="", fc="grey_l", ec="grey", dashed=False, tsize=10.5, bsize=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=C[fc], ec=C[ec], lw=1.6, ls="--" if dashed else "-"))
    ty = y + h - 0.035 if body else y + h / 2
    ax.text(x + w / 2, ty, title, ha="center", va="top" if body else "center",
            fontsize=tsize, fontweight="bold", color=C["ink"])
    if body:
        ax.text(x + w / 2, y + h - 0.075, body, ha="center", va="top", fontsize=bsize,
                color="#2D3748", linespacing=1.45)


def arrow(ax, p, q, color="grey", dashed=False, text=None, toff=(0, 0.012)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4,
                                 color=C[color], ls="--" if dashed else "-"))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha="center",
                va="bottom", fontsize=7.8, color=C[color], style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT(name), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[+] figures/" + name)


# ---- Figure 1.1: flood exposure ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 2.6))
exp, rest = 94.424, 164.109 - 94.424
ax.barh([0], [exp], color=C["blue"], height=0.5)
ax.barh([0], [rest], left=[exp], color="#CBD5E0", height=0.5)
ax.text(exp / 2, 0, f"Exposed to flood risk\n{exp:.2f} million (57.5%)", ha="center", va="center",
        color="white", fontweight="bold", fontsize=10)
ax.text(exp + rest / 2, 0, f"Not exposed\n{rest:.2f} million (42.5%)", ha="center", va="center",
        color=C["ink"], fontsize=10)
ax.set_xlim(0, 164.109); ax.set_yticks([]); ax.set_xlabel("Population of Bangladesh (millions; total 164.1; > 0.15 m inundation, 1-in-100-year flood)")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
save(fig, "figure_1_1_flood_stats.png")

# ---- Figure 1.2: workflow and scope -------------------------------------------------
fig, ax = canvas(11, 5.2)
box(ax, 0.01, 0.56, 0.2, 0.36, "Social-media posts", "disaster photograph\n+ accompanying text\n(Bengali)",
    "blue_l", "blue")
box(ax, 0.26, 0.56, 0.2, 0.36, "Preprocessing", "resize 224 × 224, normalise\nBanglaBERT tokenisation\n(max 128 tokens)",
    "grey_l", "grey")
ax.add_patch(FancyBboxPatch((0.495, 0.08), 0.255, 0.9, boxstyle="round,pad=0.01,rounding_size=0.02",
                            fc="none", ec=C["orange"], lw=2.2))
ax.text(0.6225, 0.955, "Scope of this thesis", ha="center", va="top", fontsize=9, color=C["orange"],
        fontweight="bold")
box(ax, 0.51, 0.56, 0.225, 0.33, "DRISHTI-Bn", "ViT-Base/16 + BanglaBERT\nLoRA (r = 8) adapters\nclassification head",
    "orange_l", "orange")
box(ax, 0.51, 0.12, 0.225, 0.33, "Analysis outputs", "Grad-CAM heat map\ncalibrated confidence\nBengali text (experimental)",
    "purple_l", "purple")
box(ax, 0.79, 0.56, 0.2, 0.36, "Damage category", "Severe Damage\nHumanitarian Rescue\nAffected People",
    "green_l", "green")
box(ax, 0.79, 0.12, 0.2, 0.33, "Responder use", "triage dashboard,\nalerts, resource planning\n(not implemented)",
    "grey_l", "grey", dashed=True)
arrow(ax, (0.21, 0.74), (0.26, 0.74)); arrow(ax, (0.46, 0.74), (0.51, 0.74))
arrow(ax, (0.735, 0.74), (0.79, 0.74), "green"); arrow(ax, (0.6225, 0.56), (0.6225, 0.45), "purple")
arrow(ax, (0.89, 0.56), (0.89, 0.45), "grey", dashed=True)
arrow(ax, (0.735, 0.285), (0.79, 0.285), "grey", dashed=True)
save(fig, "figure_1_2_system_pipeline.png")

# ---- Figure 2.1: LoRA -----------------------------------------------------------------
fig, ax = canvas(9, 3.6)
box(ax, 0.02, 0.38, 0.13, 0.24, "Input x", "x ∈ ℝ^d", "grey_l", "grey")
box(ax, 0.27, 0.6, 0.27, 0.3, "Pre-trained W₀ (frozen)", "W₀ ∈ ℝ^(d × d)\nno gradient", "blue_l", "blue")
box(ax, 0.22, 0.08, 0.16, 0.3, "A (trainable)", "A ∈ ℝ^(r × d)\nrandom init", "orange_l", "orange")
box(ax, 0.44, 0.08, 0.16, 0.3, "B (trainable)", "B ∈ ℝ^(d × r)\ninitialised to 0", "orange_l", "orange")
ax.add_patch(matplotlib.patches.Ellipse((0.72, 0.5), 0.045, 0.11, fc=C["green_l"], ec=C["green"], lw=1.6))
ax.text(0.72, 0.5, "+", ha="center", va="center", fontsize=16, color=C["green"], fontweight="bold")
box(ax, 0.83, 0.38, 0.15, 0.24, "Output h", "h ∈ ℝ^d", "purple_l", "purple")
arrow(ax, (0.15, 0.55), (0.27, 0.72)); arrow(ax, (0.15, 0.45), (0.22, 0.25))
arrow(ax, (0.38, 0.23), (0.44, 0.23), "orange")
ax.text(0.41, 0.42, "rank r ≪ d", ha="center", fontsize=8.5, color=C["orange"], style="italic")
arrow(ax, (0.54, 0.75), (0.69, 0.53)); arrow(ax, (0.6, 0.23), (0.69, 0.47), "orange", text="× α / r", toff=(0.03, -0.01))
arrow(ax, (0.765, 0.5), (0.83, 0.5))
ax.text(0.5, 0.97, "h = W₀x + (α / r)·BAx      (this study: r = 8, α = 16, on query and value projections)",
        ha="center", va="top", fontsize=10)
save(fig, "figure_2_1_lora.png")

# ---- Figure 3.1: architecture ---------------------------------------------------------------
fig, ax = canvas(12, 6.4)
box(ax, 0.01, 0.62, 0.14, 0.26, "Image", "RGB, 224 × 224\n16 × 16 patches", "grey_l", "grey")
box(ax, 0.01, 0.2, 0.14, 0.26, "Bengali text", "machine-translated\ntweet, ≤ 128 tokens", "grey_l", "grey")
box(ax, 0.2, 0.6, 0.22, 0.3, "ViT-Base/16 encoder", "12 layers, 12 heads, d = 768\nfrozen weights\n+ LoRA on query/value",
    "blue_l", "blue")
box(ax, 0.2, 0.18, 0.22, 0.3, "BanglaBERT encoder", "ELECTRA, 12 layers, d = 768\nfrozen weights\n+ LoRA on query/value",
    "green_l", "green")
box(ax, 0.47, 0.42, 0.17, 0.18, "Concatenate", "[v_CLS ; t_CLS]\n∈ ℝ^1536", "grey_l", "grey", bsize=8.2)
box(ax, 0.68, 0.4, 0.15, 0.22, "MLP head", "1536 → 512 → 3\nBatchNorm, ReLU,\nDropout 0.3", "orange_l", "orange")
box(ax, 0.87, 0.4, 0.12, 0.22, "Softmax", "Severe Damage\nHumanitarian Rescue\nAffected People", "orange_l", "orange",
    bsize=7.8)
box(ax, 0.47, 0.72, 0.43, 0.22, "Caption decoder (multi-task variant only)",
    "3 Transformer decoder layers, 8 heads, d_ff = 2048\ncross-attention over 196 ViT patch tokens\nbeam search (k = 3) → Bengali tokens",
    "purple_l", "purple", dashed=True, tsize=9.8)
arrow(ax, (0.15, 0.75), (0.2, 0.75)); arrow(ax, (0.15, 0.33), (0.2, 0.33))
arrow(ax, (0.42, 0.68), (0.5, 0.6), "blue", text="v_CLS (pooler)", toff=(0.075, -0.005))
arrow(ax, (0.42, 0.36), (0.5, 0.42), "green", text="t_CLS", toff=(0.045, -0.03))
arrow(ax, (0.42, 0.84), (0.47, 0.84), "purple", dashed=True)
ax.text(0.445, 0.95, "196 patch\ntokens", ha="center", va="bottom", fontsize=7.8, color=C["purple"], style="italic")
arrow(ax, (0.64, 0.51), (0.68, 0.51)); arrow(ax, (0.83, 0.51), (0.87, 0.51))
ax.text(0.755, 0.33, "L_cls = cross-entropy", ha="center", fontsize=8.5, color=C["orange"])
ax.text(0.685, 0.665, "L_cap = token NLL;  L = L_cls + 0.5·L_cap", ha="center", fontsize=8.5, color=C["purple"])
ax.text(0.5, 0.04, "Solid: CLS-Only model (main; 1.38 M trainable parameters).   "
        "Dashed: caption branch added in the multi-task model (74.79 M trainable).",
        ha="center", fontsize=8.8, color=C["grey"])
save(fig, "figure_3_1_architecture.png")

# ---- Figure 3.3: explanation and reliability analysis ---------------------------------------
fig, ax = canvas(12, 5.6)
box(ax, 0.01, 0.4, 0.14, 0.22, "Test image", "grouped test set\n(n = 375)", "grey_l", "grey")
box(ax, 0.2, 0.4, 0.15, 0.22, "Trained model", "CLS-Only (seed 42)\nmulti-task v3 for\ncross-attention", "orange_l", "orange")
box(ax, 0.42, 0.74, 0.25, 0.2, "Grad-CAM", "gradients on tokens entering\nthe last ViT block → 14 × 14 map", "blue_l", "blue")
box(ax, 0.42, 0.5, 0.25, 0.2, "Attention Rollout", "12 layers, head mean,\n90% lowest weights discarded", "blue_l", "blue")
box(ax, 0.42, 0.26, 0.25, 0.2, "Decoder cross-attention", "3 layers × 8 heads,\nexplains caption tokens only", "purple_l", "purple",
    dashed=True)
box(ax, 0.42, 0.02, 0.25, 0.2, "Temperature scaling", "single T fitted on validation\nset (n = 373) by NLL", "green_l", "green")
box(ax, 0.73, 0.5, 0.26, 0.44, "Faithfulness test", "remove top 10 / 25 / 50% patches\nranked by each map\n→ drop in predicted-class probability\n→ compare with random patches\n\nagreement: top-10% patch IoU",
    "red_l", "red")
box(ax, 0.73, 0.02, 0.26, 0.3, "Calibration metrics", "ECE, MCE (10 bins)\nNLL, Brier score\nreliability diagram", "green_l", "green")
arrow(ax, (0.15, 0.51), (0.2, 0.51))
for y in (0.84, 0.6, 0.36, 0.12):
    arrow(ax, (0.35, 0.51), (0.42, y), "purple" if y == 0.36 else "green" if y == 0.12 else "blue",
          dashed=y == 0.36)
arrow(ax, (0.67, 0.84), (0.73, 0.78), "red"); arrow(ax, (0.67, 0.6), (0.73, 0.66), "red")
arrow(ax, (0.67, 0.36), (0.73, 0.55), "red", dashed=True); arrow(ax, (0.67, 0.12), (0.73, 0.17), "green")
save(fig, "figure_3_3_xai_pipeline.png")
