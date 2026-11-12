"""
Phase 5 — thesis figures from the saved results (local, no GPU)
==============================================================
  figure_4_2_baseline_comparison_grouped.png  baselines, grouped split, seed points + mean ± std
  figure_4_3_ablation_original_split.png      ablation, original split, single seed (relabelled, C4–C6)
  figure_4_9_faithfulness_deletion.png        deletion test: Grad-CAM vs Rollout vs Random

Usage: python src/make_phase5_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)

# Reference palette (dataviz skill, light mode) — validated blue / orange; gray = neutral reference
BLUE, ORANGE, GRAY, LIGHTGRAY = "#2a78d6", "#eb6834", "#8a8984", "#c9c8c3"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 300,
})

# ---------------------------------------------------------------- Fig 4.2
res = pd.read_csv(R("results", "defense_fixes", "phase3", "phase3_seed_results.csv"))
order = [("ViT-Only (last 2 blocks + pooler)", "ViT-Only\n(ViT-Base, last 2 blocks)"),
         ("CrisisViT-style (ViT-L, last 4 blocks + pooler)", "CrisisViT-style\n(ViT-Large, last 4 blocks)"),
         ("CLS-Only (ViT+BanglaBERT+LoRA r=8)", "DRISHTI-Bn\n(ViT-Base + BanglaBERT, LoRA r = 8)")]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for y, (key, label) in enumerate(order):
    g = res[res.model == key]
    m = g.macro_f1.mean(); s = g.macro_f1.std(ddof=1) if len(g) > 1 else 0
    ours = key.startswith("CLS-Only")
    ax.barh(y, m, height=0.55, color=BLUE if ours else LIGHTGRAY, zorder=2)
    if len(g) > 1:
        ax.errorbar(m, y, xerr=s, color=INK, capsize=4, lw=1.2, zorder=3)
    ax.scatter(g.macro_f1, [y] * len(g), s=22, color=INK, zorder=4, edgecolor=SURFACE, linewidth=0.8)
    params = g.trainable_params.iloc[0] / 1e6
    txt = f"{m:.2f}" + (f" ± {s:.2f}" if len(g) > 1 else " (1 seed)") + f"   |   {params:.2f}M trainable"
    ax.text(78.2, y, txt, va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order])
ax.set_xlim(65, 78); ax.set_xticks(range(65, 79, 2))
ax.set_xlabel("Macro F1 (%) on the grouped test set (n = 375)")
ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(R("figure_4_2_baseline_comparison_grouped.png"), bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 4.3
abl = [("No-LoRA (head only)", 77.74, GRAY), ("LoRA r = 4", 79.41, GRAY), ("LoRA r = 16", 79.14, GRAY),
       ("No-CrossAttn decoder", 81.88, GRAY), ("LoRA r = 8 — run A", 81.47, BLUE),
       ("LoRA r = 8 — run B (CLS-Only)", 83.21, BLUE)]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for y, (lab, v, c) in enumerate(abl):
    ax.barh(y, v, height=0.55, color=c, zorder=2)
    ax.text(v + 0.25, y, f"{v:.2f}", va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(abl))); ax.set_yticklabels([a[0] for a in abl])
ax.set_xlim(70, 86); ax.set_xlabel("Macro F1 (%) — original un-grouped split (n = 376), single seed")
ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
ax.annotate("", xy=(83.21, 5.42), xytext=(81.47, 5.42),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
ax.text(82.34, 5.55, "same config:\n1.74 pp apart", ha="center", va="bottom", fontsize=8.5, color=INK)
plt.tight_layout()
plt.savefig(R("figure_4_3_ablation_original_split.png"), bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 4.9
fa = pd.read_csv(R("results", "defense_fixes", "phase4", "phase4_faithfulness.csv"))
base_acc = 88.2667
styles = {"Grad-CAM": (BLUE, "o"), "Rollout": (ORANGE, "s"), "Random": (GRAY, "^")}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for meth, (c, mk) in styles.items():
    g = fa[fa.method == meth].sort_values("frac")
    x = [0] + list(g.frac * 100)
    for ax, ycol, y0 in [(axes[0], "mean_prob_drop", 0.0), (axes[1], "accuracy_after", base_acc)]:
        y = [y0] + list(g[ycol])
        ax.plot(x, y, color=c, lw=2, marker=mk, ms=7, mec=SURFACE, mew=1.2, zorder=3)
        name = "Attention Rollout" if meth == "Rollout" else meth
        ax.text(x[-1] + 1.5, y[-1], name, va="center", fontsize=9.5, color=INK)
axes[0].set_ylabel("Drop in predicted-class\nprobability (pp)")
axes[1].set_ylabel("Accuracy after deletion (%)")
for ax in axes:
    ax.set_xlabel("Top patches removed (%)"); ax.set_xticks([0, 10, 25, 50]); ax.set_xlim(-2, 66)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
axes[0].set_title("(a) Confidence drop", fontsize=11); axes[1].set_title("(b) Accuracy", fontsize=11)
plt.tight_layout()
plt.savefig(R("figure_4_9_faithfulness_deletion.png"), bbox_inches="tight"); plt.close()
print("[+] Saved figure_4_2_baseline_comparison_grouped.png, figure_4_3_ablation_original_split.png, "
      "figure_4_9_faithfulness_deletion.png")
