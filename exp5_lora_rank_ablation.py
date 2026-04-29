import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════
#  HARDCODED DATA  — Experiment 5: LoRA Rank Ablation
# ═══════════════════════════════════════════════════════════════

# ── LoRA results ────────────────────────────────────────────────
ranks            = [1,      2,      4,      8,      16]
trainable_params = [9_216,  18_432, 36_864, 73_728, 147_456]
task_a_accuracy  = [0.1236, 0.0981, 0.1224, 0.1427, 0.1389]

# ── Full fine-tuning results ─────────────────────────────────────
fft_params   = 10_696_704   # total model params (all trainable)
fft_acc_a    = 0.0769       # Task A  7.69 %
fft_acc_b    = 0.8840       # Task B 88.40 %  (reference)

# ── Random baseline ──────────────────────────────────────────────
random_baseline = 0.10      # 10 % (1 in 10 speakers)

# ═══════════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════════
BG      = "#0e1117"
PANEL   = "#161b22"
GRID    = "#21262d"
FG      = "#e6edf3"
MUTED   = "#8b949e"

C_LORA  = "#58a6ff"   # blue  – LoRA curve
C_FFT   = "#f78166"   # coral – full fine-tuning
C_BASE  = "#6e7681"   # grey  – random baseline
C_BEST  = "#3fb950"   # green – best LoRA rank
C_ANN   = "#d2a8ff"   # lavender – annotation text

plt.rcParams.update({
    "font.family":       "monospace",
    "text.color":        FG,
    "axes.labelcolor":   FG,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "axes.edgecolor":    GRID,
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "grid.color":        GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
})

# ═══════════════════════════════════════════════════════════════
#  FIGURE  (2 panels side-by-side)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         gridspec_kw={"wspace": 0.38})
fig.patch.set_facecolor(BG)

best_idx = task_a_accuracy.index(max(task_a_accuracy))   # r=8

# ──────────────────────────────────────────────────────────────
#  PANEL 1 – Accuracy vs Trainable Params  (log-x)
# ──────────────────────────────────────────────────────────────
ax = axes[0]

# Random baseline
ax.axhline(random_baseline, color=C_BASE, lw=1.1, ls=":",
           label=f"Random baseline ({random_baseline*100:.0f}%)", zorder=1)

# Full fine-tuning horizontal dashed line
ax.axhline(fft_acc_a, color=C_FFT, lw=1.2, ls="--", alpha=0.55, zorder=1)

# LoRA curve
ax.plot(trainable_params, task_a_accuracy,
        color=C_LORA, lw=2.2, zorder=3,
        marker="o", ms=9,
        markeredgecolor="white", markeredgewidth=0.9,
        label="LoRA – Task A")

# Highlight best LoRA point
ax.scatter([trainable_params[best_idx]], [task_a_accuracy[best_idx]],
           color=C_BEST, s=180, zorder=6,
           edgecolors="white", linewidths=1.1)

# Annotate each LoRA rank
for i, (r, p, a) in enumerate(zip(ranks, trainable_params, task_a_accuracy)):
    color  = C_BEST if i == best_idx else C_ANN
    weight = "bold"  if i == best_idx else "normal"
    offset = (0, 16) if i != 1 else (0, -22)   # r=2 dips – push label down
    ax.annotate(f"r={r}\n{a*100:.1f}%",
                xy=(p, a), xytext=offset,
                textcoords="offset points",
                ha="center", fontsize=8,
                color=color, fontweight=weight)

# Full fine-tuning scatter
ax.scatter([fft_params], [fft_acc_a],
           color=C_FFT, s=150, zorder=5, marker="D",
           edgecolors="white", linewidths=0.9,
           label=f"Full fine-tuning ({fft_acc_a*100:.1f}%)")
ax.annotate(f"Full FT\n{fft_acc_a*100:.1f}%\n10.7 M params",
            xy=(fft_params, fft_acc_a),
            xytext=(-80, 28), textcoords="offset points",
            color=C_FFT, fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color=C_FFT, lw=0.85))

# Axes
ax.set_xscale("log")
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x < 1e5 else
                      (f"{x/1e3:.0f}K" if x < 1e6 else f"{x/1e6:.1f}M")))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlim(5_000, 30_000_000)
ax.set_ylim(0.0, 0.22)

ax.set_xlabel("Trainable Parameters  (log scale)", fontsize=10, labelpad=8)
ax.set_ylabel("Task A Accuracy", fontsize=10, labelpad=8)
ax.set_title("Accuracy vs. Trainable Parameters\n(Experiment 5 – LoRA Rank Ablation)",
             fontsize=11, fontweight="bold", pad=12)

# Secondary x-axis: rank labels
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(trainable_params)
ax2.set_xticklabels([f"r={r}" for r in ranks], fontsize=8, color=C_LORA)
ax2.tick_params(top=True, length=4, color=C_LORA)
for sp in ax2.spines.values():
    sp.set_visible(False)

ax.legend(loc="upper left", fontsize=8.5,
          framealpha=0.2, facecolor=PANEL, edgecolor=GRID)

# ──────────────────────────────────────────────────────────────
#  PANEL 2 – Bar chart: Task A accuracy per rank
# ──────────────────────────────────────────────────────────────
ax3 = axes[1]

bar_labels  = [f"r={r}" for r in ranks] + ["Full FT"]
bar_values  = task_a_accuracy + [fft_acc_a]
bar_colors  = [C_BEST if i == best_idx else C_LORA
               for i in range(len(ranks))] + [C_FFT]

x = np.arange(len(bar_labels))
bars = ax3.bar(x, bar_values, color=bar_colors,
               edgecolor="white", linewidth=0.6, width=0.55, zorder=3)

# Random baseline line
ax3.axhline(random_baseline, color=C_BASE, lw=1.2, ls=":",
            label=f"Random baseline ({random_baseline*100:.0f}%)", zorder=4)

# Value labels on bars
for bar, val in zip(bars, bar_values):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.004,
             f"{val*100:.1f}%",
             ha="center", va="bottom", fontsize=8.5,
             color=FG, fontweight="bold")

ax3.set_xticks(x)
ax3.set_xticklabels(bar_labels, fontsize=9)
ax3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax3.set_ylim(0, 0.22)
ax3.set_xlabel("Configuration", fontsize=10, labelpad=8)
ax3.set_ylabel("Task A Accuracy", fontsize=10, labelpad=8)
ax3.set_title("Task A Accuracy by Rank\nvs. Full Fine-Tuning",
              fontsize=11, fontweight="bold", pad=12)
ax3.legend(fontsize=8.5, framealpha=0.2,
           facecolor=PANEL, edgecolor=GRID)

# Param count below each bar
param_labels = [f"{p//1000}K" for p in trainable_params] + ["10.7M"]
for xi, pl in zip(x, param_labels):
    ax3.text(xi, -0.018, pl, ha="center", va="top",
             fontsize=7, color=MUTED, transform=ax3.get_xaxis_transform())

# ──────────────────────────────────────────────────────────────
#  Super-title
# ──────────────────────────────────────────────────────────────
fig.suptitle("Experiment 5 — LoRA Rank Ablation  |  Task A Speaker Identification",
             fontsize=13, fontweight="bold", color=FG, y=1.02)

plt.tight_layout()

os.makedirs("plots", exist_ok=True)
out = "plots/exp5_lora_rank_ablation.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
