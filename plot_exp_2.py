"""
Part 2 plots for SFT analysis report.

Directory layout:
  Training logs:
    training_logs/sft_A.log
    training_logs/sft_B.log
    training_logs/sft_combined.log
  Evaluation logs:
    evaluation_logs/baseline_for_task_A.log
    evaluation_logs/baseline_for_task_B.log
    evaluation_logs/sft_A_for_task_A.log
    evaluation_logs/sft_A_for_task_B.log
    evaluation_logs/sft_B_for_task_A.log
    evaluation_logs/sft_B_for_task_B.log
    evaluation_logs/sft_combined_for_task_A.log
    evaluation_logs/sft_combined_for_task_B.log

Produces 4 PNG files in ./plots/:
  1_training_curves.png        – all 3 runs on one figure
  2_accuracy_comparison.png    – grouped bar: single vs multi-task
  3_catastrophic_forgetting.png – val loss before/after SFT
  4_results_table.png          – full assignment table
"""

import re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── PATHS (edit here if needed) ───────────────────────────────────────────────
TRAIN_LOGS = {
    'SFT-A':     'training_logs/sft_A.log',
    'SFT-B':     'training_logs/sft_B.log',
    'SFT-Comb':  'training_logs/sft_combined.log',
}
EVAL_LOGS = {
    ('Baseline',  'A'): 'evaluation_logs/baseline_for_task_A.log',
    ('Baseline',  'B'): 'evaluation_logs/baseline_for_task_B.log',
    ('SFT-A',     'A'): 'evaluation_logs/sft_A_for_task_A.log',
    ('SFT-A',     'B'): 'evaluation_logs/sft_A_for_task_B.log',
    ('SFT-B',     'A'): 'evaluation_logs/sft_B_for_task_A.log',
    ('SFT-B',     'B'): 'evaluation_logs/sft_B_for_task_B.log',
    ('SFT-Comb',  'A'): 'evaluation_logs/sft_combined_for_task_A.log',
    ('SFT-Comb',  'B'): 'evaluation_logs/sft_combined_for_task_B.log',
}
# Hardcoded values we already know (used when log file is missing/unrun)
KNOWN_ACC = {
    ('SFT-A',    'A'): 7.69,
    ('SFT-B',    'B'): 88.40,
    ('SFT-Comb', 'A'): 16.67,
}
# Val loss on original Shakespeare (from catastrophic forgetting analysis)
VAL_LOSS = {
    'Baseline': 1.7617,
    'SFT-A':    2.0632,
    'SFT-B':    1.7965,
    'SFT-Comb': 1.8259,
}
OUT = 'plots'

# ── palette ───────────────────────────────────────────────────────────────────
PAL = {
    'SFT-A':    '#4C72B0',
    'SFT-B':    '#DD8452',
    'SFT-Comb': '#55A868',
    'Baseline': '#C44E52',
}
GRID = '#efefef'

# ── helpers ───────────────────────────────────────────────────────────────────
def parse_train_log(path):
    steps, tr, vl = [], [], []
    pat = re.compile(r'step\s+(\d+):\s+train loss\s+([\d.]+),\s+val loss\s+([\d.]+)')
    try:
        with open(path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    tr.append(float(m.group(2)))
                    vl.append(float(m.group(3)))
        print(f"  Loaded {len(steps)} steps from {path}")
    except FileNotFoundError:
        print(f"  [warn] not found: {path}")
    return steps, tr, vl

def parse_acc(path):
    pat = re.compile(r'Accuracy:\s+([\d.]+)%')
    try:
        with open(path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    return float(m.group(1))
    except FileNotFoundError:
        pass
    return None

def smooth(values, w=15):
    if len(values) < w:
        return values
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='valid')

os.makedirs(OUT, exist_ok=True)

# ── load all training logs ────────────────────────────────────────────────────
print("Loading training logs...")
train_data = {name: parse_train_log(path) for name, path in TRAIN_LOGS.items()}

# ── load all eval accuracies ──────────────────────────────────────────────────
print("Loading evaluation logs...")
acc = {}
for key, path in EVAL_LOGS.items():
    v = parse_acc(path)
    acc[key] = v if v is not None else KNOWN_ACC.get(key)
    if acc[key] is not None:
        print(f"  {key[0]:12s} Task {key[1]}: {acc[key]:.2f}%")
    else:
        print(f"  {key[0]:12s} Task {key[1]}: — (not yet evaluated)")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Training curves (all 3 on one figure, train + val)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
fig.suptitle('Fine-tuning Loss Curves', fontsize=14, fontweight='bold', y=1.02)

for ax, (name, (steps, tr, vl)) in zip(axes, train_data.items()):
    col = PAL[name]
    if steps:
        w = 15
        ax.plot(steps, tr, color=col, alpha=0.2, lw=1.0)
        ax.plot(steps, vl, color=col, alpha=0.2, lw=1.0, ls='--')
        ax.plot(steps[w-1:], smooth(tr, w), color=col, lw=2.2, label='Train (smoothed)')
        ax.plot(steps[w-1:], smooth(vl, w), color=col, lw=2.2, ls='--', label='Val (smoothed)')
        best_val = min(vl)
        ax.axhline(best_val, color=col, alpha=0.4, lw=1.0, ls=':')
        ax.text(steps[-1], best_val + 0.02, f'best={best_val:.3f}',
                ha='right', va='bottom', fontsize=8, color=col)
    else:
        ax.text(0.5, 0.5, 'Log not found', ha='center', va='center',
                transform=ax.transAxes, color='grey')

    ax.set_title(name, fontsize=12, fontweight='bold', color=col)
    ax.set_xlabel('Training Step', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_facecolor(GRID)
    ax.grid(True, color='white', lw=0.8)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
p1 = f'{OUT}/1_training_curves.png'
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved {p1}')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Accuracy comparison: Random | Baseline | SFT-A | SFT-B | Multi-task
# ═══════════════════════════════════════════════════════════════════════════════
setups = ['Random\nBaseline', 'Pretrained\n(no SFT)', 'Single-task A', 'Single-task B', 'Multi-task\nA+B']
task_a = [10.0,
          acc.get(('Baseline','A')),
          acc.get(('SFT-A','A')),
          acc.get(('SFT-B','A')),
          acc.get(('SFT-Comb','A'))]
task_b = [50.0,
          acc.get(('Baseline','B')),
          acc.get(('SFT-A','B')),
          acc.get(('SFT-B','B')),
          acc.get(('SFT-Comb','B'))]

# Colours per group
group_colors = ['#aaaaaa', PAL['Baseline'], PAL['SFT-A'], PAL['SFT-B'], PAL['SFT-Comb']]

plot_a = [v if v is not None else 0 for v in task_a]
plot_b = [v if v is not None else 0 for v in task_b]

x = np.arange(len(setups))
w = 0.35

fig, ax = plt.subplots(figsize=(13, 5.5))
b1 = ax.bar(x - w/2, plot_a, w, label='Task A  (Speaker ID)',
            color=[c for c in group_colors],
            edgecolor='white', linewidth=1.2, alpha=1.0)
b2 = ax.bar(x + w/2, plot_b, w, label='Task B  (Verse/Prose)',
            color=[c for c in group_colors],
            edgecolor='white', linewidth=1.2, alpha=0.55)

# Value labels
for bars, vals in [(b1, plot_a), (b2, plot_b)]:
    for bar, val, orig in zip(bars, vals, task_a if bars is b1 else task_b):
        if orig is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 2,
                    '—', ha='center', va='bottom', fontsize=10, color='grey')

# Legend: solid = Task A, faded = Task B
import matplotlib.patches as mpatches
leg_a = mpatches.Patch(color='#555555', label='Task A  (Speaker ID)')
leg_b = mpatches.Patch(color='#555555', alpha=0.45, label='Task B  (Verse/Prose)')
ax.legend(handles=[leg_a, leg_b], fontsize=9, loc='upper left')

ax.set_xticks(x)
ax.set_xticklabels(setups, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_ylim(0, 110)
ax.set_title('Task Accuracy: Random / Baseline / Single-task / Multi-task SFT',
             fontsize=13, fontweight='bold')
ax.set_facecolor(GRID)
ax.grid(axis='y', color='white', lw=0.8)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
p2 = f'{OUT}/2_accuracy_comparison.png'
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {p2}')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Catastrophic forgetting
# ═══════════════════════════════════════════════════════════════════════════════
models  = list(VAL_LOSS.keys())
losses  = list(VAL_LOSS.values())
colors  = [PAL.get(m, '#888888') for m in models]
base    = VAL_LOSS['Baseline']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, losses, color=colors, width=0.5,
              edgecolor='white', linewidth=1.5)

for bar, v, m in zip(bars, losses, models):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    if m != 'Baseline':
        delta = v - base
        sign  = '+' if delta >= 0 else ''
        col   = '#e74c3c' if delta > 0.1 else '#27ae60'
        ax.text(bar.get_x() + bar.get_width()/2, v / 2,
                f'{sign}{delta:.4f}', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

ax.axhline(base, color=PAL['Baseline'], ls=':', lw=2, alpha=0.6,
           label=f'Baseline ({base:.4f})')
ax.set_ylabel('Val Loss on Original Shakespeare', fontsize=11)
ax.set_title('Catastrophic Forgetting Analysis\n'
             'Val Loss on Original Shakespeare Before/After SFT',
             fontsize=12, fontweight='bold')
ax.set_facecolor(GRID)
ax.grid(axis='y', color='white', lw=0.8)
ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=9)
ax.set_ylim(0, max(losses) * 1.2)

plt.tight_layout()
p3 = f'{OUT}/3_catastrophic_forgetting.png'
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {p3}')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Results table (as required by assignment)
# ═══════════════════════════════════════════════════════════════════════════════
def fmt(v):
    return f'{v:.1f}%' if v is not None else '—'

rows = [
    ('Random baseline',    fmt(10.0),                   fmt(50.0),                   '—'),
    ('Pretrained (no SFT)',fmt(acc.get(('Baseline','A'))),fmt(acc.get(('Baseline','B'))),f"{VAL_LOSS['Baseline']:.4f}"),
    ('Single-task A',      fmt(acc.get(('SFT-A','A'))), fmt(acc.get(('SFT-A','B'))), f"{VAL_LOSS['SFT-A']:.4f}"),
    ('Single-task B',      fmt(acc.get(('SFT-B','A'))), fmt(acc.get(('SFT-B','B'))), f"{VAL_LOSS['SFT-B']:.4f}"),
    ('Multi-task A+B',     fmt(acc.get(('SFT-Comb','A'))),fmt(acc.get(('SFT-Comb','B'))),f"{VAL_LOSS['SFT-Comb']:.4f}"),
]
col_labels = ['Setup', 'Task A Acc.', 'Task B Acc.', 'Shakespeare Val Loss']

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.2)

for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#2d3e50')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

alt = ['#f0f4f8', '#ffffff']
for i in range(1, len(rows)+1):
    for j in range(len(col_labels)):
        tbl[(i, j)].set_facecolor(alt[i % 2])
        if i == 1:  # random baseline row — grey italic
            tbl[(i, j)].set_text_props(color='grey', style='italic')

ax.set_title('Results Table', fontsize=13, fontweight='bold', pad=16)
plt.tight_layout()
p4 = f'{OUT}/4_results_table.png'
plt.savefig(p4, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {p4}')

print(f'\nAll done. Plots saved to ./{OUT}/')