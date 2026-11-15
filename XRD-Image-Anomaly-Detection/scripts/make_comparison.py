import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = pd.read_csv(sys.argv[1]); P = R.pivot(index='class', columns='model', values='ap')
P = P.sort_values('Full pipeline')
BG='#151515'; INK='#e6e6e3'; MUTED='#8f8e86'; GRID='#2a2a27'; FULL='#3987e5'; SIMP='#d95926'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK})
fig, ax = plt.subplots(figsize=(11, 5.4), facecolor=BG); ax.set_facecolor(BG)
y = np.arange(len(P))
for i, (c, r) in enumerate(P.iterrows()):
    ax.plot([r['Simple brightness stats'], r['Full pipeline']], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle='round')
    ax.plot([r['Random guessing']] * 2, [i - 0.28, i + 0.28], color=MUTED, lw=2.5, zorder=2, solid_capstyle='round')
ax.scatter(P['Simple brightness stats'], y, s=95, color=SIMP, zorder=3, edgecolor=BG, linewidth=2)
ax.scatter(P['Full pipeline'], y, s=110, color=FULL, zorder=4, edgecolor=BG, linewidth=2)
for i, v in enumerate(P['Full pipeline']):
    ax.text(v + 0.022, i, f'{v:.2f}', va='center', color=INK, fontsize=10, fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels([c.lower().capitalize() for c in P.index], color=INK)
ax.set_xlim(0, 1.06); ax.set_xticks(np.linspace(0, 1, 6)); ax.tick_params(axis='x', colors=MUTED, length=0); ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color=GRID, lw=0.7); ax.set_axisbelow(True)
[s.set_visible(False) for s in ax.spines.values()]
ax.set_xlabel('average precision on held-out test images (1.0 = perfect)', color=MUTED)
m = R.groupby('model')['ap'].mean()
leg = [Line2D([0],[0], color=MUTED, lw=2.5, label=f"Random guessing  (avg {m['Random guessing']:.2f})"),
       Line2D([0],[0], marker='o', color='none', markerfacecolor=SIMP, markersize=10, label=f"Brightness stats only  (avg {m['Simple brightness stats']:.2f})"),
       Line2D([0],[0], marker='o', color='none', markerfacecolor=FULL, markersize=10, label=f"Full pipeline  (avg {m['Full pipeline']:.2f})")]
ax.legend(handles=leg, frameon=False, labelcolor=INK, loc='lower center', bbox_to_anchor=(0.45, 0.99), ncol=3, fontsize=10.5)
fig.suptitle('The physics features beat both baselines on every problem', x=0.02, ha='left', y=0.97, color=INK, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(sys.argv[2], dpi=130, facecolor=BG)
