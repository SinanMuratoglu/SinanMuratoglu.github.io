import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
imp = pd.read_csv(sys.argv[1], index_col=0).clip(lower=0)
share = imp / imp.sum()                       # each class sums to 1
feats = []
for c in share.columns:
    for f in share[c].nlargest(3).index:
        if f not in feats: feats.append(f)
nice = {"ring_frac_sectors_lit":"ring goes all the way around","ring_radius_frac":"ring radius",
 "ring_max_prom":"ring sharpness","ring_prom_2":"2nd ring sharpness","bg_level":"background level",
 "bg_slope":"background slope","bg_inner":"inner background","diffuse_humps":"broad humps",
 "diffuse_energy":"diffuse texture","sector_range":"one side brighter","sector_h1":"left/right imbalance",
 "sector_h2":"quadrupole imbalance","sector_spread_by_r":"azimuthal spread","spot_frac":"Bragg spot density",
 "spot_count":"Bragg spot count","art_saturated_frac":"saturated pixels","art_line_score":"straight streaks",
 "art_masked_frac":"dead area","int_p99":"brightest 1%","int_mean":"mean intensity","int_std":"intensity spread",
 "center_dy":"beam centre offset (y)","center_dx":"beam centre offset (x)"}
order = ["Ice Ring","Background Ring","Strong Background","Diffuse Scattering","Loop Scattering","Non-uniform Detector","Artifact"]
M = share.loc[feats, order].T.values
BG='#151515'; INK='#e6e6e3'; MUTED='#8f8e86'
cmap = LinearSegmentedColormap.from_list('seq', ['#1d1d1b', '#1c3f6e', '#2a78d6', '#9cc6f5'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK})
fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=BG)
ax.set_facecolor(BG)
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=M.max(), aspect='auto')
# 2px surface gap between cells
ax.set_xticks(np.arange(-.5, len(feats)), minor=True); ax.set_yticks(np.arange(-.5, len(order)), minor=True)
ax.grid(which='minor', color=BG, lw=2.5); ax.tick_params(which='minor', length=0)
ax.set_xticks(range(len(feats))); ax.set_xticklabels([nice.get(f, f) for f in feats], rotation=40, ha='right', color=MUTED)
ax.set_yticks(range(len(order))); ax.set_yticklabels([o.lower().capitalize() for o in order], color=INK)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
# label only each class's top feature
for i in range(len(order)):
    j = int(np.argmax(M[i])); ax.text(j, i, f"{M[i, j]:.0%}", ha='center', va='center',
        color='#0b0b0b' if M[i, j] > 0.45 * M.max() else INK, fontsize=9.5, fontweight='bold')
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015); cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); cb.ax.tick_params(colors=MUTED, length=0)
cb.set_label("share of that problem's importance", color=MUTED)
ax.set_title("What the model looks at for each problem", loc='left', color=INK, fontsize=13, pad=12)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130, facecolor=BG)
