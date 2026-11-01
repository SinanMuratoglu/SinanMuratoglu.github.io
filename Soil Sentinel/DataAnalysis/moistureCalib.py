import statistics

import matplotlib.pyplot as plt

dataAir = [53955, 54023, 53989, 54007, 53936, 54039, 53957, 53954, 54022, 53991, 54019, 53997, 53979, 53947, 53972, 54031, 53942, 54004, 53959, 53952, 54011, 53952, 53948, 53988, 53974, 54021, 54038, 54004, 53967, 54043, 54003, 53990, 53995, 53979, 54010, 53981, 53938, 53971, 54032, 53994, 53931, 53989, 53931, 54035, 54033, 53973, 53976, 53991, 54032, 53931, 53949, 54012, 54021, 54011, 53968, 53967, 53985, 53968, 53954, 53992, 53973, 54003, 54006, 54029, 53955, 53999, 53935, 53942, 54026, 54010, 53945, 53993, 54015, 53990, 53988, 53971, 53983, 54020, 53975, 53967, 53987, 54013, 53963, 53976, 54044, 53948, 54031, 53960, 53967, 53964, 54018, 53976, 53984, 54041, 53961, 53978, 54018, 54032, 53979, 53980]
dataWater = [21999, 22254, 22123, 22278, 21912, 22051, 21845, 22044, 21899, 22259, 21845, 22227, 22255, 22004, 22139, 22282, 22156, 22213, 22178, 22089, 21878, 22218, 22285, 22118, 22004, 22158, 22030, 22272, 21899, 21817, 22266, 22210, 22223, 22111, 22246, 21931, 21802, 21832, 22141, 21840, 22127, 21867, 22061, 21873, 22256, 22203, 21989, 22072, 22256, 22271, 22194, 22193, 22041, 22149, 21954, 21837, 22214, 22262, 22213, 21854, 22199, 21832, 22142, 21806, 22063, 21973, 21973, 21937, 22181, 21806, 21968, 21944, 21826, 22272, 21835, 21994, 22126, 21849, 21807, 21996, 22146, 22209, 21971, 22258, 22052, 22044, 22186, 22056, 21834, 21854, 21870, 22208, 22167, 22108, 22017, 21974, 22141, 22261, 22247, 22008]
dataWet = [32217, 31943, 31981, 31745, 31397, 31969, 31409, 31993, 31987, 31330, 31775, 31740, 32096, 31367, 31709, 31528, 31368, 31474, 32142, 31741, 32069, 31311, 32024, 31522, 32044, 31549, 31759, 31908, 31405, 31905, 32033, 32164, 32223, 32041, 31968, 31540, 31808, 31350, 32292, 31316, 31532, 32043, 32244, 32299, 31580, 32281, 31676, 32092, 32151, 31828, 31303, 31576, 32239, 31727, 31405, 31624, 31549, 31976, 31536, 31729, 31850, 31939, 31414, 32263, 31928, 31768, 31431, 32156, 31982, 31715, 31901, 31954, 31600, 31332, 32031, 31606, 31763, 31532, 31862, 31571, 32189, 31770, 32243, 31507, 31910, 31447, 32239, 31306, 32239, 31819, 31539, 32055, 31855, 31430, 32275, 31550, 31916, 32081, 32067, 31440]
dataDry = [42836, 42863, 42832, 43123, 43103, 43103, 42954, 43065, 42889, 42899, 42881, 43010, 43141, 42994, 42823, 43108, 42929, 42825, 43136, 43016, 42934, 42949, 42843, 42802, 42825, 43178, 42814, 43035, 43103, 42888, 43171, 43044, 43051, 43195, 43113, 43096, 43005, 43083, 43093, 43158, 42826, 43036, 43073, 43160, 43030, 43117, 43145, 43122, 42884, 43169, 42833, 42870, 43058, 43091, 42998, 43196, 42868, 43074, 42864, 42969, 42971, 42897, 43076, 42879, 42903, 42888, 42918, 42865, 43105, 42943, 42916, 42962, 42988, 42821, 42835, 42881, 43009, 42920, 43145, 43098, 42965, 43006, 43139, 42852, 42981, 43174, 42890, 42910, 42878, 42948, 42841, 42945, 42947, 43173, 42954, 43200, 43140, 43105, 42925, 42959]

data = [
    ("Air", dataAir),
    ("Dry Soil", dataDry),
    ("Wet Soil", dataWet),
    ("Water", dataWater),
]
THRESHOLD = 38000
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def jitter(n, center=0):
    return [center + ((k * 37) % 21 - 10) / 50 for k in range(n)]


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


for name, v in data:
    print("%-9s %4d %8.0f %7.0f" % (name, len(v), statistics.mean(v),
                                    statistics.stdev(v) if len(v) > 1 else 0))

fig1, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
style(ax)

for i, (name, v) in enumerate(data):
    ax.scatter(jitter(len(v), i), v, s=22, color=COLORS[i], alpha=0.7, linewidths=0)
    m = statistics.mean(v)
    ax.plot([i - 0.3, i + 0.3], [m, m], color=TEXT, linewidth=1)
    ax.text(i + 0.33, m, "mean: %.0f" % m, va="center", fontsize=9, color=TEXT)

ax.axhline(THRESHOLD, color=TEXT_2, linestyle="--", linewidth=1.2)
ax.text(len(data) - 0.5, THRESHOLD, "dry threshold (%d) " % THRESHOLD,
        ha="right", va="bottom", fontsize=9, color=TEXT_2)
ax.set_xticks(range(len(data)))
ax.set_xticklabels([name for name, _ in data], color=TEXT)
ax.set_xlim(-0.6, len(data) - 0.2)
ax.set_ylabel("Raw sensor reading", color=TEXT_2)
ax.set_title("Moisture sensor readings by condition",
             loc="left", color=TEXT)
fig1.tight_layout()
fig1.savefig("strip_plot.png", dpi=200)
print("Saved strip_plot.png")

fig2, axes = plt.subplots(1, len(data), figsize=(11, 4.5), facecolor=SURFACE)

for i, (ax, (name, v)) in enumerate(zip(axes, data)):
    style(ax)
    ax.scatter(jitter(len(v)), v, s=22, color=COLORS[i], alpha=0.6, linewidths=0, zorder=2)
    ax.boxplot(v, positions=[0], widths=0.5, showfliers=False,
               patch_artist=True, zorder=3,
               boxprops={"facecolor": "none", "edgecolor": TEXT, "linewidth": 1.2},
               medianprops={"color": TEXT, "linewidth": 2},
               whiskerprops={"color": TEXT, "linewidth": 1.2},
               capprops={"color": TEXT, "linewidth": 1.2})
    ax.set_title("%s\nmean %.0f, sd %.0f" % (name, statistics.mean(v), statistics.stdev(v)),
                 color=TEXT, fontsize=10)
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)
    ax.spines["bottom"].set_visible(False)

axes[0].set_ylabel("Raw sensor reading", color=TEXT_2)
fig2.suptitle("Moisture sensor readings by condition",
              x=0.01, ha="left", color=TEXT, fontsize=12)
fig2.tight_layout()
fig2.savefig("strip_box_panels.png", dpi=200)
print("Saved strip_box_panels.png")

plt.show()