"""
Longevity Village Analysis: graphical abstract.

All numbers are typed in from the manuscript results (no data are read).

Usage:
    cd /path/to/kme-microbiome-analysis
    python -m ml.longevity_graphical_abstract [OUT_PREFIX]
Writes OUT_PREFIX.png, OUT_PREFIX.tiff (300 dpi) and OUT_PREFIX_525px.png;
default OUT_PREFIX is ml/output/longevity/graphical_abstract.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from ml.longevity_config import LONGEVITY_OUT

LONGEVITY_OUT.mkdir(parents=True, exist_ok=True)
out = sys.argv[1] if len(sys.argv) > 1 else str(LONGEVITY_OUT / "graphical_abstract")
plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(12, 7.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 120); ax.set_ylim(5, 79); ax.axis("off")


def box(x, y, w, h, face, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=face, ec=edge, lw=2.5))


def text(x, y, s, size=11, color="#333333", **kw):
    ax.text(x, y, s, ha="center", va="top", fontsize=size, color=color, linespacing=1.35, **kw)


# Title
text(60, 77.5, "Host-Adapted Lactobacilli and Fiber-Degrader Depletion in the\n"
               "Gut Microbiome of Korean Longevity-Village Elders",
     size=16, color="black", weight="bold")

# Cohort boxes
box(2, 43, 38, 25, "#BBDEFB", "#1565C0")
text(21, 66, "DS Older Adults (n = 192)", size=13, color="#1565C0", weight="bold")
text(21, 62, "Health-checkup cohort, Naju\naged 60–85", size=10.5)
text(21, 55.5, "Higher:", size=10.5, weight="bold")
text(21, 52.5, "• Lachnospira, Monoglobus\n• Lachnoclostridium,\n  Lachnospiraceae UCG-004",
     size=10, style="italic")
text(21, 46.6, "• Fiber-degrader index (d = −0.55)", size=10)

box(80, 43, 38, 25, "#FFCDD2", "#C62828")
text(99, 66, "Rural Cohort (n = 1,256)", size=13, color="#C62828", weight="bold")
text(99, 62, "Ten longevity villages, Jeollanam-do\naged 63–105; traditional lifestyle", size=10.5)
text(99, 55.5, "Higher:", size=10.5, weight="bold")
text(99, 52.5, "• Lactobacillus, Limosilactobacillus\n• Megasphaera, Olsenella\n• Escherichia-Shigella",
     size=10, style="italic")

# Comparison arrow
ax.annotate("", xy=(79.5, 55.5), xytext=(40.5, 55.5),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=2.5))
text(60, 62, "70 genera differ\n(48 survive rarefaction)", size=11, color="#555555", weight="bold")
text(60, 53.5, "same kit, primers,\nsequencing, pipeline", size=9.5, color="#777777", style="italic")

# Key findings
box(12, 11, 96, 29, "#FFF9C4", "#F9A825")
text(60, 38, "Key Findings", size=14, color="#E65100", weight="bold")
cols = [
    (28, "Taxonomic",
     "• Host-adapted lactobacilli\n  enriched in RC\n• Kimchi fermenters\n  not enriched\n"
     "• Survives age-matching,\n  sex stratification, rarefaction"),
    (60, "Functional (PICRUSt2)",
     "• Butyrate genes 12% lower\n  (rarefied), kinase route only;\n  CoA-transferase unchanged\n"
     "• Lactose catabolism 1.8× higher\n• Fermentation/glycolysis: not\n  supported after depth\n  normalization"),
    (92, "Study Design",
     "• All ten villages longevity-\n  designated; no within-study\n  longevity contrast\n"
     "• Sequencing run confounded\n  with cohort\n• PICRUSt2 restates taxonomy,\n  not independent evidence"),
]
for x, head, body in cols:
    text(x, 33, head, size=11.5, weight="bold")
    ax.text(x - 13, 29.5, body, ha="left", va="top", fontsize=9.8, color="#444444", linespacing=1.35)
for x in (44, 76):
    ax.plot([x, x], [14, 31], color="#CCCCCC", lw=1)

text(60, 8, "2,084 participants  •  16S rRNA V4  •  Jeollanam-do, South Korea",
     size=10, color="#888888", style="italic")

fig.savefig(f"{out}.png", dpi=300)
fig.savefig(f"{out}.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
im = Image.open(f"{out}.png")
im.resize((525, round(525 * im.height / im.width)), Image.LANCZOS).save(f"{out}_525px.png")
