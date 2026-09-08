"""Generate the three figures for WRITEUP.md.

PLAN.md M5 caps the writeup at three plots. Each is regenerated from the saved
result JSON — never hand-drawn, never typed in — so a figure and the table beside
it cannot drift apart:

    python scripts/make_plots.py

Design follows the bundled data-viz method. Notes on the choices that matter:

- **Categorical slots 1-3 of the reference palette, unchanged.** That subset is
  documented as all-pairs validated in both light and dark modes (CVD ΔE 9.2
  light / 9.4 dark, normal-vision 24.0 / 20.9). Nothing here invents a hue.
- **Light and dark variants of every figure.** Dark mode is *selected* from the
  palette's own dark steps, not an automatic inversion, and WRITEUP.md serves
  them through `<picture>` so a GitHub reader in dark mode does not get a white
  slab.
- **Every figure has a table beside it in the writeup.** Two light-mode slots we
  use sit below 3:1 contrast on the light surface, so the method's relief rule
  applies: ship direct labels or a table view. We ship both.
- **One y-axis, recessive solid gridlines, no per-point labels.**
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "plots"

# Reference palette, slots 1-3. Light / dark are the same hues stepped for their
# own surface — not a flip.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_soft": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_soft": "#c3c2b7",
        "grid": "#333331",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def style(theme: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["ink"],
            "axes.labelcolor": theme["ink_soft"],
            "xtick.color": theme["ink_soft"],
            "ytick.color": theme["ink_soft"],
            "axes.edgecolor": theme["grid"],
            "grid.color": theme["grid"],
            # Solid, thin, recessive. Dashed gridlines read as data.
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.dpi": 160,
        }
    )


def tidy(ax, theme: dict) -> None:
    """Recessive chrome: horizontal grid only, no top/right spines."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(theme["grid"])
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(length=0)


def rounded_bar(ax, x, width, height, color, radius=0.012):
    """A bar with 4px-ish rounded data-ends, anchored to the baseline."""
    patch = FancyBboxPatch(
        (x - width / 2, 0),
        width,
        max(height - radius * 100, 0.1),
        boxstyle=f"round,pad=0,rounding_size={radius * 100}",
        linewidth=0,
        facecolor=color,
        mutation_aspect=0.008,
    )
    ax.add_patch(patch)


def load_contamination() -> dict:
    path = sorted(RESULTS.glob("contamination/*.json"))[0]
    return json.loads(path.read_text(encoding="utf-8"))


def figure_1(theme_name: str) -> Path:
    """Headline: recall@10 by eval condition, both retrievers."""
    theme = THEMES[theme_name]
    style(theme)
    data = load_contamination()

    conditions = ["human", "synth-a", "synth-b"]
    labels = {
        "human": "human-authored\n(BEIR)",
        "synth-a": "corpus-generated\n(typical prompt)",
        "synth-b": "corpus-generated\n(paraphrase-instructed)",
    }
    retrievers = ["bm25", "dense"]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.36
    for r_i, retriever in enumerate(retrievers):
        report = next(
            r for r in data["reports"] if r["chunker"] == "fixed" and r["retriever"] == retriever
        )
        by_name = {c["name"]: c for c in report["conditions"]}
        for c_i, cond in enumerate(conditions):
            est = by_name[cond]["estimates"]["recall@10"]
            mean, lo, hi = est["mean"] * 100, est["lo"] * 100, est["hi"] * 100
            # 2px surface gap between adjacent bars.
            x = c_i + (r_i - 0.5) * (width + 0.02)
            ax.bar(
                x,
                mean,
                width=width,
                color=theme["series"][c_i],
                alpha=1.0 if retriever == "bm25" else 0.55,
                linewidth=0,
                zorder=2,
            )
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="none",
                ecolor=theme["ink_soft"],
                elinewidth=1.4,
                capsize=4,
                zorder=3,
            )
            # Selective direct labels: the value, not a label on every element.
            ax.text(
                x,
                hi + 1.6,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=theme["ink"],
                zorder=4,
            )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([labels[c] for c in conditions], fontsize=9)
    ax.set_ylabel("recall@10 (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title(
        "Same index, same retriever — only the queries differ",
        loc="left",
        pad=14,
        color=theme["ink"],
    )
    ax.text(
        0,
        1.015,
        "SciFact · fixed-token chunks · bars are 95% bootstrap CIs",
        transform=ax.transAxes,
        fontsize=9,
        color=theme["ink_soft"],
        va="bottom",
    )
    # Opacity carries the retriever; the legend says so rather than relying on it.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=theme["ink_soft"], alpha=1.0, linewidth=0),
        plt.Rectangle((0, 0), 1, 1, facecolor=theme["ink_soft"], alpha=0.55, linewidth=0),
    ]
    # Above the axes, sharing the subtitle band. Inside the plot it landed on
    # top of the bars.
    ax.legend(
        handles,
        ["BM25", "dense (MiniLM)"],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1, 1.0),
        fontsize=9,
        labelcolor=theme["ink_soft"],
        ncols=2,
    )
    tidy(ax, theme)
    fig.tight_layout()

    out = OUT / f"fig1-conditions-{theme_name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_2(theme_name: str) -> Path:
    """The overlap gradient, and the ceiling that hides it."""
    theme = THEMES[theme_name]
    style(theme)
    data = load_contamination()
    report = next(
        r for r in data["reports"] if r["chunker"] == "fixed" and r["retriever"] == "bm25"
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    series = [
        ("synth-a", "typical prompt", theme["series"][1]),
        ("synth-b", "paraphrase-instructed", theme["series"][2]),
    ]
    for name, label, color in series:
        rows = report["deciles"][name]
        x = [r["decile"] for r in rows]
        mean = [r["estimates"]["recall@10"]["mean"] * 100 for r in rows]
        lo = [r["estimates"]["recall@10"]["lo"] * 100 for r in rows]
        hi = [r["estimates"]["recall@10"]["hi"] * 100 for r in rows]
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(x, mean, color=color, linewidth=2, zorder=3)
        ax.plot(x, mean, "o", color=color, markersize=5, zorder=4)
        # Direct-label the line end instead of a legend box.
        ax.text(
            x[-1] + 0.16,
            mean[-1],
            label,
            color=color,
            fontsize=9,
            va="center",
            ha="left",
            fontweight="bold",
        )

    # The human baseline is the thing both are measured against.
    human = next(c for c in report["conditions"] if c["name"] == "human")
    baseline = human["estimates"]["recall@10"]["mean"] * 100
    ax.axhline(baseline, color=theme["ink_soft"], linewidth=1, zorder=2)
    ax.text(
        -0.35,
        baseline + 1.6,
        f"human-authored queries ({baseline:.1f})",
        fontsize=9,
        color=theme["ink_soft"],
        va="bottom",
    )

    ax.set_xlabel("query–gold lexical overlap decile  (0 = lowest)")
    ax.set_ylabel("recall@10 (%)")
    ax.set_xticks(range(10))
    ax.set_xlim(-0.4, 11.6)
    ax.set_ylim(40, 104)
    ax.set_title(
        "Retrievability tracks lexical overlap — even where the mean effect is gone",
        loc="left",
        pad=14,
        color=theme["ink"],
    )
    ax.text(
        0,
        1.015,
        "SciFact · BM25 · shaded bands are 95% bootstrap CIs",
        transform=ax.transAxes,
        fontsize=9,
        color=theme["ink_soft"],
        va="bottom",
    )
    tidy(ax, theme)
    fig.tight_layout()

    out = OUT / f"fig2-overlap-gradient-{theme_name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_3(theme_name: str) -> Path:
    """M4: what share of mined 'negatives' clear a relevance bar."""
    theme = THEMES[theme_name]
    style(theme)

    rows = []
    for path in sorted(RESULTS.glob("negatives/*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        rows.append(d)
    rows.sort(key=lambda d: (-d["enrichment"], d["dataset"]))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.36
    for i, d in enumerate(rows):
        for j, (key, color) in enumerate(
            (("background_rate", theme["series"][0]), ("suspected_fraction", theme["series"][1]))
        ):
            x = i + (j - 0.5) * (width + 0.02)
            value = d[key] * 100
            ax.bar(x, value, width=width, color=color, linewidth=0, zorder=2)
            ax.text(
                x,
                value + 1.6,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=theme["ink"],
                zorder=4,
            )
    ax.set_xticks(range(len(rows)))
    # Enrichment belongs in the tick label. Drawn as free text below the axis it
    # landed on top of the tick text.
    ax.set_xticklabels(
        [
            f"{d['dataset']} · {d['retriever']}\n{d['enrichment']:.0f}× enrichment\n"
            f"filter AUC {d['calibration_auc']:.2f}"
            for d in rows
        ],
        fontsize=9,
    )
    ax.set_ylabel("share clearing the relevance bar (%)")
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(
        "Mined “hard negatives” mostly look relevant",
        loc="left",
        pad=14,
        color=theme["ink"],
    )
    ax.text(
        0,
        1.015,
        "same calibrated cross-encoder threshold applied to both groups · × = enrichment",
        transform=ax.transAxes,
        fontsize=9,
        color=theme["ink_soft"],
        va="bottom",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=theme["series"][0], linewidth=0),
        plt.Rectangle((0, 0), 1, 1, facecolor=theme["series"][1], linewidth=0),
    ]
    # Below the axis: this chart's title runs nearly full width, so the band
    # above the plot has no room for a legend.
    ax.legend(
        handles,
        ["random corpus documents", "top-ranked unjudged (mined)"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        fontsize=9,
        labelcolor=theme["ink_soft"],
        ncols=2,
    )
    tidy(ax, theme)
    fig.tight_layout()

    out = OUT / f"fig3-false-negatives-{theme_name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme_name in ("light", "dark"):
        for fn in (figure_1, figure_2, figure_3):
            print(f"wrote {fn(theme_name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
