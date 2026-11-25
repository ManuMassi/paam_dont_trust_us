from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    FONTSIZE = 22
    SHOW_DREBIN = False  # False to plot only the fusion methods
    SHOW_CI = False  # True to add the 95% bootstrap CI of the deferral as a band

    results_dir = (
        Path(__file__).parent.parent.parent / "results" / "Ensemble" / "tables"
    )
    output_path = results_dir

    methods = [
        "Drebin",
        "Tests 1 and 3 (voting)",
        "Test 2 (weighted avg.)",
        "Test 2 without RAMDA",
    ]
    if not SHOW_DREBIN:
        methods = methods[1:]

    title = {
        "Drebin": "Drebin",
        "Tests 1 and 3 (voting)": "Majority voting",
        "Test 2 (weighted avg.)": "$F_1$-weighted avg.",
        "Test 2 without RAMDA": "$F_1$-weighted avg. w/o Ramda",
    }

    df = pd.read_csv(results_dir / "table2_min_deferral.csv")
    targets = np.sort(df["required_f1_and_recall"].unique())

    deferral = {m: [] for m in methods}
    ci_low = {m: [] for m in methods}
    ci_high = {m: [] for m in methods}

    for m in methods:
        rows = df[df["method"] == m].sort_values("required_f1_and_recall")
        if rows.empty:
            raise ValueError(f"{m} not found in table2_min_deferral.csv")
        deferral[m] = rows["test_deferred_pct"].values
        ci_low[m] = rows["test_deferred_pct_ci_low"].values
        ci_high[m] = rows["test_deferred_pct_ci_high"].values

    sns.set_theme(style="whitegrid")

    method_palette = dict(zip(methods, sns.color_palette("tab10", len(methods))))
    linestyle = {m: "-" for m in methods}
    if SHOW_DREBIN:
        method_palette["Drebin"] = "gray"
        linestyle["Drebin"] = "--"

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:

        if SHOW_CI:
            ax.fill_between(
                targets,
                ci_low[method],
                ci_high[method],
                color=method_palette[method],
                alpha=0.15,
                linewidth=0,
            )

        ax.plot(
            targets,
            deferral[method],
            marker="o",
            markersize=4,
            linewidth=2,
            color=method_palette[method],
            linestyle=linestyle[method],
        )

    ax.set_xlabel("Quality target ($F_1$ and recall $\\geq$ target)", fontsize=FONTSIZE - 2)
    ax.set_ylabel("Deferred test samples (%)", fontsize=FONTSIZE - 2)
    ax.set_xticks(targets)
    ax.set_xticklabels([f"{t:.2f}" for t in targets])
    ax.set_ylim(bottom=-0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)

    ax.tick_params(axis="x", labelsize=FONTSIZE - 2)
    ax.tick_params(axis="y", labelsize=FONTSIZE - 2)

    # ---------- legend ----------

    method_handles = [
        Line2D(
            [0], [0],
            color=method_palette[m],
            lw=2,
            linestyle=linestyle[m],
            label=title[m],
        )
        for m in methods
    ]

    fig.legend(
        handles=method_handles,
        loc="upper center",
        ncol=3,
        fontsize=15,
        bbox_to_anchor=(0.55, 1.07)
    )

    plt.tight_layout()

    name = "deferral_targets" if SHOW_DREBIN else "deferral_targets_fusion_only"
    if SHOW_CI:
        name += "_ci"

    plt.savefig(
        output_path / f"{name}.pdf",
        dpi=600,
        bbox_inches="tight",
    )
