"""Generate the final paper figure for the ESM2/ProtT5 P/N comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Values used in the final manuscript figure. ProtT5 values are the retained
# server results; ESM2 values retain the full precision stored by the sweep.
MLP_N_RESULTS = {
    "ProtT5": {
        "reduced_dim": [2, 4, 6, 8, 16, 32],
        "test_accuracy": [0.6878, 0.7003, 0.7043, 0.6990, 0.70975, 0.7125],
    },
    "ESM2": {
        "reduced_dim": [2, 4, 6, 8, 16, 32],
        "test_accuracy": [
            0.6873428331936295,
            0.691533948030176,
            0.6811958647666946,
            0.6892986867840178,
            0.6809164571109249,
            0.6837105336686226,
        ],
    },
}

MLP_P_RESULTS = {
    "ProtT5": {
        "reduced_dim": [2, 4, 6, 8, 16, 32],
        "test_accuracy": [0.62325, 0.63575, 0.64825, 0.65375, 0.6555, 0.653],
    },
    "ESM2": {
        "reduced_dim": [2, 4, 6, 8, 16, 32],
        "test_accuracy": [
            0.6286672254819782,
            0.648225761385862,
            0.6311818943839062,
            0.647108130762783,
            0.668622520257055,
            0.6728136350936016,
        ],
    },
}

DISPLAY_DIMENSIONS = [2, 4, 6, 8, 16, 32]
F_REPRESENTATION_ACCURACY = {"ProtT5": 0.7083, "ESM2": 0.6781223805532272}


def plot_mlp_pn_accuracy(output_path: Path) -> None:
    """Render one PNG or PDF using the exact final-manuscript styling."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(4.6, 3.9), dpi=220)
    model_styles = {"ProtT5": {"color": "#4C78A8"}, "ESM2": {"color": "#E45756"}}
    f_line_colors = {"ProtT5": "#2F5F98", "ESM2": model_styles["ESM2"]["color"]}
    representation_results = {"N": MLP_N_RESULTS, "P": MLP_P_RESULTS}
    representation_styles = {
        "N": {"marker": "s", "linestyle": "-"},
        "P": {"marker": "o", "linestyle": "--"},
    }
    model_x_offsets = {"ProtT5": -0.12, "ESM2": 0.12}
    representation_x_offsets = {"P": -0.025, "N": 0.025}
    dimension_positions = {
        dimension: index for index, dimension in enumerate(DISPLAY_DIMENSIONS)
    }
    annotation_offsets = {
        ("N", "ProtT5"): 8,
        ("N", "ESM2"): -9,
        ("P", "ProtT5"): -9,
        ("P", "ESM2"): 8,
    }
    special_annotation_offsets = {
        ("N", "ProtT5", 2): (8, 11),
        ("N", "ESM2", 2): (10, -4),
        ("N", "ProtT5", 4): (-6, 3),
        # Keep the K=6 N labels close to their points so they do not meet midway.
        ("N", "ProtT5", 6): (-5, -5),
        ("N", "ESM2", 6): (5, 5),
        ("N", "ProtT5", 8): (0, -4),
        ("N", "ESM2", 8): (2, -7),
        ("N", "ESM2", 16): (0, 10),
        ("N", "ESM2", 32): (0, 10),
        ("P", "ProtT5", 2): (10, -10),
        # The K=6 P labels are placed outside the two nearby curves.
        ("P", "ProtT5", 6): (0, 10),
        ("P", "ESM2", 6): (0, -10),
        # These two nearby K=8 labels are placed away from each other.
        ("P", "ProtT5", 8): (0, 10),
        ("P", "ESM2", 8): (0, -10),
        ("P", "ESM2", 16): (-2, 4),
        ("P", "ESM2", 32): (0, -10),
    }

    for representation, results in representation_results.items():
        for model, values in results.items():
            x_positions = [
                dimension_positions[dimension]
                + model_x_offsets[model]
                + representation_x_offsets[representation]
                for dimension in values["reduced_dim"]
            ]
            axis.plot(
                x_positions,
                values["test_accuracy"],
                linewidth=2.2,
                markersize=5.5,
                marker=representation_styles[representation]["marker"],
                linestyle=representation_styles[representation]["linestyle"],
                **model_styles[model],
            )
            for dimension, x_value, y_value in zip(
                values["reduced_dim"], x_positions, values["test_accuracy"]
            ):
                x_offset, y_offset = special_annotation_offsets.get(
                    (representation, model, dimension),
                    (0, annotation_offsets[(representation, model)]),
                )
                axis.annotate(
                    f"{y_value:.4f}",
                    xy=(x_value, y_value),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if y_offset > 0 else "top",
                    fontsize=9,
                )

    for model, accuracy in F_REPRESENTATION_ACCURACY.items():
        color = f_line_colors[model]
        axis.axhline(
            accuracy,
            color=color,
            linestyle="--",
            linewidth=2.2,
            alpha=1.0,
            zorder=1,
        )
        y_offset = 7 if model == "ProtT5" else -14
        axis.annotate(
            f"{model} F = {accuracy:.4f}",
            xy=(0.015, accuracy),
            xycoords=("axes fraction", "data"),
            xytext=(0, y_offset),
            textcoords="offset points",
            color=color,
            fontsize=9.5,
            ha="left",
            va="bottom" if y_offset > 0 else "top",
        )

    axis.set_xlabel(r"Reduced feature dimension $K$", labelpad=6)
    axis.set_ylabel("Test accuracy")
    axis.set_xticks(
        range(len(DISPLAY_DIMENSIONS)),
        [str(dimension) for dimension in DISPLAY_DIMENSIONS],
    )
    axis.set_xlim(-0.42, len(DISPLAY_DIMENSIONS) - 0.58)
    axis.set_ylim(0.608, 0.722)
    axis.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
    axis.legend(
        handles=[
            Line2D([0], [0], color=model_styles["ProtT5"]["color"], lw=2.2, label="ProtT5"),
            Line2D([0], [0], color=model_styles["ESM2"]["color"], lw=2.2, label="ESM2"),
            Line2D([0], [0], color="black", marker="s", linestyle="-", markersize=5.5, label="N"),
            Line2D([0], [0], color="black", marker="o", linestyle="--", markersize=5.5, label="P"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="F"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=5,
        columnspacing=0.9,
        handletextpad=0.45,
        borderaxespad=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.27)
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the PNG and PDF (default: this script's directory).",
    )
    args = parser.parse_args()
    for suffix in ("png", "pdf"):
        output_path = args.output_dir / f"prott5_esm2_mlp_pn_accuracy.{suffix}"
        plot_mlp_pn_accuracy(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
