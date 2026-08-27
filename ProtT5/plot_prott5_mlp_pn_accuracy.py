import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DISPLAY_DIMENSIONS = [2, 4, 6, 8, 16, 32]


def selected_series(frame, representation):
    rows = frame[
        (frame["representation_type"] == representation)
        & (frame["reduce_dim"].isin(DISPLAY_DIMENSIONS))
        & (frame["used_in_paper_figure"])
    ].sort_values("reduce_dim")
    dimensions = rows["reduce_dim"].astype(int).tolist()
    if dimensions != DISPLAY_DIMENSIONS:
        raise ValueError(
            f"{representation} dimensions are {dimensions}, expected {DISPLAY_DIMENSIONS}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Plot the ProtT5 P/N accuracy series used in the paper figure."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/results/prott5_representation_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/prott5_mlp_pn_accuracy.png"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    p_rows = selected_series(frame, "P")
    n_rows = selected_series(frame, "N")
    f_accuracy = float(
        frame.loc[frame["representation"] == "F", "test_accuracy"].iloc[0]
    )

    fig, axis = plt.subplots(figsize=(6.4, 4.4), dpi=220)
    axis.plot(
        p_rows["reduce_dim"],
        p_rows["test_accuracy"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="P",
    )
    axis.plot(
        n_rows["reduce_dim"],
        n_rows["test_accuracy"],
        marker="s",
        linestyle="-",
        linewidth=2,
        label="N",
    )
    axis.axhline(f_accuracy, color="black", linestyle=":", linewidth=1.6, label="F")
    for rows in (p_rows, n_rows):
        for _, row in rows.iterrows():
            axis.annotate(
                f"{row['test_accuracy']:.4f}",
                (row["reduce_dim"], row["test_accuracy"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_xticks(DISPLAY_DIMENSIONS)
    axis.set_xlabel("Reduced dimension")
    axis.set_ylabel("Test accuracy")
    axis.set_title("ProtT5 MLP P/N-representation accuracy")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
