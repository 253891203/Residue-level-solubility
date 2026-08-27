from __future__ import annotations

import argparse
from pathlib import Path

from utils.data_utils import length_stats, load_netsolp_csv, resolve_path, write_json


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check NetSolP CSV columns, labels, folds, and sequence lengths.")
    parser.add_argument("--train_csv", default="../data/netsolp/PSI_Biology_solubility_trainset.csv")
    parser.add_argument("--test_csv", default="../data/netsolp/NESG_testset.csv")
    parser.add_argument("--out_dir", default="outputs/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(ROOT, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_info = load_netsolp_csv(resolve_path(ROOT, args.train_csv), "train")
    test_df, test_info = load_netsolp_csv(resolve_path(ROOT, args.test_csv), "nesg")

    fold_message = "未发现原始 fold 列，因此使用分层 5 折划分"
    if train_info["fold_column"] is not None:
        fold_counts = train_df["fold"].value_counts().sort_index()
        fold_message = f"Found original fold column '{train_info['fold_column']}' and will use it: {fold_counts.to_dict()}"
        train_info["fold_distribution"] = {str(k): int(v) for k, v in fold_counts.items()}
    else:
        train_info["fold_strategy"] = fold_message

    stats = {
        "train": length_stats(train_df),
        "NESG_test": length_stats(test_df),
        "note": "All sequences are retained. No filtering or truncation is performed by this check.",
    }
    report = {
        "train": train_info,
        "NESG_test": test_info,
        "sequence_length_stats": stats,
        "fold_message": fold_message,
        "independent_test_rule": "NESG_testset.csv is reserved for final independent testing only.",
    }
    write_json(out_dir / "sequence_length_stats.json", stats)
    write_json(out_dir / "data_check_report.json", report)

    print("Train rows:", len(train_df))
    print("Train columns:", train_info["columns"])
    print("Train label column:", train_info["label_column"], train_info["label_distribution"])
    print("Train sequence column:", train_info["sequence_column"])
    print("Train length stats:", stats["train"])
    print(fold_message)
    print("NESG rows:", len(test_df))
    print("NESG columns:", test_info["columns"])
    print("NESG label column:", test_info["label_column"], test_info["label_distribution"])
    print("NESG sequence column:", test_info["sequence_column"])
    print("NESG length stats:", stats["NESG_test"])
    print("Wrote:", out_dir / "data_check_report.json")


if __name__ == "__main__":
    main()
