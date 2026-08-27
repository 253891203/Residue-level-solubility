import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_pdbsol_csv, round_up_to, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Check PDBSol CSV columns and sequence stats.")
    parser.add_argument("--train_csv", default="../data/pdbsol/train.csv")
    parser.add_argument("--valid_csv", default="../data/pdbsol/valid.csv")
    parser.add_argument("--test_csv", default="../data/pdbsol/test.csv")
    parser.add_argument("--round_to", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    paths = {
        "train": PROJECT_ROOT / args.train_csv,
        "valid": PROJECT_ROOT / args.valid_csv,
        "test": PROJECT_ROOT / args.test_csv,
    }
    frames = [load_pdbsol_csv(path, split, logger) for split, path in paths.items()]
    max_len = max(int(df["length"].max()) for df in frames)
    pad_len = round_up_to(max_len, args.round_to)
    logger.info("Global max length=%d | pad_len=%d", max_len, pad_len)


if __name__ == "__main__":
    main()
