# Data acquisition and local layout

No third-party sequence dataset is redistributed in this repository. Download each dataset from its original provider, review the provider's current terms, and place the files in the paths below. The repository-level `.gitignore` keeps all files under `data/` untracked except this document.

## Expected directory tree

```text
data/
├── uesolds/
│   ├── train.fasta
│   ├── validation.fasta
│   └── test.fasta
├── netsolp/
│   ├── PSI_Biology_solubility_trainset.csv
│   └── NESG_testset.csv
├── pdbsol/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── fgnnsol/
│   ├── eSol_train.csv
│   ├── eSol_test.csv
│   ├── S.cerevisiae_test.csv
│   ├── eval_data/fastaEval/*.fasta
│   └── test_data/fastaTest/*.fasta
└── protsolm_external/                 # optional external benchmark
    └── ExternalTest.csv
```

Paths are relative to the repository root. Generated embeddings, processed manifests, checkpoints and caches remain outside `data/` and are ignored separately.

## UESolDS / PLM_Sol splits

- Official source: <https://github.com/Violet969/PLM_Sol/tree/main/embedding_dataset>
- Upstream filenames: `Train_dataset.fasta`, `validation_dataset.fasta`, and `test_dataset.fasta`.
- Local filenames: rename them to `train.fasta`, `validation.fasta`, and `test.fasta` under `data/uesolds/`.
- Used by: `ESM2/`, `ProtT5/`, `PLM_Sol/`, and `Explainability/`.

The similarly named upstream `datasets/` directory contains loading code, not the FASTA split files.

## PSI Biology and NESG / NetSolP

- Official service and data page: <https://services.healthtech.dtu.dk/services/NetSolP-1.0/>
- Official code repository: <https://github.com/teevee112/NetSolP-1.0>
- Required files: `PSI_Biology_solubility_trainset.csv` and `NESG_testset.csv`.
- Local destination: `data/netsolp/` without renaming.

The PSI Biology file is used for five-fold model development; `NESG_testset.csv` is reserved for independent testing.

## PDBSol / ProtSolM

- Official source: <https://github.com/tyang816/ProtSolM/tree/main/data/PDBSol>
- Required files: `train.csv`, `valid.csv`, and `test.csv`.
- Local destination: `data/pdbsol/` without renaming.

The upstream ProtSolM repository is licensed separately. Obtain the files from the upstream project instead of copying them into a public fork of this repository.

## eSOL and S. cerevisiae / FGNNSol

- Official source: <https://github.com/SCrownJ/FGNNSol/tree/main/dataset>
- CSV directory: <https://github.com/SCrownJ/FGNNSol/tree/main/dataset/csvFile>
- Required CSV files: `eSol_train.csv`, `eSol_test.csv`, and `S.cerevisiae_test.csv`.
- Required split directories: `eval_data/fastaEval/` and `test_data/fastaTest/`.
- Local destination: place the CSV files directly in `data/fgnnsol/` and retain the FASTA directory names above.

From `FGNNSol/`, `python 00_fetch_official_fasta.py` can sparse-check out the two FASTA directories into the expected root-level data location. It does not download the CSV files. `python 01_check_and_prepare_data.py` creates reproducible local manifests under `FGNNSol/prepared/`; those are intentionally untracked.

## Optional ProtSolM external benchmark

The merged `PROTSOLM/` workflow can evaluate upstream `ExternalTest.csv`. If needed, obtain it from the original ProtSolM repository and place it at `data/protsolm_external/ExternalTest.csv`. Its component datasets remain third-party material and are not redistributed here.

## Redistribution statement

These URLs are provenance and acquisition instructions only. Copyright and dataset licenses remain with their respective owners. Any license later selected for this repository applies only to original repository code and documentation unless a file explicitly states otherwise; it does not relicense third-party datasets or pretrained models.
