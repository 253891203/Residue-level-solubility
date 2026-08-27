# Simple yet Effective: The Critical Role of Residue-Level Information in Protein Solubility Prediction

This repository contains the paper's implementation and final machine-readable results. It compares pooled protein-level representations with representations that retain residue positions, and evaluates a standard Transformer on complete ESM2 residue-embedding matrices.

Third-party datasets, pretrained encoder files, generated residue embeddings and trained checkpoints are intentionally not stored in the Git repository tree. See [`data/README.md`](data/README.md) for verified upstream sources and the required local layout.

The paper's trained downstream checkpoints are distributed separately as the GitHub Release asset `Residue-level-solubility-checkpoints-v1.zip`. Extract that archive into the repository root to restore the original `outputs/checkpoints/` paths. Keeping weights in a Release avoids placing large binaries in Git history while preserving direct evaluation and masking-analysis reproducibility.

## Main results

### Residue-representation experiments

| PLM | Representation | Input size | ACC | F1 |
| --- | --- | ---: | ---: | ---: |
| ESM2-650M | F | 1,280 | 0.6781 | 0.7381 |
| ESM2-650M | S | 500 | 0.6097 | 0.7041 |
| ESM2-650M | P-4 | 2,000 | 0.6482 | 0.7129 |
| ESM2-650M | N-4 | 2,000 | **0.6915** | **0.7461** |
| ProtT5 | F | 1,024 | 0.7083 | 0.7405 |
| ProtT5 | S | 500 | 0.5815 | 0.6744 |
| ProtT5 | P-8 | 4,000 | 0.6538 | 0.6937 |
| ProtT5 | N-32 | 16,000 | **0.7125** | **0.7441** |

Complete sweeps are retained in `ESM2/outputs/results/esm2_representation_results.csv` and `ProtT5/outputs/results/prott5_representation_results.csv`.

### Benchmark experiments

| Dataset | Directory | ACC | F1 | Precision | Recall | AUC | MCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NESG | `NetSolP/` | 0.734 | 0.816 | 0.729 | 0.926 | 0.774 | 0.396 |
| PDBSol-test | `PROTSOLM/` | 0.802 | 0.809 | 0.810 | 0.808 | 0.891 | 0.604 |
| UESolDS | `PLM_Sol/` | 0.751 | 0.760 | 0.733 | 0.789 | 0.831 | 0.503 |
| eSOL | `FGNNSol/` | 0.786 | 0.770 | 0.777 | 0.764 | 0.885 | 0.571 |

The eSOL values are the mean of seeds 2024--2028. Other rows retain the evaluation protocol reported in the manuscript.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/README.md` | Third-party acquisition, provenance and expected local filenames |
| `ESM2/` | ESM2-650M F/S/P/N experiments and final result tables |
| `ProtT5/` | ProtT5 F/S/P/N experiments and final result tables |
| `NetSolP/` | PSI Biology five-fold training code and NESG results |
| `PROTSOLM/` | Merged PDBSol/external-test workflows and mean-pool ablation |
| `PLM_Sol/` | Full-length UESolDS Transformer workflow and reported metrics |
| `FGNNSol/` | eSOL regression, five-seed predictions and aggregate results |
| `Explainability/` | Masking, amino-acid ranking, AFDB mapping and structural tables |
| `figures/representation_comparison/` | Final ESM2/ProtT5 P/N figure, vector PDF, and `plot_mlp_pn_accuracy.py` generator |

Experiment directories use numbered scripts for execution order and `outputs/results/` for retained results. Fold counts differ by benchmark and are deliberately preserved.

## Installation

Python 3.10 or newer is recommended. A CUDA GPU is required for practical ESM2-650M/ProtT5 embedding generation and full-matrix training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The encoder identifiers are `esm2_t33_650M_UR50D` (layer 33, 1,280 residue features) and `Rostlab/prot_t5_xl_half_uniref50-enc` (1,024 residue features). Their weights are downloaded by the relevant libraries or supplied through model-path options; they are not part of this repository.

## Reproduction

First populate the ignored root `data/` directory exactly as described in [`data/README.md`](data/README.md). Run commands from each named experiment directory.

### ESM2 and ProtT5 representation experiments

```bash
cd ESM2
python 01_generate_esm2_embeddings.py --device-ids 0,1
python 02_train_fsp_representations.py --embedding-dir embeddings --representations F,S,P --dims 1,2,4,6,8,16,32
python 03_train_n_representations.py --embedding-dir embeddings --dims 1,2,4,6,8,16,32

cd ../ProtT5
python 01_generate_prott5_embeddings.py
python 02_train_fsp_representations.py --representations F,S,P --dims 1,2,4,6,8,16,32
python 02_train_fsp_representations.py --representations N --dims 1,2,4,6,8
python 03_train_n_representations.py --dims 16,32
```

Only sequences at most 500 residues long are retained in these preliminary experiments. The post-filter train/validation/test sizes are 60,953/3,450/3,579.

Regenerate the final cross-model manuscript figure from the repository root with:

```bash
python figures/representation_comparison/plot_mlp_pn_accuracy.py
```

This writes `prott5_esm2_mlp_pn_accuracy.png` and the vector PDF beside the script. The per-model diagnostic plotters remain in `ESM2/plot_esm2_mlp_pn_accuracy.py` and `ProtT5/plot_prott5_mlp_pn_accuracy.py`.

### NetSolP: PSI Biology to NESG

```bash
cd NetSolP
python 00_check_data.py
python 01_embed_esm2_650m.py --allow_online
python 02_train_transformer_5fold.py
python 03_eval_nesg.py
```

### PDBSol and mean-pool ablation

`PROTSOLM/` is the merged release; duplicate local/server copies have been removed.

```bash
cd PROTSOLM
python scripts/01_check_pdbsol_csv.py
python scripts/02_embed_pdbsol_esm2.py --out_dir outputs/embeddings --device cuda
python scripts/02b_verify_embeddings.py --embedding_dir outputs/embeddings
python scripts/03_train_transformer.py --embedding_dir outputs/embeddings --epochs 30 --patience 5 --batch_size 8 --seed 42
python scripts/03b_train_transformer_meanpool_ablation.py --embedding_dir outputs/embeddings
```

The optional external workflow reads `../data/protsolm_external/ExternalTest.csv` by default:

```bash
python scripts/02_embed_external_esm2.py
python scripts/04_eval_transformer.py --target both --model_set all
```

### Full-length UESolDS Transformer

```bash
cd PLM_Sol
python 01_generate_embeddings.py --output-dir embeddings --device-ids 0,1
python 02_train_transformer.py --embedding-dir embeddings --output-dir outputs
python 03_evaluate.py --checkpoint outputs/checkpoints/best_model.pt --embedding-dir embeddings
```

The upstream training FASTA contains 70,031 rows. The paper's legacy dictionary-key preprocessing yields 69,831 effective entries because 200 repeated identifiers overwrite earlier records within the same shard. `python 01_generate_embeddings.py --audit-only` checks this without producing embeddings.

### eSOL / FGNNSol comparison

```bash
cd FGNNSol
python 00_fetch_official_fasta.py
python 01_check_and_prepare_data.py
python 02_embed_esm2_650m.py --embedding_dir cache/esm2_650m_embeddings --resume True
python 03_train_transformer_regression.py --embedding_dir cache/esm2_650m_embeddings --output_dir outputs --seeds 2024 2025 2026 2027 2028
python 04_evaluate_fixed_test_sets.py --embedding_dir cache/esm2_650m_embeddings --output_dir outputs
```

### Masking and structural analyses

From `Explainability/`, generate the ignored intermediates and use a locally trained PLM_Sol checkpoint:

```bash
python scripts/generate_masked_esm2_embeddings_2500.py --output-dir masked_embeddings
python scripts/batch_annotate_and_rank_amino_acid_2500.py \
  --train-h5 ../PLM_Sol/embeddings/train_esm2_650m_maxlen2500.h5 \
  --test-h5 ../PLM_Sol/embeddings/test_esm2_650m_maxlen2500.h5 \
  --masked-h5 masked_embeddings/masked_test_esm2_embeddings_2500.h5 \
  --masked-index-csv masked_embeddings/masked_test_esm2_index_2500.csv \
  --model-path ../PLM_Sol/outputs/checkpoints/best_model.pt \
  --results-dir outputs/masking
```

AFDB coordinate downloads are ignored. The committed structural tables contain 6,240 key-site records; 5,200 were resolved to AFDB residues and 3,137 resolved sites were surface-exposed at RSA >= 25%. The manuscript's Top/Bottom 25% comparison is in `Explainability/outputs/structure/top_bottom_25pct/`.

## Included and excluded artifacts

Included are original analysis/training code, small configurations, final metrics, histories, aggregate tables, per-sample predictions, masking transitions, structural summaries, and result figures.

Excluded are all third-party raw or trivially converted datasets; `.pt`, `.pth`, and `.ckpt` weights; `.npy`, `.npz`, `.h5`, and `.hdf5` arrays; generated embeddings; downloaded AFDB coordinates; caches, logs and packaging archives. These rules are enforced by `.gitignore`. Result files may record historical checkpoint filenames for provenance even though the binaries are not distributed.

## Directional key-site ranking

The final frequency-corrected ranking is included as `Explainability/outputs/masking/directional_key_site_ranking_2500.csv`, together with its summary JSON and plot. It was produced by `rank_directional_key_site_contribution_2500.py` and gives the manuscript ordering `H > D > E > G > S > Q > K > W > Y > L > R > N > F > T > V > A > P > C > I > M`. The older class-balanced, key-site-only ranking uses a different denominator and is intentionally not included as the paper result.

The manuscript structure figure (`overall_framework.png`) is not required to run the code and is not included. It can be added later only if the authors want the repository landing page to reproduce the paper overview.

## License and third-party materials

No repository-wide software license has yet been selected. Add the authors' chosen license before public release. Any future repository license must cover original code/documentation only and must not be presented as relicensing UESolDS, PSI Biology, NESG, PDBSol, eSOL/FGNNSol data, pretrained encoders, or other third-party materials.

## Citation

```bibtex
@article{peng_residue_solubility,
  title   = {Simple yet Effective: The Critical Role of Residue-Level Information in Protein Solubility Prediction},
  author  = {Peng, Junhao and Hao, Xinru and Liang, Zhi and Zhang, Sihai},
  note    = {Manuscript}
}
```
