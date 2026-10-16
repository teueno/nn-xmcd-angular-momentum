# nn-xmcd-angular-momentum
Code, data, and trained models for neural-network-based reconstruction of spin and orbital angular momentum from synthetic XAS/XMCD spectra.

This repository provides the complete code, model configurations, and evaluation scripts used in the manuscript:
Tetsuro Ueno "Neural-network-based reconstruction of spin and orbital angular momentum from X-ray magnetic circular dichroism spectra" [arXiv:2607.11058](https://arxiv.org/abs/2607.11058) [cond-mat.mtrl-sci].


## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.


## Citation

If you use this software, please cite both the software repository and the associated article.

Software DOI

https://doi.org/10.5281/zenodo.23409837

Repository

https://github.com/teueno/nn-xmcd-angular-momentum

The associated manuscript is currently available as:

Tetsuro Ueno
"Neural-network-based reconstruction of spin and orbital angular momentum from X-ray magnetic circular dichroism spectra"
[arXiv:2607.11058](https://arxiv.org/abs/2607.11058) [cond-mat.mtrl-sci]
https://arxiv.org/abs/2607.11058


## Repository Structure

```text
configs/
├── dataset/
│   ├── Fe_d6.json
│   ├── Co_d7.json
│   └── Ni_d8.json
│
└── training/
    ├── Fe_d6.json
    ├── Co_d7.json
    └── Ni_d8.json

data/
├── processed/
├── splits/
└── preprocessed/

figures/
├── Figure2_XAS_XMCD_multi_element.pdf
├── Figure2_XAS_XMCD_multi_element.png
├── Figure3_Sz_Lz_multi.pdf
├── Figure3_Sz_Lz_multi.png
├── Figure3_Sz_Lz_multi.pdf
├── Figure3_Sz_Lz_multi.png
├── Figure5_learning_curves.pdf
└── Figure5_learning_curves.png

models
├── Fe_d6/
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── selected_model.pt
│   └── training_config.json
│
├── Co_d7/
└── Ni_d8/

notebooks/
├── figure_2_xas_and_xmcd.ipynb
├── figure_3_parameter_dependences_of_sz_and_lz.ipynb
├── figure_4_parity_plots.ipynb
└── figure_5_learning_curve.ipynb

quanty/
├── configs/
├── runs/
├── templates/
└── run_quanty.py

results/
├── audit/
├── Fe_d6/
├── Co_d7/
└── Ni_d8/

src/
├── __init__.py
├── audit_dataset.py
├── data_loader.py
├── evaluate.py
├── metrics.py
├── model.py
├── prepare_dataset.py
├── preprocess.py
├── split_dataset.py
├── train.py
└── utils.py

.gitignore

CITATION.cff

environment.yml

LICENSE

README.md

requirements.txt
```


## Data Availability

To keep the GitHub repository lightweight and maintainable, several large files are not included in this repository.

The following files are excluded via `.gitignore`:

```text
quanty/runs/*/spec/*.spec

data/processed/*.npz
data/splits/*.npz
data/preprocessed/*.npz

models/**/*.pt
models/**/*.pth
```

These files include:

- Quanty-generated XAS/XMCD spectra (**.spec`)
- Processed machine-learning datasets (`*.npz`)
- Preprocessed datasets and scaling parameters *`*.npz`)
- Trained neural-network checkpoint* (`*.pt`, `*.pth`)

The source code, configuration files, notebooks, valuation results, and metadata required to reproduce the workflow are included in this repository.

Large datasets, spectra, and trained models will be archived separately through Zenodo.


# Installation

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- PyTorch
- Git
- Quanty (see https://www.quanty.org for installation instructions)

The code has been tested with CPU execution. For maximum reproducibility, CPU training is recommended.

---

## Clone repository

```bash
mkdir -p ~/github_test
cd ~/github_test

git clone \
https://github.com/teueno/nn-xmcd-angular-momentum.git

cd nn-xmcd-angular-momentum
```

## Create a Python environment

### Conda

```bash
conda create -n xmcd_nn python=3.11
conda activate xmcd_nn
```

### venv

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## Install dependencies

### Using requirements.txt

```bash
pip install -r requirements.txt
```

---

## Verify installation

```bash
python -c "import torch, numpy, pandas; print(torch.__version__)"
```

---



# Workflow

The complete workflow consists of:

```text
Quanty
↓
prepare_dataset.py
↓
split_dataset.py
↓
audit_dataset.py
↓
preprocess.py
↓
train.py
↓
evaluate.py
```

---

# 1. Generate Quanty Spectra

Run Quanty calculations and collect synthetic XAS/XMCD spectra.

```bash
python quanty/run_quanty.py --config quanty/configs/Fe_d6.json
```

Output:

```text
quanty/runs/Fe_d6/logs/
quanty/runs/Fe_d6/lua/
quanty/runs/Fe_d6/metadata/
quanty/runs/Fe_d6/spec/
quanty/runs/Fe_d6/status/
quanty/runs/quanty_results_Fe_d6.csv
```

In the following, the same workflow applies to:

```text
Co_d7
Ni_d8
```

---

# 2. Prepare Dataset

Create the machine-learning dataset from Quanty outputs.

```bash
python src/prepare_dataset.py --config configs/dataset/Fe_d6.json
```

Output:

```text
data/processed/
├── Fe_d6_dataset.npz
├── Fe_d6_metadata.csv
├── Fe_d6_accepted.csv
├── Fe_d6_excluded.csv
└── Fe_d6_summary.json
```

---

# 3. Create a Reproducible Split

Generate fixed train, validation, and test subsets.

```bash
python src/split_dataset.py --config configs/dataset/Fe_d6.json
```

Output:

```text
data/splits/
├── Fe_d6_split.csv
├── Fe_d6_split_indices.npz
└── Fe_d6_split_summary.json
```

Default split ratios:

```text
Training   : 70%
Validation : 15%
Test       : 15%
```

---

# 4. Audit Dataset

Check for:

- duplicate spectra
- duplicate parameter combinations
- train/validation/test leakage
- identical spectra with inconsistent labels

```bash
python src/audit_dataset.py --config configs/dataset/Fe_d6.json
```

Output:

```text
results/audit/
├── Fe_d6_audit_summary.json
└── Fe_d6_duplicate_groups.csv
```

A successful audit should report:

```json
"audit_status": "pass"
```

---

# 5. Fit Preprocessing on the Training Subset Only

Input spectra are standardized independently at each energy channel using statistics calculated from the training subset only.

Target variables (`Sz`, `Lz`) are also standardized using the training subset only.

```bash
python src/preprocess.py --config configs/dataset/Fe_d6.json
```

Output:

```text
data/preprocessed/
├── Fe_d6_preprocessed.npz
├── Fe_d6_scalers.npz
└── Fe_d6_preprocess_summary.json
```

---

# 6. Verify DataLoader

Optional validation step.

```bash
python src/data_loader.py --config configs/training/Fe_d6.json
```

Typical output:

```text
DataLoader setup completed successfully.
```

Checks:

- sample counts
- tensor shapes
- batch sizes
- device selection

---

# 7. Verify Model Definition

Optional model self-test.

```bash
python src/model.py
```

Successful execution should produce:

```text
Model self-test completed successfully.
```

---

# 8. Train the Neural Network

```bash
python src/train.py --config configs/training/Fe_d6.json
```

Output:

```text
models/Fe_d6/
├── best_model.pt
├── final_model.pt
├── selected_model.pt
└── training_config.json

results/Fe_d6/
├── training_history.csv
└── training_summary.json
```

Notes:

- Training uses fixed train/validation/test subsets.
- Input preprocessing is fitted on the training subset only.
- Early stopping monitors validation loss.
- `best_model.pt` corresponds to the lowest validation loss.
- `selected_model.pt` is used for all evaluations and manuscript figures.

---

# 9. Evaluate the Trained Model

```bash
python src/evaluate.py --config configs/training/Fe_d6.json
```

Output:

```text
results/Fe_d6/
├── predictions.csv
├── metrics.json
├── residual_summary.csv
└── evaluation_summary.json
```

---

# Evaluation Metrics

For each target (`Sz`, `Lz`), the following metrics are reported:

```text
SSE
SST
MSE
RMSE
NRMSE
MAE
Bias
R²
Pearson r
Pearson r²
```

Definitions:

```text
Residual = y_pred − y_true

SSE = Σ(y_pred − y_true)²

SST = Σ(y_true − mean(y_true))²

R² = 1 − SSE / SST

RMSE = sqrt(MSE)

NRMSE = RMSE / std_population(y_true)

Bias = mean(y_pred − y_true)
```

---

# Predictions

`predictions.csv` contains the complete test subset:

```text
sample_id
prefix

sz_true_standardized
sz_pred_standardized

sz_true
sz_pred

sz_residual

lz_true_standardized
lz_pred_standardized

lz_true
lz_pred

lz_residual
```

Additional metadata columns from `*_accepted.csv` are automatically merged when available.

---

# Complete Example

For Fe:

```bash
python src/prepare_dataset.py --config configs/dataset/Fe_d6.json

python src/split_dataset.py --config configs/dataset/Fe_d6.json

python src/audit_dataset.py --config configs/dataset/Fe_d6.json

python src/preprocess.py --config configs/dataset/Fe_d6.json

python src/train.py --config configs/training/Fe_d6.json

python src/evaluate.py --config configs/training/Fe_d6.json
```

Replace `Fe_d6` with:

```text
Co_d7
Ni_d8
```

to reproduce the corresponding datasets and models.

---

# Re-running a Step

All processing scripts support:

```bash
--overwrite
```

Example:

```bash
python src/train.py --config configs/training/Fe_d6.json --overwrite
```

---

# Reproducibility

The repository records:

- configuration files
- train/validation/test assignments
- preprocessing statistics
- training history
- evaluation metrics
- SHA-256 hashes of inputs and outputs

This provides a complete audit trail from Quanty-generated spectra to the final neural-network predictions.


# Figure Reproduction

All figures reported in the manuscript can be reproduced using the Jupyter notebooks provided in:

```text
notebooks/
```

The following notebooks generate the figures used in the paper:

| Figure | Notebook |
|----------|----------|
| Figure 2 | `figure_2_xas_and_xmcd.ipynb` |
| Figure 3 | `figure_3_parameter_dependences_of_sz_and_lz.ipynb` |
| Figure 4 | `figure_4_parity_plots.ipynb` |
| Figure 5 | `figure_5_learning_curves.ipynb` |

The notebooks use the outputs generated by the training and evaluation workflow, including:

```text
results/
├── Fe_d6/
├── Co_d7/
└── Ni_d8/
```

Generated figures are written to:

```text
figures/
```

as both PNG and PDF files.

## Figure generation workflow

The figures should be generated after the complete machine-learning workflow has finished:

```text
prepare_dataset.py
↓
split_dataset.py
↓
audit_dataset.py
↓
preprocess.py
↓
train.py
↓
evaluate.py
↓
Jupyter notebooks
```

The notebooks read the saved outputs from `results/` and reproduce the manuscript figures directly from the evaluation results