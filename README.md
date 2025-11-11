# SED Predict

SED Predict is a research-oriented toolkit for predicting the Spitzer MIPS 24 μm magnitude of
young stellar objects (YSOs) from shorter-wavelength photometry.  It bundles feature engineering,
model training, inference, posterior sampling, and diagnostic plotting into a single workflow so
that astronomers can quantify both predictive performance and astrophysical uncertainties.

## Project highlights

* **Unified feature engineering** – `data_loader.py` constructs a consistent set of base and
  derived features (colour indices, Galactic longitude sine/cosine terms, and quadratic terms)
  before any model is trained or evaluated.【F:data_loader.py†L1-L82】
* **Multiple regression back-ends** – Gradient-boosted trees (XGBoost), NGBoost probabilistic
  boosting, and an MLP regressor are implemented behind a common interface, each returning training
  histories for later diagnostics.【F:train.py†L1-L153】【F:xgboost_model.py†L1-L66】【F:ngboost_model.py†L1-L43】【F:mlp_model.py†L1-L38】
* **Posterior decomposition** – The `posterior.py` module combines model uncertainty with a simple
  inclination-driven perturbation model (after Whitney et al. 2003) to produce posterior samples and
  per-stage uncertainty budgets.【F:posterior.py†L1-L77】
* **Rich visual diagnostics** – `plots.py` generates loss curves, residual analyses, spatial error
  maps, colour–colour diagrams, and posterior visualisations, making it easy to interrogate model
  behaviour and the resulting uncertainty estimates.【F:plots.py†L1-L196】【F:plots.py†L217-L315】

## Repository layout

```
SED_predict/
├── data_loader.py        # Shared feature engineering and train/test/inference splits
├── train.py              # Configurable training entry point
├── inference.py          # Batch inference and diagnostic plotting
├── xgboost_model.py      # Gradient boosted tree training utilities
├── ngboost_model.py      # NGBoost training utilities
├── mlp_model.py          # Feed-forward neural network training utilities
├── posterior.py          # Posterior sampling & uncertainty quantification helpers
├── plots.py              # Matplotlib / seaborn visualisations for training & inference
├── inlist                # INI-style configuration consumed by train.py & inference.py
└── outputs/              # Default location for trained artefacts and plots (gitignored)
```

The repository does not ship training data; configure `inlist` to point at a CSV file with the
expected columns.

## Installation

1. Create a Python 3.9+ virtual environment.
2. Install dependencies:

   ```bash
   pip install numpy pandas scikit-learn xgboost ngboost seaborn matplotlib joblib
   ```

   NGBoost may require a working C compiler; consult the
   [NGBoost documentation](https://github.com/stanfordmlgroup/ngboost) if build issues appear.

## Configuration

The `inlist` file supplies both training and inference defaults. Important keys include:

* `paths.data_file` – CSV containing the source catalogue used for training or inference.
* `hyperparameters.*` – Learning rate, number of boosting stages, and tree depth used by the
  selected model.【F:train.py†L20-L76】
* `general.test_size` / `general.val_size` – Fractions used when `data_loader.load_and_split_data`
  derives train/validation/test partitions.【F:data_loader.py†L40-L82】
* `general.model_type` – One of `xgboost`, `ngboost`, or `mlp` determining which back-end is
  trained or loaded for inference.【F:train.py†L27-L116】【F:inference.py†L97-L157】
* `general.output_dir` – Directory for trained artefacts, plots, and posterior samples.
* `columns.feature_columns` / `columns.target_column` – Optional overrides for the feature matrix
  and regression target used during training and inference. Leave blank to fall back to the default
  astrophysical bands and MIPS 24 μm magnitude.【F:train.py†L33-L71】【F:inference.py†L107-L176】【F:data_loader.py†L6-L108】

Override any value on the command line (see below) without editing the file.

## Training workflow

```bash
cd SED_predict
python train.py
```

The training script performs the following steps:

1. Parse the configuration to locate the input CSV and choose a model family.【F:train.py†L20-L56】
2. Build a feature matrix with engineered colour indices and trigonometric terms, dropping rows with
   incomplete data and splitting into train/validation/test partitions.【F:data_loader.py†L1-L82】
3. Train the requested model while collecting per-iteration RMSE history for diagnostics.
4. Evaluate the model on the held-out test set, printing RMSE/MAE metrics and saving the estimator
   (and scaler for the MLP variant) under `outputs/` by default.【F:train.py†L56-L136】
5. Generate evaluation plots (actual vs predicted, feature importance, residual analyses, spatial
   errors, posterior comparisons) and store them alongside the trained model.【F:train.py†L136-L196】
6. Sample posterior distributions that combine model error with disk inclination effects, summarise
   uncertainty by spectral class, and persist both the posterior draws and CSV aggregates for later
   study.【F:train.py†L196-L240】【F:posterior.py†L1-L77】

Adjust hyperparameters or the model type in `inlist` to explore alternative learners.

## Inference workflow

Use `inference.py` to run predictions against new catalogues:

```bash
python inference.py --config inlist --data-file path/to/new_sources.csv
```

The script reloads the trained artefacts, recomputes feature engineering, and writes a CSV of
predictions (optionally including standard deviations when using NGBoost). It will also regenerate a
suite of inference-focused diagnostics—colour–colour diagrams, residual distributions (when truth is
available), posterior comparisons, and sky maps coloured by the predicted band.【F:inference.py†L1-L239】

Key options:

* `--model-type` – Override the model flavour recorded in the config.
* `--model-path` / `--scaler-path` – Manually point to saved artefacts (especially if relocating
  outputs between machines).
* `--output-dir` / `--output-file` – Control where plots and CSV predictions are saved.

## Input data expectations

The loader anticipates columns covering Galactic coordinates, near-IR magnitudes, Spitzer IRAC
channels, a spectral index (`alpha`), and the MIPS24 target magnitude. Missing features are safely
filled with NaNs before engineered quantities are computed, while invalid rows are removed during
training to keep splits clean.【F:data_loader.py†L6-L82】

At inference time, rows are retained unless *all* feature values are missing, and the target column is
optional. The helper also recalculates the spectral index (`alpha`) from Ks and I4 magnitudes whenever
possible so plots remain consistent.【F:data_loader.py†L58-L72】【F:inference.py†L20-L101】

## Outputs

Training and inference runs populate the configured `output_dir` with:

* `*_model.joblib` (and, for the MLP, `*_scaler.joblib`)
* CSVs capturing posterior/uncertainty summaries and inference predictions
* PNG plots for all diagnostics (training: no prefix; inference: `inf_` prefix)
* `posteriors.npy` containing numpy arrays of posterior samples for downstream science analysis

These artefacts enable detailed auditing of predictive quality and uncertainty decomposition.

## Extending the toolkit

The modular structure makes it straightforward to add new regression models or alternative posterior
assumptions:

* Implement a trainer/evaluator pair similar to `xgboost_model.py` and register it in `train.py` and
  `inference.py`.
* Extend `generate_posterior` to incorporate more sophisticated inclination priors or additional
  astrophysical noise sources.
* Add new visualisations by appending helper functions to `plots.py` and invoking them from the
  training or inference scripts.

Contributions and experiment-specific forks are welcome—feel free to adapt the pipeline to your own
SED studies.
