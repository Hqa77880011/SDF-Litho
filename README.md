# SDF-Litho

Implementation of **SDF-Litho: Continuous Signed-Distance Surrogate Learning for Layout-to-Wafer Contour Prediction under Fixed Process Conditions**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate layouts and fixed-process contours using Gaussian filtering, thresholding and binary morphology; SDF distances are in pixels.

```bash
sdf-litho demo --output data/generated.npz --count 96 --config config.json --seed 13
sdf-litho prepare --input data/generated.npz --output data/prepared
```

`x[N,1,H,W]` is a binary layout, `y` the target wafer mask, `sdf` the signed target distance, and `corner` a corner-region weight. Obtain wafer masks from a fixed characterized simulator or registered SEM contours. The demo constructs its own fixed-process contours and derives all geometry targets.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
sdf-litho train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
sdf-litho evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_edge`, `no_corner`, `no_weight`. Reuse the prepared data and training seed:

```bash
sdf-litho train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
sdf-litho evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
sdf-litho plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
