# cs289a-bank-churn-prediction

CS289A Path 2 experiments for Kaggle Playground S4E1, following `doc/Project Proposal.pdf`.

## Run in Google Colab

1. Upload/open `bank_churn.ipynb` in Colab. CPU works; a free GPU can accelerate the MLP.
2. Put the labeled CSV at `MyDrive/CS189/CS289A_Graduate_Project/data/train.csv`, or change `PROJECT_DIR` to your Drive shortcut. No Kaggle API or credentials are needed.
3. Run all cells. The default `MODE="smoke"` uses 10% of the fixed training partition, 30 forest trees, and at most 3 MLP epochs. Test evaluation is disabled.
4. For the actual study, set `MODE="full"` and enable the desired `RUN_EXPERIMENTS` flags. The full baseline uses all training rows. Hyperparameter candidates and ablations may take considerably longer on a CPU; completed model checkpoints are reused.
5. Review validation outputs. Only then set `RUN_FINAL_TEST=True` and rerun in full mode. The notebook freezes one configuration and optional validation-tuned threshold per model before test evaluation. Test results are cached separately.

The optional residual-network extension is deferred. The required four models, feature-group ablation, data-size ablation, and MLP architecture/dropout experiments are implemented.

## Reproducibility and persistence

- Seed: 42. Stratified row partitions: approximately 70% train, 15% validation, 15% test (integer rounding applies).
- Share the exact CSV and saved `results/split_*.npz` / `.json` files among team members. Content and row order are fingerprinted. Identical package versions also matter.
- All preprocessing fits on each experiment's training rows only. Validation selects models and early-stopping epochs. Test rows never select settings.
- `models/`: configuration-hashed sklearn pipelines or PyTorch checkpoints, preprocessing, metadata, original training time, and best MLP epoch/state.
- `results/`: deduplicated validation CSV, separate test CSV, training-history CSVs, split indices, PNGs, and frozen final-selection metadata.
- Changing data, split, features, preprocessing settings, hyperparameters, device, or recorded package versions produces a different checkpoint identity. Increment `IMPLEMENTATION_VERSION` when changing training implementation logic.
- Load only trusted checkpoints. Avoid concurrent writes to the same Drive experiment. Atomic replacement limits partial writes but cannot provide cross-user locking or guaranteed Drive synchronization.
- Random forest parallel summation and device/library differences can introduce tiny floating-point differences. Timings depend on hardware and runtime load.

## Local checks

`tests/validate_notebook.py` validates notebook JSON/schema and Python syntax, executes the notebook with a substituted local loader, checks checkpoint reuse without fitting, configuration invalidation, result upserts, and the smoke/test guard. Synthetic execution additionally tests frozen final selection and cached test results. All test artifacts use a temporary directory and are removed.

```bash
python tests/validate_notebook.py --synthetic
python tests/validate_notebook.py --data /absolute/path/to/train.csv
```

Requires the notebook's Python packages plus `nbformat`. Local CSV checks never evaluate the held-out test partition. These checks do not verify Google authentication, Drive mount/sync behavior, or CUDA execution. Full searches and ablations must be run in Colab; no experimental conclusions are supplied here.

Keep raw datasets, credentials, and model checkpoints out of version control. Do not include smoke or synthetic measurements in the project report.
