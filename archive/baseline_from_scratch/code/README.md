# Paper Pipelines

This directory contains the standardized pipelines for turning the current
project progress into conference-paper evidence.

## 1. Dataset audit

Run:

```bash
conda activate myenv
python main/dataset_audit_pipeline.py
```

Outputs:

- `reports/paper_metrics/dataset/dataset_summary.json`
- `reports/paper_metrics/dataset/class_distribution.csv`
- `reports/paper_metrics/dataset/objects_long.csv`
- `reports/paper_metrics/dataset/dimension_distribution.csv`
- `reports/paper_metrics/dataset/split_300.csv`
- `reports/paper_metrics/dataset/split_1000.csv`
- `reports/paper_metrics/dataset/split_3000.csv`
- `reports/paper_metrics/dataset/paper_dataset_summary.md`

Use these artifacts for the dataset section of the paper: data integrity,
class imbalance, object counts, angle ranges, image dimensions, and clip-level
splitting.

## 2. Low-resource detection experiment

Run a quick smoke test:

```bash
conda activate myenv
python main/pytorch_implementation/edge_detection_experiment.py --dataset-sizes 1000 --epochs 1 --max-train-batches 1 --benchmark-batches 1
```

Run the paper experiments:

```bash
conda activate myenv
python main/pytorch_implementation/edge_detection_experiment.py --dataset-sizes 1000 1500 2000 2500 3000 --epochs 8
```

Outputs:

- `reports/paper_metrics/experiments/experiment_summary.csv`
- `reports/paper_metrics/experiments/paper_experiment_summary.md`
- `reports/paper_metrics/experiments/result_subset_*.json`
- `reports/paper_metrics/experiments/history_subset_*.json`
- `reports/paper_metrics/experiments/model_subset_*.pt`

The experiment summary contains the main paper table:

- dataset size and train/validation/test image counts
- CPU training time
- CPU latency and FPS
- peak RAM
- model size and parameter count
- held-out test precision, recall, F1, mean matched IoU, and Macro AP50 with oriented IoU
- learning curves, per-class AP charts, and a cross-dataset metric chart

## Recommended paper framing

The current contribution is strongest as a dataset-and-baseline applied ML
paper:

1. Real urban-intersection vehicle data from Peru.
2. Oriented bounding-box annotation analysis.
3. Lightweight CNN baseline for 9-class vehicle detection.
4. Accuracy/compute trade-off across 1,000, 1,500, 2,000, 2,500, and 3,000 images.
5. CPU evidence that motivates future Raspberry Pi deployment.

Each experiment uses its matching `data/dataset_<size>/` directory and keeps
entire clips in exactly one split. This prevents adjacent video frames leaking
between training, validation, and held-out testing.
