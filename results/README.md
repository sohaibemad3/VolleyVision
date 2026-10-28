# Experimental Results

This directory contains the training and evaluation results for the six baselines implemented in the Volleyball Group Activity Recognition project.

## Directory Structure

```text
results/
├── training/
│   ├── baseline1/
│   ├── baseline4/
│   ├── baseline5/
│   ├── baseline6/
│   ├── baseline7/
│   └── baseline8/
└── test/
    ├── baseline1/
    ├── baseline4/
    ├── baseline5/
    ├── baseline6/
    ├── baseline7/
    └── baseline8/
```

## Training Results

The `training/` directory contains the training artifacts for each baseline, including the available training logs and learning curves.

These artifacts may include:

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy
- Training logs containing metrics recorded across epochs

The learning curves help analyze model convergence, compare training behavior, and identify potential overfitting.

## Test Results

The `test/` directory contains the evaluation artifacts for each baseline on the test set.

These artifacts may include:

- **Evaluation metrics:** Quantitative measurements of model performance, such as accuracy and macro F1-score, saved in `metrics.json` when available.
- **Confusion matrix:** A visualization of the classification performance across the eight group activity classes.

## Baseline Comparison

Each baseline has a separate directory to keep its experimental results organized and make comparisons easier.

For a fair comparison, baselines should use the same test split, class mapping, and evaluation procedure where applicable.

Training and validation results are kept separate from test results to distinguish model learning behavior from final test-set evaluation.

**Note:** All reported metrics should reflect actual experimental results. Missing or unavailable values are not inferred or fabricated.


