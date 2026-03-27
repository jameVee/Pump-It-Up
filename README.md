# Pump It Up: Data Mining the Water Table

This repository contains a machine learning workflow for the **Pump It Up** classification problem: predicting the operating condition of water pumps in Tanzania from tabular metadata.

## Project Contents

- `code.ipynb` — main notebook with data loading, EDA, feature engineering, model training, and tuning.
- `training_set_values.csv` — training features.
- `training_set_labels.csv` — target labels for the training set.
- `test_set_values.csv` — test features for inference/submission.
- `Presentation Slide.pdf` — project presentation material.

## Problem Statement

The goal is to classify each water point into one of three categories:

- `functional`
- `functional needs repair`
- `non functional`

## Workflow Summary

The notebook follows this high-level process:

1. **Load and merge data** from training values and labels.
2. **Investigate and visualize** class balance and feature behavior.
3. **Preprocess features**, including:
   - dropping low-value or high-noise columns,
   - handling missing values,
   - categorical wrangling for high-cardinality fields (for example, funder/installer grouping),
   - creating derived features.
4. **Train baseline models** including Logistic Regression, KNN, Decision Tree, Random Forest, and XGBoost.
5. **Tune hyperparameters** with `RandomizedSearchCV` and stratified cross-validation.
6. **Prepare test-set transformations** aligned with training preprocessing.

## Requirements

Use Python 3.9+ and install the following common dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## How to Run

1. Ensure the CSV files remain in the repository root (same directory as `code.ipynb`).
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `code.ipynb` and run cells from top to bottom.

## Notes

- Keep preprocessing consistent between train and test data.
- If you generate prediction files, save them with clear names (for example, including model name and date).
- For reproducibility, set and keep explicit random seeds in training and CV steps.

---
