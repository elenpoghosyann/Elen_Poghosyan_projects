# Assignment 3 — Decision Trees & Ensemble Methods
## YSU CS2020 Machine Learning

### Dataset
UCI Credit Default dataset (30,000 samples, 23 features, binary classification)
Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

### Files
| File | Description |
|------|-------------|
| `decision_tree.py` | CART Decision Tree from scratch |
| `random_forest.py` | Random Forest from scratch (uses decision_tree.py) |
| `experiments.py`   | All 3 experiments + figures |
| `requirements.txt` | Python dependencies |

### How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments (auto-downloads dataset)
python experiments.py
```

Figures are saved to `../figures/`:
- `model_comparison.png`
- `hyperparameter_tuning.png`
- `feature_importance.png`
- `learning_curves.png`

### Sanity checks
```bash
python decision_tree.py   # Should print Iris accuracy > 0.95
python random_forest.py   # Should print Iris accuracy > 0.95
```
