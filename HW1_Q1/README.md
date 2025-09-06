# HW1 — Q1: Classification Metrics (Binary Case)

Implement classic binary classification metrics from scratch.

- Core (50 pts): compute metrics from a 2×2 confusion matrix — Precision, Recall, F1, Specificity, Accuracy.
- BONUS (20 pts): build an ROC curve from scores and compute AUC two ways (Trapezoid & Rank/Mann–Whitney).
- No metric libraries: Do not call any function that directly computes metrics (e.g., `sklearn.metrics.*`, `roc_curve`, `auc`, `roc_auc_score`, etc.).
- Allowed tools: Python standard library; for numeric integration you may use NumPy (including `numpy.trapz`).

Rounding & Safety
- Round base metrics to 4 decimals; round AUC to 6 decimals.
- Use safe-divide: if a denominator is zero, return 0.0.

Robustness
- Your submission will be executed on multiple inputs. Runtime errors may lead to a grade reduction for Q1.


---

## Repository Layout (student package)

```
data/
  confusion.csv
  scores.csv

tests/
  test_confusion.py
  test_roc.py

main_confusion_student.py
main_roc_student.py
pytest.ini
requirements.txt
README.md  (this file)
```

---

## Functions to Implement

In `main_confusion_student.py` (Core)
- `precision(cm)`
- `recall(cm)`
- `f1_score(cm)`
- `specificity(cm)`
- `accuracy(cm)`

> Each function accepts the full 2×2 confusion matrix: `cm = [[TN, FP], [FN, TP]]`.

In `main_roc_student.py` (BONUS)
- `roc_curve(pairs)` — sweep thresholds over unique scores (descending), group ties together, include (0,0) and (1,1); return TPR/FPR sequence.
- `auc_from_curve(curve)` — trapezoidal AUC over (FPR, TPR).
- `auc_from_ranks(pairs)` — rank/Mann–Whitney AUC (probability a random positive > random negative; handle ties with average ranks).


---

## Data Files (What They Mean)

### `data/confusion.csv`
A 2×2 confusion matrix with rows = ACTUAL and columns = PREDICTED:
```
TN,FP
FN,TP
```
Example provided in this assignment:
``` 
90,10
5,95
```
⇒ TN=90, FP=10, FN=5, TP=95.

### `data/scores.csv`
Two columns, no header:
```
score,label
```
- `score` (float): larger = more 'positive' and it is labeled 1 when above a chosen threshold.
- `label` (int): ground-truth class in `{0,1}`


---

## How to Run from CLI

Core metrics
```bash
python main_confusion_student.py data/confusion.csv
```

BONUS ROC/AUC
```bash
python main_roc_student.py data/scores.csv
```


---

## Running Tests

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

If a function is not implemented yet, tests may catch `NotImplementedError` and skip/return early until you complete it.


---

## Policy (read carefully)

- BONUS scope: ROC/AUC is BONUS +20 on top of the base 50 points.
- No metric libraries: Do not call prebuilt metric functions in any language (e.g., `sklearn.metrics.*`, `roc_curve`, `auc`, `roc_auc_score`, R `pROC::roc`, MATLAB `perfcurve`, etc.).
- Allowed helpers: Python stdlib; NumPy for arrays & `numpy.trapz`; Pandas for I/O.
- Robustness: Code must not crash on valid inputs; handle edge cases and ties. Runtime errors may lead to a grade reduction for Q1.
- Reproducibility: Do not change function names, file paths, or I/O contract.
