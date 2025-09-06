"""
HW1 — Q1 (Core 50 pts): Confusion‑Matrix Metrics (NO metric libraries)

Usage:
    python main_confusion_student.py data/confusion.csv

Data file: confusion.csv (2×2 matrix; rows = ACTUAL, cols = PREDICTED)
    TN,FP
    FN,TP

Library policy:
- Implement ALL metrics from scratch. Do NOT use any metric function from any library
  (e.g., sklearn.metrics.*). Standard library is OK.
- Safe divide: if a denominator is 0 → return 0.0.
- Display rounding: print values to 4 decimals.

Outputs (in this order):
    Precision=..., Recall=..., F1-score=..., Specificity=..., Accuracy=...
"""

import sys, csv
from typing import List

def safe_div(n: float, d: float) -> float:
    """Return n/d, or 0.0 if d == 0 (safe divide)."""
    return n / d if d else 0.0

def TN(cm): return cm[0][0]
def FP(cm): return cm[0][1]
def FN(cm): return cm[1][0]
def TP(cm): return cm[1][1]

# ----- TODO: implement these; tests will import and call them -----
def precision(cm: List[List[int]]) -> float:
    """TP / (TP + FP) with safe divide; cm is [[TN,FP],[FN,TP]]."""
    raise NotImplementedError

def recall(cm: List[List[int]]) -> float:
    """TP / (TP + FN) with safe divide; cm is [[TN,FP],[FN,TP]]."""
    raise NotImplementedError

def f1_score(cm: List[List[int]]) -> float:
    """2PR / (P + R) with safe divide; compute P and R from cm."""
    raise NotImplementedError

def specificity(cm: List[List[int]]) -> float:
    """TN / (TN + FP) with safe divide; cm is [[TN,FP],[FN,TP]]."""
    raise NotImplementedError

def accuracy(cm: List[List[int]]) -> float:
    """(TP + TN) / Total with safe divide; cm is [[TN,FP],[FN,TP]]."""
    raise NotImplementedError
# ---------------------------------------------------------------

def read_confusion(path: str) -> List[List[int]]:
    """Read a 2x2 confusion matrix from CSV as [[TN,FP],[FN,TP]]."""
    with open(path, newline="") as f:
        rows = [[int(x) for x in row] for row in csv.reader(f) if row]
    assert len(rows) == 2 and len(rows[0]) == 2 and len(rows[1]) == 2, "confusion.csv must be 2x2"
    return rows

def main():
    if len(sys.argv) != 2:
        print("Usage: python main_confusion_student.py data/confusion.csv")
        raise SystemExit(1)
    cm = read_confusion(sys.argv[1])

    # Compute metrics (students must implement functions above)
    p = precision(cm)
    r = recall(cm)
    f1 = f1_score(cm)
    spec = specificity(cm)
    acc = accuracy(cm)

    print(f"Precision={p:.4f}, Recall={r:.4f}, F1-score={f1:.4f}, Specificity={spec:.4f}, Accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
