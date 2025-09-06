# References:
# ROC overview: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
# Rank-based AUC / Mann–Whitney: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test

"""HW (Student - BONUS 20 pts): ROC curve & AUC (NO sklearn)

This is a BONUS section worth up to +20 points:
  - AUC (trapezoid): 10
  - AUC (rank / Mann–Whitney): 10

Run:
  python main_roc_student.py scores.csv

====================================
DATA FILES — WHAT THEY MEAN
====================================

1) confusion.csv  (used in Q1 core metrics; format is 2×2 matrix)

    TN,FP
    FN,TP

- Rows = ACTUAL class:
    • Row 1 (top)    - Actual Negatives
    • Row 2 (bottom) - Actual Positives

- Columns = PREDICTED class:
    • Column 1 (left)  - Predicted Negative
    • Column 2 (right) - Predicted Positive

- Example:
    90,10
    5,95
  Meaning:
    TN=90 (true negatives), FP=10 (false positives),
    FN=5 (false negatives), TP=95 (true positives).

You will use this file in the core part of Q1 to compute:
Precision, Recall, F1-score, Specificity, and Accuracy.

------------------------------------

2) scores.csv  (used in this ROC/AUC BONUS file; two columns, NO header)

    score,label

- score (float): larger = more 'positive' and it is labeled 1 when above a chosen threshold.
- label (int): ground-truth class, where 1 = positive, 0 = negative

- Example:
    0.95,1
    0.90,1
    0.85,1
    0.80,0
    0.70,1

Interpretation:
- (0.95,1) means the example has model score 0.95 and true label = 1 (positive).
- For ROC, vary a threshold τ and predict positive if and only if score ≥ τ.
- Construct ROC points (TPR vs FPR) across all distinct thresholds
  (handle ties together), then compute:
    • AUC by trapezoidal rule
    • AUC by rank/Mann–Whitney method
"""


import sys, csv
from typing import List, Tuple, Dict

Pair = Tuple[float, int]

def read_scores(path: str) -> List[Pair]:
    pairs: List[Pair] = []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row: 
                continue
            s = float(row[0]); y = int(row[1])
            assert y in (0,1), "label must be 0 or 1"
            pairs.append((s,y))
    return pairs

# ----- TODO: Implement the following (BONUS) -----
def roc_curve(pairs: List[Pair]) -> List[Dict[str, float]]:
    """Return list of dicts: {'thr','tpr','fpr','tp','fp','fn','tn'}.
    Threshold rule: predict positive if score >= thr.
    Handle ties by processing equal scores together.
    Include (0,0) and (1,1) endpoints.
    """
    raise NotImplementedError

def auc_from_curve(curve: List[Dict[str, float]]) -> float:
    """Trapezoidal AUC over points (FPR,TPR)."""
    raise NotImplementedError

def auc_from_ranks(pairs: List[Pair]) -> float:
    """Rank-based AUC (Mann–Whitney).
    Equivalent to probability a random positive has higher score than a random negative."""
    raise NotImplementedError

def main():
    if len(sys.argv) != 2:
        print("Usage: python main_roc_student.py data/scores.csv")
        raise SystemExit(1)
    pairs = read_scores(sys.argv[1])
    curve = roc_curve(pairs)
    auc_tr = auc_from_curve(curve)
    auc_rk = auc_from_ranks(pairs)
    print("ROC/AUC BONUS (no any programming package sklearn)\n")
    print(f"AUC (trapezoid)        = {auc_tr:.6f}")
    print(f"AUC (rank/Mann-Whitney)= {auc_rk:.6f}")
    print("\nFirst few ROC points (thr, TPR, FPR):")
    for pt in curve[:5]:
        print(f"thr={pt['thr']}, tpr={pt['tpr']:.3f}, fpr={pt['fpr']:.3f}")

if __name__ == "__main__":
    main()
