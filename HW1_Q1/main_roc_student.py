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
import os #just used for testing

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
    roc = []
    actual_positives = sum(pair[1] for pair in pairs)
    actual_negatives = len(pairs) - actual_positives
    thresholds = sorted({s for s, _ in pairs}, reverse=True)
    thresholds.append(min(thresholds) - 1e-6)
    for thr in thresholds:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for pair in pairs:
            if pair[0] >= thr and pair[1] == 1:
                tp += 1
            elif pair[0] >= thr and pair[1] == 0:
                fp += 1
            elif pair[0] < thr and pair[1] == 1:
                fn += 1
            elif pair[0] < thr and pair[1] == 0:
                tn += 1
            else:
                raise ValueError('You did something wrong')
        tpr = 0 if actual_positives == 0 else tp/actual_positives
        fpr = 0 if actual_negatives == 0 else fp/actual_negatives
        roc.append({'thr':thr, 'tpr':tpr, 'fpr':fpr, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn})
    return roc

### test
#script_dir = os.path.dirname(os.path.abspath(__file__))
#csv_path = os.path.join(script_dir, 'data', 'scores.csv')
#print(roc_curve(read_scores(csv_path)))

def auc_from_curve(curve: List[Dict[str, float]]) -> float:
    curve_sorted = sorted(curve, key=lambda r: r['fpr'])
    auc = 0.0
    for i in range(1, len(curve_sorted)):
        x1 = curve_sorted[i - 1]['fpr']
        x2 = curve_sorted[i]['fpr']
        y1 = curve_sorted[i - 1]['tpr']
        y2 = curve_sorted[i]['tpr']
        auc += (x2 - x1) * (y1 + y2) / 2
        #print('left fpr, right fpr, bottom tpr, top trp' + str([x1,x2,y1,y2]))
    return round(auc, 6)

def auc_from_ranks(pairs: List[Pair]) -> float:
    pairs_sorted = sorted(pairs)
    scores = [s for s, _ in pairs_sorted]
    labels = [y for _, y in pairs_sorted]
    ranks = []
    i = 0
    while i < len(pairs_sorted):
        score_i = pairs_sorted[i][1]
        j = i
        while j + 1 < len(pairs_sorted) and pairs_sorted[j + 1][0] == score_i:
            j += 1
        avg_rank_for_i_through_j = sum(range(i+1, j+2))/len(range(i+1, j+2))
        for k in range(i, j+1):
            ranks.append(avg_rank_for_i_through_j)
        i = j + 1

    labels_1_ranks_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    p = sum(1 for _, label in pairs if label == 1)
    n = sum(1 for _, label in pairs if label == 0)
    if p == 0 or n == 0:
        raise ValueError("Need both 0s and 1s.")

    auc = (labels_1_ranks_sum - p * (p + 1) / 2) / (p * n)
    return auc

### test
#script_dir = os.path.dirname(os.path.abspath(__file__))
#csv_path = os.path.join(script_dir, 'data', 'scores.csv')
#print('fromcurve: ' + str(auc_from_curve(roc_curve(read_scores(csv_path)))))
#print('fromranks: ' + str(auc_from_ranks(read_scores(csv_path))))

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