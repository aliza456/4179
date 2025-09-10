# QID: Q1
# ENTRYPOINT: knn_predict

import math
from collections import Counter

def knn_predict(train_X, train_y, test_X, k):
    """
    k-NN with Euclidean distance.
    - For each test sample, compute distances to all training samples.
    - Choose the k nearest after sorting neighbors by (distance, label).
    - Predict by majority vote over those k labels; break ties by smallest label.
    """
    preds = []
    for x in test_X:
        dists = []
        for (tx, ty) in zip(train_X, train_y):
            d = math.sqrt(sum((xi - ti) ** 2 for xi, ti in zip(x, tx)))
            dists.append((d, ty))
        # sort by (distance asc, label asc)
        dists.sort(key=lambda t: (t[0], t[1]))
        k_labels = [lab for _, lab in dists[:k]]
        counts = Counter(k_labels)
        # choose label with max count; tie -> smallest label
        best_count = max(counts.values())
        candidates = [lab for lab, c in counts.items() if c == best_count]
        preds.append(min(candidates))
    return preds
