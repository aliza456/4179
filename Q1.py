# QID: Q7
# ENTRYPOINT: standardize_columns

import math

def standardize_columns(X):
    """
    Column-wise z-score (population std, ddof=0).
    If a column's std == 0, that entire column becomes zeros.
    Each value rounded to 4 decimals.
    """
    if not X:
        return []
    n, m = len(X), len(X[0])
    # Transpose
    cols = list(zip(*X))
    means = []
    stds = []
    for j in range(m):
        col = cols[j]
        mu = sum(col) / n
        means.append(mu)
        var = sum((x - mu) ** 2 for x in col) / n  # population variance
        stds.append(math.sqrt(var))

    out = []
    for i in range(n):
        row = []
        for j in range(m):
            s = stds[j]
            if s == 0:
                row.append(0.0)
            else:
                z = (X[i][j] - means[j]) / s
                # round to 4 decimals
                row.append(round(z, 4))
        out.append(row)
    return out
