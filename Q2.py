# QID: Q8
# ENTRYPOINT: top_k_cosine

import math

def top_k_cosine(query, docs, k):
    """
    Return indices of top-k docs by cosine similarity to query.
    - Cosine = (q·d)/(||q||*||d||)
    - If any norm is 0, similarity treated as 0 (no crash).
    - Break ties by smaller index.
    """
    qnorm = math.sqrt(sum(q*q for q in query))
    sims = []
    for idx, d in enumerate(docs):
        dnorm = math.sqrt(sum(x*x for x in d))
        if qnorm == 0 or dnorm == 0:
            sim = 0.0
        else:
            dot = sum(qi*di for qi, di in zip(query, d))
            sim = dot / (qnorm * dnorm)
        sims.append((sim, idx))
    # Sort by similarity desc, then index asc
    sims.sort(key=lambda t: (-t[0], t[1]))
    return [idx for _, idx in sims[:k]]
