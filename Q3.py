# QID: Q3
# ENTRYPOINT: bow_transform

def bow_transform(corpus, vocab):
    """
    Tokenize by whitespace (input already lowercase), count vocab term
    occurrences per document. Ignore tokens not in vocab.
    Return matrix of shape [len(corpus)][len(vocab)].
    """
    vindex = {term: i for i, term in enumerate(vocab)}
    out = [[0] * len(vocab) for _ in corpus]
    for di, doc in enumerate(corpus):
        for tok in doc.split():
            if tok in vindex:
                out[di][vindex[tok]] += 1
    return out
