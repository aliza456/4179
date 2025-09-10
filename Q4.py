# QID: Q6
# ENTRYPOINT: perceptron_epoch

def perceptron_epoch(X, y, w, b, lr):
    """
    Single epoch (one pass) of binary perceptron with labels in {-1, +1}.
    Update when y_i * (w·x + b) <= 0.
    Return [w, b] with w as list[float] and b as float.
    """
    w = [float(v) for v in w]
    b = float(b)

    for xi, yi in zip(X, y):
        activation = sum(wj * xj for wj, xj in zip(w, xi)) + b
        if yi * activation <= 0:
            for j in range(len(w)):
                w[j] += lr * yi * xi[j]
            b += lr * yi
    return [w, b]
