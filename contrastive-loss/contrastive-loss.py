import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    y = np.array(y, dtype=float)

    # Compute L2 distance
    distances = np.linalg.norm(a - b, axis=-1)

    # Compute loss per sample
    loss = y * (distances ** 2) + \
           (1 - y) * (np.maximum(0, margin - distances) ** 2)

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        return float(loss)