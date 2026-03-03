import numpy as np

def hinge_loss(y_true, x, margin=1.0, reduction="mean"):
    y_true = np.array(y_true, dtype=float)
    x = np.array(x, dtype=float)

    # Compute hinge loss per sample
    losses = np.maximum(0, margin - y_true * x)

    if reduction == "mean":
        return float(np.mean(losses))
    elif reduction == "sum":
        return float(np.sum(losses))
    else:
        return losses