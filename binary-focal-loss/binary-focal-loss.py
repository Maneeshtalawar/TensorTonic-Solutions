import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    predictions = np.array(predictions, dtype=float)
    targets = np.array(targets, dtype=float)

    # Compute p_t
    p_t = targets * predictions + (1 - targets) * (1 - predictions)

    # Compute focal loss
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)

    return float(np.mean(loss))