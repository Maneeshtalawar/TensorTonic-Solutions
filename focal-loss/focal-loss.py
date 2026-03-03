import numpy as np

def focal_loss(p, y, gamma=2.0):
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # Compute focal loss
    loss = - (
        y * ((1 - p) ** gamma) * np.log(p) +
        (1 - y) * (p ** gamma) * np.log(1 - p)
    )

    return float(np.mean(loss))