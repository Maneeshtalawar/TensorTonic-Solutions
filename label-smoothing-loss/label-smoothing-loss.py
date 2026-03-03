import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    predictions = np.array(predictions, dtype=float)

    K = len(predictions)

    # Create smoothed target distribution
    smooth_target = np.full(K, epsilon / K)
    smooth_target[target] += (1 - epsilon)

    # Compute cross-entropy
    loss = -np.sum(smooth_target * np.log(predictions))

    return float(loss)