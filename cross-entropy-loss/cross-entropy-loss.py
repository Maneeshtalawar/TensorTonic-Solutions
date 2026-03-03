import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=float)

    N = y_true.shape[0]

    # Select probabilities of correct classes
    correct_probs = y_pred[np.arange(N), y_true]

    # Compute average negative log likelihood
    loss = -np.mean(np.log(correct_probs))

    return float(loss)