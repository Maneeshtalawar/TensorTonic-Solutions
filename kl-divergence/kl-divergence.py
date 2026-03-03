import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # Add epsilon for numerical stability
    p = p + eps
    q = q + eps

    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))

    return float(kl)