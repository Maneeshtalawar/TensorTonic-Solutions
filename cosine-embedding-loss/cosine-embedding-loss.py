import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    # Compute cosine similarity
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)

    cos_sim = dot / (norm1 * norm2)

    # Compute loss
    if label == 1:
        loss = 1 - cos_sim
    else:  # label == -1
        loss = max(0, cos_sim - margin)

    return float(loss)