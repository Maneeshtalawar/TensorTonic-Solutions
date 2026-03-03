import numpy as np

def dice_loss(p, y, eps=1e-8):
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # Flatten arrays (important for segmentation masks)
    p = p.flatten()
    y = y.flatten()

    # Compute intersection
    intersection = np.sum(p * y)

    # Compute Dice coefficient
    dice = (2.0 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)

    # Dice loss
    return float(1.0 - dice)