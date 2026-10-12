"""Compact reference implementation of the vertical seam operation."""

import numpy as np
from scipy import ndimage


def energy(image, lung_mask=None, weight=0.0):
    """Normalized Sobel magnitude plus a finite lung-mask penalty."""
    x = np.asarray(image, dtype=np.float32)
    gx = ndimage.sobel(x, axis=1, mode="reflect")
    gy = ndimage.sobel(x, axis=0, mode="reflect")
    gradient = np.sqrt(gx**2 + gy**2)
    maximum = gradient.max()
    base = (np.zeros_like(gradient, dtype=np.float32)
            if maximum <= 1e-8 else gradient / (maximum + 1e-8))
    if lung_mask is not None:
        if lung_mask.shape != x.shape:
            raise ValueError("Image and lung mask must have the same shape")
        base = base + np.asarray(lung_mask, dtype=np.float32) * float(weight)
    return np.asarray(base, dtype=np.float32)


def minimum_seam(pixel_energy):
    """Return a minimum-energy vertical path; ties prefer left, up, right."""
    dp = np.asarray(pixel_energy, dtype=np.float64).copy()
    if dp.ndim != 2 or not all(dp.shape):
        raise ValueError("Energy must be a nonempty two-dimensional array")
    height, width = dp.shape
    for row in range(1, height):
        previous = dp[row - 1]
        left = np.r_[np.inf, previous[:-1]]
        right = np.r_[previous[1:], np.inf]
        dp[row] += np.minimum(np.minimum(left, previous), right)

    seam = np.empty(height, dtype=np.int64)
    seam[-1] = np.argmin(dp[-1])  # leftmost final-row minimum
    for row in range(height - 2, -1, -1):
        column = seam[row + 1]
        candidates = [dp[row, column - 1] if column else np.inf,
                      dp[row, column],
                      dp[row, column + 1] if column + 1 < width else np.inf]
        seam[row] = np.clip(column + np.argmin(candidates) - 1, 0, width - 1)
    return seam


def remove_seam(array, seam):
    """Remove one selected pixel from every row of a two-dimensional array."""
    height, width = array.shape
    keep = np.ones((height, width), dtype=bool)
    keep[np.arange(height), seam] = False
    return array[keep].reshape(height, width - 1)


def retarget(image, lung_mask, seams=200, weight=10.0):
    """Return the narrower image and deleted original columns (seam, row)."""
    current = np.asarray(image).copy()
    mask = np.asarray(lung_mask, dtype=bool).copy()
    if current.ndim != 2 or mask.shape != current.shape:
        raise ValueError("Image and mask must be matching 2-D arrays")
    height, width = current.shape
    if not 0 <= seams < width:
        raise ValueError("Number of seams must be between 0 and width - 1")
    original_columns = np.tile(np.arange(width), (height, 1))
    paths = np.empty((seams, height), dtype=np.int64)
    for step in range(seams):
        seam = minimum_seam(energy(current, mask, weight))
        paths[step] = original_columns[np.arange(height), seam]
        current = remove_seam(current, seam)
        mask = remove_seam(mask, seam)
        original_columns = remove_seam(original_columns, seam)
    return current, paths
