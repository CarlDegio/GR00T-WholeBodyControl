"""Label-independent descriptors and distances for T1's fixed image cohort."""
import numpy as np


def unit(x):
    x = np.asarray(x, np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-12) or not np.isfinite(x).all():
        raise ValueError('Nonfinite or zero feature vector')
    return x / norm


def descriptors(tokens):
    """640x480 images become 224x168 plus 28px padding: exclude 2 patch rows/side."""
    x = np.asarray(tokens, np.float32).reshape(-1, 16, 16, tokens.shape[-1])[:, 2:14]
    cells = x.reshape(len(x), 3, 4, 4, 4, x.shape[-1]).mean(axis=(2, 4))
    spatial = unit(unit(cells).reshape(len(x), -1))
    return {'spatial': spatial.astype(np.float32),
            'global': unit(x.mean(axis=(1, 2))).astype(np.float32)}


def neighbor_distances(query, reference, k=3, exclude=None):
    d = np.clip(1 - unit(query) @ unit(reference).T, 0, 2)
    if exclude is not None:
        d[np.arange(len(d)), np.asarray(exclude)] = np.inf
    order = np.argsort(d, axis=1, kind='stable')[:, :k]
    nearest = np.take_along_axis(d, order, axis=1)
    if not np.isfinite(nearest).all():
        raise ValueError('Not enough independent reference episodes')
    return nearest.mean(axis=1), order


def outcome_fraction(success, numerator, denominator):
    # Existing reporting convention; raw annotations are kept separately.
    if success is True:
        return 1.0
    if numerator is None or denominator is None:
        return None
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError('Invalid explicit Progress fraction')
    return numerator / denominator
