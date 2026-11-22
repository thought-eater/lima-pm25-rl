"""Cobertura, anchura y CRPS por instante para el campo latente sintético."""

from statistics import NormalDist

import numpy as np
from scipy.special import ndtr

Z95 = NormalDist().inv_cdf(0.975)


def scores(truth, mean, std):
    """Medias espaciales por instante; el CRPS de una masa puntual es |error|."""
    if (
        truth.shape != mean.shape
        or truth.shape != std.shape
        or not all(np.isfinite(x).all() for x in (truth, mean, std))
        or (std < 0).any()
    ):
        raise ValueError("Campo o posterior no válido")
    error = np.abs(truth - mean)
    z = np.divide(error, std, out=np.zeros_like(error), where=std > 0)
    crps = np.where(
        std > 0,
        error * (2 * ndtr(z) - 1)
        + std / np.sqrt(np.pi) * (np.sqrt(2) * np.exp(-0.5 * z * z) - 1),
        error,
    )
    return {
        "coverage95_pct": 100 * (error <= Z95 * std).mean(axis=1),
        "width95_ug_m3": 2 * Z95 * std.mean(axis=1),
        "crps_ug_m3": crps.mean(axis=1),
    }
