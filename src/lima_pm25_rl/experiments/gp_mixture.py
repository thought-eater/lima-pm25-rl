"""Combinación marginal local de dos GP; evidencia prequential, sin verdad privada."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateConfig:
    window_batches: int = 3
    max_evidence_age_s: float = 3600.0
    locality_radius_m: float = 500.0
    log_score_clip: float = 4.0
    minimum_fast_weight: float = 0.05

    def __post_init__(self):
        if type(self.window_batches) is not int or self.window_batches < 2:
            raise ValueError("La adaptación requiere al menos dos lotes de revisitas")
        for value in (
            self.max_evidence_age_s,
            self.locality_radius_m,
            self.log_score_clip,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "Edad, radio y límite de puntuación deben ser positivos"
                )
        if not 0 < self.minimum_fast_weight < 0.5:
            raise ValueError("El límite de pesos debe estar entre 0 y 0.5")


def log_predictive_density(value, mean, std, noise):
    """Densidad de la LECTURA: sumar ruido instrumental a la varianza latente."""
    value, mean, std, noise = np.broadcast_arrays(value, mean, std, noise)
    if (
        not all(np.isfinite(a).all() for a in (value, mean, std, noise))
        or (std < 0).any()
        or (noise < 0).any()
    ):
        raise ValueError(
            "Predicciones y ruido deben ser finitos y desviaciones no negativas"
        )
    variance = np.maximum(std**2 + noise**2, 1e-12)
    return -0.5 * (np.log(2 * np.pi * variance) + (value - mean) ** 2 / variance)


def mix_marginals(slow_mean, slow_std, fast_mean, fast_std, fast_weight):
    """Momentos de una mezcla, no varianza de dos estimaciones independientes."""
    slow_mean, slow_std, fast_mean, fast_std, weight = np.broadcast_arrays(
        slow_mean, slow_std, fast_mean, fast_std, fast_weight
    )
    if (
        not all(
            np.isfinite(a).all()
            for a in (slow_mean, slow_std, fast_mean, fast_std, weight)
        )
        or (slow_std < 0).any()
        or (fast_std < 0).any()
        or (weight < 0).any()
        or (weight > 1).any()
    ):
        raise ValueError("Momentos o pesos de mezcla inválidos")
    mean = (1 - weight) * slow_mean + weight * fast_mean
    variance = (
        (1 - weight) * slow_std**2
        + weight * fast_std**2
        + weight * (1 - weight) * (slow_mean - fast_mean) ** 2
    )
    return mean, np.sqrt(variance)


class LocalEvidenceGate:
    """Pesos heurísticos por punto, independientes de grilla, fases o regiones reales.

    Un lote aporta una media local de razones log-predictivas, no un producto
    de evidencias supuestamente independientes. Sin tres lotes locales recientes
    se conserva 50/50. Cada misión crea una instancia nueva.
    """

    def __init__(self, config=None, *, history_duration_s=21600):
        self.config = config if config is not None else GateConfig()
        if not np.isfinite(history_duration_s) or history_duration_s <= 0:
            raise ValueError("El historial debe ser positivo y finito")
        self.history_duration_s = history_duration_s
        self.seen = {}
        self.batches = []
        self.last_time = -np.inf

    def weights(self, time_s, positions):
        points = np.asarray(positions, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Se requieren coordenadas métricas finitas")
        if not np.isfinite(time_s) or time_s < self.last_time:
            raise ValueError("No se puede consultar un peso anterior al historial")
        count = np.zeros(len(points), dtype=int)
        total = np.zeros(len(points))
        for stamp, xy, scores in reversed(self.batches):
            if not 0 <= time_s - stamp <= self.config.max_evidence_age_s:
                continue
            distance = np.linalg.norm(points[:, None, :] - xy[None, :, :], axis=2)
            kernel = np.maximum(1 - distance / self.config.locality_radius_m, 0) ** 2
            mass = kernel.sum(axis=1)
            available = (mass > 0) & (count < self.config.window_batches)
            # La atenuación máxima mantiene influencia débil cerca del borde;
            # duplicar sensores en el mismo punto no amplifica su confianza.
            local = (kernel @ scores) / np.maximum(mass, 1e-300) * kernel.max(axis=1)
            total[available] += local[available]
            count[available] += 1
        weight = np.full(len(points), 0.5)
        enough = count == self.config.window_batches
        weight[enough] = 1 / (1 + np.exp(-total[enough] / self.config.window_batches))
        return np.clip(
            weight, self.config.minimum_fast_weight, 1 - self.config.minimum_fast_weight
        ), count

    def observe(self, time_s, records, slow_log_density, fast_log_density):
        """Recibir un lote admitido por core, con pronósticos hechos ANTES del lote."""
        if not np.isfinite(time_s) or time_s <= self.last_time:
            raise ValueError("Los lotes deben llegar una sola vez y en orden temporal")
        if any(r.simulation_time_s != time_s for r in records):
            raise ValueError("Las mediciones deben ser simultáneas")
        if len({(r.sensor_id, r.timestamp_utc) for r in records}) != len(records):
            raise ValueError("Lecturas duplicadas en el lote")
        slow = np.asarray(slow_log_density)
        fast = np.asarray(fast_log_density)
        if (
            slow.shape != (len(records),)
            or fast.shape != slow.shape
            or not np.isfinite([slow, fast]).all()
        ):
            raise ValueError("Puntuaciones predictivas incompatibles con las lecturas")
        self.seen = {
            p: t for p, t in self.seen.items() if time_s - t <= self.history_duration_s
        }
        scores = np.clip(
            fast - slow, -self.config.log_score_clip, self.config.log_score_clip
        )
        eligible = []
        grouped = {}
        for r, score in zip(records, scores, strict=True):
            key = (r.crs, r.x_m, r.y_m)
            revisit = key in self.seen and self.seen[key] < time_s
            eligible.append(revisit)
            if revisit:
                grouped.setdefault((r.x_m, r.y_m), []).append(float(score))
        # Solo después de evaluar TODO el lote se registran sus visitas.
        for r in records:
            self.seen[r.crs, r.x_m, r.y_m] = time_s
        self.batches = [
            b for b in self.batches if time_s - b[0] <= self.config.max_evidence_age_s
        ]
        if grouped:
            self.batches.append(
                (
                    time_s,
                    np.asarray(list(grouped)),
                    np.asarray([np.mean(v) for v in grouped.values()]),
                )
            )
        self.last_time = time_s
        return np.asarray(eligible, dtype=bool), scores
