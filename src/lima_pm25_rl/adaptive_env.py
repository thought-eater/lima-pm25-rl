"""Adaptación de R1 de Barrionuevo a una flota central y campo dinámico."""

from dataclasses import asdict, dataclass

import numpy as np
from lima_pm25.estimation import AdaptiveMapUpdater, GPFitConfig
from lima_pm25.simulation import MeasurementBatch, SensorType

from .env import LimaPM25Env


@dataclass(frozen=True, slots=True)
class ModelChangeReward:
    mean_weight: float = 1.0
    uncertainty_weight: float = 0.5
    influence_radius_m: float = 2000.0

    def __post_init__(self):
        if any(
            not np.isfinite(v) or v < 0
            for v in (self.mean_weight, self.uncertainty_weight)
        ):
            raise ValueError("Los pesos de cambio deben ser finitos y no negativos")
        if self.mean_weight + self.uncertainty_weight <= 0:
            raise ValueError("Se requiere al menos una contribución positiva")
        if not np.isfinite(self.influence_radius_m) or self.influence_radius_m <= 0:
            raise ValueError("El radio de influencia debe ser positivo")

    def protocol(self):
        return {
            "mode": "model_change",
            "version": 1,
            **asdict(self),
            "reference": "same_time_previous_history_plus_new_fixed; both_fits_start_from_same_previous_parameters",
            "normalization": "active_cell_count_times_original_prior_stddev_for_both_terms",
            "support": "union_of_influence_disks_at_new_mobile_measurement_coordinates",
            "uncertainty": "absolute_change_of_standard_deviation_per_cell",
        }


def model_change_terms(before, after, grid, positions, settings, scale):
    """Absolutos por celda; unión de áreas sin multiplicar solapamientos."""
    mask = grid.active_mask
    x, y = np.meshgrid(grid.x_centers_m, grid.y_centers_m)
    influence = np.zeros(grid.shape, dtype=bool)
    for px, py in positions:
        influence |= (x - px) ** 2 + (y - py) ** 2 <= settings.influence_radius_m**2
    influence &= mask
    denominator = int(mask.sum()) * scale
    return (
        float(
            np.abs(after.mean_ug_m3[influence] - before.mean_ug_m3[influence]).sum()
            / denominator
        ),
        float(
            np.abs(
                after.standard_deviation_ug_m3[influence]
                - before.standard_deviation_ug_m3[influence]
            ).sum()
            / denominator
        ),
    )


class AdaptiveGPEnv(LimaPM25Env):
    """GP reajustado por lote; sin acceso a la verdad para aprender o recompensar."""

    def __init__(
        self, *, gp_fit: GPFitConfig, model_change: ModelChangeReward, **kwargs
    ):
        if kwargs.get("concentration_priority", 0) != 0:
            raise ValueError(
                "R1 usa cambio del modelo; concentration_priority debe ser cero"
            )
        self.gp_fit = gp_fit
        self.model_change = model_change
        super().__init__(**kwargs)

    @property
    def reward_protocol(self):
        return self.model_change.protocol()

    def _reset_estimator(self):
        config = self.estimator_config
        self._adaptive = AdaptiveMapUpdater(
            self.domain.grid,
            config.reconstruction_config,
            config.history_duration,
            self.gp_fit,
        )
        self._fixed_adaptive = AdaptiveMapUpdater(
            self.domain.grid,
            config.reconstruction_config,
            config.history_duration,
            self.gp_fit,
        )
        self._tick_gain = self._mean_change = self._std_change = 0.0
        self._fit_iterations = self._fit_attempts = self._fit_nonconverged = 0
        self._fit_seconds = 0.0
        self.gp_trace = []

    def _assimilate(self, batch, *, reward_weights=None):
        if batch is None:
            return 0.0
        before, after = self._adaptive.update(batch)
        self._records.extend(batch.records)
        fixed = tuple(r for r in batch.records if r.sensor_type is SensorType.FIXED)
        fixed_after = None
        if fixed:
            _, fixed_after = self._fixed_adaptive.update(
                MeasurementBatch(batch.simulation_time_s, batch.timestamp_utc, fixed),
                reference=False,
            )
        for branch, result in (
            ("counterfactual", before),
            ("factual", after),
            ("fixed_only", fixed_after),
        ):
            if result is not None:
                self._fit_iterations += result.fit.iterations
                self._fit_attempts += int(result.fit.attempted)
                self._fit_nonconverged += int(
                    result.fit.attempted and not result.fit.converged
                )
                self._fit_seconds += result.fit.seconds
                self.gp_trace.append(
                    {
                        "elapsed_s": batch.simulation_time_s,
                        "branch": branch,
                        **asdict(result.parameters),
                        **asdict(result.fit),
                    }
                )
        if before is None or after is None:
            return 0.0
        eligible_keys = set(after.posterior.observations.used_measurement_keys)
        positions = [
            (r.x_m, r.y_m)
            for r in batch.records
            if r.sensor_type is SensorType.MOBILE
            and (r.sensor_id, r.timestamp_utc) in eligible_keys
        ]
        base = self.estimator_config.reconstruction_config
        mean, std = model_change_terms(
            before.posterior,
            after.posterior,
            self.domain.grid,
            positions,
            self.model_change,
            base.prior_stddev_ug_m3,
        )
        self._mean_change += mean
        self._std_change += std
        mask = self.domain.grid.active_mask
        self._tick_gain += float(
            np.mean(
                before.posterior.standard_deviation_ug_m3[mask] ** 2
                - after.posterior.standard_deviation_ug_m3[mask] ** 2
            )
            / base.prior_stddev_ug_m3**2
        )
        return (
            self.model_change.mean_weight * mean
            + self.model_change.uncertainty_weight * std
        )

    def step(self, action):
        self._tick_gain = self._mean_change = self._std_change = 0.0
        self._fit_iterations = self._fit_attempts = self._fit_nonconverged = 0
        self._fit_seconds = 0.0
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(
            uncertainty_gain=self._tick_gain,
            model_mean_change=self._mean_change,
            model_uncertainty_change=self._std_change,
            gp_fit_iterations=self._fit_iterations,
            gp_fit_attempts=self._fit_attempts,
            gp_fit_nonconverged=self._fit_nonconverged,
            gp_fit_seconds=self._fit_seconds,
            gp_parameters=asdict(self._adaptive.parameters),
        )
        return observation, reward, terminated, truncated, info

    def _predict(self, timestamp=None, records=None):
        if records is not None:
            raise ValueError(
                "Un subhistorial adaptativo necesita un estimador independiente"
            )
        timestamp = timestamp or self._simulation.clock.utc_at(
            self._simulation.state.simulation_time_s
        )
        return self._adaptive.predict(timestamp)

    def predict_fixed(self):
        now = self._simulation.clock.utc_at(self._simulation.state.simulation_time_s)
        return self._fixed_adaptive.predict(now)

    def _observation(self):
        result = super()._observation()
        mask = self.domain.grid.active_mask
        std = self._posterior.standard_deviation_ug_m3[mask]
        scale = self.estimator_config.reconstruction_config.prior_stddev_ug_m3
        result["maps"][3, mask] = std / (std + scale)
        return result
