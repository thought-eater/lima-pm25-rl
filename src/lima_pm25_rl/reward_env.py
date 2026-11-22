"""Recompensa experimental con verdad privada del simulador, solo en episodios sintéticos."""

from lima_pm25.estimation import (
    ReconstructionRequest,
    select_reconstruction_observations,
)
from lima_pm25.simulation import SensorType

from .env import LimaPM25Env

SYNTHETIC_ERROR_REWARD = {
    "mode": "synthetic_error",
    "version": 1,
    "gain": "sum(pre_tick_gp_weights * (squared_error_before - squared_error_after)) / (active_cells * prior_variance)",
    "reference": "same_time_previous_history_plus_new_fixed",
    "truth_access": "simulator_reward_scalar_only",
    "negative_gains": "retained",
}


def error_reward_protocol(protocol):
    return {**protocol, "reward": dict(SYNTHETIC_ERROR_REWARD)}


class SyntheticErrorRewardEnv(LimaPM25Env):
    """Solo cambia la ganancia escalar. Observaciones, GP, info y restricciones coinciden.

    La referencia conserva móviles anteriores y agrega las fijas nuevas. La
    verdad se consulta al final del tick; nunca se entrega al reconstructor,
    a observaciones ni a info. Los pesos proceden del GP anterior a la acción.
    """

    @property
    def reward_protocol(self):
        return dict(SYNTHETIC_ERROR_REWARD)

    def __init__(self, *, episode_factory, **kwargs):
        def synthetic_only(seed):
            episode = episode_factory(seed)
            if episode.historical_fixed is not None:
                raise ValueError(
                    "La recompensa de error sintético no admite replay histórico"
                )
            return episode

        self._tick_uncertainty_gain = 0.0
        super().__init__(episode_factory=synthetic_only, **kwargs)

    def reset(self, *, seed=None, options=None):
        self._tick_uncertainty_gain = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._tick_uncertainty_gain = 0.0
        observation, reward, terminated, truncated, info = super().step(action)
        # El diagnóstico sigue representando varianza, no se renombra el error
        # como incertidumbre ni se publica una segunda señal de verdad en info.
        info["uncertainty_gain"] = self._tick_uncertainty_gain
        return observation, reward, terminated, truncated, info

    def _assimilate(self, batch, *, reward_weights=None):
        if batch is None:
            return 0.0
        gp = self.estimator_config.reconstruction_config
        eligible, _ = select_reconstruction_observations(
            ReconstructionRequest(
                grid=self.domain.grid,
                reconstruction_timestamp_utc=batch.timestamp_utc,
                measurements=batch.records,
            ),
            gp,
        )
        mobile = any(r.sensor_type is SensorType.MOBILE for r in eligible)
        reference = None
        if mobile:
            assert self._simulation is not None
            if batch.simulation_time_s != self._simulation.state.simulation_time_s:
                raise ValueError(
                    "La recompensa móvil necesita el campo del mismo instante"
                )
            fixed = tuple(r for r in batch.records if r.sensor_type is SensorType.FIXED)
            reference = self._predict(batch.timestamp_utc, (*self._records, *fixed))
        before_snapshot = self._updater.current_snapshot
        gain = super()._assimilate(batch, reward_weights=reward_weights)
        self._tick_uncertainty_gain += gain
        after = self._updater.current_snapshot
        if not mobile or after is before_snapshot:
            return 0.0
        assert reference is not None and after is not None
        mask = self.domain.grid.active_mask
        truth = self._simulation.state.concentration_ug_m3[mask]
        improvement = (reference.mean_ug_m3[mask] - truth) ** 2 - (
            after.mean_ug_m3[mask] - truth
        ) ** 2
        if reward_weights is not None:
            improvement = reward_weights[mask] * improvement
        return float(improvement.sum() / (mask.sum() * gp.prior_stddev_ug_m3**2))


def error_environment(env):
    """Otra instancia sobre el mismo dominio/fábrica, con simulación e historial propios."""
    return SyntheticErrorRewardEnv(
        domain=env.domain,
        episode_factory=env._episode_factory,
        estimator_config=env.estimator_config,
        initial_nodes=env.initial_nodes,
        mobile_noise_stddev_ug_m3=env.mobile_noise_stddev_ug_m3,
        invalid_action_penalty=env.invalid_action_penalty,
        concentration_priority=env.concentration_priority,
        distance_cost_per_km=env.distance_cost_per_km,
        arrival_strategy=env.arrival_strategy,
        render_mode=env.render_mode,
    )
