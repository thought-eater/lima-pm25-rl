"""Base de Gymnasium que reutiliza el dominio y la física de lima-pm25-core."""

from typing import Any, NoReturn

import gymnasium as gym
from lima_pm25.domain import AmbientBaseline, SpatialDomain
from lima_pm25.simulation import PM25PhysicsConfig, PM25State, PM25Transport


class LimaPM25Env(gym.Env[object, object]):
    """Entorno preliminar, sin observaciones, acciones ni avance interactivo.

    Recibe un dominio ya preparado y parámetros físicos explícitos. No descarga
    cartografía, carga archivos ni ejecuta un episodio completo. El dominio se
    conserva como información estática de solo lectura; el motor y el campo
    pertenecen a cada instancia. El campo verdadero permanece en ``_state`` y
    no constituye una observación del agente.
    """

    # Shared capabilities; Gymnasium's instance annotation rules out ClassVar.
    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        *,
        domain: SpatialDomain,
        physics: PM25PhysicsConfig,
        initial_baseline: AmbientBaseline,
    ) -> None:
        super().__init__()
        if not isinstance(domain, SpatialDomain):
            raise TypeError("domain debe ser un SpatialDomain de lima-pm25-core")
        if not isinstance(physics, PM25PhysicsConfig):
            raise TypeError("physics debe ser un PM25PhysicsConfig de lima-pm25-core")
        if not isinstance(initial_baseline, AmbientBaseline):
            raise TypeError(
                "initial_baseline debe ser un AmbientBaseline de lima-pm25-core"
            )

        self.domain = domain
        self.physics = physics
        self.initial_baseline = initial_baseline
        self._transport = PM25Transport(domain.grid, physics)
        self._state: PM25State
        self._initialize_simulation()

    def _initialize_simulation(self, *, seed: int | None = None) -> None:
        """Reinicia solo la física y el generador de Gymnasium para el futuro reset.

        No produce observaciones ni avanza el transporte. Una semilla explícita
        reinicia el generador; ``None`` conserva uno previamente inicializado.
        """
        super().reset(seed=seed)
        self._state = self._transport.initialize_uniform(
            self.initial_baseline,
            simulation_time_s=0.0,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> NoReturn:
        """Pendiente de la observación inicial; no modifica el estado interno."""
        raise NotImplementedError(
            "reset() está pendiente de definir la observación inicial; "
            "el campo verdadero del simulador no se devuelve al agente."
        )

    def step(self, action: object) -> NoReturn:
        """Pendiente de acciones, avance temporal, recompensa y fin del episodio."""
        raise NotImplementedError(
            "step() está pendiente de definir las acciones, el avance temporal, "
            "la recompensa y las condiciones de término."
        )
