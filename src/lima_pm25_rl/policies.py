"""Referencias de muestreo con la misma coordinación y restricciones del agente."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .env import LimaPM25Env, Observation
from .simulation import VEHICLE_COUNT


@dataclass(frozen=True)
class PlanningDecision:
    """Contexto conocido antes de elegir; no contiene mediciones futuras."""

    vehicle: int
    action: int
    mask: np.ndarray
    reserved_cells: tuple[tuple[int, int], ...]


def vehicle_input(
    observation: Observation,
    vehicle: int,
    reserved_cells: Iterable[tuple[int, int]] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Vista condicionada al vehículo, sin alterar posiciones ni anticipar lecturas."""
    maps = observation["maps"]
    reserved = np.zeros_like(maps[0])
    for cell in reserved_cells:
        reserved[cell] = 1
    others = sum(maps[4 + index] for index in range(VEHICLE_COUNT) if index != vehicle)
    view = np.concatenate(
        (maps[:4], maps[4 + vehicle][None], others[None], reserved[None])
    )
    auxiliary = np.concatenate(
        (observation["remaining"], observation["vehicles"].ravel())
    )
    return view.copy(), auxiliary.copy()


def sequence_input(
    observations: Sequence[Observation],
    vehicle: int,
    reserved_cells: Iterable[tuple[int, int]],
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ventana causal; al inicio repite el primer mapa sin producir mediciones."""
    if history_length < 1 or not observations:
        raise ValueError("La ventana necesita observaciones y una longitud positiva")
    window = list(observations)[-history_length:]
    window = [window[0]] * (history_length - len(window)) + window
    inputs = [
        vehicle_input(
            item, vehicle, reserved_cells if index == history_length - 1 else ()
        )
        for index, item in enumerate(window)
    ]
    return np.stack([item[0] for item in inputs]), np.stack(
        [item[1] for item in inputs]
    )


class BaselinePolicy:
    """Aleatoria, greedy por incertidumbre o barrido por filas sobre destinos viales.

    El barrido alterna este/oeste al cambiar de fila y norte/sur al llegar al
    borde. En una red irregular es una referencia geométrica, no una garantía
    de cobertura exhaustiva. Las tres políticas reservan destinos en orden 0–2.
    """

    def __init__(self, strategy: str, *, seed: int = 0) -> None:
        if strategy not in {"random", "greedy", "sweep"}:
            raise ValueError("strategy debe ser random, greedy o sweep")
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.horizontal = [2, 6, 2]
        self.vertical = [0, 0, 0]

    def act(self, env: LimaPM25Env, observation: Observation) -> np.ndarray:
        actions = np.zeros(VEHICLE_COUNT, dtype=np.int64)
        reserved = set()
        for vehicle in range(VEHICLE_COUNT):
            available = np.flatnonzero(env.action_masks(reserved)[vehicle])
            if not len(available):
                continue
            if self.strategy == "random":
                action = int(self.rng.choice(available))
            elif self.strategy == "greedy":
                values = []
                for candidate in available:
                    route = env.route(vehicle, int(candidate))
                    assert route is not None
                    values.append(observation["maps"][3, *route.cell])
                best = available[np.isclose(values, np.max(values))]
                action = int(self.rng.choice(best))
            else:
                horizontal, vertical = self.horizontal[vehicle], self.vertical[vehicle]
                if horizontal in available:
                    action = horizontal
                elif vertical in available:
                    action = vertical
                    self.horizontal[vehicle] = (horizontal + 4) % 8
                else:
                    self.vertical[vehicle] = (vertical + 4) % 8
                    preferred = self.vertical[vehicle]
                    action = preferred if preferred in available else int(available[0])
                    self.horizontal[vehicle] = (horizontal + 4) % 8
            actions[vehicle] = action
            route = env.route(vehicle, action)
            assert route is not None
            reserved.add(route.cell)
        return actions
