"""Memoria acotada de episodios: un mapa por tick, ventanas construidas al muestrear."""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .env import Observation
from .policies import HISTORY_MODES, PlanningDecision, candidate_inputs, sequence_input


@dataclass
class EpisodeExperience:
    observations: list[Observation]
    decisions: list[tuple[PlanningDecision, ...]]
    rewards: list[float]
    done: list[bool]


class SequenceReplay:
    """Evicta episodios completos, nunca une ventanas de episodios distintos.

    La capacidad se expresa en ticks físicos; hay hasta tres transiciones de
    planificación por tick y una observación terminal adicional por episodio.
    No se guardan copias de las ventanas ni de las tres perspectivas. Cada
    decisión conserva máscaras, pendientes y asignaciones anteriores para
    reconstruir todos los candidatos del siguiente contexto de planificación.
    """

    def __init__(
        self,
        capacity_steps: int,
        history_length: int,
        gamma: float,
        *,
        seed: int,
        history_mode: str = "history",
    ) -> None:
        if capacity_steps < 1 or history_length < 1 or not 0 <= gamma <= 1:
            raise ValueError(
                "La capacidad y el historial deben ser positivos; gamma debe estar en [0,1]"
            )
        if history_mode not in HISTORY_MODES:
            raise ValueError("history_mode debe ser history o repeat_current")
        self.capacity_steps = capacity_steps
        self.history_length = history_length
        self.history_mode = history_mode
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.episodes: deque[EpisodeExperience] = deque()
        self.steps = 0
        self._indices: list[tuple[EpisodeExperience, int, int]] = []

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def nbytes(self) -> int:
        """Bytes de arrays almacenados; excluye el overhead de objetos Python."""
        return sum(
            value.nbytes
            for episode in self.episodes
            for observation in episode.observations
            for value in observation.values()
        ) + sum(
            decision.mask.nbytes
            for episode in self.episodes
            for tick in episode.decisions
            for decision in tick
        )

    def add(self, episode: EpisodeExperience) -> None:
        count = len(episode.rewards)
        if (
            not count
            or count > self.capacity_steps
            or len(episode.observations) != count + 1
            or len(episode.decisions) != count
            or len(episode.done) != count
            or not episode.done[-1]
            or any(episode.done[:-1])
        ):
            raise ValueError(
                "El buffer requiere un episodio completo que quepa en la capacidad configurada"
            )
        if not any(episode.decisions):
            raise ValueError("El episodio no contiene decisiones controlables")
        for tick in episode.decisions:
            for decision in tick:
                if (
                    decision.mask.shape != (3, 8)
                    or not decision.mask[decision.vehicle, decision.action]
                ):
                    raise ValueError("No se puede aprender de una acción enmascarada")
        self.episodes.append(episode)
        self.steps += count
        while self.steps > self.capacity_steps:
            self.steps -= len(self.episodes.popleft().rewards)
        self._indices = [
            (item, step, choice)
            for item in self.episodes
            for step, choices in enumerate(item.decisions)
            for choice in range(len(choices))
        ]

    def transition(self, index: int) -> dict[str, Any]:
        episode, step, choice_index = self._indices[index]
        decision = episode.decisions[step][choice_index]
        current = sequence_input(
            episode.observations[: step + 1],
            decision.vehicle,
            decision.reserved_cells,
            self.history_length,
            decision.pending,
            decision.assigned_positions,
            decision.mask[decision.vehicle],
            history_mode=self.history_mode,
        )
        physical = choice_index == len(episode.decisions[step]) - 1
        done = physical and episode.done[step]
        if not physical:
            following = episode.decisions[step][choice_index + 1]
            next_step = step
        elif not done:
            if not episode.decisions[step + 1]:
                raise ValueError(
                    "Una transición no terminal necesita una siguiente decisión"
                )
            following = episode.decisions[step + 1][0]
            next_step = step + 1
        else:
            following = PlanningDecision(
                0, 0, np.zeros((3, 8), dtype=np.int8), (), (), ((0.0, 0.0),) * 3
            )
            next_step = step + 1
        following_input = candidate_inputs(
            episode.observations[: next_step + 1],
            following.reserved_cells,
            self.history_length,
            following.pending,
            following.assigned_positions,
            following.mask,
            history_mode=self.history_mode,
        )
        return {
            "observations": current,
            "next_observations": following_input,
            "action": decision.action,
            "reward": episode.rewards[step] if physical else 0.0,
            "discount": self.gamma if physical else 1.0,
            "done": done,
            "next_mask": following.mask.copy(),
        }

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Any]:
        if not 0 < batch_size <= len(self):
            raise ValueError("El lote debe caber en las transiciones almacenadas")
        transitions = [
            self.transition(int(i))
            for i in self.rng.choice(len(self), batch_size, replace=False)
        ]

        def tensors(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            values = tuple(
                torch.as_tensor(
                    np.stack([item[name][part] for item in transitions]), device=device
                )
                for part in (0, 1)
            )
            return values[0], values[1]

        return {
            "observations": tensors("observations"),
            "next_observations": tensors("next_observations"),
            "actions": torch.tensor(
                [item["action"] for item in transitions], device=device
            ),
            "rewards": torch.tensor(
                [item["reward"] for item in transitions],
                dtype=torch.float32,
                device=device,
            ),
            "discounts": torch.tensor(
                [item["discount"] for item in transitions],
                dtype=torch.float32,
                device=device,
            ),
            "done": torch.tensor([item["done"] for item in transitions], device=device),
            "next_masks": torch.as_tensor(
                np.stack([item["next_mask"] for item in transitions]), device=device
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        """Solo tensores y tipos básicos, compatibles con torch.load(weights_only=True)."""
        return {
            "capacity_steps": self.capacity_steps,
            "history_length": self.history_length,
            "history_mode": self.history_mode,
            "gamma": self.gamma,
            "rng": self.rng.bit_generator.state,
            "episodes": [
                {
                    "observations": [
                        {name: torch.from_numpy(value) for name, value in item.items()}
                        for item in episode.observations
                    ],
                    "decisions": [
                        [
                            {
                                "vehicle": item.vehicle,
                                "action": item.action,
                                "mask": torch.from_numpy(item.mask),
                                "reserved_cells": item.reserved_cells,
                                "pending": item.pending,
                                "assigned_positions": item.assigned_positions,
                            }
                            for item in tick
                        ]
                        for tick in episode.decisions
                    ],
                    "rewards": episode.rewards,
                    "done": episode.done,
                }
                for episode in self.episodes
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (
            self.capacity_steps,
            self.history_length,
            self.gamma,
            self.history_mode,
        )
        if (
            state["capacity_steps"],
            state["history_length"],
            state["gamma"],
            state.get("history_mode", "history"),
        ) != expected:
            raise ValueError(
                "La configuración del buffer no coincide con el checkpoint"
            )
        self.episodes.clear()
        self.steps = 0
        self._indices.clear()
        for item in state["episodes"]:
            self.add(
                EpisodeExperience(
                    [
                        {
                            name: value.cpu().numpy().copy()
                            for name, value in observation.items()
                        }
                        for observation in item["observations"]
                    ],
                    [
                        tuple(
                            PlanningDecision(
                                decision["vehicle"],
                                decision["action"],
                                decision["mask"].cpu().numpy().copy(),
                                tuple(
                                    tuple(cell) for cell in decision["reserved_cells"]
                                ),
                                tuple(decision["pending"]),
                                tuple(
                                    tuple(point)
                                    for point in decision["assigned_positions"]
                                ),
                            )
                            for decision in tick
                        )
                        for tick in item["decisions"]
                    ],
                    item["rewards"],
                    item["done"],
                )
            )
        self.rng.bit_generator.state = state["rng"]
