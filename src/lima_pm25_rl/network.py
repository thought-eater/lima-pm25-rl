"""BiLG-D3QN: CNN → Bi-LSTM → Bi-GRU → valor/ventaja, con objetivo Double DQN.

Implementación inspirada en Fu et al.; tamaños explícitos, no una reproducción
de sus resultados. Las secuencias recibidas deben terminar en la decisión
actual. No se usan estados futuros ni observaciones de otros episodios.
"""

from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .env import LimaPM25Env, Observation
from .policies import PlanningDecision, sequence_input
from .simulation import VEHICLE_COUNT


class BiLGD3QN(nn.Module):
    def __init__(
        self, *, channels: int = 7, auxiliary_size: int = 7, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            128 + auxiliary_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.gru = nn.GRU(
            2 * hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.value = nn.Sequential(
            nn.Linear(2 * hidden_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(2 * hidden_size, 64), nn.ReLU(), nn.Linear(64, 8)
        )

    def forward(self, maps: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        if (
            maps.ndim != 5
            or auxiliary.ndim != 3
            or maps.shape[:2] != auxiliary.shape[:2]
            or maps.shape[1] == 0
        ):
            raise ValueError(
                "Se esperan mapas [lote, tiempo, canales, filas, columnas] y auxiliares [lote, tiempo, variables]"
            )
        batch, length = maps.shape[:2]
        features = self.encoder(maps.flatten(0, 1)).reshape(batch, length, -1)
        sequence, _ = self.lstm(torch.cat((features, auxiliary), dim=-1))
        _, hidden = self.gru(sequence)
        # h_n, no output[:, -1]: resume ambas direcciones de la ventana histórica.
        summary = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        advantage = self.advantage(summary)
        return self.value(summary) + advantage - advantage.mean(dim=-1, keepdim=True)


def double_dqn_loss(
    online: BiLGD3QN,
    target: BiLGD3QN,
    *,
    observations: tuple[torch.Tensor, torch.Tensor],
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_observations: tuple[torch.Tensor, torch.Tensor],
    next_masks: torch.Tensor,
    done: torch.Tensor,
    discounts: torch.Tensor,
) -> torch.Tensor:
    """Online selecciona, target evalúa; el final de misión no hace bootstrap.

    done incluye término o agotamiento del presupuesto de este experimento.
    discounts vale 1 entre subdecisiones sin avance físico y gamma después de
    un tick. Cada ventana debe provenir del mismo episodio y usar la perspectiva
    de la decisión correspondiente. No basta con barajar fotogramas aislados.
    """
    predicted = online(*observations).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        scores = online(*next_observations).masked_fill(~next_masks.bool(), -torch.inf)
        selected = scores.argmax(dim=-1)
        evaluated = target(*next_observations).gather(1, selected[:, None]).squeeze(1)
        bootstrap = (~done.bool()) & next_masks.bool().any(dim=-1)
        expected = rewards + discounts * torch.where(bootstrap, evaluated, 0.0)
    return F.smooth_l1_loss(predicted, expected)


class NeuralController:
    """Una red central compartida, tres elecciones condicionadas por reservas.

    El historial se actualiza una vez por tick físico, no una vez por vehículo.
    Llamar reset() al comenzar cada episodio. No entrena automáticamente.
    """

    def __init__(
        self, network: BiLGD3QN, *, history_length: int = 4, seed: int = 0
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length debe ser positivo")
        self.network = network
        self.history_length = history_length
        self.history = deque(maxlen=history_length)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.history.clear()

    def act(
        self, env: LimaPM25Env, observation: Observation, *, epsilon: float = 0.0
    ) -> np.ndarray:
        return self.plan(env, observation, epsilon=epsilon)[0]

    def plan(
        self, env: LimaPM25Env, observation: Observation, *, epsilon: float = 0.0
    ) -> tuple[np.ndarray, tuple[PlanningDecision, ...]]:
        """Elige las tres direcciones y conserva máscaras/reservas para aprender."""
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon debe estar entre 0 y 1")
        if observation["remaining"][0] == 1:
            self.history.clear()
        self.history.append({name: value.copy() for name, value in observation.items()})
        actions = np.zeros(VEHICLE_COUNT, dtype=np.int64)
        decisions = []
        reserved = set()
        device = next(self.network.parameters()).device
        for vehicle in range(VEHICLE_COUNT):
            mask = env.action_masks(reserved)[vehicle]
            available = np.flatnonzero(mask)
            if not len(available):
                continue
            if self.rng.random() < epsilon:
                action = int(self.rng.choice(available))
            else:
                maps_array, auxiliary_array = sequence_input(
                    list(self.history), vehicle, reserved, self.history_length
                )
                maps = torch.as_tensor(maps_array[None], device=device)
                auxiliary = torch.as_tensor(auxiliary_array[None], device=device)
                with torch.no_grad():
                    scores = self.network(maps, auxiliary)[0].cpu().numpy()
                action = int(available[np.argmax(scores[available])])
            actions[vehicle] = action
            decisions.append(
                PlanningDecision(vehicle, action, mask.copy(), tuple(sorted(reserved)))
            )
            route = env.route(vehicle, action)
            assert route is not None
            reserved.add(route.cell)
        return actions, tuple(decisions)
