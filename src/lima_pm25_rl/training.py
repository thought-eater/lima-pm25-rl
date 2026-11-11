"""Entrenador Double DQN recurrente, sin frameworks ni ejecución implícita."""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO

import numpy as np
import torch

from .env import LimaPM25Env, Observation
from .evaluation import evaluate_episode
from .network import NeuralController, build_network, double_dqn_loss
from .policies import HISTORY_MODES
from .replay import EpisodeExperience, SequenceReplay


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str = "bilg"
    seed: int = 42
    history_length: int = 4
    history_mode: str = "history"
    hidden_size: int = 64
    capacity_steps: int = 2000
    batch_size: int = 16
    learning_starts: int = 128
    updates_per_step: int = 1
    target_update_interval: int = 200
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gradient_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    device: str = "auto"
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if self.history_mode not in HISTORY_MODES:
            raise ValueError("history_mode debe ser history o repeat_current")
        if self.architecture not in {"bilg", "window"}:
            raise ValueError("architecture debe ser bilg o window")
        for name in (
            "history_length",
            "hidden_size",
            "capacity_steps",
            "batch_size",
            "updates_per_step",
            "target_update_interval",
            "epsilon_decay_steps",
            "cpu_threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} debe ser un entero positivo")
        for name in ("seed", "learning_starts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} debe ser un entero no negativo")
        if self.capacity_steps < self.batch_size:
            raise ValueError("La capacidad debe ser al menos el tamaño del lote")
        if (
            not 0 <= self.gamma <= 1
            or not 0 <= self.epsilon_end <= self.epsilon_start <= 1
        ):
            raise ValueError(
                "gamma y las probabilidades de exploración deben estar en [0,1]"
            )
        if any(
            not np.isfinite(value) or value <= 0
            for value in (self.learning_rate, self.gradient_clip)
        ):
            raise ValueError(
                "La tasa de aprendizaje y el límite de gradiente deben ser positivos y finitos"
            )
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device debe ser auto, cpu o cuda")


class Trainer:
    """Recoge un episodio y después optimiza con episodios completos anteriores.

    El número de actualizaciones es proporcional a los ticks físicos, no al
    número de vehículos. Los checkpoints se producen exclusivamente entre
    episodios e incluyen el buffer y todos los generadores usados para aprender.
    """

    def __init__(
        self,
        env: LimaPM25Env,
        config: TrainingConfig,
        *,
        protocol: dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.config = config
        self.protocol = json.loads(json.dumps(protocol or {}, sort_keys=True))
        if self.protocol.get("reward") != getattr(env, "reward_protocol", None):
            raise ValueError("La recompensa del entorno no coincide con el protocolo")
        torch.set_num_threads(config.cpu_threads)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        name = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        if name == "cuda" and not torch.cuda.is_available():
            raise ValueError("Se solicitó CUDA pero no está disponible")
        self.device = torch.device(name)
        self.online = build_network(
            config.architecture,
            hidden_size=config.hidden_size,
            history_length=config.history_length,
        ).to(self.device)
        self.target = deepcopy(self.online).eval().requires_grad_(False)
        for module in self.target.modules():
            if isinstance(module, torch.nn.RNNBase):
                module.flatten_parameters()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.controller = NeuralController(
            self.online,
            history_length=config.history_length,
            seed=config.seed + 1,
            history_mode=config.history_mode,
        )
        self.replay = SequenceReplay(
            config.capacity_steps,
            config.history_length,
            config.gamma,
            seed=config.seed + 2,
            history_mode=config.history_mode,
        )
        self.completed_episodes = 0
        self.physical_steps = 0
        self.updates = 0
        self.target_updates = 0
        self.best_validation_variance = float("inf")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def epsilon(self) -> float:
        fraction = max(0.0, 1 - self.physical_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_end + fraction * (
            self.config.epsilon_start - self.config.epsilon_end
        )

    def episode_seed(self, episode_index: int) -> int:
        return int(
            np.random.SeedSequence([self.config.seed, 0, episode_index]).generate_state(
                1
            )[0]
        )

    def train_episode(self) -> dict[str, Any]:
        start = perf_counter()
        seed = self.episode_seed(self.completed_episodes)
        observation, info = self.env.reset(seed=seed, options={"scenario_seed": seed})
        assert self.env._simulation is not None
        if self.env._simulation.is_historical_replay:
            raise ValueError("El replay histórico final no puede usarse para entrenar")
        expected_steps = round(self.env._simulation.duration_s / 300)
        if expected_steps > self.config.capacity_steps:
            raise ValueError("Un episodio completo no cabe en el replay buffer")
        self.controller.reset()
        self.online.eval()
        initial_variance = info["integrated_variance"]
        variances = [initial_variance]
        information_gain = 0.0
        distance_cost = 0.0
        experience = EpisodeExperience([self._snapshot(observation)], [], [], [])
        epsilon_start = self.epsilon
        choices = 0
        while True:
            action, decisions = self.controller.plan(
                self.env, observation, epsilon=self.epsilon
            )
            observation, reward, terminated, truncated, info = self.env.step(action)
            if info["invalid_actions"].any():
                raise RuntimeError(
                    "El controlador produjo una acción enmascarada; no se aprende de ella"
                )
            experience.decisions.append(decisions)
            experience.rewards.append(float(reward))
            experience.done.append(bool(terminated or truncated))
            experience.observations.append(self._snapshot(observation))
            variances.append(info["integrated_variance"])
            information_gain += info["uncertainty_gain"]
            distance_cost += info["distance_cost"]
            self.physical_steps += 1
            choices += len(decisions)
            if terminated or truncated:
                break
        environment_seconds = perf_counter() - start
        self.replay.add(experience)
        losses = []
        optimize_start = perf_counter()
        for _ in range(len(experience.rewards) * self.config.updates_per_step):
            loss = self.optimize()
            if loss is not None:
                losses.append(loss)
        self.completed_episodes += 1
        return {
            "episode": self.completed_episodes,
            "scenario_seed": seed,
            "physical_steps": self.physical_steps,
            "episode_steps": len(experience.rewards),
            "planning_decisions": choices,
            "reward": sum(experience.rewards),
            "information_gain": information_gain,
            "distance_cost": distance_cost,
            "epsilon_start": epsilon_start,
            "epsilon_end": self.epsilon,
            "loss": float(np.mean(losses)) if losses else None,
            "optimizer_updates": self.updates,
            "target_updates": self.target_updates,
            "invalid_actions": 0,
            "distance_m": float(info["distance_m"].sum()),
            "initial_integrated_variance": initial_variance,
            "final_integrated_variance": info["integrated_variance"],
            "mean_integrated_variance": float(
                np.trapezoid(variances) / len(experience.rewards)
            ),
            "replay_steps": self.replay.steps,
            "replay_transitions": len(self.replay),
            "replay_array_bytes": self.replay.nbytes,
            "environment_seconds": environment_seconds,
            "optimization_seconds": perf_counter() - optimize_start,
            "seconds": perf_counter() - start,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0,
        }

    def optimize(self) -> float | None:
        if (
            self.physical_steps < self.config.learning_starts
            or len(self.replay) < self.config.batch_size
        ):
            return None
        self.online.train()
        loss = double_dqn_loss(
            self.online,
            self.target,
            **self.replay.sample(self.config.batch_size, self.device),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("La pérdida no es finita")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_updates += 1
        return float(loss.detach().cpu())

    def validate(self, env: LimaPM25Env, seeds: list[int]) -> dict[str, float]:
        if not seeds:
            raise ValueError("La validación necesita semillas predefinidas")
        env.reset(seed=seeds[0])
        assert env._simulation is not None
        if env._simulation.is_historical_replay:
            raise ValueError(
                "El replay histórico final no se utiliza para seleccionar modelos"
            )
        was_training = self.online.training
        rng = self._torch_rng()
        self.online.eval()
        try:
            metrics = [
                evaluate_episode(
                    env,
                    NeuralController(
                        self.online,
                        history_length=self.config.history_length,
                        seed=seed,
                        history_mode=self.config.history_mode,
                    ),
                    seed=seed,
                    comparison=True,
                )
                for seed in seeds
            ]
        finally:
            self.online.train(was_training)
            self._restore_torch_rng(rng)
        return {
            "reward": float(np.mean([item["reward"] for item in metrics])),
            "mean_integrated_variance": float(
                np.mean([item["mean_integrated_variance"] for item in metrics])
            ),
            "final_integrated_variance": float(
                np.mean([item["final_integrated_variance"] for item in metrics])
            ),
            "invalid_actions": float(
                np.sum([item["invalid_actions"] for item in metrics])
            ),
        }

    def save_checkpoint(self, path: Path | str) -> None:
        self._save(
            path,
            {
                **self._policy_state(),
                "kind": "training",
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "completed_episodes": self.completed_episodes,
                "physical_steps": self.physical_steps,
                "updates": self.updates,
                "target_updates": self.target_updates,
                "best_validation_variance": self.best_validation_variance,
                "replay": self.replay.state_dict(),
                "controller_rng": self.controller.rng.bit_generator.state,
                "torch_rng": self._torch_rng(),
            },
        )

    def export_policy(self, path: Path | str) -> None:
        self._save(path, {**self._policy_state(), "kind": "policy"})

    def load_checkpoint(self, path: Path | str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("version") != 2 or state.get("kind") != "training":
            raise ValueError(
                "Se necesita un checkpoint de entrenamiento del formato actual"
            )
        actual = asdict(TrainingConfig(**state["training_config"]))
        expected = asdict(self.config)
        for key in ("device", "cpu_threads"):
            actual.pop(key)
            expected.pop(key)
        if actual != expected or state["protocol"] != self.protocol:
            raise ValueError(
                "El protocolo o los hiperparámetros no coinciden con el checkpoint"
            )
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.completed_episodes = state["completed_episodes"]
        self.physical_steps = state["physical_steps"]
        self.updates = state["updates"]
        self.target_updates = state["target_updates"]
        self.best_validation_variance = state["best_validation_variance"]
        self.replay.load_state_dict(state["replay"])
        self.controller.rng.bit_generator.state = state["controller_rng"]
        self.controller.reset()
        self._restore_torch_rng(state["torch_rng"])

    @staticmethod
    def _snapshot(observation: Observation) -> Observation:
        return {name: value.copy() for name, value in observation.items()}

    def _policy_state(self) -> dict[str, Any]:
        return {
            "version": 2,
            "training_config": asdict(self.config),
            "protocol": self.protocol,
            "online": self.online.state_dict(),
            "torch_version": str(torch.__version__),
            "device": str(self.device),
            "completed_episodes": self.completed_episodes,
            "physical_steps": self.physical_steps,
        }

    @staticmethod
    def _save(path: Path | str, state: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def _torch_rng(self) -> dict[str, Any]:
        return {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if self.device.type == "cuda"
            else [],
        }

    def _restore_torch_rng(self, state: dict[str, Any]) -> None:
        torch.set_rng_state(state["cpu"])
        if (
            self.device.type == "cuda"
            and len(state["cuda"]) == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all(state["cuda"])


def load_policy(
    path: Path | str | BinaryIO, *, device: str = "auto", seed: int = 0
) -> tuple[NeuralController, dict[str, Any]]:
    """Carga segura de pesos y contexto, sin restaurar ni ejecutar el entrenamiento."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != 2 or state.get("kind") not in {"policy", "training"}:
        raise ValueError("El archivo no es una política del formato actual")
    config = TrainingConfig(**state["training_config"])
    name = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    network = build_network(
        config.architecture,
        hidden_size=config.hidden_size,
        history_length=config.history_length,
    ).to(name)
    network.load_state_dict(state["online"])
    network.eval()
    return NeuralController(
        network,
        history_length=config.history_length,
        seed=seed,
        history_mode=config.history_mode,
    ), state["protocol"]
