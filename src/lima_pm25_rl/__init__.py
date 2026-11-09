import gymnasium as gym

from .env import LimaPM25Env
from .episodes import Episode, GeneratedEpisodes, HistoricalReplay

__all__ = ["Episode", "GeneratedEpisodes", "HistoricalReplay", "LimaPM25Env"]

gym.register(
    id="lima_pm25_rl/LimaPM25-v0",
    entry_point="lima_pm25_rl.env:LimaPM25Env",
)
