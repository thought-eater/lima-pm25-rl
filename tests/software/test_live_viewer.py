import json

import numpy as np
import torch

from lima_pm25_rl.live_viewer import LiveViewer
from lima_pm25_rl.network import BiLGD3QN, NeuralController


def test_live_comparison_keeps_scenario_cache_and_no_samples_in_fixed(env):
    torch.manual_seed(1)
    controller = NeuralController(BiLGD3QN(hidden_size=8).eval())
    app = LiveViewer.__new__(LiveViewer)
    app.policies = {"fijo": (env, controller)}
    app.known = {42, 43}
    app.seed = None
    app.branches = {}
    app.threshold = 8.0
    app.indices = np.arange(int(env.domain.grid.active_mask.sum()))[None, :]
    app.limits = {"concentration": [0, 100], "std": [0, 55], "error": [-60, 60]}
    app.latlon = lambda points: np.asarray(points).tolist()
    first = app.frame(42, "fijo", "neural", 0)
    step = app.frame(42, "fijo", "neural", 1)
    neural, fixed, greedy = [
        json.loads(app.frame(42, "fijo", s, 1)) for s in ("neural", "fixed", "greedy")
    ]
    assert (
        fixed["values"]["truth"]
        == greedy["values"]["truth"]
        == neural["values"]["truth"]
    )
    assert fixed["samples"] == [r for r in neural["samples"] if r["type"] == "fixed"]
    assert not fixed["routes"] and not fixed["q_trace"] and not greedy["q_trace"]
    assert json.loads(first)["q_trace"] and neural["q_trace"]
    assert (
        app.frame(42, "fijo", "neural", 0) == first
        and app.frame(42, "fijo", "neural", 1) == step
    )
    assert len(app.branches["fijo", "neural"].frames) == 2
    app.frame(43, "fijo", "neural", 0)
    assert app.frame(42, "fijo", "neural", 1) == step


def test_q_trace_matches_uninstrumented_plan_including_reservations(env):
    torch.manual_seed(13)
    net = BiLGD3QN(hidden_size=8).eval()
    reference, watched = NeuralController(net), NeuralController(net)
    obs, _ = env.reset(seed=10)
    for _ in range(3):
        trace = []
        expected, _ = reference.plan(env, obs)
        actual, decisions = watched.plan(env, obs, q_trace=trace)
        np.testing.assert_array_equal(expected, actual)
        assert len(trace) == len(decisions)
        for d, r in zip(decisions, trace, strict=True):
            np.testing.assert_array_equal(d.mask, r["valid"])
            assert (
                r["chosen"]
                == np.where(r["valid"], r["q"], -np.inf).argmax()
                == 8 * d.vehicle + d.action
            )
            assert len(r["reserved"]) == r["order"]
        obs, *_ = env.step(actual)
