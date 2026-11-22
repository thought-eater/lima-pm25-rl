import numpy as np
import pytest

from lima_pm25_rl.evaluation import evaluate_episode
from lima_pm25_rl.experiments.speed_experiment import LimitedRoutes
from lima_pm25_rl.policies import BaselinePolicy


@pytest.mark.parametrize("strategy", ["random", "sweep", "greedy"])
def test_cap_is_local_and_common_and_terminated_ticks_do_not_move(make_env, strategy):
    env = make_env()
    original = make_env()
    env.router = LimitedRoutes(env.router, 1)
    original.reset(seed=11)
    assert original.action_masks().any()
    frames = []
    metrics = evaluate_episode(
        env,
        BaselinePolicy(strategy),
        seed=11,
        comparison=True,
        observe_tick=lambda e, p, d: frames.append(
            (e._simulation.state.simulation_time_s, d)
        ),
    )
    assert not env.action_masks().any()
    assert metrics["invalid_actions"] == metrics["distance_m"] == 0
    assert metrics["early_termination"] and metrics["elapsed_s"] == 900
    assert [t for t, _ in frames] == [0, 300, 600, 900]
    assert np.array_equal(frames[1][1], [0, 0, 0])
    assert frames[2][1] is frames[3][1] is None
    assert metrics["mobile_measurements"] == 3


def test_exact_distance_is_admissible_and_shorter_budget_excludes_route(env):
    env.reset(seed=2)
    direction = np.flatnonzero(env.action_masks()[0])[0]
    node = env._nodes[0]
    route = env.router.plan(node, direction)
    assert LimitedRoutes(env.router, route.distance_m).plan(node, direction) == route
    assert (
        LimitedRoutes(env.router, route.distance_m - 1e-7).plan(node, direction) is None
    )
