import json
from copy import deepcopy
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
import torch
from shapely.geometry import LineString

from lima_pm25_rl import Episode, LimaPM25Env
from lima_pm25_rl.arrival_experiment import (
    compare_outputs,
    run_case_trial,
    run_full_trial,
)
from lima_pm25_rl.network import BiLGD3QN, NeuralController
from lima_pm25_rl.routing import RoadRouter


@pytest.fixture
def arrival_domain(domain):
    graph = nx.MultiDiGraph(crs=domain.grid.crs)
    points = {
        "start": (100, 100),
        "dead": (250, 100),
        "live": (350, 150),
        "next": (500, 150),
        "held-a": (100, 700),
        "held-b": (700, 700),
    }
    for name, (x, y) in points.items():
        graph.add_node(name, x=x, y=y)
    for u, v, coordinates in (
        ("start", "dead", ((100, 100), (250, 100))),
        ("start", "live", ((100, 100), (100, 150), (350, 150))),
        ("live", "next", ((350, 150), (500, 150))),
    ):
        graph.add_edge(u, v, geometry=LineString(coordinates))
    return replace(domain, road_graph=graph)


def test_selects_shortest_continuing_stop_in_same_cell_without_graph_mutation(
    arrival_domain,
):
    router = RoadRouter(arrival_domain)
    original = arrival_domain.road_graph.copy()
    decoded = router.graph.copy()
    shortest = router.plan("start", 2)
    continuing = router.plan("start", 2, prefer_continuation=True)
    assert shortest.cell == continuing.cell == (0, 1)
    assert not router.has_continuation(shortest.destination)
    assert router.has_continuation(continuing.destination)
    lengths = nx.single_source_dijkstra_path_length(
        router.graph, "start", weight="length"
    )
    expected = min(
        lengths[node]
        for node in router.graph.predecessors(router._goals[(0, 1)])
        if node in lengths and router.has_continuation(node)
    )
    assert continuing.distance_m == pytest.approx(expected)
    assert continuing.distance_m > shortest.distance_m
    assert router.plan("start", 2) == shortest
    assert nx.utils.graphs_equal(original, arrival_domain.road_graph)
    assert nx.utils.graphs_equal(decoded, router.graph)
    for node in set(router.graph) - set(router._goals.values()):
        if router.cell(node) is not None:
            assert router.has_continuation(node) == any(
                router.plan(node, a) is not None for a in range(8)
            )


def test_fallback_preserves_direction_when_continuing_candidates_are_unreachable(
    arrival_domain,
):
    graph = arrival_domain.road_graph.copy()
    graph.remove_edge("start", "live", 0)
    router = RoadRouter(replace(arrival_domain, road_graph=graph))
    shortest = router.plan("start", 2)
    assert router.has_continuation("live")
    assert router.plan("start", 2, prefer_continuation=True) == shortest
    assert router.plan(shortest.destination, 6, prefer_continuation=True) is None


def test_last_tick_uses_nearest_stop_and_step_executes_the_route_seen_by_policy(
    arrival_domain,
    template,
    estimator_config,
):
    def environment(duration, strategy):
        episode = Episode(
            replace(template, config=replace(template.config, duration_s=duration))
        )
        return LimaPM25Env(
            domain=arrival_domain,
            episode_factory=lambda seed: episode,
            estimator_config=estimator_config,
            initial_nodes=("start", "held-a", "held-b"),
            arrival_strategy=strategy,
        )

    baseline, proposed, last = (
        environment(900, "shortest"),
        environment(900, "continuing"),
        environment(300, "continuing"),
    )
    for env in (baseline, proposed, last):
        env.reset(seed=42)
    np.testing.assert_array_equal(baseline.action_masks(), proposed.action_masks())
    assert baseline.route(0, 2) == last.route(0, 2)
    route = proposed.route(0, 2)
    assert route != baseline.route(0, 2)
    proposed.action_masks((route.cell,))
    assert proposed.route(0, 2) == route  # La continuidad no depende de reservas.
    _, _, terminated, _, info = proposed.step(np.array([2, 8, 16]))
    assert not terminated
    assert proposed.positions[0] == route.coordinates[-1]
    assert info["route_distance_m"][0] == route.distance_m
    record = proposed._simulation.mobile.history.records[0]
    assert (record.x_m, record.y_m) == route.coordinates[-1]
    assert (record.x_m, record.y_m) != arrival_domain.grid.cell_center(*route.cell)


def test_case_trial_pairs_fields_and_keeps_policy_weights(env, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    network = BiLGD3QN(hidden_size=8).eval()
    weights = deepcopy(network.state_dict())
    summary = run_case_trial(
        env,
        NeuralController(network),
        case_seed=42,
        output=tmp_path / "case",
        metadata={},
        maps=False,
    )
    assert summary["signatures_match"] and not summary["training_executed"]
    assert all(
        torch.equal(value, network.state_dict()[name])
        for name, value in weights.items()
    )
    assert env.arrival_strategy == "shortest"
    assert (tmp_path / "case/continuing.json").is_file()


def test_full_trial_pairs_all_strategies_and_rejects_unpaired_fields(env, tmp_path):
    torch.set_num_threads(1)
    policy = NeuralController(BiLGD3QN(hidden_size=8).eval())
    output = tmp_path / "full"
    summary = run_full_trial(
        env,
        policy,
        case_seeds=[42],
        policy_seeds=[0, 1],
        output=output,
        metadata={},
        maps=False,
    )
    assert summary["signatures_match"]
    assert set(summary["strategies"]) == {
        "fixed",
        "random",
        "greedy",
        "sweep",
        "neural",
    }
    assert (
        summary["strategies"]["fixed"]["mobility"]["continuing"]["held_vehicle_ticks"]
        == 0
    )
    csv_path = output / "continuing/cases.csv"
    contents = csv_path.read_text()
    # Cambiar una firma sin alterar las claves de emparejamiento.
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(contents)))
    rows[0]["field_sha256"] = "different"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="intervención"):
        compare_outputs(output)
    assert (
        json.loads((output / "config.json").read_text())["training_executed"] is False
    )


def test_case_trial_uses_common_horizon_after_early_stop(
    make_env, tmp_path, monkeypatch
):
    env = make_env()
    original = env.action_masks

    def masks(reserved=()):
        return (
            np.zeros((3, 8), dtype=np.int8) if env._steps >= 1 else original(reserved)
        )

    monkeypatch.setattr(env, "action_masks", masks)
    policy = NeuralController(BiLGD3QN(hidden_size=8).eval())
    output = tmp_path / "early"
    run_case_trial(env, policy, case_seed=42, output=output, metadata={}, maps=False)
    for variant in ("shortest", "continuing"):
        report = json.loads((output / f"{variant}.json").read_text())
        assert report["metrics"]["decision_end_elapsed_s"] == 300
        assert report["metrics"]["elapsed_s"] == 900
        assert len(report["ticks"]) == 1
        assert len(report["arrivals_m"]) == 1
        assert len(report["fixed_measurements"]) == 2
