import io
import tarfile
from copy import deepcopy
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
import torch
from shapely.geometry import LineString

from lima_pm25_rl.evaluation import evaluate_episode
from lima_pm25_rl.mobility_diagnostics import (
    archive_files,
    audit_arrival,
    compare_fixed_measurements,
    destination_connectivity,
    trace_episode,
)
from lima_pm25_rl.network import BiLGD3QN, NeuralController
from lima_pm25_rl.routing import RoadRouter


def test_closest_stop_can_trap_vehicle_despite_a_continuing_stop_in_same_cell(domain):
    graph = nx.MultiDiGraph(crs=domain.grid.crs)
    positions = {
        "start": (100, 100),
        "dead": (250, 100),
        "live": (350, 150),
        "next": (500, 150),
    }
    for node, (x, y) in positions.items():
        graph.add_node(node, x=x, y=y)
    graph.add_edge(
        "start", "dead", geometry=LineString((positions["start"], positions["dead"]))
    )
    graph.add_edge(
        "start", "live", geometry=LineString(((100, 100), (100, 150), (350, 150)))
    )
    graph.add_edge(
        "live", "next", geometry=LineString((positions["live"], positions["next"]))
    )
    domain = replace(domain, road_graph=graph)
    router = RoadRouter(domain)
    original = graph.copy()
    decoded = router.graph.copy()
    chosen = router.plan("start", 2)
    report = audit_arrival(domain, router, "start", 2)
    assert report["chosen_connectivity"]["continuation_directions"] == []
    assert report["chosen_connectivity"]["original_reachable_nodes"] == 1
    assert report["chosen_connectivity"]["forward_endpoint_reachable_in_decoder"]
    alternative = report["nearest_continuing_candidate"]
    assert alternative["cell"] == report["target_cell"] == [0, 1]
    assert "E" in alternative["continuation_directions"]
    assert report["additional_distance_m"] > 0
    assert router.plan("start", 2) == chosen
    assert nx.utils.graphs_equal(graph, original)
    assert nx.utils.graphs_equal(router.graph, decoded)


def test_audit_distinguishes_a_discarded_segment_at_the_buffer_boundary(domain):
    graph = nx.MultiDiGraph(crs=domain.grid.crs)
    graph.add_node("start", x=100, y=100)
    graph.add_node("outside", x=1000, y=100)
    graph.add_edge("start", "outside", geometry=LineString(((100, 100), (1000, 100))))
    domain = replace(domain, road_graph=graph)
    router = RoadRouter(domain)
    node = "start"
    for _ in range(3):
        node = router.plan(node, 2).destination
    report = destination_connectivity(domain, router, node)
    assert report["continuation_directions"] == []
    assert report["reachable_physical_nodes"] == 1
    assert not report["forward_endpoint_reachable_in_decoder"]
    assert report["original_component"][0]["inside_permitted_buffer"] is False


def test_trace_preserves_policy_weights_and_reproduces_standard_evaluation(make_env):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    network = BiLGD3QN(hidden_size=8).eval()
    weights = deepcopy(network.state_dict())
    policy = NeuralController(network)
    report = trace_episode(make_env(), policy, case_seed=42, vehicle=0)
    reference = evaluate_episode(
        make_env(), NeuralController(network), seed=42, comparison=True
    )
    for key in (
        "reward",
        "final_integrated_variance",
        "mean_integrated_variance",
        "distance_m",
        "synthetic_rmse_ug_m3",
    ):
        assert report["metrics"][key] == pytest.approx(reference[key])
    assert all(
        torch.equal(value, network.state_dict()[name])
        for name, value in weights.items()
    )
    assert all(
        report["signatures"][key] == reference[key] for key in report["signatures"]
    )
    assert not network._forward_hooks
    assert len(report["ticks"]) == 3
    assert report["first_arrival_audit"] is not None
    for tick in report["ticks"]:
        for stage in tick["stages"]:
            scores = np.asarray(stage["q_values"])
            masks = np.asarray(stage["valid_masks"], dtype=bool)
            best = int(np.argmax(np.where(masks, scores, -np.inf)))
            assert best // 8 == stage["vehicle"]
        assert not any(tick["invalid_actions"])


def test_archive_reader_never_extracts_member_paths_and_rejects_ambiguity(tmp_path):
    archive = tmp_path / "input.tar.zst"
    with tarfile.open(archive, "w:zst") as tar:
        for name, content in (
            ("../../policy-best.pt", b"weights"),
            ("ignored.txt", b"ignore"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    assert archive_files(archive, {"policy-best.pt"}) == {"policy-best.pt": b"weights"}
    assert list(tmp_path.iterdir()) == [archive]
    with pytest.raises(ValueError, match="Faltan"):
        archive_files(archive, {"missing.json"})
    duplicate = tmp_path / "duplicate.tar"
    with tarfile.open(duplicate, "w") as tar:
        for name in ("one/config.json", "two/config.json"):
            member = tarfile.TarInfo(name)
            member.size = 2
            tar.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="ambiguo"):
        archive_files(duplicate, {"config.json"})


def test_fixed_comparison_accepts_coordinate_roundoff_but_not_changed_values_or_sources():
    actual = {
        "sensor_id": "FIXED",
        "timestamp_utc": "2026-02-05T07:00:00+00:00",
        "x_m": 285390.34287870256,
        "y_m": 8667920.939161547,
        "simulation_time_s": 0.0,
        "value_ug_m3": 25.0,
        "noise_stddev_ug_m3": 0.0,
        "provenance": "synthetic",
    }
    reference = {**actual, "x_m": "285390.34287870245"}
    result = compare_fixed_measurements([actual], [reference])
    assert result["match"]
    assert 0 < result["max_absolute_differences"]["x_m"] < 1e-6
    assert not compare_fixed_measurements(
        [actual], [{**reference, "value_ug_m3": 25.1}]
    )["match"]
    assert not compare_fixed_measurements(
        [actual], [{**reference, "provenance": "observed"}]
    )["match"]
    assert not compare_fixed_measurements([actual], [reference, reference])["match"]
