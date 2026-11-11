from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from lima_pm25.simulation import MeasurementRecord

from lima_pm25_rl.gp_mixture import (
    GateConfig,
    LocalEvidenceGate,
    log_predictive_density,
    mix_marginals,
)


def reading(time, x=100, sensor="MOBILE_0"):
    return MeasurementRecord(
        sensor_id=sensor,
        sensor_type="mobile",
        timestamp_utc=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=time),
        simulation_time_s=time,
        x_m=x,
        y_m=100,
        crs="EPSG:32718",
        value_ug_m3=30,
        quality_status="accepted",
        noise_stddev_ug_m3=1,
        position_source_id="test",
        provenance="Prueba de causalidad",
    )


def test_predictive_density_includes_measurement_noise_once():
    expected = -0.5 * (np.log(2 * np.pi * 13) + 25 / 13)
    assert log_predictive_density(25, 20, 2, 3) == pytest.approx(expected)
    assert np.isfinite(log_predictive_density(20, 20, 0, 0))
    with pytest.raises(ValueError):
        log_predictive_density(20, 20, -1, 0)


def test_mixture_variance_contains_disagreement_and_does_not_assume_independence():
    mean, std = mix_marginals(10, 2, 30, 2, 0.5)
    assert mean == 20 and std == pytest.approx(np.sqrt(104))
    for weight in (0, 0.2, 0.5, 1):
        mean, std = mix_marginals(-5, 3, -5, 3, weight)
        assert mean == -5 and std == 3
    assert mix_marginals(10, 2, 30, 4, 0) == (10, 2)
    assert mix_marginals(10, 2, 30, 4, 1) == (30, 4)
    with pytest.raises(ValueError):
        mix_marginals(10, 2, 30, 4, 1.1)


def test_first_visits_and_same_cell_different_points_do_not_teach_persistence():
    gate = LocalEvidenceGate()
    used, _ = gate.observe(
        0, [reading(0), reading(0, sensor="MOBILE_1")], [0, 0], [4, 4]
    )
    assert not used.any()  # Dos lecturas simultáneas no son una revisita.
    for t, x in ((300, 110), (600, 120), (900, 130)):
        used, _ = gate.observe(t, [reading(t, x)], [0], [4])
        assert not used.any()  # Todos estos puntos pertenecen a la misma celda.
    weight, count = gate.weights(900, [(100, 100)])
    assert weight[0] == 0.5 and count[0] == 0
    used, _ = gate.observe(1200, [reading(1200, sensor="MOBILE_2")], [0], [4])
    assert used[0]  # Revisita del punto real, aunque llegue otro sensor idéntico.
    assert gate.weights(1200, [(100, 100)])[0][0] == 0.5


def test_three_batches_locality_forgetting_and_reset():
    gate = LocalEvidenceGate()
    gate.observe(0, [reading(0)], [0], [4])
    for t in (300, 600):
        gate.observe(t, [reading(t)], [0], [4])
        assert gate.weights(t, [(100, 100)])[0][0] == 0.5
    gate.observe(900, [reading(900)], [0], [40])  # Se limita la razón a 4.
    weight, count = gate.weights(900, [(100, 100), (350, 100), (600, 100)])
    np.testing.assert_allclose(weight, [0.95, 1 / (1 + np.exp(-1)), 0.5])
    np.testing.assert_array_equal(count, [3, 3, 0])
    for t in (1200, 1500, 1800):
        gate.observe(t, [reading(t)], [0], [-4])
    assert gate.weights(1800, [(100, 100)])[0][0] == pytest.approx(0.05)
    assert gate.weights(6000, [(100, 100)])[0][0] == 0.5
    assert LocalEvidenceGate().weights(1800, [(100, 100)])[0][0] == 0.5
    with pytest.raises(ValueError):
        gate.observe(1800, [reading(1800)], [0], [0])
    with pytest.raises(ValueError):
        gate.weights(1700, [(100, 100)])


def test_simultaneous_replicas_do_not_multiply_evidence():
    gates = [LocalEvidenceGate(), LocalEvidenceGate()]
    for t in (0, 300, 600, 900):
        gates[0].observe(t, [reading(t)], [0], [1])
        records = [reading(t, sensor=f"MOBILE_{v}") for v in range(3)]
        gates[1].observe(t, records, [0] * 3, [1] * 3)
    np.testing.assert_allclose(
        gates[0].weights(900, [(100, 100), (300, 100)])[0],
        gates[1].weights(900, [(100, 100), (300, 100)])[0],
    )
    assert gates[0].weights(900, [(100, 100)])[0][0] == pytest.approx(
        1 / (1 + np.exp(-1))
    )


def test_gate_requires_valid_parameters_and_ordered_unique_batches():
    with pytest.raises(ValueError):
        GateConfig(window_batches=1)
    with pytest.raises(ValueError):
        GateConfig(locality_radius_m=0)
    with pytest.raises(ValueError):
        LocalEvidenceGate(history_duration_s=-1)
    gate = LocalEvidenceGate()
    with pytest.raises(ValueError, match="duplicadas"):
        gate.observe(0, [reading(0), reading(0)], [0, 0], [0, 0])
    with pytest.raises(ValueError, match="simultáneas"):
        gate.observe(300, [reading(0)], [0], [0])
