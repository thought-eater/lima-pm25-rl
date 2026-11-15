import numpy as np
import pytest

from lima_pm25_rl.hotspot_metrics import (
    aggregate_missions,
    cell_detection_events,
    frame_metrics,
    summarize_mission,
)


def score(truth, mean, threshold=10):
    return frame_metrics(
        truth,
        mean,
        np.ones(len(truth)),
        threshold=threshold,
        cell_area_m2=40000,
        coordinates=np.column_stack(
            (np.arange(len(truth)) * 200, np.zeros(len(truth)))
        ),
    )


def test_omissions_and_false_alarms_do_not_cancel_deficit():
    row = score([20, 20, 0, 0], [5, 25, 15, 0])
    assert row["hot_cells"] == 2
    assert row["true_positive_cells"] == row["false_negative_cells"] == 1
    assert row["hot_deficit_sum_ug_m3"] == 15
    assert row["false_high_area_km2"] == pytest.approx(0.04)
    result = summarize_mission([row, row], [0, 300])
    assert result["temporal_hot_deficit_ug_m3"] == 7.5
    assert result["hot_cell_time_recall"] == 0.5


def test_no_hotspot_is_undefined_not_perfect_detection():
    row = score([0, 1], [20, 0])
    result = summarize_mission([row, row], [0, 300])
    assert result["temporal_hot_mse"] is None
    assert result["temporal_hot_deficit_ug_m3"] is None
    assert result["hot_cell_time_recall"] is None
    assert result["mean_false_high_area_km2"] == pytest.approx(0.04)


def test_labelling_whole_map_high_preserves_false_alarm_cost():
    row = score([20, 0, 0], [30, 30, 30])
    assert row["true_positive_cells"] == 1
    assert row["hot_deficit_sum_ug_m3"] == 0
    assert row["false_high_area_km2"] == pytest.approx(0.08)


def test_local_mse_is_pooled_over_cell_time_not_averaged_rmse():
    first = score([20, 0], [10, 0])
    last = score([20, 20], [20, 20])
    result = summarize_mission([first, last], [0, 300])
    assert result["temporal_hot_mse"] == pytest.approx(100 / 3)


def test_peak_ties_do_not_depend_on_array_order():
    a = score([20, 20, 0], [15, 10, 25])
    assert a["peak_signed_error_ug_m3"] == -7.5
    assert a["peak_location_distance_m"] == 200


def test_delays_report_never_detected_and_left_censored_separately():
    truth = np.array([[20, 0, 0], [20, 20, 20], [0, 20, 20], [0, 0, 0]])
    mean = np.array([[20, 0, 0], [20, 0, 0], [0, 20, 0], [0, 20, 20]])
    result = cell_detection_events(truth, mean, np.arange(4) * 300, 10)
    assert result["cell_events_left_censored"] == 1
    assert result["cell_events_detected"] == 1
    assert result["cell_events_never_detected"] == 1
    assert result["cell_event_delay_detected_mean_s"] == 300
    assert result["cell_event_missed_fraction"] == 0.5


def test_aggregation_averages_replicas_then_cases_and_only_then_sqrt():
    rows = [
        {"strategy": "neural", "case_seed": 1, "policy_seed": 0, "final_mse": 1.0},
        {"strategy": "neural", "case_seed": 1, "policy_seed": 1, "final_mse": 9.0},
        {"strategy": "neural", "case_seed": 2, "policy_seed": 0, "final_mse": 25.0},
    ]
    cases, summaries, _ = aggregate_missions(rows)
    assert len(cases) == 2
    assert summaries[0]["final_mse"] == 15
    assert summaries[0]["final_rmse_ug_m3"] == pytest.approx(np.sqrt(15))


@pytest.mark.parametrize("times", [[300, 600], [0, 0], [0]])
def test_invalid_clock_is_rejected(times):
    with pytest.raises(ValueError):
        summarize_mission([score([1], [1])] * len(times), times)


@pytest.mark.parametrize("mean", [[np.nan], [np.inf], [1, 2]])
def test_invalid_fields_are_rejected(mean):
    with pytest.raises(ValueError):
        score([1], mean)
