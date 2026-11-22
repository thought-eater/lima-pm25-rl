from types import SimpleNamespace

import numpy as np
import pytest

from lima_pm25_rl.validation_audit import scores, segments, speed_summary


def test_gaussian_central_score_and_degenerate_limit():
    y = np.array([[0.0, 4.0], [0.0, -4.0]])
    result = scores(y, np.zeros_like(y), np.array([[2.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(result["coverage95_pct"], 50)
    np.testing.assert_allclose(result["width95_ug_m3"], 3.91992797)
    np.testing.assert_allclose(
        result["crps_ug_m3"], (2 * (np.sqrt(2) - 1) / np.sqrt(np.pi) + 4) / 2
    )


def test_road_detour_holds_and_ended_ticks_have_distinct_denominators():
    routes = {0: [(0, 0), (0, 400), (10, 400), (10, 0)], 1: [(2, 0)], 2: [(3, 0)]}
    recs = [
        SimpleNamespace(
            sensor_id=f"MOBILE_{v}", simulation_time_s=t, x_m=p[-1][0], y_m=p[-1][1]
        )
        for v, p in routes.items()
        for t in (300, 600)
    ]
    rows = segments(routes, recs, np.arange(4) * 300, 600, 810)
    assert speed_summary(rows)["moving_median_kmh"] == pytest.approx(9.72)
    assert speed_summary(rows)["held_pct_active"] == pytest.approx(100 * 5 / 6)
    assert sum(r["required_kmh"] is None for r in rows) == 3
    with pytest.raises(ValueError, match="distancia"):
        segments(routes, recs, np.arange(4) * 300, 600, 10)
