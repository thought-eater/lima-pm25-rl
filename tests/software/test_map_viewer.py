import json
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from lima_pm25.domain import build_grid
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box

from lima_pm25_rl import map_viewer as viewer


def grid():
    original = build_grid(box(278000, 8660000, 278800, 8660800), "EPSG:32718", 200)
    mask = original.active_mask.copy()
    mask[0, 0] = False
    return replace(original, active_mask=mask)


def test_reprojection_preserves_cell_centers_and_inactive_mask():
    domain = grid()
    indices, bounds = viewer.raster_projection(domain, 300)
    mercator = Transformer.from_crs(4326, 3857, always_xy=True)
    west, south = mercator.transform(bounds[0][1], bounds[0][0])
    east, north = mercator.transform(bounds[1][1], bounds[1][0])
    native = Transformer.from_crs(domain.crs, 3857, always_xy=True)
    expected = np.full(domain.shape, -1, dtype=int)
    expected[domain.active_mask] = np.arange(domain.active_mask.sum())
    height, width = indices.shape
    for row in range(domain.shape[0]):
        for column in range(domain.shape[1]):
            x, y = native.transform(*domain.cell_center(row, column))
            px = int((x - west) / (east - west) * width)
            py = int((north - y) / (north - south) * height)
            assert indices[py, px] == expected[row, column]


def test_colors_preserve_signed_error_and_transparency():
    rgba = viewer.colorize(
        np.array([-5.0, 0.0, 5.0]), np.array([[-1, 0, 1, 2]]), [-5.0, 5.0], "RdBu_r"
    )
    assert rgba[0, 0, 3] == 0
    assert np.all(rgba[0, 1:, 3] == 255)
    assert rgba[0, 1, 2] > rgba[0, 1, 0]  # Negativo: sobreestimación azul.
    assert rgba[0, 3, 0] > rgba[0, 3, 2]  # Positivo: subestimación roja.


def test_route_prefixes_never_show_future_or_extend_terminated_mission():
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (20.0, 10.0)]
    records = [
        SimpleNamespace(sensor_id="MOBILE_0", simulation_time_s=t, x_m=x, y_m=y)
        for t, x, y in [(300, 10, 0), (600, 20, 10)]
    ]
    frames = viewer.route_frames({0: points}, records, [0, 300, 600, 900])
    assert frames[0]["ends"] == [0, 1, 3, 3]
    bad = [SimpleNamespace(sensor_id="MOBILE_0", simulation_time_s=300, x_m=99, y_m=99)]
    with pytest.raises(ValueError, match="recorrido"):
        viewer.route_frames({0: points}, bad, [0, 300])


def test_viewer_generates_four_maps_all_frames_and_explicit_provenance(
    tmp_path, monkeypatch
):
    domain = grid()
    count = int(domain.active_mask.sum())
    truth = np.tile(np.arange(count), (3, 1)).astype(float)
    mission = {
        "id": "example",
        "study": "Prueba </script>",
        "case": 42,
        "strategy": "BiLG",
        "strategy_id": "neural",
        "episode": 2000,
        "start": "2026-02-01T00:00:00+00:00",
        "decision_end": 300,
        "times": np.array([0.0, 300.0, 600.0]),
        "truth": truth,
        "mean": truth + 2,
        "std": np.ones_like(truth),
        "routes": [],
        "samples": [],
        "metrics": [{"rmse": 2.0, "hot_rmse": None, "distance_km": 0.0, "recall": None}]
        * 3,
        "source": {"policy_file": "example.pt"},
    }
    monkeypatch.setattr(
        viewer,
        "_geometry",
        lambda root: (domain, None, [], [], [[-12.1, -77.1], [-12.0, -77.0]]),
    )
    monkeypatch.setattr(viewer, "load_missions", lambda *args: ([mission], 25.0))
    settings = {"core_root": ".", "raster_width": 100}
    output = tmp_path / "maps"
    page = viewer.generate_viewer(settings, tmp_path, output)
    text = page.read_text()
    assert len(re.findall(r"= L\.map\(", text)) == 4
    embedded = re.search(
        r'<script type="application/json" id="viewer-data">(.*?)</script>',
        text,
        re.DOTALL,
    ).group(1)
    data = json.loads(embedded)
    assert data["missions"][0]["study"] == "Prueba </script>"
    assert "</script>" not in embedded
    assert data["limits"]["concentration"] == [0.0, float(count + 1)]
    assert data["limits"]["error"] == [-2.0, 2.0]
    for images in data["missions"][0]["images"].values():
        assert len(images) == 3
        for image in images:
            with Image.open(output / image) as png:
                assert png.mode == "RGBA" and png.width == 100
    metadata = json.loads((output / "config.json").read_text())
    assert not any(
        metadata[k]
        for k in ("training_executed", "simulation_executed", "gp_fit_executed")
    )
    assert metadata["error"] == "synthetic_truth_minus_GP_mean"
    with pytest.raises(FileExistsError):
        viewer.generate_viewer(settings, tmp_path, output)
