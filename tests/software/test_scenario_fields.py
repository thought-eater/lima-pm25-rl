import json
from dataclasses import replace

import numpy as np
import pytest
from lima_pm25.domain import PM25SourceRole
from lima_pm25.simulation import EmissionProfile, PM25Transport, RoadLineSource
from shapely.geometry import LineString

from lima_pm25_rl.scenario_fields import FieldVariation


def variation(**overrides):
    return FieldVariation(
        **{
            "ambient_range_ug_m3": (13.6, 25.6),
            "road_source_scale": 20,
            "initial_plume_count": 2,
            "initial_plume_amplitude_range_ug_m3": (5, 15),
            "initial_plume_sigma_range_m": (100, 200),
            "provenance": "Hipótesis sintética exclusivamente para pruebas",
            **overrides,
        }
    )


def generated_request(template):
    return replace(
        template, episode_parameters_json=json.dumps({"partition": "training"})
    )


def test_field_changes_reproducibly_without_mutating_template_or_weather(template):
    request = generated_request(template)
    before = request.initial_state.concentration_ug_m3.copy()
    first = variation().apply(request, 42)
    second = variation().apply(
        replace(request, config=replace(request.config, seed=999)), 42
    )
    np.testing.assert_array_equal(
        first.initial_state.concentration_ug_m3,
        second.initial_state.concentration_ug_m3,
    )
    np.testing.assert_array_equal(request.initial_state.concentration_ug_m3, before)
    assert first.weather is request.weather
    assert first.model is request.model
    assert first.sensor_configs == request.sensor_configs
    assert np.ptp(first.initial_state.concentration_ug_m3) > 1
    assert not np.array_equal(
        first.initial_state.concentration_ug_m3,
        variation().apply(request, 43).initial_state.concentration_ug_m3,
    )
    metadata = json.loads(first.episode_parameters_json)["field_variation"]
    assert len(metadata["initial_plumes"]) == 2
    assert 13.6 <= metadata["ambient_ug_m3"] <= 25.6
    assert (
        first.regional_boundary.boundary_at_elapsed_s(0).concentration_ug_m3
        == metadata["ambient_ug_m3"]
    )


def test_scales_routine_and_event_without_changing_locations_or_timing(template):
    sources = tuple(
        RoadLineSource(
            source_id=name,
            line=LineString(((100, 100), (700, 100))),
            emission_rate_ug_m_s=rate,
            starts_at_s=100,
            ends_at_s=700,
            profile=EmissionProfile(((0, 0.5), (300, 1.5))),
            role=role,
            provenance="Prueba",
        )
        for name, rate, role in (
            ("routine", 100, PM25SourceRole.ROUTINE_ENVIRONMENTAL),
            ("event", 220, PM25SourceRole.COUNTERFACTUAL_EVENT),
        )
    )
    request = generated_request(replace(template, sources=sources))
    changed = variation().apply(request, 42)
    for before, after in zip(sources, changed.sources, strict=True):
        assert after.emission_rate_ug_m_s == before.emission_rate_ug_m_s * 20
        assert after.line is before.line
        assert after.profile is before.profile
        assert (after.starts_at_s, after.ends_at_s) == (100, 700)
    assert sources[0].emission_rate_ug_m_s == 100


@pytest.mark.parametrize(
    "overrides",
    [
        {"ambient_range_ug_m3": (25, 13)},
        {"road_source_scale": float("nan")},
        {"initial_plume_sigma_range_m": (0, 10)},
        {"initial_plume_count": True},
        {"provenance": ""},
    ],
)
def test_invalid_field_parameters_fail_explicitly(overrides):
    with pytest.raises(ValueError):
        variation(**overrides)


def test_field_variation_requires_a_generated_partition(template):
    with pytest.raises(ValueError, match="partición"):
        variation().apply(template, 42)


def test_inactive_cells_remain_zero_in_irregular_domain(template):
    active = template.model.grid.active_mask.copy()
    active[0, 0] = False
    grid = replace(template.model.grid, active_mask=active)
    model = PM25Transport(grid, template.model.physics)
    request = generated_request(replace(template, model=model))
    result = variation().apply(request, 42)
    assert result.initial_state.concentration_ug_m3[0, 0] == 0
    assert (result.initial_state.concentration_ug_m3[active] > 0).all()
