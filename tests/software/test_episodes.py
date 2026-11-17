from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from lima_pm25.domain import ScenarioFamilyPurpose
from lima_pm25.estimation import HistoricalPM25Dataset, HistoricalPM25Record
from pyproj import Transformer

from lima_pm25_rl import GeneratedEpisodes, HistoricalReplay


@pytest.mark.parametrize("mode", ["historical", "historical_perturbed", "synthetic"])
def test_all_episode_weather_families_delegate_to_core_and_respect_partitions(
    template, historical_weather, generation_config, mode
):
    factory = GeneratedEpisodes(
        template,
        historical_weather,
        generation_config,
        generation_config.training_end_utc,
        weather_mode=mode,
    )
    first, repeated = factory(42), factory(42)
    assert (
        generation_config.training_start_utc
        <= first.request.config.start_utc
        < generation_config.training_end_utc
    )
    assert first.historical_fixed is None
    assert (
        first.request.episode_parameters_json
        == repeated.request.episode_parameters_json
    )
    assert (
        first.request.scenario_identity.family_purpose
        is ScenarioFamilyPurpose.STRESS_COUNTERFACTUAL
    )
    evaluated = replace(factory, partition="evaluation")(42)
    assert evaluated.request.config.start_utc >= generation_config.evaluation_start_utc
    assert evaluated.request.config.start_utc != first.request.config.start_utc


def test_evaluation_cannot_overlap_gp_fitting(
    template, historical_weather, generation_config
):
    with pytest.raises(ValueError, match="solapa"):
        GeneratedEpisodes(
            template,
            historical_weather,
            generation_config,
            generation_config.evaluation_start_utc + timedelta(hours=1),
        )


def test_replay_uses_every_available_observed_catalog_record_without_imputation(
    template, historical_weather
):
    start = datetime(2025, 1, 4, tzinfo=UTC)
    lon, lat = Transformer.from_crs(
        template.model.grid.crs, "EPSG:4326", always_xy=True
    ).transform(700, 700)
    records = (
        HistoricalPM25Record(
            "FIXED", start, lon, lat, None, 30, "test", "test", "senamhi_observed"
        ),
        HistoricalPM25Record(
            "FIXED",
            start + timedelta(minutes=5),
            lon,
            lat,
            None,
            99,
            "test",
            "test",
            "imputed_short_linear",
        ),
        HistoricalPM25Record(
            "CERES", start, lon, lat, None, None, "test", "test", "missing_structural"
        ),
    )
    pm25 = HistoricalPM25Dataset(records, "unit-fixture", "0" * 64, "UTC")
    fit_end = datetime(2025, 1, 3, tzinfo=UTC)
    factory = HistoricalReplay(
        template, historical_weather, pm25, start, fit_end, fit_end
    )
    episode = factory(42)
    assert len(episode.historical_fixed) == 1
    assert episode.historical_fixed[0].value_ug_m3 == 30
    assert episode.historical_fixed[0].sensor_id == "FIXED"
    assert episode.historical_fixed[0].simulation_time_s == 0
    assert (
        episode.request.scenario_identity.family_purpose
        is ScenarioFamilyPurpose.HISTORICAL_REPLAY
    )
    with pytest.raises(ValueError, match="reservado"):
        replace(factory, policy_fit_end_utc=start + timedelta(hours=1))
