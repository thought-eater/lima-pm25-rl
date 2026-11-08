import socket
from pathlib import Path

import gymnasium as gym
import networkx as nx
import numpy as np
import pytest
from lima_pm25.domain import AmbientBaseline, DomainConfig, SpatialDomain, build_grid
from lima_pm25.simulation import PM25PhysicsConfig, PM25State, PM25Transport
from shapely.geometry import LineString, Polygon

from lima_pm25_rl import LimaPM25Env


@pytest.fixture
def domain() -> SpatialDomain:
    boundary = Polygon([(0, 0), (600, 0), (0, 400)])
    grid = build_grid(boundary, "EPSG:32718", 200)
    graph = nx.MultiDiGraph(crs=grid.crs)
    graph.add_node(1, x=100.0, y=100.0)
    graph.add_node(2, x=300.0, y=100.0)
    graph.add_edge(
        1,
        2,
        length=200.0,
        geometry=LineString([(100, 100), (300, 100)]),
    )
    return SpatialDomain(
        config=DomainConfig(cell_size_m=200),
        road_graph=graph,
        boundary=boundary,
        grid=grid,
        graph_source="synthetic_test",
        grid_source="synthetic_test",
    )


@pytest.fixture
def physics() -> PM25PhysicsConfig:
    return PM25PhysicsConfig(
        mixing_height_m=200,
        diffusivity_m2_s=10,
        dry_deposition_velocity_m_s=0.001,
        wet_scavenging_per_mm=0.1,
    )


@pytest.fixture
def initial_baseline() -> AmbientBaseline:
    return AmbientBaseline(
        provider_id="test-initial",
        concentration_ug_m3=8,
        provenance="Condición inicial sintética para pruebas de software",
    )


@pytest.fixture
def env(
    domain: SpatialDomain,
    physics: PM25PhysicsConfig,
    initial_baseline: AmbientBaseline,
) -> LimaPM25Env:
    return LimaPM25Env(
        domain=domain,
        physics=physics,
        initial_baseline=initial_baseline,
    )


def test_environment_composes_core_types_without_rebuilding_domain(
    env: LimaPM25Env,
    domain: SpatialDomain,
    physics: PM25PhysicsConfig,
    initial_baseline: AmbientBaseline,
) -> None:
    assert isinstance(env, gym.Env)
    assert env.domain is domain
    assert env.domain.road_graph is domain.road_graph
    assert env.physics is physics
    assert env.initial_baseline is initial_baseline
    assert isinstance(env._transport, PM25Transport)
    assert env._transport.grid is domain.grid
    assert env._transport.physics is physics


def test_initial_field_uses_the_supplied_baseline_and_active_mask(
    env: LimaPM25Env,
) -> None:
    state = env._state
    mask = env.domain.grid.active_mask
    assert isinstance(state, PM25State)
    assert state.grid is env.domain.grid
    assert state.simulation_time_s == 0
    assert state.ambient_baseline is env.initial_baseline
    assert state.concentration_ug_m3.shape == env.domain.grid.shape
    assert np.any(~mask)
    assert np.all(state.concentration_ug_m3[mask] == 8)
    assert np.all(state.concentration_ug_m3[~mask] == 0)
    assert not state.concentration_ug_m3.flags.writeable


def test_internal_initialization_restores_the_initial_field_and_time(
    env: LimaPM25Env,
) -> None:
    concentration = np.zeros(env.domain.grid.shape)
    concentration[env.domain.grid.active_mask] = 25
    env._state = env._transport.initialize_from_array(
        concentration,
        simulation_time_s=120,
    )
    previous = env._state

    result = env._initialize_simulation(seed=42)

    assert result is None
    assert env._state is not previous
    assert env._state.simulation_time_s == 0
    assert env._state.ambient_baseline is env.initial_baseline
    mask = env.domain.grid.active_mask
    assert np.all(env._state.concentration_ug_m3[mask] == 8)
    assert np.all(previous.concentration_ug_m3[mask] == 25)
    assert previous.simulation_time_s == 120
    assert not np.shares_memory(
        env._state.concentration_ug_m3,
        previous.concentration_ug_m3,
    )


def test_instances_own_separate_transport_state_and_random_generators(
    env: LimaPM25Env,
) -> None:
    other = LimaPM25Env(
        domain=env.domain,
        physics=env.physics,
        initial_baseline=env.initial_baseline,
    )
    env._initialize_simulation(seed=42)
    other._initialize_simulation(seed=42)
    assert env._transport is not other._transport
    assert env._state is not other._state
    assert env.np_random is not other.np_random
    assert not np.shares_memory(
        env._state.concentration_ug_m3,
        other._state.concentration_ug_m3,
    )
    np.testing.assert_array_equal(
        env.np_random.random(8),
        other.np_random.random(8),
    )
    current_other_state = other._state
    other_rng_state = other.np_random.bit_generator.state

    env._initialize_simulation(seed=7)

    assert other._state is current_other_state
    assert other.np_random_seed == 42
    assert other.np_random.bit_generator.state == other_rng_state


def test_explicit_seed_reproduces_the_gymnasium_random_sequence(
    env: LimaPM25Env,
) -> None:
    env._initialize_simulation(seed=42)
    first_draw = env.np_random.random(8)

    env._initialize_simulation(seed=42)

    assert env.np_random_seed == 42
    np.testing.assert_array_equal(env.np_random.random(8), first_draw)


def test_none_seed_preserves_the_existing_random_sequence(env: LimaPM25Env) -> None:
    env._initialize_simulation(seed=42)
    reference = np.random.default_rng(42)
    np.testing.assert_array_equal(env.np_random.random(8), reference.random(8))
    generator = env.np_random

    env._initialize_simulation()

    assert env.np_random is generator
    assert env.np_random_seed == 42
    np.testing.assert_array_equal(env.np_random.random(8), reference.random(8))


def test_invalid_seed_does_not_replace_the_field_or_generator(env: LimaPM25Env) -> None:
    state = env._state
    generator = env.np_random

    with pytest.raises(gym.error.Error):
        env._initialize_simulation(seed=-1)

    assert env._state is state
    assert env.np_random is generator


@pytest.mark.parametrize("parameter", ["domain", "physics", "initial_baseline"])
def test_constructor_rejects_inputs_that_are_not_core_objects(
    env: LimaPM25Env,
    parameter: str,
) -> None:
    arguments = {
        "domain": env.domain,
        "physics": env.physics,
        "initial_baseline": env.initial_baseline,
    }
    arguments[parameter] = None

    with pytest.raises(TypeError, match=parameter):
        LimaPM25Env(**arguments)


def test_reset_is_explicitly_pending_and_does_not_expose_or_change_truth(
    env: LimaPM25Env,
) -> None:
    state = env._state
    generator = env.np_random
    rng_state = generator.bit_generator.state

    with pytest.raises(NotImplementedError, match="observación inicial"):
        env.reset(seed=42)

    assert env._state is state
    assert env.np_random is generator
    assert generator.bit_generator.state == rng_state


def test_step_is_explicitly_pending_without_advancing_the_simulation(
    env: LimaPM25Env,
) -> None:
    state = env._state
    rng_state = env.np_random.bit_generator.state

    with pytest.raises(NotImplementedError, match="acciones"):
        env.step(object())

    assert env._state is state
    assert env.np_random.bit_generator.state == rng_state


def test_no_placeholder_spaces_registration_or_render_modes(env: LimaPM25Env) -> None:
    assert not hasattr(env, "action_space")
    assert not hasattr(env, "observation_space")
    assert env.spec is None
    assert env.render_mode is None
    assert env.metadata["render_modes"] == []
    with pytest.raises(NotImplementedError):
        env.render()
    env.close()
    env.close()


def test_construction_and_initialization_do_not_download_or_create_files(
    domain: SpatialDomain,
    physics: PM25PhysicsConfig,
    initial_baseline: AmbientBaseline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_network(*args: object, **kwargs: object) -> None:
        pytest.fail("El entorno no debe abrir conexiones de red")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "socket", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    original_graph = domain.road_graph.copy()
    original_mask = domain.grid.active_mask.copy()
    env = LimaPM25Env(
        domain=domain,
        physics=physics,
        initial_baseline=initial_baseline,
    )

    env._initialize_simulation(seed=42)
    env.close()

    assert list(tmp_path.iterdir()) == []
    assert nx.utils.graphs_equal(domain.road_graph, original_graph)
    np.testing.assert_array_equal(domain.grid.active_mask, original_mask)
