from copy import deepcopy

import pytest
import torch

from lima_pm25_rl.network import BiLGD3QN, NeuralController, double_dqn_loss
from lima_pm25_rl.policies import AUXILIARY_SIZE


def test_bilg_forward_backward_and_target_isolation():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    online = BiLGD3QN(hidden_size=16)
    target = deepcopy(online)
    maps, auxiliary = torch.randn(2, 3, 7, 8, 8), torch.randn(2, 3, AUXILIARY_SIZE)
    output = online(maps, auxiliary)
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    previous = next(online.parameters()).detach().clone()
    loss = double_dqn_loss(
        online,
        target,
        observations=(maps, auxiliary),
        actions=torch.tensor([0, 1]),
        rewards=torch.tensor([1.0, 2.0]),
        next_observations=(
            maps[:, None].expand(-1, 3, -1, -1, -1, -1),
            auxiliary[:, None].expand(-1, 3, -1, -1),
        ),
        next_masks=torch.ones(2, 3, 8, dtype=torch.bool),
        done=torch.tensor([False, True]),
        discounts=torch.full((2,), 0.99),
    )
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in target.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in online.parameters()
    )
    optimizer.step()
    assert not torch.equal(previous, next(online.parameters()))


def test_double_target_selects_with_online_masks_invalid_actions_and_stops_bootstrap():
    class Scores(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, maps, auxiliary):
            return self.values.expand(maps.shape[0], -1)

    online = Scores([2, 1000, 3, 0, 0, 0, 0, 0])
    target = Scores([5, 2000, 7, 0, 0, 0, 0, 0])
    inputs = torch.zeros(2, 1), torch.zeros(2, 1)
    following = torch.zeros(2, 3, 1), torch.zeros(2, 3, 1)
    mask = torch.zeros(2, 3, 8, dtype=torch.bool)
    mask[:, 0, 0] = True
    mask[:, 0, 2] = True
    loss = double_dqn_loss(
        online,
        target,
        observations=inputs,
        next_observations=following,
        actions=torch.tensor([0, 0]),
        rewards=torch.ones(2),
        next_masks=mask,
        done=torch.tensor([False, True]),
        discounts=torch.full((2,), 0.5),
    )
    assert loss.item() == pytest.approx(1.25)


def test_central_controller_masks_every_choice_and_clears_its_history(env):
    torch.manual_seed(1)
    controller = NeuralController(BiLGD3QN(hidden_size=8), history_length=2, seed=1)
    observation, _ = env.reset(seed=42)
    for _ in range(3):
        observation, _, _, _, info = env.step(controller.act(env, observation))
        assert info["invalid_actions"].sum() == 0
    assert len(controller.history) == 2
    observation, _ = env.reset(seed=42)
    controller.act(env, observation)
    assert len(controller.history) == 1
    controller.reset()
    assert len(controller.history) == 0


def test_controller_selects_global_q_and_recomputes_after_each_reservation(env):
    class Priority(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, maps, auxiliary):
            vehicle = auxiliary[:, -1, 7:10].argmax(dim=1)
            pending = auxiliary[:, -1, 10:13].sum(dim=1)
            scores = torch.tensor([1.0, 10.0, 9.0])[vehicle]
            scores = torch.where((pending == 2) & (vehicle == 0), 12.0, scores)
            return scores[:, None].expand(-1, 8)

    observation, _ = env.reset(seed=42)
    controller = NeuralController(Priority())
    actions, decisions = controller.plan(env, observation)
    assert [item.vehicle for item in decisions] == [1, 0, 2]
    assert actions.tolist() == [item.vehicle * 8 + item.action for item in decisions]
    assert decisions[1].pending == (0, 2)
    assert not decisions[1].mask[1].any()
    assert len(decisions[1].reserved_cells) == 1
    assert decisions[1].assigned_positions[1] != (0.0, 0.0)
    _, _, _, _, info = env.step(actions)
    assert info["invalid_actions"].sum() == 0
    assert info["elapsed_s"] == 300


def test_global_double_target_selects_vehicle_and_direction_with_online():
    class Scores(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, maps, auxiliary):
            return self.values[auxiliary[:, 0].long()]

    online = Scores([[2, 3], [1, 9], [0, 1000]])
    target = Scores([[20, 50], [30, 7], [0, 2000]])
    masks = torch.tensor([[[True, True], [True, True], [False, False]]])
    loss = double_dqn_loss(
        online,
        target,
        observations=(torch.zeros(1, 1), torch.zeros(1, 1)),
        actions=torch.tensor([0]),
        rewards=torch.tensor([1.0]),
        next_observations=(torch.zeros(1, 3, 1), torch.tensor([[[0], [1], [2]]])),
        next_masks=masks,
        done=torch.tensor([False]),
        discounts=torch.tensor([0.5]),
    )
    # Online escoge (vehículo 1, dirección 1); target evalúa 7, no 50 ni 2000.
    assert loss.item() == pytest.approx(2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No hay GPU CUDA disponible")
def test_small_forward_and_backward_on_cuda():
    network = BiLGD3QN(hidden_size=8).cuda()
    output = network(
        torch.randn(1, 2, 7, 8, 8, device="cuda"),
        torch.randn(1, 2, AUXILIARY_SIZE, device="cuda"),
    )
    output.square().mean().backward()
    assert output.shape == (1, 8) and torch.isfinite(output).all()
