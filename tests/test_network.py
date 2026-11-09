from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from lima_pm25_rl.network import BiLGD3QN, NeuralController, double_dqn_loss


def test_bilg_forward_backward_and_target_isolation():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    online = BiLGD3QN(hidden_size=16)
    target = deepcopy(online)
    maps, auxiliary = torch.randn(2, 3, 7, 8, 8), torch.randn(2, 3, 7)
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
        next_observations=(maps, auxiliary),
        next_masks=torch.ones(2, 8, dtype=torch.bool),
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
    mask = torch.tensor([[True, False, True, False, False, False, False, False]] * 2)
    loss = double_dqn_loss(
        online,
        target,
        observations=inputs,
        next_observations=inputs,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No hay GPU CUDA disponible")
def test_small_forward_and_backward_on_cuda():
    network = BiLGD3QN(hidden_size=8).cuda()
    output = network(
        torch.randn(1, 2, 7, 8, 8, device="cuda"), torch.randn(1, 2, 7, device="cuda")
    )
    output.square().mean().backward()
    assert output.shape == (1, 8) and torch.isfinite(output).all()
