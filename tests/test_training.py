import torch

from kvstreams.compaction import build_policy
from kvstreams.config import CacheConfig, ModelConfig, RolloutConfig, SamplingConfig
from kvstreams.engine import KVStreamEngine
from kvstreams.envs import build_env
from kvstreams.model import TinyLM
from kvstreams.training import RLConfig, RLTrainer, SFTTrainer, group_advantages, policy_loss


def small_stack(tok, engine_cls=KVStreamEngine):
    torch.manual_seed(0)
    model = TinyLM(
        ModelConfig(
            vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64
        )
    )
    eng = engine_cls(
        model,
        tok,
        build_policy("keep_recent", keep_last=1),
        CacheConfig(block_size=16, num_blocks=512),
        RolloutConfig(context_budget=96),
        SamplingConfig(max_new_tokens=6),
    )
    return model, eng


def test_group_advantages():
    adv = group_advantages(torch.tensor([1.0, 0.0, 1.0, 1.0]), group_size=2)
    assert torch.allclose(adv[:2], torch.tensor([1.0, -1.0], dtype=torch.float64), atol=1e-4)
    assert torch.allclose(adv[2:], torch.zeros(2, dtype=torch.float64))


def test_policy_loss_gradient_sign():
    logp = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    adv = torch.tensor([[1.0, -1.0]])
    mask = torch.tensor([[True, True]])
    loss, m = policy_loss(logp, logp.detach(), logp.detach(), adv, mask)
    loss.backward()
    # Positive advantage -> increase log-prob (negative gradient of the loss).
    assert logp.grad[0, 0] < 0 < logp.grad[0, 1]
    assert m["tis_mean"] == 1.0


def test_tis_weight_is_capped():
    logp = torch.zeros(1, 1, requires_grad=True)
    loss, m = policy_loss(
        logp,
        logp.detach(),
        torch.full((1, 1), -10.0),
        torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        tis_cap=2.0,
    )
    assert m["tis_mean"] == 2.0


def test_sft_reduces_loss(tok):
    model, eng = small_stack(tok)
    sft = SFTTrainer(
        model, eng, lambda s: build_env("recall", seed=s, num_distractors=3), lr=1e-2, batch_size=4
    )
    losses = sft.train(30, log_fn=None)
    assert losses[-1] < losses[0] * 0.7


def test_rl_step_runs_with_near_zero_mismatch(tok):
    model, eng = small_stack(tok)
    trainer = RLTrainer(
        model,
        eng,
        lambda s: build_env("recall", seed=s, num_distractors=3),
        RLConfig(tasks_per_step=2, group_size=2),
        log_fn=None,
    )
    hist = trainer.train(2)
    assert len(hist) == 2
    assert all(h["compactions"] > 0 for h in hist)
    assert all(h["mismatch_kl"] < 1e-6 for h in hist)
