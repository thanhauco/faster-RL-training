import torch

from kvstreams.rope import RotaryEmbedding, apply_rope


def _rot(rope, x, pos):
    cos, sin = rope(torch.tensor([pos]))
    return apply_rope(x[None, None], cos, sin)[0, 0]


def test_scores_depend_only_on_relative_position():
    torch.manual_seed(0)
    rope = RotaryEmbedding(16).double()
    q, k = torch.randn(16, dtype=torch.float64), torch.randn(16, dtype=torch.float64)
    a = _rot(rope, q, 1000) @ _rot(rope, k, 990)
    b = _rot(rope, q, 10) @ _rot(rope, k, 0)
    assert torch.allclose(a, b, atol=1e-10)


def test_wrong_position_after_compaction_changes_scores():
    """Re-using a key rotated for slot 1000 as if it were at slot 500 breaks attention."""
    torch.manual_seed(0)
    rope = RotaryEmbedding(16).double()
    q, k = torch.randn(16, dtype=torch.float64), torch.randn(16, dtype=torch.float64)
    k_cached = _rot(rope, k, 1000)  # rotated at its original logical position
    correct = _rot(rope, q, 1010) @ k_cached  # query continues the original counter
    physical = _rot(rope, q, 510) @ k_cached  # query numbered by the compacted physical slot
    reference = _rot(rope, q, 1010) @ _rot(rope, k, 1000)
    assert torch.allclose(correct, reference)
    assert not torch.allclose(physical, reference)


def test_rotation_preserves_norm():
    rope = RotaryEmbedding(8).double()
    x = torch.randn(3, 2, 8, dtype=torch.float64)
    cos, sin = rope(torch.tensor([0, 7, 123]))
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1))
