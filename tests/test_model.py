import torch

from kvstreams.model import causal_mask
from kvstreams.paged_cache import PagedKVCache


def _cache(model, bs=4):
    cfg = model.cfg
    return PagedKVCache(
        cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, block_size=bs, num_blocks=64, dtype=model.dtype
    )


def test_cached_forward_matches_full_forward(model64):
    torch.manual_seed(1)
    tokens = torch.randint(0, model64.cfg.vocab_size, (23,))
    full = model64(tokens[None])[0]
    cache = _cache(model64)
    cache.add_sequence(0)
    outs = []
    for chunk in (tokens[:10], tokens[10:11], tokens[11:20], tokens[20:]):
        start = cache.num_tokens(0)
        outs.append(
            model64.forward_cached(chunk, torch.arange(start, start + len(chunk)), cache, 0)
        )
    assert torch.allclose(torch.cat(outs), full, atol=1e-10)


def test_eviction_matches_masked_forward(model64):
    """Evicting a block from the live cache == masking it out for later queries."""
    torch.manual_seed(2)
    tokens = torch.randint(0, model64.cfg.vocab_size, (20,))
    cache = _cache(model64)
    cache.add_sequence(0)
    model64.forward_cached(tokens[:12], torch.arange(12), cache, 0)
    cache.evict_positions(0, [(4, 8)])
    tail = model64.forward_cached(tokens[12:], torch.arange(12, 20), cache, 0)

    mask = causal_mask(20).clone()
    mask[12:, 4:8] = False
    ref = model64(tokens[None], torch.arange(20)[None], mask[None])[0, 12:]
    assert torch.allclose(tail, ref, atol=1e-10)


def test_renumbering_positions_breaks_equivalence(model64):
    torch.manual_seed(3)
    tokens = torch.randint(0, model64.cfg.vocab_size, (16,))
    keep = torch.cat([torch.arange(0, 4), torch.arange(8, 16)])
    mask = causal_mask(16).clone()
    mask[8:, 4:8] = False
    # Compaction-time mask, original positions.
    a = model64(tokens[None], torch.arange(16)[None], mask[None])[0, -1]
    # Rebuilt shorter sequence, re-numbered positions: different states.
    b = model64(tokens[keep][None])[0, -1]
    assert not torch.allclose(a, b, atol=1e-6)
