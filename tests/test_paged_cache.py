import pytest
import torch

from kvstreams.paged_cache import BlockAllocator, OutOfBlocksError, PagedKVCache


def make_cache(bs=4, nb=16):
    return PagedKVCache(
        n_layers=1, n_kv_heads=1, head_dim=2, block_size=bs, num_blocks=nb, dtype=torch.float64
    )


def fill(cache, seq, positions):
    pos = torch.tensor(positions)
    slots = cache.append_slots(seq, pos)
    kv = pos.double()[:, None, None].expand(-1, 1, 2).clone()
    cache.write(0, slots, kv, kv)


def test_allocator_reuse_and_double_free():
    a = BlockAllocator(2)
    b0, b1 = a.allocate(), a.allocate()
    with pytest.raises(OutOfBlocksError):
        a.allocate()
    a.free(b0)
    assert a.allocate() == b0
    with pytest.raises(ValueError):
        a.free(5)
    a.free(b1)
    with pytest.raises(ValueError):
        a.free(b1)


def test_gather_follows_block_table_order():
    c = make_cache()
    c.add_sequence(0)
    fill(c, 0, range(10))
    k, _ = c.gather(0, 0)
    assert k[:, 0, 0].tolist() == list(range(10))
    assert len(c.block_table(0)) == 3
    c.check_invariants(0)


def test_evict_blocks_joins_table_without_holes():
    c = make_cache()
    c.add_sequence(0)
    fill(c, 0, range(14))
    table = c.block_table(0)
    c.evict_blocks(0, [1])
    assert c.block_table(0) == [table[0], table[2], table[3]]
    assert c.num_tokens(0) == 10
    # Physical slot 4 now holds logical position 8: storage and position are decoupled.
    assert c.positions(0).tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13]
    k, _ = c.gather(0, 0)
    assert k[:, 0, 0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13]
    c.check_invariants(0)
    assert c.allocator.num_free == 16 - 3


def test_cannot_evict_partial_last_block():
    c = make_cache()
    c.add_sequence(0)
    fill(c, 0, range(6))
    with pytest.raises(ValueError):
        c.evict_blocks(0, [1])


def test_evict_positions_shrinks_to_whole_blocks():
    c = make_cache()
    c.add_sequence(0)
    fill(c, 0, range(16))
    # Request 2..11: only block [4..7] and [8..11] are fully covered.
    evicted = c.evict_positions(0, [(2, 12)])
    assert evicted == [(4, 12)]
    assert c.positions(0).tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    assert c.stats.shrunk_tokens == 2


def test_append_after_eviction_continues_logical_positions():
    c = make_cache()
    c.add_sequence(0)
    fill(c, 0, range(8))
    c.evict_positions(0, [(0, 4)])
    fill(c, 0, [8, 9])
    assert c.positions(0).tolist() == [4, 5, 6, 7, 8, 9]
    c.check_invariants(0)


def test_truncate_releases_partial_block():
    c = make_cache()
    c.add_sequence(0)
    fill(c, 0, range(6))
    c.truncate(0, 2)
    assert c.num_tokens(0) == 4 and len(c.block_table(0)) == 1
    fill(c, 0, [4, 5])
    assert c.positions(0).tolist() == list(range(6))


def test_free_sequence_returns_blocks():
    c = make_cache()
    c.add_sequence(0)
    fill(c, 0, range(9))
    c.free_sequence(0)
    assert c.allocator.num_free == 16
