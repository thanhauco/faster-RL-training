import pytest
import torch

from kvstreams.compaction import build_policy
from kvstreams.config import CacheConfig, ModelConfig, RolloutConfig, SamplingConfig
from kvstreams.engine import build_engine
from kvstreams.model import TinyLM
from kvstreams.tokenizer import CharTokenizer


@pytest.fixture(scope="session")
def tok():
    return CharTokenizer()


@pytest.fixture(scope="session")
def model64(tok):
    torch.manual_seed(0)
    return TinyLM(ModelConfig(vocab_size=tok.vocab_size, init_std=0.1)).double()


@pytest.fixture
def make_engine(model64, tok):
    def _make(
        engine="kvstream",
        policy="keep_recent",
        block_size=16,
        pad=True,
        commit=True,
        budget=160,
        max_new_tokens=12,
        record_full=False,
        **policy_kwargs,
    ):
        if not policy_kwargs and policy in ("keep_recent", "markovian", "summary"):
            policy_kwargs = {"keep_last": 2 if policy == "keep_recent" else 1}
        return build_engine(
            engine,
            model64,
            tok,
            build_policy(policy, **policy_kwargs),
            CacheConfig(
                block_size=block_size,
                num_blocks=1024,
                pad_to_block=pad,
                commit_partial_blocks=commit,
            ),
            RolloutConfig(context_budget=budget, record_full_logprobs=record_full),
            SamplingConfig(max_new_tokens=max_new_tokens),
        )

    return _make
