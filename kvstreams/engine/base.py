"""Shared multi-turn rollout loop.

Subclasses differ only in two hooks:

* :meth:`BaseEngine._positions` – which RoPE positions new tokens get.
* :meth:`BaseEngine._evict` – how a compaction plan is executed.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from ..chat import ChatTemplate
from ..compaction.events import CompactionEvent, CompactionLog
from ..compaction.policies import CompactionPlan, CompactionPolicy
from ..compaction.segments import Segment
from ..config import CacheConfig, RolloutConfig, SamplingConfig
from ..envs.base import Env
from ..model import TinyLM
from ..paged_cache import PagedKVCache
from ..tokenizer import CharTokenizer
from ..utils.spans import Span
from .rollout import Rollout, RolloutStats


class _Run:
    """Mutable state of a single rollout."""

    def __init__(self, seq_id: int, generator: Optional[torch.Generator]) -> None:
        self.seq_id = seq_id
        self.gen = generator
        self.tokens: list[int] = []
        self.loss_mask: list[bool] = []
        self.inf_logprobs: list[float] = []
        self.full_logprobs: dict[int, torch.Tensor] = {}
        self.segments: list[Segment] = []
        self.log = CompactionLog()
        self.stats = RolloutStats()
        self.completions: list[str] = []
        # Stream indices whose KV is currently live, in cache order.
        self.context: list[int] = []
        # Stream indices of an uncommitted partial block awaiting recomputation.
        self.pending: list[int] = []


class BaseEngine:
    name = "base"

    def __init__(
        self,
        model: TinyLM,
        tokenizer: CharTokenizer,
        policy: CompactionPolicy,
        cache_cfg: Optional[CacheConfig] = None,
        rollout_cfg: Optional[RolloutConfig] = None,
        sampling: Optional[SamplingConfig] = None,
    ) -> None:
        self.model = model
        self.tok = tokenizer
        self.template = ChatTemplate(tokenizer)
        self.policy = policy
        self.cache_cfg = cache_cfg or CacheConfig()
        self.rollout_cfg = rollout_cfg or RolloutConfig()
        self.sampling = sampling or SamplingConfig()
        self.cache = PagedKVCache.from_config(
            model.cfg, self.cache_cfg, device=model.device, dtype=model.dtype
        )
        self._next_seq = 0

    @property
    def block_size(self) -> int:
        return self.cache_cfg.block_size

    # ------------------------------------------------------------------ hooks
    def _positions(self, run: _Run, idx: list[int]) -> list[int]:
        raise NotImplementedError

    def _evict(self, run: _Run, spans: list[Span]) -> list[Span]:
        raise NotImplementedError

    # ------------------------------------------------------------------ public
    def rollout(
        self,
        env: Env,
        generator: Optional[torch.Generator] = None,
        teacher_forcing: bool = False,
    ) -> Rollout:
        """Play one episode of ``env``.

        With ``teacher_forcing`` the assistant turns are taken from ``env.oracle()``
        instead of being sampled (log-probs are still recorded). Compaction behaves
        identically in both modes.
        """
        was_training = self.model.training
        self.model.eval()
        seq_id = self._next_seq
        self._next_seq += 1
        self.cache.add_sequence(seq_id)
        run = _Run(seq_id, generator)
        t0 = time.perf_counter()
        reward = 0.0
        try:
            with torch.no_grad():
                start = env.reset()
                self._run_prompt(run, start.system)
                user = start.user
                for _ in range(self.rollout_cfg.max_turns):
                    prefix = self.template.user_turn(user)
                    forced = env.oracle() if teacher_forcing else None
                    self._maybe_compact(
                        run, env, len(prefix), self._reserve(forced), teacher_forcing
                    )
                    text = self._run_turn(run, prefix, "turn", forced)
                    run.completions.append(text)
                    step = env.step(text)
                    if step.done:
                        reward = step.reward
                        break
                    user = step.user
        finally:
            self.cache.free_sequence(seq_id)
            self.model.train(was_training)
        run.stats.wall_time = time.perf_counter() - t0
        return Rollout(
            engine=self.name,
            tokens=run.tokens,
            loss_mask=run.loss_mask,
            inf_logprobs=run.inf_logprobs,
            segments=run.segments,
            log=run.log,
            stats=run.stats,
            reward=reward,
            completions=run.completions,
            block_size=self.block_size,
            temperature=self.sampling.temperature,
            full_logprobs=run.full_logprobs,
            env_seed=getattr(env, "seed", None),
        )

    # ------------------------------------------------------------------ internals
    def _reserve(self, forced: Optional[str]) -> int:
        gen = (
            len(self.tok.encode(forced)) + 1
            if forced is not None
            else self.sampling.max_new_tokens + 1
        )
        pad = self.block_size - 1 if self.cache_cfg.pad_to_block else 0
        return gen + pad

    def _live(self, run: _Run) -> int:
        return self.cache.num_tokens(run.seq_id) + len(run.pending)

    def _append(self, run: _Run, toks: list[int], loss: bool = False) -> list[int]:
        start = len(run.tokens)
        run.tokens.extend(int(t) for t in toks)
        run.loss_mask.extend([loss] * len(toks))
        run.inf_logprobs.extend([0.0] * len(toks))
        return list(range(start, len(run.tokens)))

    def _forward(self, run: _Run, idx: list[int], kind: str) -> torch.Tensor:
        dev = self.model.device
        toks = torch.tensor([run.tokens[i] for i in idx], dtype=torch.long, device=dev)
        pos = torch.tensor(self._positions(run, idx), dtype=torch.long, device=dev)
        logits = self.model.forward_cached(toks, pos, self.cache, run.seq_id)
        run.context.extend(idx)
        st = run.stats
        st.tokens_forwarded += len(idx)
        if kind == "decode":
            st.decode_tokens += len(idx)
        elif kind == "reprefill":
            st.reprefill_tokens += len(idx)
        else:
            st.prefill_tokens += len(idx)
        st.peak_live_tokens = max(st.peak_live_tokens, self.cache.num_tokens(run.seq_id))
        return logits

    def _logp(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(logits / self.sampling.temperature, dim=-1)

    def _record(self, run: _Run, idx: int, logp_row: torch.Tensor) -> None:
        run.inf_logprobs[idx] = float(logp_row[run.tokens[idx]])
        if self.rollout_cfg.record_full_logprobs:
            run.full_logprobs[idx] = logp_row.detach().cpu().clone()

    def _sample(self, run: _Run, logits: torch.Tensor) -> tuple[int, torch.Tensor]:
        logp = self._logp(logits)
        if self.sampling.greedy:
            return int(logp.argmax()), logp
        tok = torch.multinomial(logp.exp(), 1, generator=run.gen)
        return int(tok), logp

    def _pad(self, run: _Run) -> None:
        if not self.cache_cfg.pad_to_block:
            return
        pads = self.template.padding(len(run.tokens), self.block_size)
        if pads:
            idx = self._append(run, pads)
            self._forward(run, idx, "prefill")
            run.stats.padding_tokens += len(pads)

    def _uncommit_tail(self, run: _Run) -> None:
        """Emulate vLLM: a trailing partial block is not committed and is recomputed later."""
        if self.cache_cfg.commit_partial_blocks:
            return
        tail = self.cache.num_tokens(run.seq_id) % self.block_size
        if tail:
            self.cache.truncate(run.seq_id, tail)
            run.pending = run.context[-tail:]
            del run.context[-tail:]

    def _run_prompt(self, run: _Run, system: str) -> None:
        idx = self._append(run, self.template.system(system))
        self._forward(run, idx, "prefill")
        self._pad(run)
        run.segments.append(Segment(0, "prompt", 0, len(run.tokens), protected=True))
        self._uncommit_tail(run)

    def _run_turn(self, run: _Run, prefix: list[int], kind: str, forced: Optional[str]) -> str:
        seg_start = len(run.tokens)
        new_idx = self._append(run, prefix)
        pending, run.pending = run.pending, []
        if pending:
            run.stats.recomputed_tokens += len(pending)
            if any(ev.time > pending[0] and ev.evicted for ev in run.log):
                run.stats.stale_recomputed_tokens += len(pending)
        logits = self._forward(run, pending + new_idx, "prefill")[-1]

        end = self.tok.end_id
        generated: list[int] = []
        if forced is not None:
            target = self.tok.encode(forced) + [end]
            idx = self._append(run, target, loss=True)
            chunk = self._forward(run, idx, "prefill")
            logp = self._logp(torch.cat([logits[None], chunk[:-1]], dim=0))
            for k, i in enumerate(idx):
                self._record(run, i, logp[k])
            generated = target
        else:
            for _ in range(self.sampling.max_new_tokens):
                tok, logp = self._sample(run, logits)
                (i,) = self._append(run, [tok], loss=True)
                self._record(run, i, logp)
                generated.append(tok)
                logits = self._forward(run, [i], "decode")[-1]
                if tok == end:
                    break
            else:
                # Length cap hit: close the message with a forced (untrained) end tag.
                idx = self._append(run, [end])
                self._forward(run, idx, "decode")

        self._pad(run)
        run.segments.append(Segment(len(run.segments), kind, seg_start, len(run.tokens)))
        self._uncommit_tail(run)
        return self.tok.decode(generated)

    def _maybe_compact(
        self, run: _Run, env: Env, incoming: int, reserve: int, teacher_forcing: bool
    ) -> None:
        budget = self.rollout_cfg.context_budget
        required = self._live(run) + incoming + reserve
        if required <= budget:
            return
        plan = self.policy.plan(run.segments, self._live(run), budget, required)
        if plan.summary_request is not None:
            forced = env.summary_oracle() if teacher_forcing else None
            prefix = self.template.user_turn(plan.summary_request)
            run.completions.append(self._run_turn(run, prefix, "summary", forced))
        if plan.spans:
            self._compact(run, plan)
        if self._live(run) + incoming + reserve > budget:
            run.stats.budget_overflows += 1

    def _compact(self, run: _Run, plan: CompactionPlan) -> None:
        pending = set(run.pending)
        if any(s <= p < e for p in pending for s, e in plan.spans):
            raise RuntimeError(
                "compaction policy tried to evict uncommitted tokens of the latest turn"
            )
        t = len(run.tokens)
        live_before = self._live(run)
        evicted = self._evict(run, plan.spans)
        if not evicted:
            return
        event = CompactionEvent(
            time=t,
            evicted=evicted,
            requested=list(plan.spans),
            live_before=live_before,
            live_after=self._live(run),
            policy=self.policy.name,
            segments=list(plan.segments),
        )
        run.log.append(event)
        run.stats.compactions += 1
        run.stats.evicted_tokens += event.num_evicted
        for seg in run.segments:
            if not seg.alive:
                continue
            covered = sum(max(0, min(seg.end, e) - max(seg.start, s)) for s, e in evicted)
            if covered >= seg.length:
                seg.evicted_at = t
                seg.partially_evicted = False
            elif covered:
                seg.partially_evicted = True
