"""Non-invasive capture of FVLMoE (LIMoEBlock / MoeLayer) per-token routing.

OFF by default. When ``ENABLED`` is False the instrumentation branch in
``MoeLayer._scatter_to_experts`` is not taken at trace time, so the compiled graph is
identical to the original and inference is byte-identical. When ``ENABLED`` is True
(set by scripts/eval/eval_expert_routing.py *before* the first ``policy.infer`` so the
jitted graph includes the callback), each MoE call ships the per-token top-1 expert id
and router probability to a host-side buffer via ``jax.debug.callback`` -- a pure side
effect that does not feed back into the model outputs.

What is read (see flaxformer routing.RouterIndices for TokensChooseScatterRouter):
  expert id    = router_indices.dispatch_indices[..., 0, 0]   # top-1 preferred expert
  router prob  = router_indices.combine_weights[..., 0]        # top-1 softmax gate prob

Token order is preserved from the MoE input, so for batch=1 the FVLMoE force token --
which pi0_force appends LAST in concat([prefix_out, force_tokens]) -- is the last real
token, i.e. column ``seq_length - 1`` of each captured record.
"""

from __future__ import annotations

import contextlib
from functools import partial

import numpy as np

# Module-level gate, read at trace time. Toggle via enable()/disable(). Default off.
ENABLED = False

# Host-side buffer of captured records (one per MoE call; the diffusion sampler calls the
# MoE once per step, so a single infer appends num_steps identical records).
_BUFFER: list[dict] = []


def enable() -> None:
    global ENABLED
    ENABLED = True


def disable() -> None:
    global ENABLED
    ENABLED = False


def clear() -> None:
    _BUFFER.clear()


@contextlib.contextmanager
def frame():
    """Clear the buffer, run one inference inside the ``with``, and expose the records.

    Does NOT toggle ENABLED: call enable() once before the first infer so the jitted graph
    is traced with the callback present, then use this per frame to collect records.
    """
    _BUFFER.clear()
    yield _BUFFER


def _record(expert_flat, prob_flat, *, n_real: int, batch_size: int, seq_length: int) -> None:
    """Host callback: keep the real (un-padded) tokens, reshaped to [batch, seq]."""
    e = np.asarray(expert_flat).reshape(-1)[:n_real].reshape(batch_size, seq_length)
    p = np.asarray(prob_flat).reshape(-1)[:n_real].reshape(batch_size, seq_length)
    _BUFFER.append({"expert": e, "prob": p, "batch_size": batch_size, "seq_length": seq_length})


def emit(dispatch_indices, combine_weights, batch_size: int, seq_length: int) -> None:
    """Ship per-token top-1 routing to the host. Called from MoeLayer only when ENABLED.

    dispatch_indices: [num_groups, tokens_per_group, num_selected_experts, 2] (int32)
    combine_weights:  [num_groups, tokens_per_group, num_selected_experts] (float)
    """
    import jax

    expert = dispatch_indices[..., 0, 0].reshape(-1)  # top-1 expert id per padded token
    prob = combine_weights[..., 0].reshape(-1)        # top-1 gate prob per padded token
    n_real = int(batch_size) * int(seq_length)
    jax.debug.callback(
        partial(_record, n_real=n_real, batch_size=int(batch_size), seq_length=int(seq_length)),
        expert,
        prob,
    )
