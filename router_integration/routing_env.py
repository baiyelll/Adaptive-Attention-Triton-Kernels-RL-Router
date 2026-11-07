"""
Minimal AttentionRoutingEnv reconstructed from train_router.py usage.

Because the original src/env/routing_env.py was never committed to GitHub,
this implementation provides the same public interface so PPO training and
the real-latency path can run.

Interface used by train_router.py:
  - env.reset() -> (obs, info)
  - env.step(action) -> (obs, reward, terminated, truncated, info)
  - env.observation_space.shape
  - env.action_space.n
  - env.max_steps_per_episode
  - env.get_episode_summary()
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from router_integration.latency_lut import LatencyLUT, VARIANT_NAMES
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# metrics helpers (copied logic so we do not depend on the external package layout)
def _estimate_flops(seq_len, hidden_dim, n_heads, attention_type, window_size=64):
    d_k = hidden_dim // n_heads
    if attention_type == 0:
        return float(2 * seq_len * seq_len * d_k)
    if attention_type == 1:
        return float(2 * seq_len * window_size * d_k)
    return float(seq_len * hidden_dim * hidden_dim)


def _compute_routing_reward(
    loss_delta, flops_baseline, flops_current, latency, alpha=1.0, beta=0.1, gamma=0.05
):
    flops_saving = (flops_baseline - flops_current) / max(flops_baseline, 1e-6)
    latency_bonus = max(0.0, (100.0 - latency) / 100.0) if latency < 100 else 0.0
    return float(-alpha * loss_delta - beta * flops_saving + gamma * latency_bonus)


def _compute_compact_observation(
    layer_idx, seq_len, hidden_states, perplexity_delta, max_seq_len=512
):
    h = hidden_states.detach().float().reshape(-1, hidden_states.shape[-1])
    h_mean = h.mean(dim=0).mean().item()
    h_std = h.std(dim=0).mean().item()
    h_norm = (h**2).mean(dim=1).sqrt().mean().item()
    return np.array(
        [
            layer_idx / max(1, 6),
            seq_len / max_seq_len,
            np.clip(h_mean, -10, 10),
            np.clip(h_std, 0, 10),
            h_norm,
            perplexity_delta,
            np.sin(layer_idx),
            np.cos(seq_len / max_seq_len),
            np.tanh(h_mean),
            np.tanh(h_std),
        ],
        dtype=np.float32,
    )


class _Space:
    def __init__(self, n=None, shape=None):
        self.n = n
        self.shape = shape


class AttentionRoutingEnv:
    """
    Gymnasium-style env that routes one layer at a time through a frozen transformer.

    When `latency_lut` is provided, the reward uses real measured latency.
    Otherwise a cheap FLOPs-derived proxy is used (same spirit as the original
    simulated latency).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: Iterator,
        config,
        device: torch.device,
        latency_lut: Optional[LatencyLUT] = None,
        use_real_latency: bool = False,
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.latency_lut = latency_lut
        self.use_real_latency = use_real_latency and latency_lut is not None

        self.n_layer = int(getattr(config, "n_layer", 6))
        self.n_embd = int(getattr(config, "n_embd", 128))
        self.n_head = int(getattr(config, "n_head", 4))
        self.block_size = int(getattr(config, "block_size", 256))
        av = getattr(config, "attention_variants", {}) or {}
        if isinstance(av, dict):
            self.window_size = int(av.get("window_size", 64))
        else:
            self.window_size = 64

        reward_cfg = getattr(config, "reward", {}) or {}
        if isinstance(reward_cfg, dict):
            self.alpha = float(reward_cfg.get("alpha", 1.0))
            self.beta = float(reward_cfg.get("beta", 0.1))
            self.gamma = float(reward_cfg.get("gamma", 0.05))
        else:
            self.alpha, self.beta, self.gamma = 1.0, 0.1, 0.05

        rl_cfg = getattr(config, "rl", {}) or {}
        if isinstance(rl_cfg, dict):
            self.max_steps_per_episode = int(rl_cfg.get("max_steps_per_episode", 256))
        else:
            self.max_steps_per_episode = 256

        self.action_space = _Space(n=3)
        self.observation_space = _Space(shape=(10,))

        self._batch = None
        self._targets = None
        self._baseline_loss = None
        self._layer_idx = 0
        self._actions: List[int] = []
        self._episode_rewards: List[float] = []
        self._episode_actions: List[int] = []
        self._step_count = 0
        self._hidden = None

    def _next_batch(self):
        try:
            batch = next(self.dataloader)
        except StopIteration:
            # caller should re-create the iterator; for robustness, raise
            raise RuntimeError("Dataloader exhausted — recreate iterator")
        if isinstance(batch, (list, tuple)):
            idx, targets = batch[0], batch[1] if len(batch) > 1 else batch[0]
        else:
            idx = targets = batch
        self._batch = idx.to(self.device)
        self._targets = targets.to(self.device)
        B, T = self._batch.shape
        # baseline = all full attention
        with torch.no_grad():
            loss, _ = self.model(
                self._batch, targets=self._targets, actions=[0] * self.n_layer
            )
        self._baseline_loss = float(loss.item())
        self._B, self._T = B, T
        # initial hidden approx: token embeddings
        with torch.no_grad():
            tok = self.model.token_emb(self._batch)
            pos = self.model.pos_emb(torch.arange(T, device=self.device))
            self._hidden = tok + pos

    def reset(self, *, seed=None, options=None):
        self._next_batch()
        self._layer_idx = 0
        self._actions = []
        self._episode_rewards = []
        self._episode_actions = []
        self._step_count = 0
        obs = _compute_compact_observation(
            0, self._T, self._hidden, 0.0, max_seq_len=self.block_size
        )
        return obs, {}

    def step(self, action: int):
        action = int(action)
        self._actions.append(action)
        self._episode_actions.append(action)
        self._step_count += 1

        # Run the model with the actions chosen so far (pad remaining layers with Full)
        actions_full = list(self._actions) + [0] * (self.n_layer - len(self._actions))
        with torch.no_grad():
            loss, logits = self.model(
                self._batch, targets=self._targets, actions=actions_full
            )
        current_loss = float(loss.item())
        loss_delta = current_loss - self._baseline_loss

        flops_base = (
            _estimate_flops(self._T, self.n_embd, self.n_head, 0, self.window_size)
            * self.n_layer
        )
        flops_cur = sum(
            _estimate_flops(self._T, self.n_embd, self.n_head, a, self.window_size)
            for a in actions_full
        )

        if self.use_real_latency:
            latency = self.latency_lut.config_latency(actions_full, self._B, self._T)
        else:
            # Cheap proxy: scale FLOPs into a fake "ms" so the original reward
            # formula still receives a number. At N=256 this is ~1-5 ms range.
            latency = flops_cur / max(flops_base, 1) * 5.0

        reward = _compute_routing_reward(
            loss_delta,
            flops_base,
            flops_cur,
            latency,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        self._episode_rewards.append(reward)

        self._layer_idx = len(self._actions) % self.n_layer
        # Advance "hidden" roughly via a full forward (already done); keep last logits stats
        with torch.no_grad():
            # use logits as a stand-in for hidden-state statistics
            self._hidden = logits.detach()

        obs = _compute_compact_observation(
            self._layer_idx,
            self._T,
            self._hidden,
            loss_delta,
            max_seq_len=self.block_size,
        )

        # Episode ends after we have chosen an action for every layer once
        # (or max_steps). This matches a simple per-layer routing trajectory.
        terminated = len(self._actions) >= self.n_layer
        truncated = self._step_count >= self.max_steps_per_episode
        info = {
            "loss": current_loss,
            "loss_delta": loss_delta,
            "latency_ms": latency,
            "actions": list(actions_full),
            "flops_current": flops_cur,
        }
        return obs, reward, terminated, truncated, info

    def get_episode_summary(self) -> Dict[str, Any]:
        if not self._episode_rewards:
            return {}
        counts = np.bincount(self._episode_actions, minlength=3)
        return {
            "episode_reward": float(np.sum(self._episode_rewards)),
            "avg_step_reward": float(np.mean(self._episode_rewards)),
            "action_counts": counts.tolist(),
            "n_steps": len(self._episode_rewards),
        }
