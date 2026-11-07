"""
Retrain the Hybrid Attention Router using the real-latency LUT in the reward.

This is a self-contained training script that does NOT depend on the original
repo layout. It re-implements the minimal pieces needed:

  - small HybridTransformer (same architecture)
  - PPO agent (same hyperparameters)
  - AttentionRoutingEnv with use_real_latency=True
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router_integration.latency_lut import LatencyLUT
from router_integration.routing_env import AttentionRoutingEnv
from router_integration.triton_layers import (
    TritonFullAttention,
    TritonSlidingWindowAttention,
    TritonHybridAttentionBlock,
)


# ---------------------------------------------------------------------------
# Minimal model (matches original architecture)
# ---------------------------------------------------------------------------
class LinearAttention(nn.Module):
    def __init__(self, n_head, n_embd, feature_dim=64):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.feature_proj = nn.Linear(self.head_dim, feature_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K, V = Q, Q
        Q_feat = F.elu(self.feature_proj(Q)) + 1
        K_feat = F.elu(self.feature_proj(K)) + 1
        KV = K_feat.transpose(-2, -1) @ V
        out = Q_feat @ KV
        K_sum = K_feat.sum(dim=2, keepdim=True)
        normalizer = (Q_feat @ K_sum.transpose(-2, -1)) + 1e-6
        out = out / normalizer
        return out.transpose(1, 2).contiguous().view(B, T, C)


class HybridTransformer(nn.Module):
    def __init__(
        self,
        n_layer=6,
        n_embd=128,
        n_head=4,
        block_size=256,
        vocab_size=50257,
        window_size=64,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.emb_dropout = nn.Dropout(0.0)

        class Cfg:
            pass

        cfg = Cfg()
        cfg.n_head = n_head
        cfg.n_embd = n_embd
        cfg.dropout = 0.0
        cfg.block_size = block_size
        cfg.attention_variants = {"window_size": window_size}

        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            linear = LinearAttention(n_head, n_embd, 64)
            self.blocks.append(TritonHybridAttentionBlock(i, cfg, linear))

        self.ln_final = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, actions=None):
        B, T = idx.shape
        if actions is None:
            actions = [0] * self.n_layer
        x = self.emb_dropout(
            self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        )
        for i, block in enumerate(self.blocks):
            x = block(x, action=actions[i] if i < len(actions) else 0)
        x = self.ln_final(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits


# ---------------------------------------------------------------------------
# Minimal PPO (same structure as original)
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=10, action_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class ValueNet(nn.Module):
    def __init__(self, obs_dim=10, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class SimplePPO:
    def __init__(
        self, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01
    ):
        self.device = device
        self.policy = PolicyNet().to(device)
        self.value = ValueNet().to(device)
        self.opt_p = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_v = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.ent_coef = ent_coef

    def select(self, obs_np):
        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(obs)
            value = self.value(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def update(self, traj, epochs=3, batch_size=32):
        obs = torch.from_numpy(np.array(traj["obs"])).float().to(self.device)
        acts = torch.from_numpy(np.array(traj["actions"])).long().to(self.device)
        old_lp = torch.from_numpy(np.array(traj["log_probs"])).float().to(self.device)
        rets = torch.from_numpy(np.array(traj["returns"])).float().to(self.device)
        advs = torch.from_numpy(np.array(traj["advantages"])).float().to(self.device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(epochs):
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), batch_size):
                b = idx[start : start + batch_size]
                logits = self.policy(obs[b])
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(acts[b])
                ratio = torch.exp(new_lp - old_lp[b])
                surr1 = ratio * advs[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advs[b]
                p_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.ent_coef * dist.entropy().mean()
                )
                v_loss = F.mse_loss(self.value(obs[b]).squeeze(-1), rets[b])
                self.opt_p.zero_grad()
                p_loss.backward()
                self.opt_p.step()
                self.opt_v.zero_grad()
                v_loss.backward()
                self.opt_v.step()

    def save(self, path):
        torch.save(
            {"policy": self.policy.state_dict(), "value": self.value.state_dict()}, path
        )


def collect(agent, env, num_steps):
    obs, _ = env.reset()
    traj = {
        k: [] for k in ["obs", "actions", "rewards", "values", "log_probs", "dones"]
    }
    for _ in range(num_steps):
        a, lp, v = agent.select(obs)
        next_obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        traj["obs"].append(obs)
        traj["actions"].append(a)
        traj["rewards"].append(r)
        traj["values"].append(v)
        traj["log_probs"].append(lp)
        traj["dones"].append(done)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    # GAE
    values = np.array(traj["values"] + [0.0])
    rewards = np.array(traj["rewards"])
    dones = np.array(traj["dones"])
    T = len(rewards)
    adv = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + agent.gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + agent.gamma * agent.gae_lambda * (1 - dones[t]) * gae
        adv[t] = gae
    traj["advantages"] = adv.tolist()
    traj["returns"] = (adv + np.array(traj["values"])).tolist()
    return traj


def make_synthetic_loader(B=4, T=256, vocab=50257, n_batches=200):
    """Infinite-ish synthetic batches so we do not need WikiText on Colab."""
    while True:
        for _ in range(n_batches):
            idx = torch.randint(0, vocab, (B, T))
            yield idx, idx  # targets = inputs (language-model style)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut", default="results/latency_lut.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out-dir", default="results/real_latency_runs")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    lut = LatencyLUT(args.lut)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class Cfg:
        n_layer = 6
        n_embd = 128
        n_head = 4
        block_size = 256
        attention_variants = {"window_size": 64}
        reward = {"alpha": 1.0, "beta": 0.1, "gamma": 0.05}
        rl = {"max_steps_per_episode": 256}

    for seed in args.seeds:
        print(f"\n===== Seed {seed} =====")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = HybridTransformer().cuda().eval()
        for p in model.parameters():
            p.requires_grad_(False)

        loader = make_synthetic_loader()
        env = AttentionRoutingEnv(
            model=model,
            dataloader=loader,
            config=Cfg(),
            device=torch.device("cuda"),
            latency_lut=lut,
            use_real_latency=True,
        )
        agent = SimplePPO(torch.device("cuda"))

        history = []
        for epoch in range(1, args.epochs + 1):
            traj = collect(agent, env, args.steps_per_epoch)
            agent.update(traj)
            summary = env.get_episode_summary()
            avg_r = float(np.mean(traj["rewards"]))
            history.append({"epoch": epoch, "avg_reward": avg_r, **summary})
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:3d}  avg_reward={avg_r:.4f}  actions={summary.get('action_counts')}"
                )

        ckpt = out_dir / f"router_seed{seed}.pt"
        agent.save(ckpt)
        with open(out_dir / f"history_seed{seed}.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved {ckpt}")


if __name__ == "__main__":
    main()
