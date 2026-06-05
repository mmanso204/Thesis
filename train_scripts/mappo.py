"""Strict MAPPO algorithm (Yu et al. 2022).

Subclasses SB3's PPO, overriding only `train()` to apply a separate clipped
PPO loss per agent, then average — exactly as described in the paper.

  Standard PPO:  L = clip(Π_i r_i  × A)        (product of ratios)
  Strict MAPPO:  L = (1/N) Σ_i clip(r_i × A_i)  (sum of per-agent losses)

For cooperative MARL with shared reward, A_i = A for all agents, so the
difference is purely in how clipping is applied: per-agent vs joint.
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from mappo_policy import N_AGENTS


class MAPPO(PPO):
    """Strict MAPPO: per-agent clipped PPO loss with centralized critic.

    Drop-in replacement for PPO — same constructor signature, same checkpointing.
    Requires MAPPOPolicy (which has get_per_agent_log_probs).
    """

    def train(self) -> None:
        self.policy.set_training_mode(True)

        clip_range    = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None else None
        )

        buf     = self.rollout_buffer
        n_total = buf.buffer_size * buf.n_envs

        # ── flatten (buffer_size, n_envs, ...) → (n_total, ...) ─────────────────
        # SB3's buf.get() does this via swap_and_flatten; we call _get_samples
        # directly so we must trigger it manually before any indexing.
        if not buf.generator_ready:
            for _t in ["observations", "actions", "values", "log_probs",
                       "advantages", "returns"]:
                buf.__dict__[_t] = buf.swap_and_flatten(buf.__dict__[_t])
            buf.generator_ready = True

        # ── per-agent OLD log-probs — computed BEFORE any gradient updates ──────
        with th.no_grad():
            obs_flat = th.tensor(
                buf.observations.reshape(n_total, -1),
                device=self.device, dtype=th.float32,
            )
            act_flat = th.tensor(
                buf.actions.reshape(n_total, -1).astype(np.int64),
                device=self.device, dtype=th.long,
            )
            old_pa_lps = self.policy.get_per_agent_log_probs(obs_flat, act_flat)
            # shape: (n_total, N_AGENTS)

        # ── training epochs ───────────────────────────────────────────────────
        pg_losses, vf_losses, ent_losses, clip_fracs, kl_divs = [], [], [], [], []
        continue_training = True

        for _ in range(self.n_epochs):
            if not continue_training:
                break
            indices = np.random.permutation(n_total)

            for start in range(0, n_total, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                if len(batch_idx) < self.batch_size:
                    continue

                data    = buf._get_samples(batch_idx)
                actions = data.actions  # keep float — SB3 MultiDiscrete expects float

                # centralized critic value + joint entropy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    data.observations, actions
                )
                values = values.flatten()

                # current per-agent log-probs (new policy) — long() needed for Categorical
                new_pa_lps   = self.policy.get_per_agent_log_probs(data.observations, actions.long())
                old_pa_batch = old_pa_lps[batch_idx]  # (batch, N_AGENTS)

                # normalize advantages
                adv = data.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # ── strict MAPPO: independent clipped loss per agent ──────────
                pg_loss = th.zeros(1, device=self.device)
                for i in range(N_AGENTS):
                    r_i = th.exp(new_pa_lps[:, i] - old_pa_batch[:, i])
                    pg_loss += -th.min(
                        adv * r_i,
                        adv * th.clamp(r_i, 1 - clip_range, 1 + clip_range),
                    ).mean()
                    clip_fracs.append(((r_i - 1).abs() > clip_range).float().mean().item())
                pg_loss = pg_loss / N_AGENTS

                # approximate KL for early stopping / logging
                with th.no_grad():
                    log_ratio = log_prob - data.old_log_prob
                    kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                    kl_divs.append(kl)
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # value loss (matches SB3 default with optional clipping)
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = data.old_values + th.clamp(
                        values - data.old_values, -clip_range_vf, clip_range_vf
                    )
                vf_loss  = F.mse_loss(data.returns, values_pred)
                ent_loss = -th.mean(entropy) if entropy is not None else th.mean(-log_prob)

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

                self.policy.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs

        explained_var = explained_variance(
            buf.values.flatten(), buf.returns.flatten()
        )
        self.logger.record("train/entropy_loss",         np.mean(ent_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss",           np.mean(vf_losses))
        self.logger.record("train/approx_kl",            np.mean(kl_divs))
        self.logger.record("train/clip_fraction",        np.mean(clip_fracs))
        self.logger.record("train/loss",                 loss.item())
        self.logger.record("train/explained_variance",   explained_var)
        self.logger.record("train/n_updates",            self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range",           clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf",   clip_range_vf)
