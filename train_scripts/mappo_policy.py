"""MAPPO policy for cooperative PPO with parameter sharing.

Decentralized actor: shared MLP + shared action head applied independently to
each agent's local obs, giving 7 logits each.
Centralized critic: full concatenated global obs to a single value estimate.

Each agent's action depends only on its own observation (true MAPPO).
Used in place of "MlpPolicy" with no environment changes required.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Type

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

N_AGENTS  = 2
N_ACTIONS = 7   # Discrete(7) per agent


class MAPPOExtractor(nn.Module):
    """
    Shared-weight actor applied per-agent + centralized critic.

    The observation layout is:
        [agent_0_obs | ... | agent_{N-1}_obs | global_state]
         <-- N_AGENTS * agent_obs_dim -----> <-- n_global ->

    Actor  : receives only each agent's own slice (decentralized execution).
    Critic : receives the full observation including global state (centralized).

    forward_actor returns (b, N_AGENTS * N_ACTIONS)
    forward_critic returns (b, latent_vf)
    """

    def __init__(
        self,
        obs_dim:      int,
        n_global:     int,
        net_arch:     List[int],
        activation_fn: type = nn.Tanh,
    ):
        super().__init__()
        actor_total = obs_dim - n_global          # dims belonging to actors
        assert actor_total % N_AGENTS == 0, (
            f"actor portion {actor_total} must be divisible by N_AGENTS={N_AGENTS}"
        )
        self._actor_total = actor_total
        self._agent_obs   = actor_total // N_AGENTS

        # Shared actor backbone (same weights applied to each agent's slice)
        layers: list = []
        in_dim = self._agent_obs
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), activation_fn()]
            in_dim = h
        self.actor_mlp  = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, N_ACTIONS)
        self.latent_dim_pi = N_AGENTS * N_ACTIONS

        # Centralized critic sees full obs (actor obs + global state)
        layers = []
        in_dim = obs_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), activation_fn()]
            in_dim = h
        self.critic_mlp    = nn.Sequential(*layers)
        self.latent_dim_vf = in_dim

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        b = features.shape[0]
        # Strip global state; actor sees only the per-agent portions
        actor_feats = features[:, :self._actor_total]                 # (b, N_AGENTS*agent_obs)
        obs_split   = actor_feats.reshape(b, N_AGENTS, self._agent_obs)
        flat        = obs_split.reshape(b * N_AGENTS, self._agent_obs)
        latent      = self.actor_mlp(flat)
        logits      = self.action_head(latent)
        return logits.reshape(b, N_AGENTS * N_ACTIONS)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        # Full observation including global state for better value estimates
        return self.critic_mlp(features)


class MAPPOPolicy(ActorCriticPolicy):
    """
    Usage in train_ppo.py:

        from mappo_policy import MAPPOPolicy
        model = PPO(MAPPOPolicy, vec_env, ..., policy_kwargs=dict(net_arch=[256, 256]))
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space:      spaces.Space,
        lr_schedule:       Schedule,
        net_arch:          Optional[List[int]] = None,
        activation_fn:     Type[nn.Module]     = nn.Tanh,
        n_global:          int                 = 0,
        **kwargs,
    ):
        self._mappo_arch     = net_arch or [256, 256]
        self._mappo_actfn    = activation_fn
        self._mappo_n_global = n_global   # global state dims appended after per-agent obs
        super().__init__(
            observation_space, action_space, lr_schedule,
            net_arch=[], activation_fn=activation_fn,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MAPPOExtractor(
            obs_dim       = self.features_dim,
            n_global      = self._mappo_n_global,
            net_arch      = self._mappo_arch,
            activation_fn = self._mappo_actfn,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        # Replace SB3's auto-built action_net with Identity; the extractor
        # already outputs the correct logits directly.
        self.action_net = nn.Identity()

    def get_per_agent_log_probs(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """Returns (batch, N_AGENTS) per-agent log probs for the given obs+actions.
        Used by the strict MAPPO training loop for per-agent PPO losses.
        """
        dist = self.get_distribution(obs)
        # dist.distribution is a list of N_AGENTS Categorical distributions
        return th.stack(
            [dist.distribution[i].log_prob(actions[:, i]) for i in range(N_AGENTS)],
            dim=1,
        )  # (batch, N_AGENTS)
