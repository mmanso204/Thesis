"""
Gymnasium wrapper around HouseEnv for Stable Baselines 3 PPO.

All N agents are treated as a single joint actor:
  observation = concat(state_agent_0, ..., state_agent_{N-1})  shape (N * state_dim,)
  action      = MultiDiscrete([7] * N)
  reward      = sum of all agents' shaped rewards per step  (cooperative)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLObjectProperty
from owlapy.owl_axiom import OWLObjectPropertyAssertionAxiom

from envs.environment_multi import HouseEnv, LabeledBall
from agent import Agent, NS, PROP_IS_CARRYING
from helper_functions.goals import Goal
from multigrid.core.world_object import Wall


_ROOMS = [
    "garden", "foyer", "living_room", "kitchen", "dining_room",
    "bedroom", "bathroom", "hallway", "study", "right_wing",
]

_BFS_MAX = 60


class HouseEnvSB3(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ontology_path: str,
        goal: Goal,
        num_agents: int = 3,
        max_steps: int = 6000,
        width: int = 32,
        height: int = 23,
        proximity_threshold: int = 5,
        render_mode: str = None,
        active_items: list = None,
    ):
        super().__init__()

        self.goal                = goal
        self.num_agents          = num_agents
        self.proximity_threshold = proximity_threshold
        self._n_items            = len(goal.target_items)
        self.active_items: set = set(active_items) if active_items is not None else None
        self.max_steps = max_steps

        self.env = HouseEnv(
            num_agents=num_agents,
            goal=goal,
            max_steps=4000,
            width=width,
            height=height,
            render_mode=render_mode,
        )

        self._ont = Agent(ontology_path, agent_id=0, verbose=False)

        n = self._n_items
        self._state_dim = 7 * 7 * 3 + 4 + 10 + 10 + n + 10 + 10 + 10 + n + n + 6 + 3

        self._global_state_dim = n * 4

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_agents * self._state_dim + self._global_state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([7] * num_agents)

        self._last_infos:             dict = {}
        self._ep_step_count:          int  = 0
        self._ep_pickup_count:        dict = {}
        self._ep_pickup_given:        set  = set()
        self._ep_complete_given:      bool = False
        self._ep_kitchen_bonus_given: set  = set()
        self._ep_idle_rooms_visited:  dict = {}
        self._prev_agent_pos:         dict[int, Optional[tuple]] = {}
        self._bfs_dist:               dict = {}
        self._item_bfs:               dict = {}

    def set_curriculum(self, active_items, max_steps):
        """Update the active curriculum stage. Called via VecEnv.env_method so it
        reaches every worker process when training with SubprocVecEnv."""
        self.active_items = set(active_items) if active_items is not None else None
        self.max_steps    = max_steps

    def _ont_features(self, current_room: Optional[str], keys_c, balls_d,
                      carrying: bool, agent_id: int) -> np.ndarray:
        ont       = self._ont
        known     = set(ont.get_known_rooms())
        observed  = set(ont._observed_goal_items) | set(ont._private_observed_items.get(agent_id, {}))
        delivered = ont._all_delivered

        current_oh    = [float(r == current_room)          for r in _ROOMS]
        known_bin     = [float(r in known)                 for r in _ROOMS]
        items_bin     = [float(it in observed)             for it in self.goal.target_items]
        target_oh     = [float(r == self.goal.target_room) for r in _ROOMS]
        delivered_bin = [float(it in delivered)            for it in self.goal.target_items]

        items_per_room = [
            float(any(
                ont._goal_item_room_prior.get(lbl) == room and lbl not in delivered
                for lbl in self.goal.target_items
            ))
            for room in _ROOMS
        ]

        accessible     = ont._compute_accessible_rooms(current_room)
        accessible_bin = [float(r in accessible) for r in _ROOMS]
        goal_reachable = float(self.goal.target_room in accessible)
        unreachable    = sum(
            1 for lbl in self.goal.target_items
            if lbl not in delivered
            and ont._goal_item_room_prior.get(lbl) is not None
            and ont._goal_item_room_prior[lbl] not in accessible
        ) / max(self._n_items, 1)

        others_carrying = {
            lbl for j, lbl in ont._agent_carrying.items()
            if j != agent_id and lbl is not None
        }
        others_bin = [float(it in others_carrying) for it in self.goal.target_items]

        scalars = [
            len(keys_c)  / 2.0,
            len(balls_d) / max(self._n_items, 1),
            float(carrying),
            goal_reachable,
            unreachable,
            ont._ont_reachable_cached,
        ]
        return np.array(
            current_oh + known_bin + items_bin + target_oh
            + items_per_room + accessible_bin + others_bin + delivered_bin + scalars,
            dtype=np.float32,
        )

    def _build_global_state(self) -> np.ndarray:
        """Global state for the centralized critic: exact item positions + status.
        Never split into per-agent slices — the actor never sees this."""
        delivered = self._ont._all_delivered
        carried   = {v for v in self._ont._agent_carrying.values() if v}
        feats: list = []
        for items_list in self.goal.room_items.values():
            for item in items_list:
                feats.append(float(item.label in delivered))
                feats.append(float(item.label in carried))
                feats.append(item.x / self.env.width)
                feats.append(item.y / self.env.height)
        return np.array(feats, dtype=np.float32)

    def _build_obs(self, obs_dict: dict) -> np.ndarray:
        parts = []
        for i in range(self.num_agents):
            pos      = self.env.agents[i].state.pos
            obs_data = self._ont.observations(obs_dict[i], pos, self.env, agent_id=i)
            keys_c   = self._last_infos.get(i, {}).get("agent_keys_collected", [])
            balls_d  = self._last_infos.get(i, {}).get("agent_balls_delivered", [])
            carrying = getattr(self.env.agents[i].state, "carrying", None) is not None
            img   = obs_dict[i]["image"].flatten().astype(np.float32) / 10.0
            dir_v = np.zeros(4, dtype=np.float32)
            dir_v[int(obs_dict[i].get("direction", 0)) % 4] = 1.0
            feats = self._ont_features(obs_data["Current_room"], keys_c, balls_d, carrying, i)

            px, py = int(pos[0]), int(pos[1])
            bfs_d  = self._bfs_dist.get((px, py), _BFS_MAX)
            pos_feats = np.array([
                px / self.env.width,
                py / self.env.height,
                min(bfs_d, _BFS_MAX) / _BFS_MAX,
            ], dtype=np.float32)

            parts.append(np.concatenate([img, dir_v, feats, pos_feats]))

        parts.append(self._build_global_state())
        return np.concatenate(parts, dtype=np.float32)

    def _update_ont(self, obs_dict: dict, infos: dict):
        for i in range(self.num_agents):
            pos      = self.env.agents[i].state.pos
            obs_data = self._ont.observations(obs_dict[i], pos, self.env, agent_id=i)
            self._ont.observations_to_ont(obs_data, self.env, agent_id=i)

        if self.proximity_threshold > 0:
            for _i in range(self.num_agents):
                for _j in range(_i + 1, self.num_agents):
                    pi = self.env.agents[_i].state.pos
                    pj = self.env.agents[_j].state.pos
                    if abs(pi[0]-pj[0]) + abs(pi[1]-pj[1]) <= self.proximity_threshold:
                        self._ont._share_knowledge(_i)
                        self._ont._share_knowledge(_j)

        for i in range(self.num_agents):
            now_c = getattr(self.env.agents[i].state, "carrying", None)
            g_lbl = getattr(now_c, "label", None)
            g_lbl = g_lbl if (g_lbl and g_lbl in self.goal.target_items) else None
            if self._ont._agent_carrying.get(i) != g_lbl:
                if self._ont._agent_carrying_axioms.get(i) is not None:
                    try:
                        self._ont.Agent_ont.remove_axiom(self._ont._agent_carrying_axioms[i])
                    except Exception:
                        pass
                    self._ont._agent_carrying_axioms[i] = None
                if g_lbl:
                    a_ind  = OWLNamedIndividual(IRI.create(NS, f"agent_{i}"))
                    it_ind = OWLNamedIndividual(IRI.create(NS, f"expected_{g_lbl.replace(' ', '_')}"))
                    axiom  = OWLObjectPropertyAssertionAxiom(
                        a_ind, OWLObjectProperty(IRI.create(NS, PROP_IS_CARRYING)), it_ind)
                    self._ont.Agent_ont.add_axiom(axiom)
                    self._ont._agent_carrying_axioms[i] = axiom
                self._ont._agent_carrying[i] = g_lbl

        _now_delivered = set().union(
            *(set(infos.get(i, {}).get("agent_balls_delivered", [])) for i in range(self.num_agents))
        )
        for lbl in _now_delivered - self._ont._all_delivered:
            self._ont.mark_item_delivered(lbl)
        self._ont._all_delivered = _now_delivered


    def _compute_bfs_dist(self) -> dict:
        """Single-source BFS from the kitchen centre tile.
        Using the centre (not all yellow tiles) forces agents to navigate deep
        into the kitchen rather than hovering at the entrance for carry reward."""
        from collections import deque
        yellow = [
            (x, y)
            for x in range(self.env.width)
            for y in range(self.env.height)
            if getattr(self.env.grid.get(x, y), "color", None) == "yellow"
        ]
        if not yellow:
            return {}
        cx = int(sorted(x for x, _ in yellow)[len(yellow) // 2])
        cy = int(sorted(y for _, y in yellow)[len(yellow) // 2])
        sources = [(cx, cy)]
        if not sources:
            return {}
        dist: dict = {}
        queue: deque = deque()
        for pos in sources:
            dist[pos] = 0
            queue.append(pos)
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in dist:
                    continue
                if not (0 <= nx < self.env.width and 0 <= ny < self.env.height):
                    continue
                if isinstance(self.env.grid.get(nx, ny), Wall):
                    continue
                dist[(nx, ny)] = dist[(x, y)] + 1
                queue.append((nx, ny))
        return dist

    def _compute_bfs_from(self, sx: int, sy: int) -> dict:
        """Single-source BFS from (sx, sy). Returns {(x,y): steps_from_source}."""
        from collections import deque
        dist: dict = {(sx, sy): 0}
        queue: deque = deque([(sx, sy)])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in dist:
                    continue
                if not (0 <= nx < self.env.width and 0 <= ny < self.env.height):
                    continue
                if isinstance(self.env.grid.get(nx, ny), Wall):
                    continue
                dist[(nx, ny)] = dist[(x, y)] + 1
                queue.append((nx, ny))
        return dist


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._ont.reset()
        self._ont.set_goal(self.goal)

        obs_dict, info = self.env.reset()
        self._last_infos = info

        for i in range(self.num_agents):
            self._ont._agent_carrying[i]        = None
            self._ont._agent_carrying_axioms[i] = None

        self._ep_step_count          = 0
        self._last_reach_sig         = None
        self._ep_pickup_count        = {}
        self._ep_pickup_given        = set()
        self._ep_complete_given      = False
        self._ep_kitchen_bonus_given = set()
        self._ep_idle_rooms_visited  = {i: set() for i in range(self.num_agents)}
        self._prev_agent_pos         = {
            i: (int(self.env.agents[i].state.pos[0]), int(self.env.agents[i].state.pos[1]))
            for i in range(self.num_agents)
        }
        self._bfs_dist = self._compute_bfs_dist()
        self._item_bfs = {
            item.label: self._compute_bfs_from(item.x, item.y)
            for items_list in self.goal.room_items.values()
            for item in items_list
        }
        self._ep_components = dict(
            expl=0.0, guide=0.0, pickup=0.0, pen=0.0,
            delivery=0.0, obs=0.0, lava=0.0, completion=0.0,
        )

        self._update_ont(obs_dict, info)
        obs = self._build_obs(obs_dict)
        return obs, {}

    def step(self, action):
        actions = {i: int(action[i]) for i in range(self.num_agents)}
        self._ep_step_count += 1

        prev_rooms = {
            i: set(self._ont._known_rooms_by_agent.get(i, set()))
            for i in range(self.num_agents)
        }
        prev_carrying = {
            i: getattr(self.env.agents[i].state, "carrying", None)
            for i in range(self.num_agents)
        }
        prev_balls = {
            i: set(self._last_infos.get(i, {}).get("agent_balls_delivered", []))
            for i in range(self.num_agents)
        }
        prev_all_delivered = set().union(*prev_balls.values())

        obs_next, env_rewards, terminations, truncations, infos = self.env.step(actions)
        self._last_infos = infos

        self._update_ont(obs_next, infos)

        reach_sig = (
            frozenset(self._ont._agent_room.values()),
            frozenset(self._ont._locked_doors),
        )
        if reach_sig != self._last_reach_sig:
            self._last_reach_sig = reach_sig
            self._ont._check_ont_reachability()

        _carried_labels  = {v for v in self._ont._agent_carrying.values() if v}
        _remaining_items = [
            item
            for items_list in self.goal.room_items.values()
            for item in items_list
            if item.label not in self._ont._all_delivered
            and item.label not in _carried_labels
        ]

        _active = self.active_items if self.active_items is not None else set(self.goal.target_items)

        _prev_active_del = prev_all_delivered & _active
        _curr_active_del = self._ont._all_delivered & _active
        if len(_curr_active_del) > len(_prev_active_del):
            for _j in range(self.num_agents):
                self._ep_idle_rooms_visited[_j] = set()

        total_shaping = 0.0
        for i in range(self.num_agents):
            s = 0.0

            now_carrying  = getattr(self.env.agents[i].state, "carrying", None)
            carrying_goal = (
                now_carrying is not None
                and hasattr(now_carrying, "label")
                and now_carrying.label in _active
            )
            agent_room_i = self._ont._agent_room.get(i)

            new_rooms = self._ont._known_rooms_by_agent.get(i, set()) - prev_rooms[i]
            _expl = 1.0 * len(new_rooms)
            s += _expl
            self._ep_components["expl"] += _expl

            _reexplore = 0.0
            if not carrying_goal and agent_room_i:
                room_items_here = self.goal.room_items.get(agent_room_i, [])
                room_has_undelivered = any(
                    item.label in _active
                    and item.label not in self._ont._all_delivered
                    and item.label not in _carried_labels
                    for item in room_items_here
                )
                if room_has_undelivered and agent_room_i not in self._ep_idle_rooms_visited[i]:
                    self._ep_idle_rooms_visited[i].add(agent_room_i)
                    _reexplore = 3.0
            _guide = _reexplore

            _pick = 0.0
            now_label = (getattr(now_carrying, "label", None)
                         if now_carrying is not None else None)
            prev_label = (getattr(prev_carrying[i], "label", None)
                          if prev_carrying[i] is not None else None)
            just_picked = (now_label and now_label != prev_label
                           and now_label in _active)
            if just_picked and now_label not in self._ep_pickup_given:
                self._ep_pickup_given.add(now_label)
                _pick = 20.0

            _carry = 0.0
            if carrying_goal:
                curr_pos = (int(self.env.agents[i].state.pos[0]),
                            int(self.env.agents[i].state.pos[1]))
                prev_pos = self._prev_agent_pos[i]
                curr_bfs = self._bfs_dist.get(curr_pos, _BFS_MAX)
                prev_bfs = self._bfs_dist.get(prev_pos, _BFS_MAX)
                _carry = (prev_bfs - curr_bfs) * 1.0
            _pen = _carry

            _kitchen = 0.0
            if (carrying_goal and now_label
                    and agent_room_i == self.goal.target_room
                    and now_label not in self._ep_kitchen_bonus_given):
                self._ep_kitchen_bonus_given.add(now_label)
                _kitchen = 30.0

            s += _guide + _pick + _pen + _kitchen
            self._ep_components["guide"]  += _guide
            self._ep_components["pickup"] += _pick
            self._ep_components["pen"]    += _pen
            self._ep_components["obs"]    += _kitchen

            curr_balls = set(infos[i].get("agent_balls_delivered", []))
            newly_del  = curr_balls - prev_balls[i]
            n_already  = len(_prev_active_del)
            _deliv = 0.0
            for lbl in newly_del:
                if lbl in _active:
                    _deliv  += 100.0 * (1.0 + 0.4 * n_already)
                    n_already += 1
            s += _deliv
            self._ep_components["delivery"] += _deliv

            _task_complete_stage = _active.issubset(self._ont._all_delivered)
            _comp = 0.0
            if _task_complete_stage and not self._ep_complete_given:
                _comp = 150.0
                self._ep_complete_given = True
            s += _comp
            self._ep_components["completion"] += _comp

            total_shaping += s

        for i in range(self.num_agents):
            self._prev_agent_pos[i] = (
                int(self.env.agents[i].state.pos[0]),
                int(self.env.agents[i].state.pos[1]),
            )

        obs    = self._build_obs(obs_next)
        reward = sum(float(env_rewards[i]) for i in range(self.num_agents)) + total_shaping

        terminated = any(bool(terminations[i]) for i in range(self.num_agents))
        truncated  = (any(bool(truncations[i]) for i in range(self.num_agents))
                      or self._ep_step_count >= self.max_steps)

        info_out = {
            "task_complete":     _active.issubset(self._ont._all_delivered),
            "balls_delivered":   len(self._ont._all_delivered & _active),
            "items_observed":    len(set(self._ont._observed_goal_items) & _active),
            "rooms_explored":    len(self._ont.get_known_rooms()),
            "reward_components": dict(self._ep_components) if (terminated or truncated) else None,
            "reasoner_stats":    self._ont.reasoner_stats() if (terminated or truncated) else None,
        }
        return obs, reward, terminated, truncated, info_out

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
