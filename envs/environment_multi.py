from __future__ import annotations

from typing import Optional

from multigrid.base import MultiGridEnv
from multigrid.core.grid import Grid
from multigrid.core.mission import MissionSpace
from multigrid.core.world_object import Ball, Box, Door, Floor, Goal, Key, Lava, Wall
from helper_functions.Rooms import Room, Garden, Foyer, LivingRoom, Kitchen, DiningRoom, Bedroom, Bathroom, Hallway, Study, RightWing


class LabeledKey(Key):
    def __new__(cls, color: str, label: str):
        obj = super().__new__(cls, color)
        obj.label = label
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is not None:
            self.label = getattr(obj, "label", "")


class LabeledBall(Ball):
    def __new__(cls, color: str, label: str):
        obj = super().__new__(cls, color)
        obj.label = label
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is not None:
            self.label = getattr(obj, "label", "")


class LabeledBox(Box):
    def __new__(cls, color: str, label: str):
        obj = super().__new__(cls, color)
        obj.label = label
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is not None:
            self.label = getattr(obj, "label", "")


class HouseEnv(MultiGridEnv):

    _DEFAULT_STARTS = [
        ((13, 18), 0),
        ((11, 18), 0),
        ((15, 18), 0),
        ((13, 20), 3),
    ]

    def __init__(
        self,
        num_agents: int = 2,
        goal=None,
        agent_start_configs: Optional[list[dict]] = None,
        width: int = 32,
        height: int = 23,
        max_steps: int = 10_000,
        **kwargs,
    ):
        if num_agents > len(self._DEFAULT_STARTS):
            raise ValueError(
                f"num_agents={num_agents} exceeds built-in defaults "
                f"({len(self._DEFAULT_STARTS)}). Pass agent_start_configs."
            )

        self._goal = goal
        self._agent_start_configs = agent_start_configs or []

        self._all_keys: list[str] = []
        self._all_balls: list[str] = []
        self._all_boxes: list[str] = []
        self._keys_collected: dict[int, list[str]] = {}
        self._balls_delivered: dict[int, list[str]] = {}

        _room_list = [
            Garden(), Foyer(), LivingRoom(), Kitchen(), DiningRoom(),
            Bedroom(), Bathroom(), Hallway(), Study(), RightWing(),
        ]
        self._rooms: dict[str, Room] = {r.name: r for r in _room_list}

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            agents=num_agents,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_mission(self) -> str:
        if self._goal:
            return self._goal.description
        return "Explore the house."

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self._all_keys = []
        self._all_balls = []
        self._all_boxes = []
        for room in self._rooms.values():
            room.balls.clear()
            room.boxes.clear()

        self._build_skeleton(width, height)
        self._apply_room_colors()
        self._place_doors()
        self._place_furniture()
        self._place_goal_items()

        self._keys_collected = {}
        self._balls_delivered = {}

        for i, agent in enumerate(self.agents):
            cfg = self._agent_start_configs[i] if i < len(self._agent_start_configs) else {}
            pos = cfg.get("pos", self._DEFAULT_STARTS[i][0])
            direction = cfg.get("dir", self._DEFAULT_STARTS[i][1])
            agent.state.pos = pos
            agent.state.dir = direction
            self._keys_collected[i] = []
            self._balls_delivered[i] = []

    def step(self, actions):
        prev_carrying = {i: agent.state.carrying for i, agent in enumerate(self.agents)}

        pending_ball_delivery: dict[int, tuple[str, int, int]] = {}
        for agent_id, agent in enumerate(self.agents):
            action = actions.get(agent_id) if isinstance(actions, dict) else int(actions[agent_id])
            if action != 4:
                continue
            carried = agent.state.carrying
            if not isinstance(carried, LabeledBall):
                continue
            ax, ay = agent.state.pos
            cell = self.grid.get(ax, ay)
            floor_color = cell.color if isinstance(cell, Floor) else None
            if floor_color == "yellow":
                label = getattr(carried, "label", None)
                if label and label not in self._balls_delivered.get(agent_id, []):
                    # The dropped ball lands in the cell in front of the agent
                    # (multigrid drop -> front_pos), not the agent's own cell, so
                    # record that cell for post-step cleanup.
                    fx, fy = agent.front_pos
                    pending_ball_delivery[agent_id] = (label, int(fx), int(fy))

        obs, rewards, terminations, truncations, infos = super().step(actions)

        for agent_id, agent in enumerate(self.agents):
            carrying  = agent.state.carrying
            prev      = prev_carrying[agent_id]

            if carrying is not None and prev is None and isinstance(carrying, LabeledKey):
                label = getattr(carrying, "label", None)
                if label and label not in self._keys_collected.get(agent_id, []):
                    self._keys_collected.setdefault(agent_id, []).append(label)
                    rewards[agent_id] = rewards[agent_id] + 0.1

            if agent_id in pending_ball_delivery and carrying is None:
                label, dx, dy = pending_ball_delivery[agent_id]
                self._balls_delivered.setdefault(agent_id, []).append(label)
                rewards[agent_id] = rewards[agent_id] + 0.2
                if isinstance(self.grid.get(dx, dy), LabeledBall):
                    self.grid.set(dx, dy, Floor("yellow"))

        all_balls_done = bool(self._all_balls) and set(self._all_balls) == set(self._pooled_balls())
        task_complete  = all_balls_done

        if task_complete:
            for agent_id in range(self.num_agents):
                terminations[agent_id] = True

        for agent_id in range(self.num_agents):
            infos[agent_id] = {
                "agent_keys_collected":  list(self._keys_collected.get(agent_id, [])),
                "agent_balls_delivered": list(self._balls_delivered.get(agent_id, [])),
                "all_keys_collected":    self._pooled_keys(),
                "all_balls_delivered":   self._pooled_balls(),
                "task_complete":         task_complete,
            }

        return obs, rewards, terminations, truncations, infos

    def reset(self, **kwargs):
        self._keys_collected = {i: [] for i in range(self.num_agents)}
        self._balls_delivered = {i: [] for i in range(self.num_agents)}
        obs, info = super().reset(**kwargs)
        for agent_id in range(self.num_agents):
            if agent_id not in info:
                info[agent_id] = {}
            info[agent_id].update({
                "agent_keys_collected": [],
                "agent_balls_delivered": [],
                "all_keys_collected": [],
                "all_balls_delivered": [],
                "task_complete": False,
            })
        return obs, info

    def _pooled_keys(self) -> list[str]:
        seen: set[str] = set()
        for lst in self._keys_collected.values():
            seen.update(lst)
        return list(seen)

    def _pooled_balls(self) -> list[str]:
        seen: set[str] = set()
        for lst in self._balls_delivered.values():
            seen.update(lst)
        return list(seen)

    def _unlock_garden_gate(self):
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Door) and cell.color == "green":
                    cell.is_locked = False

    def _build_skeleton(self, width, height):
        self._hwall(6, 0, width - 1)
        self._hwall(15, 0, width - 1)
        self._hwall(22, 0, width - 1)
        self._vwall(8, 7, height - 2)
        self._vwall(18, 6, height - 2)
        self._vwall(26, 7, height - 2)

    def _apply_room_colors(self):
        self._fill_room_floor("green",  1, 30, 1, 5)
        self._fill_room_floor("grey",   1,  7,  7, 14)
        self._fill_room_floor("blue",   9, 17,  7, 14)
        self._fill_room_floor("yellow", 19, 25,  7, 14)
        self._fill_room_floor("purple", 27, 31,  7, 14)
        self._fill_room_floor("red",    1,  7, 16, 18)
        self._fill_room_floor("grey",   1,  7, 19, 21)
        self._fill_room_floor("purple", 19, 25, 16, 21)
        self._fill_room_floor("red",    27, 31, 16, 21)

    def _place_doors(self):
        doors = [
            (13, 6,  "yellow", False, "front door"),
            (28, 6,  "green",  False, "garden gate"),
            (8,  10, "grey",   False, "foyer to living room"),
            (18, 10, "red",    False, "kitchen door"),
            (26, 10, "purple", False, "dining room door"),
            (8,  17, "red",    False, "bedroom door"),
            (8,  20, "grey",   False, "bathroom door"),
            (18, 18, "blue",   False, "study door"),
            (26, 18, "red",    False, "right wing door"),
            (13, 15, "grey",   False, "hallway to living room"),
            (21, 15, "yellow", False, "study to kitchen"),
        ]
        for col, row, color, locked, label in doors:
            self._place_door_helper(col, row, color, locked, label)

    def _place_furniture(self):
        """Place fixed furniture (non-interactive boxes) regardless of goal."""
        self._safe_put(LabeledBox("yellow", "plant pot"), 3, 2)
        self._all_boxes.append("plant pot")
        self._rooms["garden"].register_box("plant pot")
        self._safe_put(LabeledBox("grey", "coat rack"), 2, 8)
        self._all_boxes.append("coat rack")
        self._rooms["foyer"].register_box("coat rack")
        self._hwall(12, 10, 13)
        self._safe_put(LabeledBox("grey",   "television"),   13, 8)
        self._safe_put(LabeledBox("purple", "coffee table"), 12, 11)
        self._all_boxes.extend(["television", "coffee table"])
        self._rooms["living_room"].register_box("television")
        self._rooms["living_room"].register_box("coffee table")
        self._safe_put(LabeledBox("blue", "refrigerator"), 19, 7)
        self._safe_put(Wall(), 22, 7)
        self._safe_put(LabeledBox("blue", "kitchen sink"),  24, 7)
        self._all_boxes.extend(["refrigerator", "kitchen sink"])
        self._rooms["kitchen"].register_box("refrigerator")
        self._rooms["kitchen"].register_box("kitchen sink")
        for r in [9, 11]:
            self._hwall(r, 28, 30)
        self._safe_put(Wall(), 28, 10)
        self._safe_put(Wall(), 30, 10)
        self._hwall(17, 3, 5)
        self._safe_put(LabeledBox("grey", "nightstand"), 7, 16)
        self._all_boxes.append("nightstand")
        self._rooms["bedroom"].register_box("nightstand")
        self._safe_put(LabeledBox("grey", "toilet"), 2, 20)
        self._all_boxes.append("toilet")
        self._rooms["bathroom"].register_box("toilet")
        self._safe_put(LabeledBox("blue", "umbrella stand"), 10, 16)
        self._all_boxes.append("umbrella stand")
        self._rooms["hallway"].register_box("umbrella stand")
        self._safe_put(LabeledBox("grey", "computer monitor"), 22, 16)
        self._all_boxes.append("computer monitor")
        self._rooms["study"].register_box("computer monitor")
        self._safe_put(LabeledBox("grey", "storage shelf"), 28, 17)
        self._all_boxes.append("storage shelf")
        self._rooms["right_wing"].register_box("storage shelf")


    def _place_goal_items(self):
        """Place collectible items defined by the active goal.

        Each item spawns on a random empty floor tile inside its room (falling
        back to the goal-specified tile) so the layout varies between episodes
        instead of always using the same cell.
        """
        if not self._goal:
            return
        for room_name, items in self._goal.room_items.items():
            room = self._rooms.get(room_name)
            for gi in items:
                x, y = self._random_item_cell(room, gi.x, gi.y)
                self._safe_put(LabeledBall(gi.color, gi.label), x, y)
                self._all_balls.append(gi.label)
                if room:
                    room.register_ball(gi.label)

    def _random_item_cell(self, room, default_x: int, default_y: int):
        """Pick a random empty floor tile within `room`'s bbox.

        Only cells that are empty or plain floor are eligible, so walls, doors,
        furniture and already-placed items are skipped. Agent start tiles are
        avoided too. Falls back to the goal-specified cell when no free tile is
        available (or the room is unknown).
        """
        if room is None:
            return default_x, default_y

        starts = {tuple(self._DEFAULT_STARTS[i][0]) for i in range(self.num_agents)}
        for cfg in self._agent_start_configs:
            if isinstance(cfg, dict) and cfg.get("pos") is not None:
                starts.add(tuple(cfg["pos"]))

        c1, c2, r1, r2 = room.bbox
        candidates = []
        for x in range(max(1, c1), min(c2 + 1, self.width - 1)):
            for y in range(max(1, r1), min(r2 + 1, self.height - 1)):
                if (x, y) in starts:
                    continue
                cell = self.grid.get(x, y)
                if cell is None or isinstance(cell, Floor):
                    candidates.append((x, y))

        if not candidates:
            return default_x, default_y
        return candidates[int(self.np_random.integers(len(candidates)))]

    def _fill_room_floor(self, color: str, c1: int, c2: int, r1: int, r2: int):
        for r in range(max(0, r1), min(r2 + 1, self.height)):
            for c in range(max(0, c1), min(c2 + 1, self.width)):
                cell = self.grid.get(c, r)
                if cell is None or (isinstance(cell, Floor) and cell.color is None):
                    self.put_obj(Floor(color), c, r)

    def _vwall(self, col: int, r1: int, r2: int):
        for r in range(r1, r2 + 1):
            if not self.grid.get(col, r):
                self.put_obj(Wall(), col, r)

    def _hwall(self, row: int, c1: int, c2: int):
        for c in range(c1, c2 + 1):
            if not self.grid.get(c, row):
                self.put_obj(Wall(), c, row)

    def _place_door_helper(self, col: int, row: int, color: str, is_locked: bool, label: str):
        if 0 <= col < self.width and 0 <= row < self.height:
            self.grid.set(col, row, None)
            door = Door(color, is_locked=is_locked)
            door.label = label
            self.put_obj(door, col, row)

    def _safe_put(self, obj, col: int, row: int):
        if 0 <= col < self.width and 0 <= row < self.height:
            if not isinstance(self.grid.get(col, row), Wall):
                self.put_obj(obj, col, row)

    def get_room_names(self) -> list[str]:
        return list(self._rooms)

    def get_room_for_cell(self, x: int, y: int) -> Optional[str]:
        for room_name, room in self._rooms.items():
            if room.contains_cell(x, y):
                return room_name
        return None

    def get_all_rooms(self) -> list[str]:
        return list(self._rooms.keys())

    def get_room_bbox(self, room_name: str) -> Optional[tuple]:
        return self._rooms[room_name].bbox if room_name in self._rooms else None

    def get_room_color(self, room_name: str) -> Optional[str]:
        return self._rooms[room_name].color if room_name in self._rooms else None

    def get_room_object(self, room_name: str) -> Optional[Room]:
        return self._rooms.get(room_name)

    def get_all_walls(self) -> list[tuple[int, int]]:
        walls = []
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if cell is not None and cell.type == "wall":
                    walls.append((x, y))
        return walls

    def get_all_doors(self) -> list[tuple[int, int, Door]]:
        doors = []
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Door):
                    doors.append((x, y, cell))
        return doors

    def get_all_keys(self) -> list[tuple[int, int, LabeledKey]]:
        keys = []
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, LabeledKey):
                    keys.append((x, y, cell))
        return keys

    def get_all_balls(self) -> list[tuple[int, int, LabeledBall]]:
        balls = []
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, LabeledBall):
                    balls.append((x, y, cell))
        return balls

    def get_agent_positions(self) -> dict[int, tuple[int, int]]:
        return {i: tuple(agent.state.pos) for i, agent in enumerate(self.agents)}

    def is_door_unlocked(self, x: int, y: int) -> bool:
        cell = self.grid.get(x, y)
        if isinstance(cell, Door):
            return not cell.is_locked
        return False

    def get_task_progress(self) -> dict:
        return {
            "all_keys_collected":    self._pooled_keys(),
            "all_balls_delivered":   self._pooled_balls(),
            "total_keys":            len(self._all_keys),
            "total_balls":           len(self._all_balls),
            "keys_collected_count":  len(self._pooled_keys()),
            "balls_delivered_count": len(self._pooled_balls()),
            "task_complete":         bool(self._all_keys) and set(self._all_keys) == set(self._pooled_keys())
                                     and bool(self._all_balls) and set(self._all_balls) == set(self._pooled_balls()),
            "per_agent_keys":        {i: list(v) for i, v in self._keys_collected.items()},
            "per_agent_balls":       {i: list(v) for i, v in self._balls_delivered.items()},
        }
