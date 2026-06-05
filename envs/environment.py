from __future__ import annotations
from typing import Optional
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Floor, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv


class LabeledKey(Key):
    def __init__(self, color: str, label: str):
        super().__init__(color)
        self.label = label

class LabeledBall(Ball):
    def __init__(self, color: str, label: str):
        super().__init__(color)
        self.label = label

class LabeledBox(Box):
    def __init__(self, color: str, label: str, contains=None):
        super().__init__(color, contains)
        self.label = label


class HouseEnv(MiniGridEnv):

    def __init__(
        self,
        width: int = 32,
        height: int = 23,
        agent_start_pos: Optional[tuple] = None,
        agent_start_dir: int = 0,
        max_steps: int = 10000,
        **kwargs,
    ):
        self._agent_start_pos = agent_start_pos
        self._agent_start_dir = agent_start_dir

        self.keys_collected: list[str] = []
        self.balls_delivered: list[str] = []
        self._all_keys: list[str] = []
        self._all_balls: list[str] = []

        self._rooms = {
            "garden":       {"color": "green",  "bbox": (1, 30, 1, 5)},
            "foyer":        {"color": "grey",   "bbox": (1, 7, 7, 14)},
            "living_room":  {"color": "blue",   "bbox": (9, 17, 7, 14)},
            "kitchen":      {"color": "yellow", "bbox": (19, 25, 7, 14)},
            "dining_room":  {"color": "purple", "bbox": (27, 31, 7, 14)},
            "bedroom":      {"color": "red",    "bbox": (1, 7, 16, 18)},
            "bathroom":     {"color": "grey",   "bbox": (1, 7, 19, 21)},
            "hallway":      {"color": None,     "bbox": (9, 17, 16, 21)},
            "study":        {"color": "purple", "bbox": (19, 25, 16, 21)},
            "right_wing":   {"color": "red",    "bbox": (27, 31, 16, 21)},
        }

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "Collect all keys, deliver items to kitchen, then exit through the garden."

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self._all_keys = []
        self._all_balls = []

        self._build_skeleton(width, height)
        self._apply_room_colors()
        self._place_doors()

        self._build_garden()
        self._build_foyer()
        self._build_living_room()
        self._build_kitchen()
        self._build_dining_room()
        self._build_bedroom()
        self._build_bathroom()
        self._build_hallway()
        self._build_study()
        self._build_right_wing()

        self._safe_put(Goal(), 30, 3)

        self.agent_pos = self._agent_start_pos or (13, 18)
        self.agent_dir = self._agent_start_dir

    def _build_skeleton(self, width, height):
        self._hwall(6, 0, width - 1)    # garden / house separation
        self._hwall(15, 0, width - 1)   # upper/lower rooms separation
        self._hwall(22, 0, width - 1)   # lower boundary

        self._vwall(8, 7, height - 2)   # foyer|living AND bed+bath|hallway
        self._vwall(18, 6, height - 2)  # main vertical: kitchen+study|hallway
        self._vwall(26, 7, height - 2)  # kitchen|dining AND hallway|right-wing

    def _apply_room_colors(self):
        self._fill_room_floor("green",  1, 30, 1, 5)

        self._fill_room_floor("grey",   1,  7,  7, 14)   # Foyer
        self._fill_room_floor("blue",   9, 17,  7, 14)   # Living room
        self._fill_room_floor("yellow", 19, 25, 7, 14)   # Kitchen
        self._fill_room_floor("purple", 27, 31, 7, 14)   # Dining room

        self._fill_room_floor("red",    1,  7, 16, 18)   # Bedroom
        self._fill_room_floor("grey",   1,  7, 19, 21)   # Bathroom

        self._fill_room_floor("purple", 19, 25, 16, 21)  # Study
        self._fill_room_floor("red",    27, 31, 16, 21)  # Right wing

    def _place_doors(self):
        doors = [
            (13, 6,  "yellow", True,  "front door"),
            (28, 6,  "green",  True,  "garden gate"),
            (8,  10, "grey",   False, "foyer to living room"),
            (18, 10, "red",    False, "kitchen door"),
            (26, 10, "purple", False, "dining room door"),
            (8,  17, "red",    False, "bedroom door"),
            (8,  20, "grey",   False, "bathroom door"),
            (18, 18, "blue",   False, "study door"),
            (26, 18, "red",    False, "right wing door"),
        ]
        for col, row, color, locked, label in doors:
            self._place_door_helper(col, row, color, locked, label)

    def _build_garden(self):
        for c in range(20, 26):
            for i in range(2, 4):
                self._safe_put(Lava(), c, i)

        self._safe_put(LabeledBox("yellow", "plant pot"), 3, 2)

        for (cx, cy), label in [((6, 4), "flower pot A"), ((7, 2), "flower pot B")]:
            self._safe_put(LabeledBall("red", label), cx, cy)
            self._all_balls.append(label)

        self._safe_put(LabeledKey("green", "garden hose"), 5, 3)
        self._all_keys.append("garden hose")

    def _build_foyer(self):
        self._safe_put(LabeledBox("grey", "coat rack"), 2, 8)
        self._safe_put(LabeledBall("blue", "umbrella"), 2, 9)
        self._all_balls.append("umbrella")
        self._safe_put(LabeledKey("grey", "house key"), 3, 10)
        self._all_keys.append("house key")

    def _build_living_room(self):
        # FIX: removed _vwall(16, 8, 11) — it was cutting off part of the room
        # and the internal half-wall below now uses _hwall correctly
        self._hwall(12, 10, 13)
        self._safe_put(LabeledBox("grey", "television"), 13, 8)
        self._safe_put(LabeledBox("purple", "coffee table"), 12, 11)
        self._safe_put(LabeledBall("yellow", "floor lamp"), 9, 8)
        self._all_balls.append("floor lamp")
        self._safe_put(LabeledKey("blue", "remote control"), 12, 10)
        self._all_keys.append("remote control")

    def _build_kitchen(self):
        self._safe_put(LabeledBox("blue", "refrigerator"), 19, 7)
        self._safe_put(Wall(), 22, 7)
        self._safe_put(LabeledBox("blue", "kitchen sink"), 24, 7)
        items = [
            (LabeledKey("red",  "kitchen knife"), 21, 9),
            (LabeledBall("green", "apple"),       23, 9),
            (LabeledKey("grey", "water bottle"),  20, 11),
        ]
        for obj, x, y in items:
            self._safe_put(obj, x, y)
            if isinstance(obj, LabeledKey):
                self._all_keys.append(obj.label)
            else:
                self._all_balls.append(obj.label)

    def _build_dining_room(self):
        for r in [9, 11]:
            self._hwall(r, 28, 30)
        self._safe_put(Wall(), 28, 10)
        self._safe_put(Wall(), 30, 10)
        self._safe_put(LabeledBall("purple", "decorative vase"), 29, 10)
        self._all_balls.append("decorative vase")
        self._safe_put(LabeledKey("red", "candle holder"), 29, 8)
        self._all_keys.append("candle holder")

    def _build_bedroom(self):
        # FIX: original code drew internal walls at rows 16–17 cols 2–4,
        # but row 16 IS the room boundary wall (from _hwall(15,...) +1).
        # Use only row 17 for the internal bed outline, and only from col 3
        # so col 2 stays walkable.
        self._hwall(17, 3, 5)
        self._safe_put(LabeledBox("grey", "nightstand"), 7, 16)
        self._safe_put(LabeledKey("yellow", "alarm clock"), 6, 17)
        self._all_keys.append("alarm clock")
        self._safe_put(LabeledBall("blue", "pillow"), 3, 16)
        self._all_balls.append("pillow")

    def _build_bathroom(self):
        self._safe_put(LabeledBox("grey", "toilet"), 2, 20)
        self._safe_put(Wall(), 4, 20)
        for c in range(5, 7):
            self._safe_put(Lava(), c, 20)
        self._safe_put(LabeledBall("blue", "soap bar"), 3, 19)
        self._all_balls.append("soap bar")

    def _build_hallway(self):
        self._safe_put(LabeledBox("blue", "umbrella stand"), 10, 16)
        self._safe_put(LabeledBall("yellow", "hallway lamp"), 13, 17)
        self._all_balls.append("hallway lamp")

    def _build_study(self):
        self._safe_put(LabeledBox("grey", "computer monitor"), 22, 16)
        self._safe_put(LabeledKey("purple", "pen"), 24, 17)
        self._all_keys.append("pen")
        self._safe_put(LabeledBall("yellow", "important document"), 21, 17)
        self._all_balls.append("important document")
        self._safe_put(LabeledBall("green", "potted plant"), 20, 20)

    def _build_right_wing(self):
        self._safe_put(LabeledBox("grey", "storage shelf"), 28, 17)
        self._safe_put(LabeledKey("red", "spare key"), 29, 19)
        self._all_keys.append("spare key")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Reward for picking up any new key
        if self.carrying and isinstance(self.carrying, LabeledKey):
            if self.carrying.label not in self.keys_collected:
                self.keys_collected.append(self.carrying.label)
                reward += 0.1

        # FIX: check the agent's current position floor color after dropping,
        # since drop() places the object at the agent's position (not fwd_pos).
        if action == self.actions.drop:
            agent_cell = self.grid.get(*self.agent_pos)
            floor_color = agent_cell.color if isinstance(agent_cell, Floor) else None
            if floor_color == "yellow":
                # The dropped object is now at agent_pos (MiniGrid drops in place)
                dropped = self.grid.get(*self.agent_pos)
                if isinstance(dropped, LabeledBall):
                    if dropped.label not in self.balls_delivered:
                        self.balls_delivered.append(dropped.label)
                        reward += 0.2

        all_keys_done = set(self._all_keys) == set(self.keys_collected)
        all_balls_done = set(self._all_balls) == set(self.balls_delivered)
        if all_keys_done and all_balls_done:
            self._unlock_garden_gate()

        info.update({
            "keys_collected":       list(self.keys_collected),
            "keys_collected_count": len(self.keys_collected),
            "balls_delivered":      list(self.balls_delivered),
            "balls_delivered_count": len(self.balls_delivered),
            "task_complete":        all_keys_done and all_balls_done,
        })
        return obs, reward, terminated, truncated, info

    def _unlock_garden_gate(self):
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Door) and cell.color == "green":
                    cell.is_locked = False

    def _fill_room_floor(self, color: str, c1: int, c2: int, r1: int, r2: int):
        # FIX: use c2+1 and r2+1 so the rightmost column and bottom row are
        # included (range is exclusive on the upper bound).
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
        for room_name, room_info in self._rooms.items():
            c1, c2, r1, r2 = room_info["bbox"]
            if c1 <= x <= c2 and r1 <= y <= r2:
                return room_name
        return None

    def get_all_rooms(self) -> list[str]:
        return list(self._rooms.keys())

    def get_room_bbox(self, room_name: str) -> Optional[tuple]:
        if room_name in self._rooms:
            return self._rooms[room_name]["bbox"]
        return None

    def get_room_color(self, room_name: str) -> Optional[str]:
        if room_name in self._rooms:
            return self._rooms[room_name]["color"]
        return None

    def get_all_walls(self) -> list[tuple[int, int, Wall]]:
        walls = []
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if cell is not None and cell.type == "wall":
                    walls.append((x, y))
        return walls

    def reset(self, **kwargs):
        self.keys_collected = []
        self.balls_delivered = []
        return super().reset(**kwargs)