from __future__ import annotations


class Room:
    """Represents a named room in the house environment."""

    def __init__(
        self,
        name: str,
        color: str | None,
        bbox: tuple[int, int, int, int],
    ):
        self.name = name
        self.color = color
        self.bbox = bbox  # (col_min, col_max, row_min, row_max)
        self.balls: list[str] = []
        self.boxes: list[str] = []

    def contains_cell(self, x: int, y: int) -> bool:
        c1, c2, r1, r2 = self.bbox
        return c1 <= x <= c2 and r1 <= y <= r2

    def register_ball(self, label: str) -> None:
        self.balls.append(label)

    def register_box(self, label: str) -> None:
        self.boxes.append(label)

    def on_enter(self) -> dict:
        """Return info given to the agent upon entering this room. Override per room."""
        return {}

    def __repr__(self) -> str:
        return f"Room({self.name!r})"


class Garden(Room):
    def __init__(self):
        super().__init__("garden", "green", (1, 30, 1, 5))

    def on_enter(self) -> dict:
        return {
            "description": "An outdoor garden with green floors and lava hazards to the east. Doors visible here lead into the house.",
        }


class Foyer(Room):
    def __init__(self):
        super().__init__("foyer", "grey", (1, 7, 7, 14))

    def on_enter(self) -> dict:
        return {
            "description": "A grey entry foyer. There is a coat rack near the entrance. Doors visible here connect to adjacent rooms.",
        }


class LivingRoom(Room):
    def __init__(self):
        super().__init__("living_room", "blue", (9, 17, 7, 14))

    def on_enter(self) -> dict:
        return {
            "description": "A blue living room with a television and coffee table. Doors visible here connect to adjacent rooms.",
        }


class Kitchen(Room):
    def __init__(self):
        super().__init__("kitchen", "yellow", (19, 25, 7, 14))

    def on_enter(self) -> dict:
        return {
            "description": "A yellow kitchen with a refrigerator and sink. This is the delivery destination for goal items.",
        }


class DiningRoom(Room):
    def __init__(self):
        super().__init__("dining_room", "purple", (27, 31, 7, 14))

    def on_enter(self) -> dict:
        return {
            "description": "A purple dining room with a table arrangement. Doors visible here connect to adjacent rooms.",
        }


class Bedroom(Room):
    def __init__(self):
        super().__init__("bedroom", "red", (1, 7, 16, 18))

    def on_enter(self) -> dict:
        return {
            "description": "A red bedroom with a nightstand and bed area. Doors visible here connect to adjacent rooms.",
        }


class Bathroom(Room):
    def __init__(self):
        super().__init__("bathroom", "grey", (1, 7, 19, 21))

    def on_enter(self) -> dict:
        return {
            "description": "A grey bathroom with a toilet. Watch for lava. Doors visible here connect to adjacent rooms.",
        }


class Hallway(Room):
    def __init__(self):
        super().__init__("hallway", None, (9, 17, 16, 21))

    def on_enter(self) -> dict:
        return {
            "description": "A central hallway connecting the lower floor rooms. Doors visible here connect to adjacent rooms.",
        }


class Study(Room):
    def __init__(self):
        super().__init__("study", "purple", (19, 25, 16, 21))

    def on_enter(self) -> dict:
        return {
            "description": "A purple study with a computer monitor and storage. Doors visible here connect to adjacent rooms.",
        }


class RightWing(Room):
    def __init__(self):
        super().__init__("right_wing", "red", (27, 31, 16, 21))

    def on_enter(self) -> dict:
        return {
            "description": "A red right wing with storage shelving. Doors visible here connect to adjacent rooms.",
        }


ALL_ROOMS: list[Room] = [
    Garden(),
    Foyer(),
    LivingRoom(),
    Kitchen(),
    DiningRoom(),
    Bedroom(),
    Bathroom(),
    Hallway(),
    Study(),
    RightWing(),
]
