from dataclasses import dataclass, field


@dataclass
class GoalItem:
    label: str
    color: str   # multigrid color (red/green/blue/yellow/purple/grey)
    x: int
    y: int


@dataclass
class Goal:
    name: str
    ont_class: str
    description: str
    target_room: str
    room_items: dict[str, list[GoalItem]] = field(default_factory=dict)

    @property
    def target_items(self) -> list[str]:
        return [item.label for items in self.room_items.values() for item in items]

    @property
    def all_rooms(self) -> list[str]:
        return list(self.room_items.keys())

    def items_in_room(self, room: str) -> list[GoalItem]:
        return self.room_items.get(room, [])

    def __repr__(self) -> str:
        total = sum(len(v) for v in self.room_items.values())
        return f"Goal({self.name}: {total} items across {len(self.room_items)} rooms -> {self.target_room})"


# Goal definitions. Each GoalItem is (label, colour, grid-x, grid-y), placed to
# avoid walls and furniture. The two door keys are always placed by the
# environment regardless of the active goal.

# item label -> ontology class name, used when asserting expected items
ITEM_CLASS_MAP: dict[str, str] = {
    # Trash items
    "plastic bottle":  "PlasticBottle",
    "trash bag":       "TrashBag",
    "old newspaper":   "OldNewspaper",
    "empty can":       "EmptyCan",
    "candy wrapper":   "CandyWrapper",
    "food scrap":      "FoodScrap",
    "empty box":       "EmptyBox",
    "used napkin":     "UsedNapkin",
    "dirty plate":     "DirtyPlate",
    "dirty sock":      "DirtySock",
    "used tissue":     "UsedTissue",
    "empty shampoo":   "EmptyShampoo",
    "junk mail":       "JunkMail",
    "crumpled paper":  "CrumpledPaper",
    "used coffee cup": "UsedCoffeeCup",
    "broken toy":      "BrokenToy",
    # Food items
    "apple":           "Apple",
    "pear":            "Pear",
    "orange":          "Orange",
    "banana":          "Banana",
    "mango":           "Mango",
    "lemon":           "Lemon",
    "tomato":          "Tomato",
    "avocado":         "Avocado",
    "grapes":          "Grapes",
    "strawberry":      "Strawberry",
    "peach":           "Peach",
    "water bottle":    "WaterBottle",
    "juice box":       "JuiceBox",
    "kiwi":            "Kiwi",
    "pineapple":       "Pineapple",
    "chocolate bar":   "ChocolateBar",
    # Door keys
    "front door key":  "FrontDoorKey",
    "garden gate key": "GardenGateKey",
}

GOALS: dict[str, Goal] = {

    "collect_trash": Goal(
        name="Collect All Trash",
        ont_class="CollectTrashGoal",
        description=(
            "Someone made a mess! Collect all trash items scattered around "
            "the house and bring them to the kitchen bin."
        ),
        target_room="kitchen",
        room_items={
            "garden":      [GoalItem("plastic bottle", "blue",   6, 4),
                            GoalItem("trash bag",       "grey",   7, 2)],
            "foyer":       [GoalItem("old newspaper",   "yellow", 2, 9)],
            "living_room": [GoalItem("empty can",       "grey",   9, 8),
                            GoalItem("candy wrapper",   "purple", 12, 10)],
            "kitchen":     [GoalItem("food scrap",      "green",  21, 9),
                            GoalItem("empty box",       "grey",   23, 9)],
            "dining_room": [GoalItem("used napkin",     "yellow", 29, 8),
                            GoalItem("dirty plate",     "grey",   29, 10)],
            "bedroom":     [GoalItem("dirty sock",      "red",    3, 16),
                            GoalItem("used tissue",     "grey",   6, 17)],
            "bathroom":    [GoalItem("empty shampoo",   "blue",   3, 19)],
            "hallway":     [GoalItem("junk mail",       "yellow", 13, 17)],
            "study":       [GoalItem("crumpled paper",  "grey",   21, 17),
                            GoalItem("used coffee cup", "grey",   20, 20)],
            "right_wing":  [GoalItem("broken toy",      "red",    29, 18)],
        },
    ),

    "collect_food": Goal(
        name="Collect All Food",
        ont_class="CollectFoodGoal",
        description=(
            "Groceries have been left all over the house. Collect every "
            "food item and bring it to the kitchen."
        ),
        target_room="kitchen",
        room_items={
            "foyer":       [GoalItem("orange",    "yellow", 2,  9)],
            "living_room": [GoalItem("banana",    "yellow", 9,  8),
                            GoalItem("mango",     "yellow", 12, 10)],
            "dining_room": [GoalItem("grapes",    "yellow", 29, 8),
                            GoalItem("strawberry","yellow", 29, 10)],
            "hallway":     [GoalItem("juice box", "yellow", 13, 17)],
            "study":       [GoalItem("kiwi",      "yellow", 21, 17),
                            GoalItem("pineapple", "yellow", 20, 20)],
        },
    ),
}
