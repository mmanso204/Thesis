import random
import os
import sys
import time
from contextlib import contextmanager
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from owlapy.owl_ontology import Ontology
from multigrid.utils.obs import gen_obs_grid_vis_mask
from multigrid.core.world_object import Wall, Door
from envs.environment_multi import HouseEnv
from owlapy.iri import IRI
from owlapy.class_expression import OWLClass
from owlapy.owl_axiom import (
    OWLClassAssertionAxiom,
    OWLDeclarationAxiom,
    OWLObjectPropertyAssertionAxiom,
    OWLDataPropertyAssertionAxiom,
    OWLSameIndividualAxiom,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_reasoner import SyncReasoner
from owlapy.owl_ontology import SyncOntology
from helper_functions.goals import GOALS, Goal, ITEM_CLASS_MAP

NS = "http://www.semanticweb.org/m.manso/ontologies/2026/3/untitled-ontology-30#"

ITEM_LABEL_MAP: dict[str, tuple[str, str]] = {
    "front door key":  ("KeyObject", "FrontDoorKey"),
    "garden gate key": ("KeyObject", "GardenGateKey"),
    "apple":           ("BallObject", "Apple"),
    "avocado":         ("BallObject", "Avocado"),
    "banana":          ("BallObject", "Banana"),
    "chocolate bar":   ("BallObject", "ChocolateBar"),
    "grapes":          ("BallObject", "Grapes"),
    "juice box":       ("BallObject", "JuiceBox"),
    "kiwi":            ("BallObject", "Kiwi"),
    "lemon":           ("BallObject", "Lemon"),
    "mango":           ("BallObject", "Mango"),
    "orange":          ("BallObject", "Orange"),
    "peach":           ("BallObject", "Peach"),
    "pear":            ("BallObject", "Pear"),
    "pineapple":       ("BallObject", "Pineapple"),
    "strawberry":      ("BallObject", "Strawberry"),
    "tomato":          ("BallObject", "Tomato"),
    "water bottle":    ("BallObject", "WaterBottle"),
    "plastic bottle":  ("BallObject", "PlasticBottle"),
    "trash bag":       ("BallObject", "TrashBag"),
    "old newspaper":   ("BallObject", "OldNewspaper"),
    "empty can":       ("BallObject", "EmptyCan"),
    "candy wrapper":   ("BallObject", "CandyWrapper"),
    "food scrap":      ("BallObject", "FoodScrap"),
    "empty box":       ("BallObject", "EmptyBox"),
    "used napkin":     ("BallObject", "UsedNapkin"),
    "dirty plate":     ("BallObject", "DirtyPlate"),
    "dirty sock":      ("BallObject", "DirtySock"),
    "used tissue":     ("BallObject", "UsedTissue"),
    "empty shampoo":   ("BallObject", "EmptyShampoo"),
    "junk mail":       ("BallObject", "JunkMail"),
    "crumpled paper":  ("BallObject", "CrumpledPaper"),
    "used coffee cup": ("BallObject", "UsedCoffeeCup"),
    "broken toy":      ("BallObject", "BrokenToy"),
    "coat rack":        ("BoxObject", "CoatRack"),
    "coffee table":     ("BoxObject", "CoffeeTable"),
    "computer monitor": ("BoxObject", "ComputerMonitor"),
    "kitchen sink":     ("BoxObject", "KitchenSink"),
    "nightstand":       ("BoxObject", "Nightstand"),
    "plant pot":        ("BoxObject", "PlantPot"),
    "refrigerator":     ("BoxObject", "Refrigerator"),
    "storage shelf":    ("BoxObject", "StorageShelf"),
    "television":       ("BoxObject", "Television"),
    "toilet":           ("BoxObject", "Toilet"),
    "umbrella stand":   ("BoxObject", "UmbrellaStand"),
}

TYPE_FALLBACK: dict[str, str] = {
    "key":  "KeyObject",
    "ball": "BallObject",
    "box":  "BoxObject",
}

PROP_CONNECTS_OPEN        = "connectstoOpenDoor"
PROP_CONNECTS_CLOSED      = "connectstoClosedDoor"
PROP_ROOM_CONNECTS_OPEN   = "roomConnectedToOpenDoor"
PROP_ROOM_CONNECTS_CLOSED = "roomConnectedToClosedDoor"
PROP_LOCATED_IN           = "LocatedIn"
PROP_PROBABLY_IN          = "ProbablyInRoom"
PROP_HAS_TARGET_ROOM      = "hasDirectTargetRoom"
PROP_HAS_TARGET_OBJ       = "hasDirectTargetGoalObject"
PROP_HAS_GOAL             = "HasGoal"
PROP_RESOLVED_ROOM        = "hasResolvedTargetRoom"
PROP_RESOLVED_OBJ         = "hasResolvedTargetObject"
PROP_ACCESSIBLE_TO        = "accessibleto1"
PROP_IS_CARRYING          = "isCarrying"
PROP_IS_DELIVERED         = "isDelivered"
PROP_HAS_SUBGOAL          = "hasSubGoal"
PROP_PREVIOUS_GOAL        = "previousGoal"
PROP_CONFLICTS_WITH       = "conflictsWith"


class Agent:
    def __init__(self, common_ontology: str, agent_id: int = 0, verbose: bool = True):
        self.tbox_path = common_ontology
        self.ontology  = Ontology(common_ontology, load=True)
        self.Agent_ont = Ontology(common_ontology, load=True)
        self.agent_id  = agent_id
        self.verbose   = verbose

        self._known_cells:            set[str] = set()
        self._known_objects:          set[str] = set()
        self._known_rooms:            set[str] = set()
        self._known_doors:            set[str] = set()
        self._known_door_connections: set[str] = set()
        self._door_rooms:             dict[str, set[str]] = {}
        self._locked_doors:           set[str] = set()
        self._agent_declared:         dict[int, bool] = {}
        self._agent_room:             dict[int, str | None] = {}
        self._known_rooms_by_agent:   dict[int, set[str]] = {}
        self._known_cells_by_agent:   dict[int, set[str]] = {}
        self._goal_item_room:         dict[str, str] = {}
        self._goal_item_room_prior:   dict[str, str] = {}

        self._access_dirty:           bool = True
        self._accessibility_cache:    dict[str, set[str]] = {}
        self._asserted_accessible:    set[tuple[str, str]] = set()

        self._all_delivered:          set[str] = set()
        self._agent_carrying:         dict[int, str | None] = {}
        self._agent_carrying_axioms:  dict[int, object]     = {}

        self._ont_reachable_cached:   float = 0.5
        self._ep_reasoner_calls:      int   = 0

        self._reasoner_time_ep:       float = 0.0
        self._reasoner_time_total:    float = 0.0
        self._reasoner_calls_ep:      int   = 0
        self._reasoner_calls_total:   int   = 0

        self._active_goal: Goal | None = None
        self._goal_ind:    OWLNamedIndividual | None = None
        self._subgoal_inds: dict[str, str] = {}
        self._observed_goal_items: dict[str, str] = {}

        self._private_item_room:      dict[int, dict[str, str]] = {}
        self._private_observed_items: dict[int, dict[str, str]] = {}


    def set_goal(self, goal: "Goal | str"):
        if isinstance(goal, str):
            goal = GOALS[goal]
        self._active_goal = goal

        goal_ind_name = f"{goal.ont_class}_agent{self.agent_id}"
        self._goal_ind = OWLNamedIndividual(IRI.create(NS, goal_ind_name))
        self.Agent_ont.add_axiom(OWLDeclarationAxiom(self._goal_ind))
        self.Agent_ont.add_axiom(OWLClassAssertionAxiom(
            self._goal_ind, OWLClass(IRI.create(NS, "Goal"))
        ))
        self.Agent_ont.add_axiom(OWLClassAssertionAxiom(
            self._goal_ind, OWLClass(IRI.create(NS, goal.ont_class))
        ))

        agent_ind = OWLNamedIndividual(IRI.create(NS, f"agent_{self.agent_id}"))
        self.Agent_ont.add_axiom(OWLDeclarationAxiom(agent_ind))
        self.Agent_ont.add_axiom(OWLClassAssertionAxiom(
            agent_ind, OWLClass(IRI.create(NS, "Agent"))
        ))
        self.Agent_ont.add_axiom(OWLObjectPropertyAssertionAxiom(
            agent_ind, OWLObjectProperty(IRI.create(NS, PROP_HAS_GOAL)), self._goal_ind
        ))

        def _add(ax):       self.Agent_ont.add_axiom(ax)
        def _ind(n):        return OWLNamedIndividual(IRI.create(NS, n))
        def _decl(i):       _add(OWLDeclarationAxiom(i))
        def _type(i, c):    _add(OWLClassAssertionAxiom(i, OWLClass(IRI.create(NS, c))))
        def _rel(s, p, o):  _add(OWLObjectPropertyAssertionAxiom(
                                s, OWLObjectProperty(IRI.create(NS, p)), o))

        if goal.target_room:
            room_ind = _ind(goal.target_room)
            _decl(room_ind)
            _type(room_ind, "Room")
            _rel(self._goal_ind, PROP_HAS_TARGET_ROOM, room_ind)
            _rel(self._goal_ind, PROP_RESOLVED_ROOM, room_ind)

        self._subgoal_inds = {}
        prev_subgoal = None
        seq_idx = 0
        for room_name, items in goal.room_items.items():
            room_prior = _ind(room_name)
            _decl(room_prior)
            _type(room_prior, "Room")
            for item in items:
                exp_name = f"expected_{item.label.replace(' ', '_')}"
                exp_ind  = _ind(exp_name)
                _decl(exp_ind)
                _type(exp_ind, "Item")
                _type(exp_ind, "GoalObject")
                if item.label in ITEM_LABEL_MAP:
                    base_cls, spec_cls = ITEM_LABEL_MAP[item.label]
                    _type(exp_ind, base_cls)
                    _type(exp_ind, spec_cls)
                _rel(exp_ind, PROP_PROBABLY_IN, room_prior)
                _rel(self._goal_ind, PROP_HAS_TARGET_OBJ, exp_ind)
                self._goal_item_room_prior[item.label] = room_name

                sg_name = f"subgoal_{item.label.replace(' ', '_')}_agent{self.agent_id}"
                sg_ind  = _ind(sg_name)
                _decl(sg_ind)
                _type(sg_ind, "Goal")
                _type(sg_ind, "FindObjectGoal")
                _rel(self._goal_ind, PROP_HAS_SUBGOAL, sg_ind)
                _rel(sg_ind, PROP_HAS_TARGET_OBJ, exp_ind)
                _rel(sg_ind, PROP_RESOLVED_ROOM, room_prior)
                if prev_subgoal is not None:
                    _rel(sg_ind, PROP_PREVIOUS_GOAL, prev_subgoal)
                prev_subgoal = sg_ind
                seq_idx += 1
                self._subgoal_inds[item.label] = sg_name

        if self.verbose:
            print(f"[Agent {self.agent_id}] Goal set → {goal.ont_class} ({self._active_goal})")


    def reset(self):
        self.Agent_ont = Ontology(self.tbox_path, load=True)
        self._known_cells.clear()
        self._known_objects.clear()
        self._known_rooms.clear()
        self._known_doors.clear()
        self._known_door_connections.clear()
        self._door_rooms.clear()
        self._locked_doors.clear()
        self._observed_goal_items.clear()
        self._active_goal = None
        self._goal_ind    = None
        self._subgoal_inds.clear()
        self._reasoner_time_ep = 0.0
        self._reasoner_calls_ep = 0
        self._agent_declared.clear()
        self._agent_room.clear()
        self._known_rooms_by_agent.clear()
        self._known_cells_by_agent.clear()
        self._goal_item_room.clear()
        self._goal_item_room_prior.clear()
        self._access_dirty = True
        self._accessibility_cache.clear()
        self._asserted_accessible.clear()
        self._all_delivered.clear()
        self._agent_carrying.clear()
        self._agent_carrying_axioms.clear()
        self._ont_reachable_cached = 0.5
        self._ep_reasoner_calls    = 0
        self._private_item_room.clear()
        self._private_observed_items.clear()

    def mark_item_delivered(self, item_label: str):
        """Assert isDelivered(item, target_room) in OWL ABox.
        Python mirror (_all_delivered) is updated separately by the caller.
        """
        if self._active_goal is None:
            return
        exp_name   = f"expected_{item_label.replace(' ', '_')}"
        item_ind   = OWLNamedIndividual(IRI.create(NS, exp_name))
        target_ind = OWLNamedIndividual(IRI.create(NS, self._active_goal.target_room))
        self.Agent_ont.add_axiom(OWLObjectPropertyAssertionAxiom(
            item_ind,
            OWLObjectProperty(IRI.create(NS, PROP_IS_DELIVERED)),
            target_ind,
        ))
        sg_name = self._subgoal_inds.get(item_label)
        if sg_name:
            self.Agent_ont.add_axiom(OWLClassAssertionAxiom(
                OWLNamedIndividual(IRI.create(NS, sg_name)),
                OWLClass(IRI.create(NS, "CompletedGoal")),
            ))

    def _get_view_exts(self, agent_pos, agent_dir, view_size: int = 7):
        if agent_dir == 0:
            topX = agent_pos[0]
            topY = agent_pos[1] - view_size // 2
        elif agent_dir == 1:
            topX = agent_pos[0] - view_size // 2
            topY = agent_pos[1]
        elif agent_dir == 2:
            topX = agent_pos[0] - view_size + 1
            topY = agent_pos[1] - view_size // 2
        else:
            topX = agent_pos[0] - view_size // 2
            topY = agent_pos[1] - view_size + 1
        return topX, topY, topX + view_size, topY + view_size

    def observations(self, observation: dict, agent_pos, env: HouseEnv, agent_id: int | None = None) -> dict:
        aid       = agent_id if agent_id is not None else self.agent_id
        agent     = env.agents[aid]
        agent_dir = agent.state.dir
        view_size = getattr(env, "agent_view_size", 7)

        room = env.get_room_for_cell(agent_pos[0], agent_pos[1])

        topX, topY, _, _ = self._get_view_exts(agent_pos, agent_dir, view_size)

        vis_mask     = gen_obs_grid_vis_mask(env.grid.state, env.agent_states, view_size)
        num_left_rot = (agent_dir + 1) % 4

        visible_objects      = []
        visible_world_coords = []
        width, height        = env.grid.width, env.grid.height

        for i_rot in range(view_size):
            for j_rot in range(view_size):
                if not vis_mask[aid, i_rot, j_rot]:
                    continue
                if   num_left_rot == 0: i, j = i_rot, j_rot
                elif num_left_rot == 1: i, j = view_size - j_rot - 1, i_rot
                elif num_left_rot == 2: i, j = view_size - i_rot - 1, view_size - j_rot - 1
                else:                   i, j = j_rot, view_size - i_rot - 1

                x, y = topX + i, topY + j
                if not (0 <= x < width and 0 <= y < height):
                    continue

                obj = env.grid.get(x, y)
                if obj is not None:
                    raw_type  = str(obj.type).replace("Type.", "")
                    norm_type = raw_type[len("labeled"):] if raw_type.startswith("labeled") else raw_type
                    visible_objects.append({
                        "type":      norm_type,
                        "color":     str(obj.color).replace("Color.", ""),
                        "label":     getattr(obj, "label", None),
                        "is_locked": getattr(obj, "is_locked", False),
                        "world_pos": (x, y),
                        "cell_name": f"Cell_{x}_{y}",
                    })
                visible_world_coords.append((x, y))

        return {
            "Agent_direction":      agent_dir,
            "Mission":              observation.get("mission", ""),
            "Agent_position":       agent_pos,
            "Current_room":         room,
            "TopX": topX, "TopY": topY,
            "BotX": topX + view_size, "BotY": topY + view_size,
            "Visible_world_coords": visible_world_coords,
            "Visible_cells":        [f"Cell_{x}_{y}" for x, y in visible_world_coords],
            "Visible_objects":      visible_objects,
        }


    def observations_to_ont(self, observations: dict, env: HouseEnv, agent_id: int | None = None):
        def cls(name):
            return OWLClass(IRI.create(NS, name))
        def ind(name):
            return OWLNamedIndividual(IRI.create(NS, name))
        def prop(name):
            return OWLObjectProperty(IRI.create(NS, name))

        def declare(individual):
            self.Agent_ont.add_axiom(OWLDeclarationAxiom(individual))

        def assert_class(individual, class_name):
            self.Agent_ont.add_axiom(OWLClassAssertionAxiom(individual, cls(class_name)))

        def assert_prop(subject, property_name, obj_individual):
            self.Agent_ont.add_axiom(OWLObjectPropertyAssertionAxiom(
                subject, prop(property_name), obj_individual
            ))

        aid = agent_id if agent_id is not None else self.agent_id

        agent_ind = ind(f"agent_{aid}")
        if not self._agent_declared.get(aid, False):
            declare(agent_ind)
            assert_class(agent_ind, "Agent")
            self._agent_declared[aid] = True

        current_room = observations["Current_room"]
        room_ind = None

        if current_room:
            room_ind = ind(current_room)

            if current_room not in self._known_rooms:
                self._known_rooms.add(current_room)
                declare(room_ind)
                assert_class(room_ind, "Room")

                room_obj = env.get_room_object(current_room)
                if room_obj:
                    info = room_obj.on_enter()
                    if info.get("description") and self.verbose:
                        print(f"[Agent {aid}] Entered '{current_room}': "
                              f"{info['description']}")

            self._known_rooms_by_agent.setdefault(aid, set()).add(current_room)

            if current_room != self._agent_room.get(aid):
                assert_prop(agent_ind, PROP_LOCATED_IN, room_ind)
                self._agent_room[aid] = current_room

        for cell_name in observations["Visible_cells"]:
            self._known_cells.add(cell_name)
            self._known_cells_by_agent.setdefault(aid, set()).add(cell_name)

        for obj in observations["Visible_objects"]:
            object_type: str = obj["type"]
            x, y             = obj["world_pos"]
            obj_key          = f"{object_type}_{x}_{y}"
            label            = obj.get("label")

            if object_type in ("floor", "wall"):
                continue

            if object_type != "door":
                if obj_key in self._known_objects:
                    continue
                self._known_objects.add(obj_key)

            if object_type == "door":
                door_label = label
                door_name  = (f"{door_label.replace(' ', '_')}"
                              if door_label else f"door_{x}_{y}")
                is_locked  = obj.get("is_locked", False)

                if current_room:
                    self._door_rooms.setdefault(door_name, set()).add(current_room)
                prev_locked = door_name in self._locked_doors
                if is_locked:
                    self._locked_doors.add(door_name)
                else:
                    self._locked_doors.discard(door_name)
                if door_name not in self._door_rooms or prev_locked != is_locked:
                    self._access_dirty = True

                door_individual = ind(door_name)
                if door_name not in self._known_doors:
                    self._known_doors.add(door_name)
                    declare(door_individual)
                    assert_class(door_individual, "Door")
                    assert_class(door_individual, "LockedDoor" if is_locked else "OpenDoor")

                if current_room and room_ind is not None:
                    conn_key = f"{door_name}|{current_room}"
                    if conn_key not in self._known_door_connections:
                        self._known_door_connections.add(conn_key)
                        connects_prop      = PROP_CONNECTS_CLOSED if is_locked else PROP_CONNECTS_OPEN
                        room_connects_prop = PROP_ROOM_CONNECTS_CLOSED if is_locked else PROP_ROOM_CONNECTS_OPEN
                        assert_prop(door_individual, connects_prop, room_ind)
                        assert_prop(room_ind, room_connects_prop, door_individual)
                        if self.verbose:
                            door_type = "locked" if is_locked else "open"
                            print(f"[Agent {aid}] Door '{door_name}' ({door_type})"
                                  f" — connects to '{current_room}'")
                continue

            is_goal_item = (label and self._active_goal
                            and label in self._active_goal.target_items)

            ind_name = (f"{label.replace(' ', '_')}_{x}_{y}"
                        if label else f"{object_type}_{x}_{y}")
            item_ind = ind(ind_name)
            declare(item_ind)
            assert_class(item_ind, "Item")

            if label in ITEM_LABEL_MAP:
                base_cls, specific_cls = ITEM_LABEL_MAP[label]
                assert_class(item_ind, base_cls)
                assert_class(item_ind, specific_cls)
            elif object_type in TYPE_FALLBACK:
                assert_class(item_ind, TYPE_FALLBACK[object_type])

            if room_ind is not None:
                assert_prop(item_ind, PROP_LOCATED_IN, room_ind)
                if is_goal_item:
                    assert_prop(item_ind, PROP_PROBABLY_IN, room_ind)

            if is_goal_item:
                exp_name = f"expected_{label.replace(' ', '_')}"
                exp_ind  = ind(exp_name)
                self.Agent_ont.add_axiom(OWLSameIndividualAxiom([item_ind, exp_ind]))
                self._private_observed_items.setdefault(aid, {})[label] = ind_name
                if current_room:
                    self._private_item_room.setdefault(aid, {})[label] = current_room

                if self._goal_ind:
                    assert_prop(self._goal_ind, PROP_HAS_TARGET_OBJ, item_ind)

                if self.verbose:
                    print(f"[Agent {aid}] ✓ Goal item observed: '{label}' "
                          f"at ({x},{y}) in room '{current_room}'")


    @contextmanager
    def _reasoner_timer(self):
        """Accumulate wall-clock time spent constructing + querying HermiT.
        Wrap every reasoner-using block so the timing stats stay accurate."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._reasoner_time_ep    += dt
            self._reasoner_time_total += dt
            self._reasoner_calls_ep    += 1
            self._reasoner_calls_total += 1

    def reasoner_stats(self) -> dict:
        """Snapshot of reasoner timing — surfaced to the training logger."""
        ce, ct = self._reasoner_calls_ep, self._reasoner_calls_total
        return {
            "reasoner_calls_ep":    ce,
            "reasoner_time_ep":     round(self._reasoner_time_ep, 4),
            "reasoner_ms_per_call": round(1000 * self._reasoner_time_ep / ce, 2) if ce else 0.0,
            "reasoner_calls_total": ct,
            "reasoner_time_total":  round(self._reasoner_time_total, 2),
        }

    def _get_reasoner(self):
        save_path = f"/tmp/agent_{self.agent_id}_{os.getpid()}_abox.owl"
        with open(os.devnull, "w") as _devnull:
            _stdout, sys.stdout = sys.stdout, _devnull
            try:
                self.Agent_ont.save(path=save_path)
            finally:
                sys.stdout = _stdout
        return SyncReasoner(save_path, reasoner="HermiT")

    def get_goal_classification(self) -> dict:
        if not self._goal_ind or not self._active_goal:
            return {}
        try:
            reasoner = self._get_reasoner()
            inferred_types = [
                t.iri.remainder
                for t in reasoner.types(self._goal_ind, direct=False)
                if t.iri.namespace == NS
            ]
            resolved = list(reasoner.object_property_values(
                self._goal_ind, prop_q(PROP_RESOLVED_ROOM)))
            return {
                "goal_class":     self._active_goal.ont_class,
                "inferred_types": inferred_types,
                "resolved_rooms": [r.iri.remainder for r in resolved],
            }
        except Exception as e:
            return {"error": str(e)}

    def _print_reasoner_summary(self):
        try:
            reasoner = self._get_reasoner()

            rooms_known    = list(reasoner.instances(cls_q("Room", NS)))
            items_observed = list(reasoner.instances(cls_q("Item", NS)))

            found_labels  = list(self._observed_goal_items.keys())
            total_targets = (len(self._active_goal.target_items)
                             if self._active_goal else 0)
            goal_type = self._active_goal.ont_class if self._active_goal else "None"

            resolved_rooms   = []
            resolved_objects = []
            if self._goal_ind:
                resolved_rooms = list(reasoner.object_property_values(
                    self._goal_ind, prop_q(PROP_RESOLVED_ROOM)))
                resolved_objects = list(reasoner.object_property_values(
                    self._goal_ind, prop_q(PROP_RESOLVED_OBJ)))

            resolved_room_names = [r.iri.remainder for r in resolved_rooms]
            resolved_obj_names  = [o.iri.remainder for o in resolved_objects]

            goal_classes = (
                [t.iri.remainder for t in reasoner.types(self._goal_ind, direct=False)
                 if t.iri.namespace == NS]
                if self._goal_ind else []
            )

            print(
                f"[Reasoner | Agent {self.agent_id}] "
                f"GoalType: {goal_type} | "
                f"Rooms known: {len(rooms_known)} | "
                f"Items in ABox: {len(items_observed)} | "
                f"Goal items found: {len(found_labels)}/{total_targets}"
                + (f" — {found_labels}" if found_labels else "")
            )
            print(
                f"[Reasoner | Agent {self.agent_id}] "
                f"HasResolvedTargetRoom: {resolved_room_names or 'none'} | "
                f"HasResolvedTargetObject ({len(resolved_objects)}): "
                + (str(resolved_obj_names[:5]) + ("…" if len(resolved_obj_names) > 5 else "")
                   if resolved_obj_names else "none")
            )
            print(
                f"[Reasoner | Agent {self.agent_id}] "
                f"Goal inferred types: {goal_classes or ['unclassified']}"
            )

            if self._active_goal and total_targets > 0:
                self._check_goal_completion(reasoner)

        except Exception as e:
            print(f"[Reasoner warning] {e}")

    def _check_goal_completion(self, reasoner):
        if not self._active_goal:
            return

        target_room   = self._active_goal.target_room
        target_items  = set(self._active_goal.target_items)
        delivered     = set()

        located_in_prop = OWLObjectProperty(IRI.create(NS, PROP_LOCATED_IN))

        for label, ind_name in self._observed_goal_items.items():
            item_ind = OWLNamedIndividual(IRI.create(NS, ind_name))
            rooms = list(reasoner.object_property_values(item_ind, located_in_prop))
            for r in rooms:
                if r.iri.remainder == target_room:
                    delivered.add(label)

        remaining = target_items - set(self._observed_goal_items.keys())
        print(
            f"[Goal reasoning] '{self._active_goal.name}' | "
            f"Observed {len(self._observed_goal_items)}/{len(target_items)} | "
            f"Delivered to '{target_room}': {len(delivered)} | "
            f"Still unseen: {len(remaining)}"
            + (f" — {list(remaining)[:5]}{'…' if len(remaining) > 5 else ''}"
               if remaining else " — ALL FOUND ✓")
        )

    @staticmethod
    def _get_adjacent_rooms(env: HouseEnv, door_x: int, door_y: int) -> list[str]:
        adjacent: set[str] = set()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = door_x + dx, door_y + dy
            if 0 <= nx < env.width and 0 <= ny < env.height:
                room = env.get_room_for_cell(nx, ny)
                if room:
                    adjacent.add(room)
        return list(adjacent)

    def get_goal_items_observed(self) -> list[str]:
        return list(self._observed_goal_items.keys())

    def get_goal_items_remaining(self) -> list[str]:
        if not self._active_goal:
            return []
        seen = set(self._observed_goal_items.keys())
        return [lbl for lbl in self._active_goal.target_items if lbl not in seen]

    def get_known_rooms(self) -> list[str]:
        return list(self._known_rooms)

    def _compute_accessible_rooms(self, start_room: str | None) -> set[str]:
        if not start_room:
            return set()
        if not self._access_dirty and start_room in self._accessibility_cache:
            return self._accessibility_cache[start_room]

        if self._access_dirty:
            self._accessibility_cache.clear()
            self._access_dirty = False
            for door_name, rooms in self._door_rooms.items():
                if door_name not in self._locked_doors:
                    rooms_list = list(rooms)
                    if len(rooms_list) == 2:
                        a, b = rooms_list
                        for src, dst in [(a, b), (b, a)]:
                            if (src, dst) not in self._asserted_accessible:
                                self._asserted_accessible.add((src, dst))
                                self.Agent_ont.add_axiom(OWLObjectPropertyAssertionAxiom(
                                    OWLNamedIndividual(IRI.create(NS, src)),
                                    OWLObjectProperty(IRI.create(NS, PROP_ACCESSIBLE_TO)),
                                    OWLNamedIndividual(IRI.create(NS, dst)),
                                ))

        accessible: set[str] = {start_room}
        queue = [start_room]
        while queue:
            room = queue.pop()
            for door_name, rooms in self._door_rooms.items():
                if room in rooms and door_name not in self._locked_doors:
                    for neighbour in rooms:
                        if neighbour not in accessible:
                            accessible.add(neighbour)
                            queue.append(neighbour)
        self._accessibility_cache[start_room] = accessible
        return accessible

    def _check_ont_reachability(self) -> None:
        if not self._goal_ind:
            return
        with self._reasoner_timer():
            try:
                reasoner = self._get_reasoner()
                types = {
                    t.iri.remainder
                    for t in reasoner.types(self._goal_ind, direct=False)
                    if t.iri.namespace == NS
                }
                if "ReachableGoal" in types:
                    self._ont_reachable_cached = 1.0
                elif "UnreachableGoal" in types:
                    self._ont_reachable_cached = 0.0
            except Exception:
                pass
        self._ep_reasoner_calls += 1

    def assert_goal_conflict(self, item_label: str, other_subgoal_name: str) -> None:
        """Assert conflictsWith between this agent's subgoal for `item_label` and
        another agent's subgoal — so HermiT infers CompetingGoal (Q7 diversion).
        conflictsWith is symmetric in the TBox, so one assertion suffices."""
        sg_name = self._subgoal_inds.get(item_label)
        if not sg_name:
            return
        self.Agent_ont.add_axiom(OWLObjectPropertyAssertionAxiom(
            OWLNamedIndividual(IRI.create(NS, sg_name)),
            OWLObjectProperty(IRI.create(NS, PROP_CONFLICTS_WITH)),
            OWLNamedIndividual(IRI.create(NS, other_subgoal_name)),
        ))

    def competency_report(self) -> dict:
        """Answer the ontology competency questions with a single DL-reasoning
        pass.  For on-demand evaluation / thesis demonstration — NOT the per-step
        hot loop (one HermiT build; cost recorded by the reasoner timer)."""
        if not self._goal_ind:
            return {}
        goal_name = self._goal_ind.iri.remainder
        ordered   = list(self._subgoal_inds.values())
        with self._reasoner_timer():
            try:
                r = self._get_reasoner()
                def insts(c):       return {i.iri.remainder for i in r.instances(cls_q(c, NS))}
                def pvals(name, p): return sorted(
                    o.iri.remainder for o in r.object_property_values(
                        OWLNamedIndividual(IRI.create(NS, name)), prop_q(p)))

                skipped    = insts("SkippedGoal")
                completed  = insts("CompletedGoal")
                subgoals   = pvals(goal_name, PROP_HAS_SUBGOAL)
                remaining  = [sg for sg in ordered if sg not in skipped]
                next_goal  = remaining[0] if remaining else None

                return {
                    "Q1_which_room":   pvals(next_goal, PROP_RESOLVED_ROOM) if next_goal else [],
                    "Q2_which_object": pvals(next_goal, PROP_HAS_TARGET_OBJ) if next_goal else [],
                    "Q3_skip":         sorted(skipped),
                    "Q4_next_goal":    next_goal,
                    "Q5_subgoals":     sorted(subgoals),
                    "Q6_reachable":    goal_name in insts("ReachableGoal"),
                    "Q7_divert":       sorted(insts("CompetingGoal")),
                    "Q8_reward":       {
                        "completed_subgoals": sorted(completed),
                        "all_done": bool(subgoals) and len(completed) >= len(subgoals),
                    },
                }
            except Exception as e:
                return {"error": str(e)}


    def _knowledge_diff(self, agent_id: int) -> dict[str, dict]:
        """Return what agent_id has privately observed that is absent from the shared pool.

        owlapy has no built-in ontology-merge API, so we diff the Python-level
        dicts here and rely on get_abox_axioms() only if a full OWL-level merge
        is ever needed (see _share_knowledge).
        """
        priv_items = self._private_item_room.get(agent_id, {})
        priv_obs   = self._private_observed_items.get(agent_id, {})
        return {
            "item_rooms": {k: v for k, v in priv_items.items() if k not in self._goal_item_room},
            "observed":   {k: v for k, v in priv_obs.items() if k not in self._observed_goal_items},
        }

    def _share_knowledge(self, agent_id: int) -> int:
        """Flush agent_id's private observations into the shared pool.

        For the OWL ABox the axioms were already asserted by observations_to_ont
        (the shared Agent_ont receives everything); what we gate here are the
        Python-level dicts that drive state features and guidance rewards.

        Returns the number of new facts added to the shared pool.
        """
        diff = self._knowledge_diff(agent_id)
        self._goal_item_room.update(diff["item_rooms"])
        self._observed_goal_items.update(diff["observed"])
        return len(diff["item_rooms"]) + len(diff["observed"])


def cls_q(name: str, ns: str) -> OWLClass:
    return OWLClass(IRI.create(ns, name))

def ind_q(name):
    return OWLNamedIndividual(IRI.create(NS, name))

def prop_q(name):
    return OWLObjectProperty(IRI.create(NS, name))


_ROOMS = [
    "garden", "foyer", "living_room", "kitchen", "dining_room",
    "bedroom", "bathroom", "hallway", "study", "right_wing",
]

_ENV_KEYS = ["front door key", "garden gate key"]


class DQNAgent(Agent):
    """
    State vector layout
    -------------------
      [0:147]        flattened 7x7x3 partial-obs image  (÷10 normalised)
      [147:151]      agent direction one-hot (4)
      [151:161]      current room one-hot    (10)
      [161:171]      known rooms binary      (10)
      [171:171+N]    goal items observed binary (N = len goal.target_items)
      [171+N:181+N]  target room one-hot     (10)
      [181+N:191+N]  goal items per room     (10)
      [191+N:201+N]  accessible rooms binary (10)
      [201+N:201+2N]  items carried by other agents (N)
      [201+2N]        keys progress           0..1
      [202+2N]        balls progress          0..1
      [203+2N]        carrying flag           0 or 1
      [204+2N]        goal room reachable     0 or 1
      [205+2N]        unreachable items frac  0..1
      [206+2N]        ont goal reachable      0/0.5/1
    """

    N_ACTIONS = 7

    def __init__(
        self,
        ont_path: str,
        goal,
        agent_id: int = 0,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update_freq: int = 500,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 100_000,
        device: str = "cpu",
        proximity_threshold: int = 0,
    ):
        super().__init__(ont_path, agent_id, verbose=False)
        self.goal               = goal
        self.proximity_threshold = proximity_threshold
        self.device      = torch.device(device)
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_freq = target_update_freq
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay

        n_items   = len(goal.target_items)
        state_dim = 7 * 7 * 3 + 4 + 10 + 10 + n_items + 10 + 10 + 10 + n_items + n_items + 6

        def _mlp():
            return nn.Sequential(
                nn.Linear(state_dim, 256), nn.ReLU(),
                nn.Linear(256, 128),       nn.ReLU(),
                nn.Linear(128, self.N_ACTIONS),
            ).to(self.device)

        self.policy_net  = _mlp()
        self.target_net  = _mlp()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer   = optim.Adam(self.policy_net.parameters(), lr=lr)
        self._buffer     = deque(maxlen=buffer_size)
        self.total_steps = 0

    def _ont_features(self, current_room, keys_c, balls_d, carrying, agent_id: int = 0) -> np.ndarray:
        known     = set(self.get_known_rooms())
        observed  = set(self._observed_goal_items) | set(self._private_observed_items.get(agent_id, {}))
        delivered = self._all_delivered

        current_oh    = [float(r == current_room) for r in _ROOMS]
        known_bin     = [float(r in known)         for r in _ROOMS]
        items_bin     = [float(it in observed)      for it in self.goal.target_items]
        target_oh     = [float(r == self.goal.target_room) for r in _ROOMS]
        delivered_bin = [float(it in delivered)     for it in self.goal.target_items]

        items_per_room = [
            float(any(
                self._goal_item_room_prior.get(lbl) == room and lbl not in delivered
                for lbl in self.goal.target_items
            ))
            for room in _ROOMS
        ]

        accessible     = self._compute_accessible_rooms(current_room)
        accessible_bin = [float(r in accessible) for r in _ROOMS]
        goal_reachable = float(self.goal.target_room in accessible)
        unreachable_items = sum(
            1 for lbl in self.goal.target_items
            if lbl not in delivered
            and self._goal_item_room_prior.get(lbl) is not None
            and self._goal_item_room_prior.get(lbl) not in accessible
        ) / max(len(self.goal.target_items), 1)

        others_carrying = {
            lbl for j, lbl in self._agent_carrying.items()
            if j != agent_id and lbl is not None
        }
        items_carried_by_others = [float(it in others_carrying) for it in self.goal.target_items]

        scalars = [
            len(keys_c)  / 2.0,
            len(balls_d) / max(len(self.goal.target_items), 1),
            float(carrying),
            goal_reachable,
            unreachable_items,
            self._ont_reachable_cached,
        ]
        return np.array(
            current_oh + known_bin + items_bin + target_oh
            + items_per_room + accessible_bin + items_carried_by_others + delivered_bin + scalars,
            dtype=np.float32,
        )

    def get_state(self, obs: dict, current_room, keys_c, balls_d, carrying, agent_id: int = 0) -> np.ndarray:
        img   = obs["image"].flatten().astype(np.float32) / 10.0
        dir_v = np.zeros(4, dtype=np.float32)
        dir_v[int(obs.get("direction", 0)) % 4] = 1.0
        return np.concatenate([img, dir_v, self._ont_features(current_room, keys_c, balls_d, carrying, agent_id)])

    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -self.total_steps / self.eps_decay
        )

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon():
            return random.randrange(self.N_ACTIONS)
        t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(t).argmax(1).item())

    def _train_step(self) -> float | None:
        if len(self._buffer) < self.batch_size:
            return None
        s, a, r, ns, d = zip(*random.sample(self._buffer, self.batch_size))

        s_t  = torch.tensor(np.array(s),                 dtype=torch.float32, device=self.device)
        ns_t = torch.tensor(np.array(ns),                dtype=torch.float32, device=self.device)
        a_t  = torch.tensor(np.array(a, dtype=np.int64), dtype=torch.long,    device=self.device)
        r_t  = torch.tensor(np.array(r, dtype=np.float32), dtype=torch.float32, device=self.device)
        d_t  = torch.tensor(np.array(d, dtype=np.float32), dtype=torch.float32, device=self.device)

        q     = self.policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r_t + self.gamma * self.target_net(ns_t).max(1).values * (1.0 - d_t)

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.policy_net.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-1.0, 1.0)
        self.optimizer.step()

        if self.total_steps % self.target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def run_episode(self, env, render: bool = False) -> dict:
        self.reset()
        self.set_goal(self.goal)
        self._check_ont_reachability()

        obs_dict, info = env.reset()
        n_agents = len(env.agents)

        self._all_delivered = set().union(
            *(set(info[i].get("agent_balls_delivered", [])) for i in range(n_agents))
        )
        for i in range(n_agents):
            self._agent_carrying[i]        = None
            self._agent_carrying_axioms[i] = None

        agent_states = []
        for i in range(n_agents):
            pos      = env.agents[i].state.pos
            obs_data = self.observations(obs_dict[i], pos, env, agent_id=i)
            self.observations_to_ont(obs_data, env, agent_id=i)
            keys_c   = info[i].get("agent_keys_collected", [])
            balls_d  = info[i].get("agent_balls_delivered", [])
            carrying = getattr(env.agents[i].state, "carrying", None) is not None
            agent_states.append(self.get_state(obs_dict[i], obs_data["Current_room"], keys_c, balls_d, carrying, agent_id=i))
        if self.proximity_threshold > 0:
            for _i in range(n_agents):
                for _j in range(_i + 1, n_agents):
                    pi = env.agents[_i].state.pos
                    pj = env.agents[_j].state.pos
                    if abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) <= self.proximity_threshold:
                        self._share_knowledge(_i)
                        self._share_knowledge(_j)

        total_reward   = 0.0
        loss_sum       = 0.0
        loss_cnt       = 0
        ont_reward_sum = 0.0
        ep_steps       = 0
        prev_observed  = set()

        rc_time      = 0.0
        rc_explore   = 0.0
        rc_carrying  = 0.0
        rc_guidance  = 0.0
        rc_pickup    = 0.0
        rc_penalties = 0.0
        rc_delivery  = 0.0
        rc_complete  = 0.0
        rc_observe   = 0.0

        prev_room_by_agent:  dict[int, str | None]      = {i: None for i in range(n_agents)}
        ep_pickup_count:     dict[tuple[int, str], int] = {}
        ep_guided_rooms:     dict[int, set[str]]        = {i: set() for i in range(n_agents)}
        ep_goal_guided:      dict[int, set[str]]        = {i: set() for i in range(n_agents)}
        ep_key_collected:    dict[int, set[str]]        = {i: set() for i in range(n_agents)}
        ep_reachable_bonus_given: bool = False
        ep_complete_given:        bool = False
        ep_door_unlocked:         set[str] = set()

        while True:
            actions = {i: self.select_action(agent_states[i]) for i in range(n_agents)}
            self.total_steps += 1
            ep_steps       += 1

            prev_rooms_by_agent    = {i: set(self._known_rooms_by_agent.get(i, set())) for i in range(n_agents)}
            prev_carrying_by_agent = {i: getattr(env.agents[i].state, "carrying", None) for i in range(n_agents)}
            prev_balls_by_agent    = {i: set(env._balls_delivered.get(i, []))            for i in range(n_agents)}
            prev_keys_by_agent     = {i: set(ep_key_collected[i])                        for i in range(n_agents)}
            prev_all_delivered     = set().union(*prev_balls_by_agent.values())
            prev_observed          = set(self._observed_goal_items)
            for _pid in range(n_agents):
                prev_observed.update(self._private_observed_items.get(_pid, {}))

            obs_next, rewards, terminations, truncations, infos = env.step(actions)
            done = bool(terminations[0]) or bool(truncations[0])

            next_states = []
            for i in range(n_agents):
                pos      = env.agents[i].state.pos
                obs_data = self.observations(obs_next[i], pos, env, agent_id=i)
                self.observations_to_ont(obs_data, env, agent_id=i)
                keys_c   = infos[i].get("agent_keys_collected", [])
                balls_d  = infos[i].get("agent_balls_delivered", [])
                carrying = getattr(env.agents[i].state, "carrying", None) is not None
                next_states.append(self.get_state(obs_next[i], obs_data["Current_room"], keys_c, balls_d, carrying, agent_id=i))

            if self.proximity_threshold > 0:
                for _i in range(n_agents):
                    for _j in range(_i + 1, n_agents):
                        pi = env.agents[_i].state.pos
                        pj = env.agents[_j].state.pos
                        if abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) <= self.proximity_threshold:
                            self._share_knowledge(_i)
                            self._share_knowledge(_j)
            else:
                pass

            self._all_delivered = set().union(
                *(set(infos[i].get("agent_balls_delivered", [])) for i in range(n_agents))
            )

            for i in range(n_agents):
                now_c  = getattr(env.agents[i].state, "carrying", None)
                g_lbl  = getattr(now_c, "label", None)
                g_lbl  = g_lbl if (g_lbl and g_lbl in self.goal.target_items) else None
                if self._agent_carrying.get(i) != g_lbl:
                    if self._agent_carrying_axioms.get(i) is not None:
                        try:
                            self.Agent_ont.remove_axiom(self._agent_carrying_axioms[i])
                        except Exception:
                            pass
                        self._agent_carrying_axioms[i] = None
                    if g_lbl:
                        a_ind  = OWLNamedIndividual(IRI.create(NS, f"agent_{i}"))
                        it_ind = OWLNamedIndividual(IRI.create(NS, f"expected_{g_lbl.replace(' ', '_')}"))
                        axiom  = OWLObjectPropertyAssertionAxiom(
                            a_ind, OWLObjectProperty(IRI.create(NS, PROP_IS_CARRYING)), it_ind)
                        self.Agent_ont.add_axiom(axiom)
                        self._agent_carrying_axioms[i] = axiom
                    self._agent_carrying[i] = g_lbl

            for i in range(n_agents):
                ep_key_collected[i] = set(infos[i].get("agent_keys_collected", []))

            now_observed = set(self._observed_goal_items)
            for _pid in range(n_agents):
                now_observed.update(self._private_observed_items.get(_pid, {}))
            newly_seen    = now_observed - prev_observed
            prev_observed = now_observed

            shapings = {}
            for i in range(n_agents):
                s = 0.0
                _time = 0.0
                _eff_item_room_i = {**self._goal_item_room, **self._private_item_room.get(i, {})}
                _eff_observed_i  = set(self._observed_goal_items) | set(self._private_observed_items.get(i, {}))

                new_rooms = self._known_rooms_by_agent.get(i, set()) - prev_rooms_by_agent[i]
                _expl = 3.0 * len(new_rooms)
                if self.goal.target_room in new_rooms:
                    _expl += 5.0
                s += _expl

                now_carrying  = getattr(env.agents[i].state, "carrying", None)
                prev_carrying = prev_carrying_by_agent[i]
                carrying_goal = (now_carrying is not None
                                 and hasattr(now_carrying, "label")
                                 and now_carrying.label in self.goal.target_items)

                agent_room_i = self._agent_room.get(i)
                in_goal_room = (agent_room_i == self.goal.target_room)
                entered_room = (agent_room_i != prev_room_by_agent[i] and agent_room_i is not None)

                _guide = 0.0
                if entered_room and in_goal_room and carrying_goal:
                    already_delivered = now_carrying.label in infos[i].get("agent_balls_delivered", [])
                    if not already_delivered and now_carrying.label not in ep_goal_guided[i]:
                        _guide += 3.0
                        if now_carrying.label in _eff_observed_i:
                            _guide += 1.0
                        ep_goal_guided[i].add(now_carrying.label)
                elif entered_room and agent_room_i and not in_goal_room:
                    if agent_room_i not in ep_guided_rooms[i]:
                        for lbl, known_room in _eff_item_room_i.items():
                            if known_room == agent_room_i and lbl not in self._all_delivered:
                                _guide += 1.0
                                ep_guided_rooms[i].add(agent_room_i)
                                break
                        if agent_room_i not in ep_guided_rooms[i]:
                            for lbl, prior_room in self._goal_item_room_prior.items():
                                if (prior_room == agent_room_i
                                        and lbl not in _eff_observed_i
                                        and lbl not in self._all_delivered):
                                    _guide += 2.0
                                    ep_guided_rooms[i].add(agent_room_i)
                                    break
                s += _guide

                just_picked_goal = (prev_carrying is None and carrying_goal)
                just_dropped_goal = (
                    prev_carrying is not None
                    and hasattr(prev_carrying, "label")
                    and prev_carrying.label in self.goal.target_items
                    and now_carrying is None
                    and agent_room_i != self.goal.target_room
                )
                _carry = 0.0
                if (carrying_goal
                        and not in_goal_room
                        and now_carrying.label not in self._all_delivered):
                    _carry += 0.1

                _pick  = 0.0
                _pen   = 0.0
                if just_picked_goal:
                    key = (i, now_carrying.label)
                    if ep_pickup_count.get(key, 0) == 0:
                        _pick += 3.0
                        if now_carrying.label in _eff_observed_i:
                            _pick += 0.5
                    elif ep_pickup_count.get(key, 0) == 1:
                        _pen -= 3.0
                    ep_pickup_count[key] = ep_pickup_count.get(key, 0) + 1
                    if now_carrying.label in prev_all_delivered:
                        _pen -= 2.0
                    for j in range(n_agents):
                        if j != i and prev_carrying_by_agent[j] is now_carrying:
                            _pen -= 2.0
                            break
                if just_dropped_goal:
                    _pen -= 1.0
                s += _carry + _pick + _pen


                curr_balls_i    = set(infos[i].get("agent_balls_delivered", []))
                newly_delivered = curr_balls_i - prev_balls_by_agent[i]
                n_already = len(prev_all_delivered)
                _deliv = 0.0
                for lbl in newly_delivered:
                    scale   = 1.0 + 0.3 * n_already
                    _deliv += 8.0 * scale
                    if lbl in _eff_observed_i:
                        _deliv += 1.5 * scale
                    n_already += 1
                s += _deliv

                newly_got_keys = ep_key_collected[i] - prev_keys_by_agent[i]
                _key = 5.0 * len(newly_got_keys)
                s += _key

                _unlock = 0.0

                _comp = 0.0
                if infos[i].get("task_complete", False) and not ep_complete_given:
                    _comp = 500.0
                    s    += _comp
                if i == n_agents - 1 and infos[i].get("task_complete", False):
                    ep_complete_given = True

                _obs = 0.3 * len(newly_seen)
                s   += _obs

                _lava = 0.0
                _pos = env.agents[i].state.pos
                for _dx, _dy in ((-1,0),(1,0),(0,-1),(0,1)):
                    _cell = env.grid.get(_pos[0]+_dx, _pos[1]+_dy)
                    if _cell is not None and _cell.type == "lava":
                        _lava -= 1.0
                        break
                s += _lava

                _ont_reach = 0.0
                if self._ont_reachable_cached == 1.0 and not ep_reachable_bonus_given:
                    _ont_reach = 10.0
                    s += _ont_reach
                ep_reachable_bonus_given = ep_reachable_bonus_given or (self._ont_reachable_cached == 1.0)

                shapings[i] = s

                rc_time      += _time
                rc_explore   += _expl
                rc_carrying  += _carry
                rc_guidance  += _guide
                rc_pickup    += _pick + _key + _unlock
                rc_penalties += _pen + _lava
                rc_delivery  += _deliv
                rc_complete  += _comp
                rc_observe   += _obs

            for i in range(n_agents):
                prev_room_by_agent[i] = self._agent_room.get(i)

            ont_reward_sum += shapings[0]

            for i in range(n_agents):
                r_i    = float(rewards[i]) + shapings[i]
                done_i = bool(terminations[i]) or bool(truncations[i])
                self._buffer.append((agent_states[i], actions[i], r_i, next_states[i], float(done_i)))

            total_reward += float(rewards[0]) + shapings[0]

            loss = self._train_step()
            if loss is not None:
                loss_sum += loss
                loss_cnt += 1

            agent_states = next_states
            if render:
                env.render()
            if done:
                break

        keys_pooled = len(set().union(*(set(infos[i].get("agent_keys_collected", [])) for i in range(n_agents))))

        return {
            "total_reward":            total_reward,
            "avg_loss":                loss_sum / max(loss_cnt, 1),
            "epsilon":                 self.epsilon(),
            "steps":                   ep_steps,
            "task_complete":           bool(infos[0].get("task_complete", False)),
            "keys_collected":          len(infos[0].get("agent_keys_collected", [])),
            "keys_pooled":             keys_pooled,
            "balls_delivered":         len(infos[0].get("all_balls_delivered", [])),
            "rooms_explored":          len(self.get_known_rooms()),
            "items_observed":          len(
                set(self._observed_goal_items)
                | set().union(*[set(d) for d in self._private_observed_items.values()])
            ),
            "items_total":             len(self.goal.target_items),
            "ont_goal_items":          list(
                set(self._observed_goal_items)
                | set().union(*[set(d) for d in self._private_observed_items.values()])
            ),
            "ont_rooms":               self.get_known_rooms(),
            "target_room":             self.goal.target_room,
            "target_room_visited":     self.goal.target_room in self._known_rooms,
            "ont_goal_class":          self.goal.ont_class,
            "ont_reward_total":        ont_reward_sum,
            "ont_reachable":           self._ont_reachable_cached,
            "rc_time":                 rc_time,
            "rc_explore":              rc_explore,
            "rc_carrying":             rc_carrying,
            "rc_guidance":             rc_guidance,
            "rc_pickup":               rc_pickup,
            "rc_penalties":            rc_penalties,
            "rc_delivery":             rc_delivery,
            "rc_complete":             rc_complete,
            "rc_observe":              rc_observe,
        }

    def save(self, path: str):
        torch.save({
            "policy":    self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps":     self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["steps"]
