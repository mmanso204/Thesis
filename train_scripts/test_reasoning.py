from envs.environment import HouseEnv, LabeledKey, LabeledBall, LabeledBox
from helper_functions.Ontology_builder import Ontology_builder
from minigrid.core.world_object import Door, Wall, Lava, Floor
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLObjectPropertyAssertionAxiom, OWLClassAssertionAxiom

ONTOLOGY_PATH = "/Users/m.manso/Downloads/thesisont_updated-2.owl"
NS = "http://www.semanticweb.org/m.manso/ontologies/2026/3/untitled-ontology-30#"


AGENTS = {
    "Agent_Alice": (13, 10),
    "Agent_Bob":   (21, 10),
    "Agent_Carol": (22, 18),
}


def build_indexes(ont):
    manager = ont.get_manager()
    axioms = manager.get_axioms(ont)

    cell_to_room = {}
    room_to_cells = {}

    for ax in axioms:
        if isinstance(ax, OWLObjectPropertyAssertionAxiom):
            if ax.get_property().iri.remainder == "InRoom":
                cell = ax.get_subject().iri.remainder
                room = ax.get_object().iri.remainder
                cell_to_room[cell] = room
                room_to_cells.setdefault(room, []).append(cell)

    return cell_to_room, room_to_cells


def get_room_of_cell(cell_to_room, x, y):
    return cell_to_room.get(f"Cell_{x}_{y}", "unknown")


def get_cells_in_room(room_to_cells, room_name):
    return room_to_cells.get(room_name, [])


def observe(env, pos):
    ax, ay = pos
    obs = {"position": pos, "room": env.get_room_for_cell(ax, ay), "visible_objects": []}
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            x, y = ax + dx, ay + dy
            if not (0 <= x < env.width and 0 <= y < env.height):
                continue
            cell = env.grid.get(x, y)
            if cell is None or isinstance(cell, Floor):
                continue
            entry = {"pos": (x, y), "type": type(cell).__name__}
            if hasattr(cell, "label"):
                entry["label"] = cell.label
            if hasattr(cell, "color"):
                entry["color"] = cell.color
            if isinstance(cell, Door):
                entry["locked"] = cell.is_locked
            obs["visible_objects"].append(entry)
    return obs


def derive(cell_to_room, room_to_cells, env, obs):
    derived = {}

    x, y = obs["position"]
    derived["my_room"] = get_room_of_cell(cell_to_room, x, y)

    visible_keys = [o for o in obs["visible_objects"] if o["type"] == "LabeledKey"]
    derived["visible_keys"] = [
        {"label": k["label"], "pos": k["pos"],
         "in_room": get_room_of_cell(cell_to_room, *k["pos"])}
        for k in visible_keys
    ]

    visible_balls = [o for o in obs["visible_objects"] if o["type"] == "LabeledBall"]
    derived["visible_balls"] = [
        {"label": b["label"], "pos": b["pos"],
         "in_room": get_room_of_cell(cell_to_room, *b["pos"])}
        for b in visible_balls
    ]

    visible_doors = [o for o in obs["visible_objects"] if o["type"] == "Door"]
    derived["nearby_doors"] = [
        {"color": d["color"], "pos": d["pos"], "locked": d["locked"],
         "connects_to": get_room_of_cell(cell_to_room, *d["pos"])}
        for d in visible_doors
    ]

    room_cells = get_cells_in_room(room_to_cells, derived["my_room"])
    derived["my_room_cell_count"] = len(room_cells)

    derived["can_pickup"] = [k["label"] for k in derived["visible_keys"]]

    return derived


def report(name, obs, derived):
    print(f"\n{'='*55}")
    print(f"  {name}  @  {obs['position']}")
    print(f"{'='*55}")

    print("\n  [STATE — raw observation]")
    print(f"    position : {obs['position']}")
    print(f"    room     : {obs['room']}")
    print(f"    visible objects ({len(obs['visible_objects'])}):")
    for o in obs["visible_objects"]:
        label = f" '{o['label']}'" if "label" in o else ""
        lock  = f" locked={o['locked']}" if "locked" in o else ""
        print(f"      {o['type']:<14} {o['color']:<8}{label}{lock}  @ {o['pos']}")

    print("\n  [DERIVED — ontology reasoning]")
    print(f"    I am in room   : {derived['my_room']}  ({derived['my_room_cell_count']} cells)")
    if derived["visible_keys"]:
        print(f"    Keys I can see :")
        for k in derived["visible_keys"]:
            print(f"      '{k['label']}'  @ {k['pos']}  (room: {k['in_room']})")
    else:
        print(f"    Keys I can see : none")
    if derived["visible_balls"]:
        print(f"    Balls I can see:")
        for b in derived["visible_balls"]:
            print(f"      '{b['label']}'  @ {b['pos']}  (room: {b['in_room']})")
    if derived["nearby_doors"]:
        print(f"    Nearby doors   :")
        for d in derived["nearby_doors"]:
            print(f"      {d['color']} door  @ {d['pos']}  locked={d['locked']}  (ontology cell room: {d['connects_to']})")
    if derived["can_pickup"]:
        print(f"    Can pick up    : {derived['can_pickup']}")
    else:
        print(f"    Can pick up    : nothing reachable")


if __name__ == "__main__":
    print("Building environment and ontology...")
    env = HouseEnv()
    env.reset()
    Ontology_builder.Ont_built = False
    Ontology_builder.onto = None
    ont = Ontology_builder.build(ontology_path=ONTOLOGY_PATH, env=env)

    manager = ont.get_manager()
    axioms = manager.get_axioms(ont)
    print(f"Ontology loaded. Total axioms: {len(axioms)}\n")

    cell_to_room, room_to_cells = build_indexes(ont)

    for agent_name, start_pos in AGENTS.items():
        obs     = observe(env, start_pos)
        derived = derive(cell_to_room, room_to_cells, env, obs)
        report(agent_name, obs, derived)