from owlapy.owl_ontology import Ontology
from owlapy.iri import IRI
from owlapy.owl_axiom import (
    OWLClassAssertionAxiom,
    OWLDeclarationAxiom,
    OWLObjectPropertyAssertionAxiom,
    OWLDataPropertyAssertionAxiom,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.class_expression import OWLClass
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_reasoner import StructuralReasoner
from envs.environment_multi import HouseEnv
from typing import List

NS = "http://www.semanticweb.org/m.manso/ontologies/2026/3/untitled-ontology-30#"

# OWL property names, must match the TBox exactly (same as agent.py)
PROP_CONNECTS_TO = "connectsTo"
PROP_LOCATED_IN  = "locatedIn"
PROP_HAS_COLOR   = "hasColor"

# box (furniture) label -> specific ontology subclass name
FURNITURE_CLASS_MAP: dict[str, str] = {
    "plant pot":        "PlantPot",
    "coat rack":        "CoatRack",
    "television":       "Television",
    "coffee table":     "CoffeeTable",
    "refrigerator":     "Refrigerator",
    "kitchen sink":     "KitchenSink",
    "nightstand":       "Nightstand",
    "toilet":           "Toilet",
    "umbrella stand":   "UmbrellaStand",
    "computer monitor": "ComputerMonitor",
    "storage shelf":    "StorageShelf",
}


class StaticOntologyBuilder:
    onto  = None
    built = False

    @classmethod
    def build_static_information(cls, ontology_path: str, env: HouseEnv) -> Ontology:
        if cls.built:
            return cls.onto

        if env.grid.get(0, 0) is None:
            env.reset()

        ont: Ontology = Ontology(ontology_path, load=True)

        # OWL classes
        room_class = OWLClass(IRI.create(NS, "Room"))
        door_class = OWLClass(IRI.create(NS, "Door"))
        cell_class = OWLClass(IRI.create(NS, "Cell"))
        wall_class = OWLClass(IRI.create(NS, "Wall"))
        item_class = OWLClass(IRI.create(NS, "Item"))
        ball_class = OWLClass(IRI.create(NS, "BallObject"))
        box_class  = OWLClass(IRI.create(NS, "BoxObject"))

        # OWL properties
        connects_to  = OWLObjectProperty(IRI.create(NS, PROP_CONNECTS_TO))
        located_in   = OWLObjectProperty(IRI.create(NS, PROP_LOCATED_IN))
        has_color    = OWLDataProperty(IRI.create(NS, PROP_HAS_COLOR))

        # Cells
        for x in range(env.width):
            for y in range(env.height):
                cell_ind = OWLNamedIndividual(IRI.create(NS, f"cell_{x}_{y}"))
                ont.add_axiom(OWLDeclarationAxiom(cell_ind))
                ont.add_axiom(OWLClassAssertionAxiom(cell_ind, cell_class))
                cell_obj = env.grid.get(x, y)
                if cell_obj and cell_obj.type == "wall":
                    ont.add_axiom(OWLClassAssertionAxiom(cell_ind, wall_class))

        # Rooms
        room_individuals: dict[str, OWLNamedIndividual] = {}
        for room_name in env.get_all_rooms():
            room_ind = OWLNamedIndividual(IRI.create(NS, room_name.replace(" ", "_")))
            ont.add_axiom(OWLDeclarationAxiom(room_ind))
            ont.add_axiom(OWLClassAssertionAxiom(room_ind, room_class))
            room_individuals[room_name] = room_ind

            color = env.get_room_color(room_name)
            if color:
                ont.add_axiom(OWLDataPropertyAssertionAxiom(
                    room_ind, has_color, OWLLiteral(color)
                ))

        # Doors + connectsTo
        door_individuals: dict[tuple, tuple] = {}
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == "door":
                    label      = getattr(cell, "label", f"door_{x}_{y}")
                    safe_label = label.replace(" ", "_")
                    door_ind   = OWLNamedIndividual(IRI.create(NS, safe_label))
                    ont.add_axiom(OWLDeclarationAxiom(door_ind))
                    ont.add_axiom(OWLClassAssertionAxiom(door_ind, door_class))
                    door_individuals[(x, y)] = (door_ind, safe_label)

        for (x, y), (door_ind, _) in door_individuals.items():
            for room_name in cls._get_adjacent_rooms(env, x, y):
                if room_name in room_individuals:
                    ont.add_axiom(OWLObjectPropertyAssertionAxiom(
                        door_ind, connects_to, room_individuals[room_name]
                    ))
                    ont.add_axiom(OWLObjectPropertyAssertionAxiom(
                        room_individuals[room_name], connects_to, door_ind
                    ))

        # Items (balls / boxes)
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is None:
                    continue
                label = getattr(cell, "label", None)
                if label is None or cell.type == "key":
                    continue
                safe_label = label.replace(" ", "_")

                if cell.type == "ball":
                    item_ind = OWLNamedIndividual(IRI.create(NS, safe_label))
                    ont.add_axiom(OWLDeclarationAxiom(item_ind))
                    ont.add_axiom(OWLClassAssertionAxiom(item_ind, ball_class))
                elif cell.type == "box":
                    item_ind = OWLNamedIndividual(IRI.create(NS, safe_label))
                    ont.add_axiom(OWLDeclarationAxiom(item_ind))
                    ont.add_axiom(OWLClassAssertionAxiom(item_ind, box_class))
                    specific = FURNITURE_CLASS_MAP.get(label)
                    if specific:
                        ont.add_axiom(OWLClassAssertionAxiom(
                            item_ind, OWLClass(IRI.create(NS, specific))
                        ))

        cls.onto  = ont
        cls.built = True
        return cls.onto

    @classmethod
    def _get_adjacent_rooms(cls, env: HouseEnv, door_x: int, door_y: int) -> List[str]:
        adjacent: set[str] = set()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = door_x + dx, door_y + dy
            if 0 <= nx < env.width and 0 <= ny < env.height:
                room = env.get_room_for_cell(nx, ny)
                if room:
                    adjacent.add(room)
        return list(adjacent)

    def display_ont(self):
        if not self.built:
            print("Ontology not built yet.")
            return

        reasoner = StructuralReasoner(self.onto)

        print("\n=== STATIC ONTOLOGY (TBox + initial ABox) ===")

        print("\nRooms")
        room_class = OWLClass(IRI.create(NS, "Room"))
        for r in reasoner.instances(room_class):
            print(f"  {r.iri.remainder}")

        print("\nDoors + connectsTo")
        door_class   = OWLClass(IRI.create(NS, "Door"))
        connects_to  = OWLObjectProperty(IRI.create(NS, PROP_CONNECTS_TO))
        for d in reasoner.instances(door_class):
            targets = [v.iri.remainder
                       for v in reasoner.object_property_values(d, connects_to)]
            print(f"  {d.iri.remainder} -> {targets}")

        print("\nItems in ontology")
        item_class = OWLClass(IRI.create(NS, "Item"))
        items = list(reasoner.instances(item_class))
        print(f"  Total: {len(items)}")
        for item in items:
            print(f"  {item.iri.remainder}")

        print("\nRoom colors (hasColor)")
        has_color = OWLDataProperty(IRI.create(NS, PROP_HAS_COLOR))
        for r in reasoner.instances(room_class):
            colors = [v.get_literal()
                      for v in reasoner.data_property_values(r, has_color)]
            if colors:
                print(f"  {r.iri.remainder} -> {colors[0]}")


if __name__ == "__main__":
    print("Building environment and static ontology...")
    env = HouseEnv(num_agents=1)
    env.reset()

    builder = StaticOntologyBuilder()
    ont = builder.build_static_information(
        ontology_path="/Users/m.manso/Downloads/thesisont_updated-2.owl",
        env=env,
    )
    print("Static ontology built.")
    builder.display_ont()