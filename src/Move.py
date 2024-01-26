
import math

import json


def apply_stat_change(target, effect):
    # Implement logic to apply stat changes to the target
    stat = effect.get("stat")
    stages = effect.get("stages", 0)
    # Apply the stat change (e.g., update target's defense)
    setattr(target, stat, getattr(target, stat) + stages)
    print(f"{target.name}'s {stat} changed by {stages} stages!")


def apply_damage(target, effect):
    # Implement logic to calculate damage and apply it to the target
    damage = effect.get("amount", 0)
    target.hp -= damage
    print(f"{target.name} took {damage} damage!")


class Move:
    def __init__(self, name, power, effects):
        self.name = name
        self.power = power
        self.effects = effects

    def apply(self, target):
        # Apply move effects
        for effect in self.effects:
            effect_type = effect.get("type")
            if effect_type == "damage":
                apply_damage(target, effect)
            elif effect_type == "stat_change":
                apply_stat_change(target, effect)


# Load moves from JSON file
def load_moves_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        moves_data = json.load(file)["moves"]

    moves = []
    for move_data in moves_data:
        moves.append(Move(**move_data))

    return moves
