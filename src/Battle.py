from Move import Move
from Dino import Dino
import math
import random
import json


def perform_battle_turn(attacker, defender, move):
    # Calculate the base damage of the move
    base_damage = move.power  # You will need more complex calculations based on stats, etc.

    # Get the type effectiveness
    effectiveness = defender.type_effectiveness(move.type)

    # Calculate the final damage
    final_damage = base_damage * effectiveness

    # Apply the damage to the defender
    defender.hp -= final_damage

    # Print out what happened
    print(
        f"{attacker.name} used {move.name} on {defender.name}! It's {effectiveness}x effective! {defender.name} "
        f"took {final_damage} damage!")

    # Apply additional effects from the move
    move.apply(defender)


def load_dinos_from_json(file_path, dino_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        dino_data = json.load(file)

    # Find the dino with the specified name
    for dino in dino_data["dinosaurs"]:
        if dino["name"] == dino_name:
            return Dino(
                name=dino["name"],
                dino_type=dino["type"],
                hp=dino["stats"]["hp"],
                attack=dino["stats"]["attack"],
                defense=dino["stats"]["defense"],
                stamina=dino["stats"]["stamina"],
                stamina_recharge=dino["stats"]["stamina_recharge"],
                speed=dino["stats"]["speed"]
            )
    return None  # Return None if no dino with the specified name was found


class Battle:
    # A global variable containing the attack chart for move effectiveness. This will be used extensively in battle
    # calculations.
    attack_chart = {
        'Lava': {'Water': 0.5, 'Air': 1.0, 'Earth': 2, 'Metal': 0.5, 'Jungle': 1.0, 'Thunder': 1.0, 'Ice': 2,
                 'Mystic': 1.0, 'Fossil': 0.5},

        'Water': {'Lava': 2, 'Air': 1.0, 'Earth': 0.5, 'Metal': 1.0, 'Thunder': 1.0, 'Jungle': 2, 'Ice': 1.0,
                  'Mystic': 2, 'Fossil': 1.0},

        'Air': {'Lava': 1.0, 'Water': 1.0, 'Earth': 0, 'Metal': 1.0, 'Jungle': 2, 'Thunder': 0.5, 'Ice': 2,
                'Mystic': 2, 'Fossil': 1.0},

        'Earth': {'Lava': 0.5, 'Water': 2, 'Air': 2, 'Metal': 1.0, 'Jungle': 0.5, 'Thunder': 0, 'Ice': 1.0,
                  'Mystic': 1.0, 'Fossil': 2},

        'Metal': {'Lava': 2, 'Water': 1.0, 'Air': 1.0, 'Earth': 1.0, 'Jungle': 0.5, 'Thunder': 2, 'Ice': 0.5,
                  'Mystic': 0.5, 'Fossil': 0.5},

        'Jungle': {'Lava': 1.0, 'Water': 0.5, 'Air': 0.5, 'Earth': 2, 'Metal': 2, 'Thunder': 1.0, 'Ice': 0.5,
                   'Mystic': 0.5, 'Fossil': 2},

        'Thunder': {'Lava': 1.0, 'Water': 0.5, 'Air': 2, 'Earth': 2, 'Metal': 0.5, 'Jungle': 1.0, 'Ice': 1.0,
                    'Mystic': 2, 'Fossil': 1.0},

        'Ice': {'Lava': 0.5, 'Water': 1.0, 'Air': 0.5, 'Earth': 1.0, 'Metal': 2, 'Jungle': 2, 'Thunder': 1.0,
                'Mystic': 0.5, 'Fossil': 2},

        'Mystic': {'Lava': 1.0, 'Water': 0.5, 'Air': 0.5, 'Earth': 1.0, 'Metal': 2, 'Jungle': 2, 'Thunder': 0.5,
                   'Ice': 2, 'Fossil': 0.5},

        'Fossil': {'Lava': 2, 'Water': 1.0, 'Air': 1.0, 'Earth': 0.5, 'Metal': 2, 'Jungle': 0.5, 'Thunder': 1.0,
                   'Ice': 0.5, 'Mystic': 2}
    }

    path_to_dinodex = "dinodex.json"

    def __init__(self):
        self.dino_pool = []

    # CREATING ROSTER FOR THE BATTLE ===================================================================================
    def select_random_dino(self):
        return random.choice(self.dino_pool)

    def add_dino_to_pool(self, dino):
        if dino is not None:
            self.dino_pool.append(dino)

    def expand_dino_pool(self, min_id, max_id, number_of_dinos_to_add):
        for _ in range(number_of_dinos_to_add):
            # Generate a random ID
            random_id = random.randint(min_id, max_id)

            # Call the load_dino_from_json function with the random ID
            dino = load_dinos_from_json(self.path_to_dinodex, random_id)

            # Add the dino to your pokemon_pool if it's not None
            self.add_dino_to_pool(dino)

    # CREATING TEAMS AND ASSIGNING MOVES ===============================================================================

    def create_team(self, team_size=6):
        team = []
        for _ in range(team_size):
            dino = self.select_random_dino()
            team.append(dino)

        return team

    def print_dino(self):
        if self.dino_pool:  # Checks if pokemon_pool is not empty
            for dino in self.dino_pool:
                if dino is not None:  # Check if the current dino is not None
                    dino.display_stats()
                else:
                    print("No Dinosaur data available.")

    # def create_team(self):
    # Step 1: select a pokemon, and remove it from the list of pokemon you can select

    # Step 2: Add moves to the pokemon from its move pool
    # Step 3: Append it to the array containing the pokemon for this team
    # Step 4: Repeat up to 5 more times for a full team


if __name__ == "__main__":
    battle = Battle()
    battle.expand_dino_pool(1, 9, 3)
    battle.print_dino()

    # TODO: loading in pokedex to battle class is now working, next, assign each pokemon moves based on its move pool
