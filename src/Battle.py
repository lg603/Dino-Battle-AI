from Move import Move
from Dino import Dino
import math
import random
import json


def load_dinos_from_json(file_path, dino_id):
    with open(file_path, 'r', encoding='utf-8') as file:
        dino_data = json.load(file)

    # Find the dino with the specified id
    for dino in dino_data["dinosaurs"]:
        if dino["id"] == dino_id:
            return Dino(
                id=dino["id"],
                name=dino["name"],
                dino_type=dino["type"],
                hp=dino["stats"]["hp"],
                attack=dino["stats"]["attack"],
                defense=dino["stats"]["defense"],
                stamina=dino["stats"]["stamina"],
                stamina_recharge=dino["stats"]["stamina_recharge"],
                speed=dino["stats"]["speed"]
            )
    return None  # Return None if no dino with the specified id was found


def load_moves_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        move_data = json.load(file)
    return [Move(move['name'], move['type'], move.get('power'), move['accuracy'], move['stamina_cost'],
                 move.get('effect')) for move in move_data['moves']]


def assign_moves_to_dino(dino, all_moves):
    # Filter moves that the dino has enough stamina to perform
    possible_moves = [move for move in all_moves if move.stamina_cost <= dino.curr_stamina]
    # Randomly assign moves to the dino
    dino.moves = random.sample(possible_moves, min(len(possible_moves), 4))  # Assign up to 4 moves


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

    path_to_dinodex = "dinos.json"
    path_to_moves = "moves.json"

    def __init__(self, min_id=1, max_id=20, debug=1):
        self.team1 = []
        self.team2 = []
        self.min_id = min_id
        self.max_id = max_id
        self.num_dinos = 6

        # Run setup function
        self.create_teams()
        self.round_number = 1
        self.debug = debug

    # CREATING ROSTER FOR THE BATTLE ===================================================================================
    def create_teams(self, team_size=3):
        team1 = []
        team2 = []
        selected_ids = set()

        while len(team1) < team_size:
            random_id = random.randint(self.min_id, self.max_id)
            if random_id not in selected_ids:
                dino = load_dinos_from_json(self.path_to_dinodex, random_id)
                if dino:
                    dino.team = 1
                    team1.append(dino)
                    selected_ids.add(random_id)

        while len(team2) < team_size:
            random_id = random.randint(self.min_id, self.max_id)
            if random_id not in selected_ids:
                dino = load_dinos_from_json(self.path_to_dinodex, random_id)
                if dino:
                    dino.team = 2
                    team2.append(dino)
                    selected_ids.add(random_id)

        self.team1 = team1
        self.team2 = team2

    def print_teams(self):
        print("Team 1:")
        for dino in self.team1:
            print(f"- {dino}")
        print("\nTeam 2:")
        for dino in self.team2:
            print(f"- {dino}")

    def print_status_chart(self):
        if self.debug:
            print("\nCurrent Status:")
            print("Team 1:")
            for dino in self.team1:
                print(f"- {dino}")
            print("\nTeam 2:")
            for dino in self.team2:
                print(f"- {dino}")
            print("\n")

    # ADD MOVES TO DINOS ON EACH TEAM, BASED ON STAMINA CAP ============================================================
    def assign_moves_to_teams(self):
        # Load all moves from the JSON file
        all_moves = load_moves_from_json(self.path_to_moves)

        # Assign moves to each dino in team1
        for dino in self.team1:
            assign_moves_to_dino(dino, all_moves)

        # Assign moves to each dino in team2
        for dino in self.team2:
            assign_moves_to_dino(dino, all_moves)

    # BATTLE SEQUENCE ==================================================================================================
    def type_effectiveness(self, attacker_type, defender_type):
        # Check if attacker_type exists in the attack_chart
        if attacker_type in self.attack_chart:
            # Check if defender_type exists for the attacker_type in the attack_chart
            if defender_type in self.attack_chart[attacker_type]:
                return self.attack_chart[attacker_type][defender_type]
            else:
                # If defender_type is not found, treat the move as 1.0x effective
                return 1.0
        else:
            # If attacker_type is not found, treat the move as 1.0x effective
            return 1.0

    def perform_battle_turn(self, attacker, defender, move):
        # Subtract the move's stamina cost from the attacker's stamina
        attacker.use_stamina(move.stamina_cost)

        if move.move_type == 1:  # Attack move
            # Calculate the base damage of the move
            base_damage = move.power if move.power else 0

            # Get the type effectiveness
            effectiveness = self.type_effectiveness(attacker.dino_type, defender.dino_type)

            # Calculate the final damage
            final_damage = math.floor(base_damage * effectiveness)

            # Apply the damage to the defender
            defender.hp = max(defender.hp - final_damage, 0)

            # Print out what happened
            if self.debug:
                # Print out what happened
                print(
                    f"{attacker.name} (Team {attacker.team}) used {move.name} on {defender.name} "
                    f"(Team {defender.team})! It's {effectiveness}x effective! {defender.name} "
                    f"took {final_damage} damage! Remaining HP: {defender.hp}")

                # Check if the defender is defeated
                if defender.hp == 0:
                    print(f"{defender.name} (Team {defender.team}) has been defeated!")

        elif move.move_type in [2, 3]:  # Stat boost/debuff move
            # Apply stat changes to the defender
            for stat, multiplier in move.effect.items():
                defender.change_stat(stat, multiplier)
                change_type = "Boosted" if multiplier > 1 else "Reduced"
                change_amount = f"{multiplier}x" if multiplier > 1 else f"{multiplier * 100}%"
                if self.debug:
                    print(
                        f"{attacker.name} used {move.name} on {defender.name}! {change_type} {defender.name}'s {stat}"
                        f" stat by {change_amount}.")

        # Recharge attacker's stamina
        attacker.recharge_stamina()

    def battle_sequence(self):
        if self.debug:
            print("\n============== BEGIN BATTLE ==============\n")
        # Assign moves to teams
        self.assign_moves_to_teams()

        # Combine both teams for sorting by speed
        all_dinos = self.team1 + self.team2

        # Continue battle until one team is out of dinos
        while any(d.hp > 0 for d in self.team1) and any(d.hp > 0 for d in self.team2):
            # Print the round number
            if self.debug:
                print(f"============== ROUND {self.round_number} ==============")

            # Sort dinos by speed, descending
            all_dinos.sort(key=lambda d: d.speed, reverse=True)

            for dino in all_dinos:
                if dino.hp > 0:
                    # Initialize defender as None
                    defender = None

                    # Filter moves that the dino has enough stamina to perform
                    available_moves = [move for move in dino.moves if move.stamina_cost <= dino.curr_stamina]

                    # Decide action based on the number of available moves
                    if len(available_moves) == 0:
                        # No moves available, dino must rest
                        dino.rest(self.debug)
                        continue  # Skip the rest of the loop for this dino
                    elif len(available_moves) == 1 and random.random() < 0.5:
                        # Only one move available, 50% chance to rest
                        dino.rest(self.debug)
                        continue  # Skip the rest of the loop for this dino
                    else:
                        # Select a random move from the available moves
                        move = random.choice(available_moves)

                    # Choose a target based on the move type
                    if move.move_type == 1:  # Attack move
                        # Target is a random dino from the opposing team
                        target_team = self.team2 if dino in self.team1 else self.team1
                        possible_defenders = [d for d in target_team if d.hp > 0]
                        if possible_defenders:
                            defender = random.choice(possible_defenders)
                    elif move.move_type == 2:  # Stat boost move
                        # Target is the attacker itself or a teammate
                        target_team = self.team1 if dino in self.team1 else self.team2
                        possible_defenders = [d for d in target_team if d.hp > 0]
                        if possible_defenders:
                            defender = random.choice(possible_defenders)
                    elif move.move_type == 3:  # Stat debuff move
                        # Target is a random dino from the opposing team
                        target_team = self.team2 if dino in self.team1 else self.team1
                        possible_defenders = [d for d in target_team if d.hp > 0]
                        if possible_defenders:
                            defender = random.choice(possible_defenders)

                    # Perform the battle turn if defender is defined
                    if defender:
                        self.perform_battle_turn(dino, defender, move)
                    else:
                        if self.debug:
                            print(f"No valid target found for {dino.name}'s move {move.name}")

            # Print status chart after each turn
            self.print_status_chart()

            # Increment the round number after each dinosaur has had their turn
            self.round_number += 1

        # Determine the winner
        winner = "Team 1" if any(d.hp > 0 for d in self.team1) else "Team 2"
        print(f"The battle is over! {winner} wins!")


if __name__ == "__main__":
    battle = Battle(debug=0)
    battle.assign_moves_to_teams()  # Assign moves to dinos in both teams
    # battle.print_teams()
    battle.battle_sequence()
