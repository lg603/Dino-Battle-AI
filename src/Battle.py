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
    return [Move(move['id'], move['name'], move['type'], move.get('power'), move['accuracy'], move['stamina_cost'],
                 move.get('effect')) for move in move_data['moves']]


def assign_moves_to_dino(dino, all_moves):
    # Filter moves that the dino has enough stamina to perform
    possible_moves = [move for move in all_moves if move.stamina_cost <= dino.curr_stamina]
    # Randomly assign moves to the dino
    dino.moves = random.sample(possible_moves, min(len(possible_moves), 4))  # Assign up to 4 moves


def calculate_hit_chance(move_accuracy, attacker_speed, defender_speed):
    # Base accuracy of the move
    base_accuracy = move_accuracy / 100.0  # Convert percentage to a decimal

    # Speed factor - this reduces the accuracy based on the defender's speed relative to the attacker's speed
    # The "+ 1" in the denominator prevents division by zero and ensures the factor is always between 0 and 1
    speed_factor = attacker_speed / (defender_speed + 1)

    # Final hit chance
    hit_chance = base_accuracy * speed_factor

    # Ensure hit chance is between 0 and 1
    hit_chance = max(0, min(hit_chance, 1))

    return hit_chance


def calculate_crit_chance(move_accuracy, attacker_speed, defender_speed):
    # Base crit chance influenced by move accuracy
    base_crit_chance = move_accuracy / 200.0  # Half of accuracy as a starting point

    # Speed factor - higher chance of crit if the attacker is faster
    speed_factor = attacker_speed / max(defender_speed, 1)  # Avoid division by zero

    # Final crit chance
    crit_chance = base_crit_chance * speed_factor

    # Ensure crit chance is between 0 and 1
    crit_chance = max(0, min(crit_chance, 1))

    return crit_chance


# ================================================= BATTLE CLASS =================================================
class Battle:
    # A global variable containing the attack chart for move effectiveness. This will be used extensively in battle
    # calculations.
    attack_chart = {
        'Lava': {'Water': 0.5, 'Air': 1.0, 'Earth': 2, 'Metal': 0.5, 'Jungle': 1.0, 'Thunder': 1.0, 'Ice': 2,
                 'Mystic': 1.0, 'Fossil': 0.5},

        'Water': {'Lava': 2, 'Air': 1.0, 'Earth': 0.5, 'Metal': 1.0, 'Thunder': 1.0, 'Jungle': 2, 'Ice': 1.0,
                  'Mystic': 2, 'Fossil': 1.0},

        'Air': {'Lava': 1.0, 'Water': 1.0, 'Earth': 0.5, 'Metal': 1.0, 'Jungle': 2, 'Thunder': 0.5, 'Ice': 2,
                'Mystic': 2, 'Fossil': 1.0},

        'Earth': {'Lava': 0.5, 'Water': 2, 'Air': 2, 'Metal': 1.0, 'Jungle': 0.5, 'Thunder': 0.5, 'Ice': 1.0,
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

    def __init__(self, min_id=1, max_id=20, debug=1, init_overall_statistics=None, init_type_wins=None, capture=False):
        self.capture = capture
        self.team1 = []
        self.team2 = []
        self.min_id = min_id
        self.max_id = max_id
        self.MAX_ROUNDS = 200
        self.num_dinos = 6

        # Run setup function
        self.round_number = 1
        self.debug = debug
        self.dino_statistics = {}  # To track extended statistics for each dino
        self.type_wins = {}  # To track wins by type
        self.total_rounds = 0  # To track total number of rounds
        self.total_battles = 0  # To track total number of battles

        # General Statistics Tracking
        self.overall_statistics = init_overall_statistics if init_overall_statistics is not None else {}
        self.type_wins = init_type_wins if init_type_wins is not None else {}

        # Winner Predictions Statistics Tracking
        self.pregame_stats = []
        self.winner = 0

        self.create_teams()

    # CREATING ROSTER FOR THE BATTLE ===================================================================================
    def create_teams(self, team_size=3):
        team1 = []
        team2 = []
        selected_ids = set()

        while len(team1) < team_size:  # Create team 1
            random_id = random.randint(self.min_id, self.max_id)
            if random_id not in selected_ids:
                dino = load_dinos_from_json(self.path_to_dinodex, random_id)
                if dino:
                    dino.team = 1
                    team1.append(dino)
                    selected_ids.add(random_id)

        while len(team2) < team_size:  # Create team 2
            random_id = random.randint(self.min_id, self.max_id)
            if random_id not in selected_ids:
                dino = load_dinos_from_json(self.path_to_dinodex, random_id)
                if dino:
                    dino.team = 2
                    team2.append(dino)
                    selected_ids.add(random_id)

        self.team1 = team1
        self.team2 = team2

        for dino in self.team1 + self.team2:
            dino.reset_stats()
            if dino.name not in self.overall_statistics:  # Set default statistics to track
                self.overall_statistics[dino.name] = {
                    'appearances': 0, 'wins': 0, 'losses': 0, 'survived': 0, 'died': 0, 'rests': 0,
                    'final_hp_sum': 0, 'final_hp_count': 0, 'final_hp_min': float('inf'), 'final_hp_max': 0,
                    'rounds_survived_sum': 0, 'rounds_survived_count': 0, 'rounds_survived_min': float('inf'),
                    'rounds_survived_max': 0,
                    'draws': 0
                }
            self.overall_statistics[dino.name]['appearances'] += 1

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

        if self.capture:  # Capturing data before the match to use in training battle winner predictor
            self.pregame_stats = self.get_pregame_stats()

    # BATTLE SEQUENCE ==================================================================================================
    def update_statistics_after_battle(self):
        # print(self.round_number)
        if self.round_number > self.MAX_ROUNDS:  # Match hit max number of rounds
            for dino in self.team1 + self.team2:
                self.overall_statistics[dino.name]['draws'] += 1
        else:  # Battle finished normally
            winning_team = self.team1 if any(d.hp > 0 for d in self.team1) else self.team2
            for dino in winning_team:
                self.type_wins[dino.dino_type] = self.type_wins.get(dino.dino_type, 0) + 1

            for dino in self.team1 + self.team2:
                dino_stats = self.overall_statistics.setdefault(dino.name, {
                    'appearances': 0, 'wins': 0, 'losses': 0, 'draws': 0, 'survived': 0, 'died': 0, 'rests': 0,
                    'final_hp_sum': 0, 'final_hp_min': float('inf'), 'final_hp_max': 0, 'final_hp_count': 0,
                    'rounds_survived_sum': 0, 'rounds_survived_min': float('inf'), 'rounds_survived_max': 0,
                    'rounds_survived_count': 0
                })

                dino_stats['appearances'] += 1

                # Update final HP stats
                dino_stats['final_hp_sum'] += dino.hp
                dino_stats['final_hp_min'] = min(dino_stats['final_hp_min'], dino.hp)
                dino_stats['final_hp_max'] = max(dino_stats['final_hp_max'], dino.hp)
                dino_stats['final_hp_count'] += 1

                # Update rounds survived stats
                rounds_survived = self.round_number if dino.hp > 0 else self.round_number - 1
                dino_stats['rounds_survived_sum'] += rounds_survived
                dino_stats['rounds_survived_min'] = min(dino_stats['rounds_survived_min'], rounds_survived)
                dino_stats['rounds_survived_max'] = max(dino_stats['rounds_survived_max'], rounds_survived)
                dino_stats['rounds_survived_count'] += 1

                if dino in winning_team:
                    dino_stats['wins'] += 1
                    self.type_wins[dino.dino_type] = self.type_wins.get(dino.dino_type, 0) + 1
                else:
                    dino_stats['losses'] += 1

                if dino.hp > 0:
                    dino_stats['survived'] += 1
                else:
                    dino_stats['died'] += 1

                dino_stats['rests'] += dino.battle_stats['rests']

            self.total_battles += 1
            self.total_rounds += self.round_number

            # Update average statistics
            for dino_name, stats in self.dino_statistics.items():
                stats['win_loss_ratio'] = stats['wins'] / stats['losses'] if stats['losses'] > 0 else float('inf')
                stats['avg_rounds_survived'] = sum(stats['rounds_survived']) / len(stats['rounds_survived'])
                stats['avg_final_hp'] = sum(stats['final_hp']) / len(stats['final_hp'])

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

    def calculate_damage(self, attacker, defender, move, crit_chance):
        # Get the type effectiveness
        effectiveness = self.type_effectiveness(attacker.dino_type, defender.dino_type)

        # Calculate base damage using the attack and defense stats
        base_damage = (move.power * (attacker.attack / defender.defense)) * effectiveness

        # Determine if it's a critical hit
        is_crit = random.random() <= crit_chance
        crit_multiplier = 1.5 if is_crit else 1

        # Calculate final damage
        final_damage = math.floor(base_damage * crit_multiplier)

        return final_damage, is_crit

    def perform_battle_turn(self, attacker, defender, move):
        # Subtract the move's stamina cost from the attacker's stamina
        attacker.use_stamina(move.stamina_cost)

        # Calculate hit chance and crit chance
        hit_chance = calculate_hit_chance(move.accuracy, attacker.speed, defender.speed)
        crit_chance = calculate_crit_chance(move.accuracy, attacker.speed, defender.speed)

        # Determine if the move hits based on its accuracy
        if random.random() <= hit_chance:  # move.accuracy is assumed to be a percentage
            if move.move_type == 1:  # Attack move
                # Calculate damage
                final_damage, is_crit = self.calculate_damage(attacker, defender, move, crit_chance)

                # Get the type effectiveness
                effectiveness = self.type_effectiveness(attacker.dino_type, defender.dino_type)

                # Apply the damage to the defender
                defender.hp = max(defender.hp - final_damage, 0)

                # Apply the damage to the defender
                defender.hp = max(defender.hp - final_damage, 0)

                # Print out what happened
                if self.debug:
                    crit_text = " Critical Hit!" if is_crit else ""
                    print(
                        f"{attacker.name} (Team {attacker.team}) used {move.name} on {defender.name} "
                        f"(Team {defender.team})! It's {effectiveness}x effective! {defender.name} "
                        f"took {final_damage} damage! {crit_text} Remaining HP: {defender.hp}")

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
                            f"{attacker.name} (Team {attacker.team}) used {move.name} on {defender.name} "
                            f"(Team {defender.team})! {change_type} {defender.name}'s {stat} stat by {change_amount}.")
        else:
            # Move missed
            if self.debug:
                print(f"{attacker.name} (Team {attacker.team})'s move {move.name} missed {defender.name}"
                      f" (Team {defender.team})!")

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

            if self.round_number > self.MAX_ROUNDS:
                if self.debug:
                    print("Battle reached maximum round limit. Declaring a draw.")
                for dino in self.team1 + self.team2:
                    # Initialize 'draws' key if not present
                    if 'draws' not in self.overall_statistics[dino.name]:
                        self.overall_statistics[dino.name]['draws'] = 0
                    self.overall_statistics[dino.name]['draws'] += 1
                return  # End the battle

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

        self.winner = 1 if any(d.hp > 0 for d in self.team1) else 2

        # Determine the winner
        if self.debug == 1:
            winners = "Team 1" if any(d.hp > 0 for d in self.team1) else "Team 2"
            print(f"The battle is over! {winners} wins!")

    def get_type_matchups(self, team1, team2):
        matchups = []
        for dino1 in team1:
            for dino2 in team2:
                effectiveness = self.type_effectiveness(dino1.dino_type, dino2.dino_type)
                matchups.append(effectiveness)
        return matchups

    def get_pregame_stats(self):
        team_1_data = [dino.get_stats() for dino in self.team1]
        team_2_data = [dino.get_stats() for dino in self.team2]

        # Flatten the lists
        team_1_data = [item for sublist in team_1_data for item in sublist]
        team_2_data = [item for sublist in team_2_data for item in sublist]

        # Get type matchups
        type_matchups = self.get_type_matchups(self.team1, self.team2)

        return team_1_data + team_2_data + type_matchups
