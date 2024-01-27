import math


class Dino:
    def __init__(self, id, name, dino_type, hp, attack, defense, stamina, stamina_recharge, speed):
        self.id = id
        self.name = name
        self.dino_type = dino_type
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.stamina_recharge = stamina_recharge
        self.speed = speed

        # Base Statistics
        self.base_hp = hp
        self.base_attack = attack
        self.base_defense = defense
        self.base_stamina = stamina
        self.base_stamina_recharge = stamina_recharge
        self.base_speed = speed

        self.max_stamina = stamina
        self.curr_stamina = stamina
        self.moves = []
        self.team = 0  # Default assignment, no team yet
        self.battle_stats = {'rests': 0}  # Initialize battle statistics

    def __str__(self):
        status = "KO" if self.hp <= 0 else "Active"
        dino_info = (f"{self.name} (Team {self.team}) - Type: {self.dino_type}, HP: {self.hp}, Attack: {self.attack}, "
                     f"Defense: {self.defense}, Current Stamina: {self.curr_stamina}, Max Stamina: {self.max_stamina}, "
                     f"Stamina Recharge: {self.stamina_recharge}, Speed: {self.speed}, Status: {status}")

        # Add moves to the string if the dinosaur has any
        if self.moves:
            moves_str = ', '.join([move.name for move in self.moves])
            dino_info += f"\nMoves: {moves_str}"
        else:
            dino_info += "\nMoves: None"

        return dino_info

    def change_stat(self, stat, multiplier):
        # print(f"Changing {stat} from {getattr(self, stat)} to {getattr(self, stat) * multiplier}")
        if stat == 'attack': # Limit stats to a reasonable range
            self.attack = max(self.base_attack * 0.2, min(math.floor(self.attack * multiplier), 1000))
        elif stat == 'defense':
            self.defense = max(self.base_defense * 0.2, min(math.floor(self.defense * multiplier), 1000))
        elif stat == 'max_stamina':
            self.max_stamina = max(self.base_stamina * 0.5, min(math.floor(self.max_stamina * multiplier), 1000))
        elif stat == 'stamina_recharge':
            self.stamina_recharge = max(self.base_stamina_recharge * 0.5, min(math.floor(self.stamina_recharge *
                                                                                         multiplier), 1000))
        elif stat == 'speed':
            self.speed = max(self.base_speed * 0.2, min(math.floor(self.speed * multiplier), 1000))

    def use_stamina(self, cost):
        self.curr_stamina = max(0, self.curr_stamina - cost)

    def recharge_stamina(self):
        self.curr_stamina += self.stamina_recharge

    def rest(self, debug):
        if debug != 0:
            print(f"{self.name} (Team {self.team}) is resting this turn and refilling its stamina.")
        self.curr_stamina = self.max_stamina
        self.battle_stats['rests'] += 1

    def reset_stats(self):
        self.attack = self.base_attack
        self.defense = self.base_defense
        self.curr_stamina = self.base_stamina
        self.max_stamina = self.base_stamina
        self.stamina_recharge = self.base_stamina_recharge
        self.speed = self.base_speed
        self.hp = self.base_hp  # Assuming max_hp is the full health of the Dino

