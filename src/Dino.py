import math


class Dino:
    def __init__(self, id, name, dino_type, hp, attack, defense, stamina, stamina_recharge, speed):
        self.id = id
        self.name = name
        self.dino_type = dino_type
        self.hp = hp
        self.base_attack = attack
        self.base_defense = defense
        self.base_stamina = stamina
        self.base_stamina_recharge = stamina_recharge
        self.base_speed = speed
        self.max_stamina = stamina
        self.attack = attack
        self.defense = defense
        self.curr_stamina = stamina
        self.stamina_recharge = stamina_recharge
        self.speed = speed
        self.moves = []

    def __str__(self):
        dino_info = (f"{self.name} ({self.dino_type}) - HP: {self.hp}, Attack: {self.attack}, Defense: {self.defense},"
                     f" Stamina: {self.curr_stamina}, Stamina Recharge: {self.stamina_recharge}, Speed: {self.speed}")

        # Add moves to the string if the dinosaur has any
        if self.moves:
            moves_str = ', '.join([move.name for move in self.moves])
            dino_info += f"\nMoves: {moves_str}"
        else:
            dino_info += "\nMoves: None"

        return dino_info

    def change_stat(self, stat, multiplier):
        if stat == 'attack':
            self.attack = max(1, math.floor(self.attack * multiplier))
        elif stat == 'defense':
            self.defense = max(1, math.floor(self.defense * multiplier))
        elif stat == 'stamina':
            self.max_stamina = max(1, math.floor(self.max_stamina * multiplier))
        elif stat == 'stamina_recharge':
            self.stamina_recharge = max(1, math.floor(self.stamina_recharge * multiplier))
        elif stat == 'speed':
            self.speed = max(1, math.floor(self.speed * multiplier))

    def use_stamina(self, cost):
        self.curr_stamina = max(0, self.curr_stamina - cost)

    def recharge_stamina(self):
        self.curr_stamina += self.stamina_recharge

    def rest(self):
        print(f"{self.name} is resting this turn and refilling its stamina.")
        self.curr_stamina = self.max_stamina
