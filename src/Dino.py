class Dino:
    def __init__(self, name, dino_type, hp, attack, defense, stamina, stamina_recharge, speed):
        self.name = name
        self.dino_type = dino_type
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.stamina = stamina
        self.stamina_recharge = stamina_recharge
        self.speed = speed

    def __str__(self):
        return (f"{self.name} ({self.dino_type}) - HP: {self.hp}, Attack: {self.attack}, Defense: {self.defense},"
                f" Stamina: {self.stamina}, Stamina Recharge: {self.stamina_recharge}, Speed: {self.speed}")


    def add_move(self, move):
        if len(self.moves) < 4 and move not in self.moves:
            self.moves.append(move)
            print(f"{self.name} learned {move.name}!")
        elif len(self.moves) == 4:
            print(f"{self.name} already knows the maximum number of moves.")
        else:
            print(f"{self.name} already knows {move.name}.")
