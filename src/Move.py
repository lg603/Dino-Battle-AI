import json


class Move:
    def __init__(self, move_id, name, move_type, power=None, accuracy=None, stamina_cost=None, effect=None):
        self.id = move_id
        self.name = name
        self.move_type = move_type  # 1: Attack, 2: Stat Boost, 3: Stat Debuff
        self.power = power
        self.accuracy = accuracy
        self.stamina_cost = stamina_cost
        self.effect = effect

    def __str__(self):
        if self.move_type == 1:
            return (f"Move: {self.name}, Type: Attack, Power: {self.power}, Accuracy: {self.accuracy}, "
                    f"Stamina Cost: {self.stamina_cost}")
        elif self.move_type == 2:
            return (f"Move: {self.name}, Type: Stat Boost, Effect: {self.effect}, Accuracy: {self.accuracy}, "
                    f"Stamina Cost: {self.stamina_cost}")
        elif self.move_type == 3:
            return (f"Move: {self.name}, Type: Stat Debuff, Effect: {self.effect}, Accuracy: {self.accuracy}, "
                    f"Stamina Cost: {self.stamina_cost}")
        else:
            return "Invalid move type"


def load_moves_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        moves = data['moves']
        move_objects = []

        for move in moves:
            move_type = move.get('type')
            name = move.get('name')
            power = move.get('power')
            accuracy = move.get('accuracy')
            stamina_cost = move.get('stamina_cost')
            effect = move.get('effect')

            move_objects.append(Move(name, move_type, power, accuracy, stamina_cost, effect))

        return move_objects

