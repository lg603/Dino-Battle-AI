from torch import nn


class BattleNet(nn.Module):
    def __init__(self):
        super(BattleNet, self).__init__()
        self.fc1 = nn.Linear(72, 128)  # 72 features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Assuming 3 output classes

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
