import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from battle_net import BattleNet


def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)

            # Ensure labels and predicted are tensors of the same type
            labels = labels.to(predicted.device).type(predicted.dtype)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Track loss over epochs
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for batch_features, batch_labels in train_loader:
            # Forward pass
            outputs = model(batch_features)
            l1_lambda = 0.001
            l1_norm = sum(p.abs().sum() for p in model.parameters())

            loss = criterion(outputs, batch_labels) + l1_lambda * l1_norm  # L1 regularization

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return epoch_losses


def plot_performance(epoch_losses):
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Training Performance')
    plt.legend()
    plt.show()
