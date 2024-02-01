import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from BattleRunner import run_simulation
from train_model import train, BattleNet, evaluate, plot_performance


def load_data_to_tensor(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Assuming the last column is the label and the rest are features
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Assuming x_tensor is your data tensor
    mean = x_tensor.mean(dim=0, keepdim=True)
    std = x_tensor.std(dim=0, keepdim=True)

    # Avoid division by zero
    std_replaced = std.where(std != 0, torch.ones_like(std))

    # Apply normalization
    x_tensor_normalized = (x_tensor - mean) / std_replaced

    # Create a TensorDataset
    dataset = TensorDataset(x_tensor_normalized, y_tensor)

    # Calculate the sizes of train and test sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Split the dataset into training and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader


def main():
    file_path = "data/battle_data.csv"
    num_battles = 30000

    # Check if the data file exists
    if not os.path.exists(file_path):
        print("Data file not found. Running battle simulations...")
        run_simulation(num_battles)

    print("Loading data...")
    train_loader, test_loader = load_data_to_tensor(file_path)

    # Initialize the model
    model = BattleNet()

    # Train the model
    print("Training model...")
    epoch_losses = train(model, train_loader, num_epochs=75)

    # Evaluate the model
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy}%")

    # Plot training performance
    plot_performance(epoch_losses)

    # Here you can add code to evaluate the model on the test_loader
    # For example: evaluate(model, test_loader)


if __name__ == "__main__":
    main()
