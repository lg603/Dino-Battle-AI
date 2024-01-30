import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from BattleRunner import run_simulation
from train_model import BattleNet, train

def load_data_to_tensor(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Assuming the last column is the label and the rest are features
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert to tensors
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create a TensorDataset
    dataset = TensorDataset(x_tensor, y_tensor)

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


# TODO: Figure out all of the funky imports