import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from BattleRunner import run_simulation
from train_model import train, BattleNet, evaluate, plot_performance
from sklearn.preprocessing import MinMaxScaler


def load_data_to_tensor(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Assuming the last column is the label and the rest are features
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Calculate min and max for each feature
    x_min = x_tensor.min(dim=0, keepdim=True).values
    x_max = x_tensor.max(dim=0, keepdim=True).values

    # Apply min-max scaling
    x_tensor_normalized = (x_tensor - x_min) / (x_max - x_min)

    # Handle any divisions by zero (if max == min)
    x_tensor_normalized[x_tensor_normalized != x_tensor_normalized] = 0

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
    num_battles = 50000

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
    epoch_losses = train(model, train_loader, num_epochs=50)

    # Evaluate the model
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy}%")

    # Plot training performance
    plot_performance(epoch_losses)

    # Here you can add code to evaluate the model on the test_loader
    # For example: evaluate(model, test_loader)


if __name__ == "__main__":
    main()

"""Look into a couple of things:
 - Adding move pools to each dino to reduce randomness of move assignment. IE, T-Rex can only learn attack boosting, 
 stamina boosting, and attack moves. Brontosaurus can only learn defensive boosting and attack moves, etc.
 - Adding more data to the features for my model. I would have to look back into my feature design (and WRITE IT DOWN
 SOMEWHERE WHAT I AM USING FOR FEATURES, AND WHAT SPOTS THEY ARE IN), and decide what to add. I think I would have to 
 add other general stats like win-rates for certain dinos, certain team-comps, etc. Also, look into how exactly NNs
 learn the associations in the data to see if there is a more optimal model for this task.
 - Try and do some math to figure out if there is an upper limit to how well a model can predict for things with 
 stochastically determined outcomes.
"""


