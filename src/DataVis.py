# visualize.py
import json
import pandas as pd
import matplotlib.pyplot as plt


def visualize_battle_statistics(json_file):
    # Load data from JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data).T

    # Sort DataFrame by 'wins' in descending order
    df_sorted = df.sort_values(by='wins', ascending=False)

    # Bar plot for wins, losses, and draws
    df_sorted[['wins', 'losses', 'draws']].plot(kind='bar', figsize=(15, 7))
    plt.title('Wins, Losses, and Draws for Each Dinosaur (Sorted by Wins)')
    plt.ylabel('Count')
    plt.xlabel('Dinosaur')
    plt.xticks(rotation=45)
    plt.show()

    # Scatter plot for average final HP vs average rounds survived
    plt.figure(figsize=(10, 6))
    plt.scatter(df['avg_final_hp'], df['avg_rounds_survived'], alpha=0.7)
    plt.title('Average Final HP vs Average Rounds Survived')
    plt.xlabel('Average Final HP')
    plt.ylabel('Average Rounds Survived')
    for i, txt in enumerate(df.index):
        plt.annotate(txt, (df['avg_final_hp'][i], df['avg_rounds_survived'][i]))
    plt.show()

    # Histogram for average rounds survived
    df['avg_rounds_survived'].plot(kind='hist', bins=20, figsize=(10, 6))
    plt.title('Histogram of Average Rounds Survived')
    plt.xlabel('Average Rounds Survived')
    plt.ylabel('Frequency')
    plt.show()

    # Box plot for final HP (min, avg, max)
    df[['final_hp_min', 'avg_final_hp', 'final_hp_max']].plot(kind='box', figsize=(10, 6))
    plt.title('Box Plot of Final HP Statistics')
    plt.ylabel('HP')
    plt.show()
