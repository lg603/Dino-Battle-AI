from Battle import Battle
import json
from DataVis import visualize_battle_statistics
import pandas as pd


def run_simulation(number_of_battles, visualize_data=False, capture_battle_stats=False):
    overall_statistics = {}  # Initialize overall statistics
    type_wins = {}  # Initialize type wins
    visualize = visualize_data
    capture_stats = capture_battle_stats
    battle_data = []

    for i in range(number_of_battles):  # Run Battles
        if i % 100 == 0:
            print(f"Battle Number {i}")
        battle = Battle(debug=0, init_overall_statistics=overall_statistics, init_type_wins=type_wins,
                        capture=capture_stats)
        battle.assign_moves_to_teams()

        # Capture data for training
        # if capture_stats:
        # print(battle.pregame_stats)

        # Run the battle
        battle.battle_sequence()

        # Capture the winner of the battle
        if capture_stats:
            # print(battle.winner)
            battle_data.append({
                "pregame_stats": battle.pregame_stats,
                "winner": battle.winner
            })

        # Update the visualized stats after a win if enabled
        if visualize:
            battle.update_statistics_after_battle()  # This method now updates overall_statistics and type_wins

    # Calculate averages and clean up statistics before saving if visualize is True
    if visualize:
        for dino_name, stats in overall_statistics.items():
            stats['avg_final_hp'] = stats['final_hp_sum'] / stats['final_hp_count'] if stats[
                                                                                           'final_hp_count'] > 0 else 0
            stats['avg_rounds_survived'] = stats['rounds_survived_sum'] / stats['rounds_survived_count'] \
                if stats['rounds_survived_count'] > 0 else 0
            # Remove the sum and count fields if they are no longer needed
            del stats['final_hp_sum'], stats['final_hp_count'], stats['rounds_survived_sum'], stats[
                'rounds_survived_count']

        # Write overall statistics to a file
        with open('data/battle_statistics.json', 'w') as file:
            json.dump(overall_statistics, file, indent=4)

        json_file = 'data/battle_statistics.json'  # Visualize the results of the battles ran
        visualize_battle_statistics(json_file)

    if capture_stats:
        return battle_data


def structure_data(battle_data):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(battle_data)

    # Assuming pregame_stats is a list of 72 elements
    # Expand each list into its own column
    stats_df = df['pregame_stats'].apply(pd.Series)

    # Concatenate with the winner column
    final_df = pd.concat([stats_df, df['winner']], axis=1)

    return final_df


if __name__ == "__main__":
    save_battle_data = True  # Toggle to save battle data to a csv for training my model
    num_battles = 30000  # Change the number of battles being run
    data = run_simulation(num_battles, capture_battle_stats=True)  # Run 100 battle simulations
    structured_data = structure_data(data)
    if save_battle_data:  # Save to CSV
        structured_data.to_csv('data/battle_data.csv', index=False)
    print(structured_data)
