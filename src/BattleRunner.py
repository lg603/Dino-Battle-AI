from Battle import Battle
import json
from DataVis import visualize_battle_statistics


def run_simulation(number_of_battles):
    overall_statistics = {}  # Initialize overall statistics
    type_wins = {}  # Initialize type wins

    for i in range(number_of_battles):  # Run Battles
        if i % 100 == 0:
            print(f"Battle Number {i}")
        battle = Battle(debug=0, init_overall_statistics=overall_statistics, init_type_wins=type_wins)
        battle.assign_moves_to_teams()
        battle.battle_sequence()
        battle.update_statistics_after_battle()  # This method now updates overall_statistics and type_wins

    # Calculate averages and clean up statistics before saving
    for dino_name, stats in overall_statistics.items():
        stats['avg_final_hp'] = stats['final_hp_sum'] / stats['final_hp_count'] if stats[
                                                                                       'final_hp_count'] > 0 else 0
        stats['avg_rounds_survived'] = stats['rounds_survived_sum'] / stats['rounds_survived_count'] \
            if stats['rounds_survived_count'] > 0 else 0
        # Remove the sum and count fields if they are no longer needed
        del stats['final_hp_sum'], stats['final_hp_count'], stats['rounds_survived_sum'], stats[
            'rounds_survived_count']

    # Write overall statistics to a file
    with open('battle_statistics.json', 'w') as file:
        json.dump(overall_statistics, file, indent=4)

    json_file = 'battle_statistics.json'  # Visualize the results of the battles ran
    visualize_battle_statistics(json_file)


if __name__ == "__main__":
    run_simulation(10000)  # Run 100 battle simulations
