import numpy as np
import os
from scipy.stats import ttest_ind
import pandas as pd

# Function to read the results.txt file and extract generation, best fitness, mean, and std
def read_results(file_path):
    generations = []
    best_fitness = []
    mean_fitness = []
    std_dev = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    generations.append(int(parts[0]))
                    best_fitness.append(float(parts[1]))
                    mean_fitness.append(float(parts[2]))
                    std_dev.append(float(parts[3]))
                except ValueError as e:
                    print(f"Error parsing line '{line}': {e}")
                    
    return np.array(generations), np.array(best_fitness), np.array(mean_fitness), np.array(std_dev)

# Function to load all best fitness and mean fitness data for multiple runs
def load_fitness_data(num_runs, experiment_name_base):
    all_best_fitness = []
    all_mean_fitness = []
    
    for run in range(1, num_runs + 1):
        experiment_name = f"{experiment_name_base}{run}"
        results_file = os.path.join(experiment_name, 'results.txt')
        
        if os.path.exists(results_file):
            generations, best_fitness, mean_fitness, _ = read_results(results_file)
            all_best_fitness.extend(best_fitness)
            all_mean_fitness.extend(mean_fitness)

    return np.array(all_best_fitness), np.array(all_mean_fitness)

# Function to calculate p-values for two datasets
def calculate_p_values(data_1, data_2):
    # Perform t-test between two datasets (best or average fitness)
    p_value = ttest_ind(data_1, data_2)[1]
    return p_value

# Usage for Enemy 1, Enemy 2, and Enemy 3 data:
num_runs = 10  # Adjust based on the number of runs

# Load data for all 3 experiment sets (only comparing between algorithms)
best_normal_enemy1, mean_normal_enemy1 = load_fitness_data(num_runs, experiment_name_base='team1_test_run_')
best_island_enemy1, mean_island_enemy1 = load_fitness_data(num_runs, experiment_name_base='team2_test_run_')

best_normal_enemy2, mean_normal_enemy2 = load_fitness_data(num_runs, experiment_name_base='team1_test_run_enemy2')
best_island_enemy2, mean_island_enemy2 = load_fitness_data(num_runs, experiment_name_base='team2_test_run_enemy2')

best_normal_enemy3, mean_normal_enemy3 = load_fitness_data(num_runs, experiment_name_base='team1_test_run_enemy3')
best_island_enemy3, mean_island_enemy3 = load_fitness_data(num_runs, experiment_name_base='team2_test_run_enemy3')

# Calculate p-values for best individual lines (Normal vs Island for each enemy)
p_value_best_normal_vs_island_enemy1 = calculate_p_values(best_normal_enemy1, best_island_enemy1)
p_value_best_normal_vs_island_enemy2 = calculate_p_values(best_normal_enemy2, best_island_enemy2)
p_value_best_normal_vs_island_enemy3 = calculate_p_values(best_normal_enemy3, best_island_enemy3)

# Calculate p-values for average lines (Normal vs Island for each enemy)
p_value_avg_normal_vs_island_enemy1 = calculate_p_values(mean_normal_enemy1, mean_island_enemy1)
p_value_avg_normal_vs_island_enemy2 = calculate_p_values(mean_normal_enemy2, mean_island_enemy2)
p_value_avg_normal_vs_island_enemy3 = calculate_p_values(mean_normal_enemy3, mean_island_enemy3)

# Create a table with the p-values (only between algorithms per enemy)
p_values_table = pd.DataFrame({
    'Enemy': [
        'Enemy 1', 
        'Enemy 2',
        'Enemy 3'
    ],
    'p-value Best (Normal vs Island)': [
        p_value_best_normal_vs_island_enemy1, 
        p_value_best_normal_vs_island_enemy2, 
        p_value_best_normal_vs_island_enemy3
    ],
    'p-value Avg (Normal vs Island)': [
        p_value_avg_normal_vs_island_enemy1, 
        p_value_avg_normal_vs_island_enemy2, 
        p_value_avg_normal_vs_island_enemy3
    ]
})

# Save p-values table to Excel
p_values_table.to_excel('p_values_comparison_table_algorithms.xlsx', index=False)

# Show the table
p_values_table
