import matplotlib.pyplot as plt
import numpy as np
import os

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

# Function to pad lists to the same length
def pad_sequences(seq_list, max_len):
    return [np.pad(seq, (0, max_len - len(seq)), constant_values=np.nan) for seq in seq_list]

# Function to plot mean lines and standard deviation clouds for multiple experiments
def plot_evolution_with_std_cloud(num_runs, experiment_name_base, label_prefix, color_best, color_avg):
    all_best_fitness = []
    all_mean_fitness = []
    max_generations = 0

    for run in range(1, num_runs + 1):
        experiment_name = f"{experiment_name_base}{run}"
        results_file = os.path.join(experiment_name, 'results.txt')
        
        if os.path.exists(results_file):
            generations, best_fitness, mean_fitness, _ = read_results(results_file)
            all_best_fitness.append(best_fitness)
            all_mean_fitness.append(mean_fitness)
            max_generations = max(max_generations, len(generations))

    all_best_fitness = pad_sequences(all_best_fitness, max_generations)
    all_mean_fitness = pad_sequences(all_mean_fitness, max_generations)
    
    all_best_fitness = np.array(all_best_fitness)
    all_mean_fitness = np.array(all_mean_fitness)

    mean_best_fitness = np.nanmean(all_best_fitness, axis=0)
    std_best_fitness = np.nanstd(all_best_fitness, axis=0)
    mean_avg_fitness = np.nanmean(all_mean_fitness, axis=0)
    std_avg_fitness = np.nanstd(all_mean_fitness, axis=0)

    plt.plot(np.arange(max_generations), mean_best_fitness, label=f'{label_prefix} - Best Fitness', color=color_best, linewidth=2)
    plt.fill_between(np.arange(max_generations), mean_best_fitness - std_best_fitness, mean_best_fitness + std_best_fitness, 
                     color=color_best, alpha=0.2)

    plt.plot(np.arange(max_generations), mean_avg_fitness, label=f'{label_prefix} - Avg Fitness', color=color_avg, linewidth=2)
    plt.fill_between(np.arange(max_generations), mean_avg_fitness - std_avg_fitness, mean_avg_fitness + std_avg_fitness, 
                     color=color_avg, alpha=0.2)

# Usage for Enemy 1 data:
num_runs = 10  # Adjust based on the number of runs

plt.figure(figsize=(10, 6))

# Plot for Evolutionary Algorithm (No Island Evolution)
plot_evolution_with_std_cloud(num_runs, experiment_name_base='team1_test_run_enemy3', label_prefix="Normal Evolution", color_best='red', color_avg='blue')

# Plot for Evolutionary Algorithm (Island Evolution)
plot_evolution_with_std_cloud(num_runs, experiment_name_base='team2_test_run_enemy3', label_prefix="Island Evolution", color_best='green', color_avg='gold')

# Adding labels, title, and legend with updated font size
plt.xlabel('Generation', fontsize=16, fontweight='bold')  # Increased font size and bold for x-axis label
plt.ylabel('Fitness', fontsize=16, fontweight='bold')  # Increased font size and bold for y-axis label
plt.title('Average Fitness Comparison on Enemy 3', fontsize=16)

# Set x-axis and y-axis limits
plt.xlim(0, 25)  # Limit x-axis to show up to generation 50
plt.ylim(0, 100)  # Adjust y-axis limits as necessary

# Set x-axis to show whole numbers
plt.xticks(np.arange(0, 26, step=5))  # Show x-axis ticks from 0 to 50 with step of 5

# Add legend with larger font size
plt.legend(loc='lower right', fontsize=16)  # Increased legend font size

# Add grid
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
