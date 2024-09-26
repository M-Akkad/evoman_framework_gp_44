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
        # Skip the first line (header)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 4:  # Ensure the line has all 4 parts
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
    max_generations = 0  # To keep track of the maximum number of generations

    for run in range(1, num_runs + 1):
        experiment_name = f"{experiment_name_base}{run}"
        results_file = os.path.join(experiment_name, 'results.txt')
        
        if os.path.exists(results_file):
            generations, best_fitness, mean_fitness, _ = read_results(results_file)
            
            all_best_fitness.append(best_fitness)
            all_mean_fitness.append(mean_fitness)
            max_generations = max(max_generations, len(generations))

    # Pad sequences to the length of the run with the most generations
    all_best_fitness = pad_sequences(all_best_fitness, max_generations)
    all_mean_fitness = pad_sequences(all_mean_fitness, max_generations)
    
    # Convert lists to numpy arrays for easier mean and std calculations
    all_best_fitness = np.array(all_best_fitness)
    all_mean_fitness = np.array(all_mean_fitness)

    # Calculate mean and standard deviation across runs, ignoring NaN values
    mean_best_fitness = np.nanmean(all_best_fitness, axis=0)
    std_best_fitness = np.nanstd(all_best_fitness, axis=0)
    mean_avg_fitness = np.nanmean(all_mean_fitness, axis=0)
    std_avg_fitness = np.nanstd(all_mean_fitness, axis=0)

    # Plot the mean line for the best fitness across runs (solid line)
    plt.plot(np.arange(max_generations), mean_best_fitness, label=f'{label_prefix} - Mean Best Fitness', color=color_best, linewidth=2)
    # Plot the standard deviation for the best fitness (no legend)
    plt.fill_between(np.arange(max_generations), mean_best_fitness - std_best_fitness, mean_best_fitness + std_best_fitness, 
                     color=color_best, alpha=0.2)

    # Plot the mean line for the average fitness across runs (solid line)
    plt.plot(np.arange(max_generations), mean_avg_fitness, label=f'{label_prefix} - Mean Avg Fitness', color=color_avg, linewidth=2)
    # Plot the standard deviation for the average fitness (no legend)
    plt.fill_between(np.arange(max_generations), mean_avg_fitness - std_avg_fitness, mean_avg_fitness + std_avg_fitness, 
                     color=color_avg, alpha=0.2)


# Usage to plot both algorithms (no island evolution vs island evolution) results in the same plot:

num_runs = 10  # Adjust to match the number of runs you performed

# Initialize the figure
plt.figure(figsize=(10, 6))

# Plot for Evolutionary Algorithm (No Island Evolution)
plot_evolution_with_std_cloud(num_runs, experiment_name_base='team1_test_run_enemy2', label_prefix="No Island Evolution", color_best='red', color_avg='blue')

# Plot for Evolutionary Algorithm (Island Evolution)
plot_evolution_with_std_cloud(num_runs, experiment_name_base='team2_test_run_enemy2', label_prefix="Island Evolution", color_best='green', color_avg='gold')

# Adding labels, title, and legend
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Average Fitness Comparison on Enemy 2')

# Set x-axis and y-axis limits
plt.xlim(0, 15)  # Limit x-axis to show up to generation 15
plt.ylim(0, 100)  # Limit y-axis to show fitness values between 0 and 100

# Set x-axis to show only whole numbers
plt.xticks(np.arange(0, 16, 5))  # Show x-axis ticks from 0 to 15 with step of 5

# Add a truncated marker (vertical dashed line with label at 15)
plt.axvline(x=15, color='black', linestyle='--', linewidth=1)  # Vertical line at generation 15
plt.text(15.2, 50, 'Values constant\nbeyond generation 15', rotation=90, verticalalignment='center', fontsize=10, color='black')

plt.legend()  # Only the lines will be in the legend, not the std deviation
plt.grid(True)

# Show the plot
plt.show()
