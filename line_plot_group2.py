import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read the average_fitness_per_generation.txt file and extract generation and mean fitness
def read_results(file_path):
    generations = []
    mean_fitness = []

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return np.array([]), np.array([])

    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            print(f"File is empty: {file_path}")
            return np.array([]), np.array([])

        for line in lines:
            parts = line.strip().split(': Average Fitness: ')
            if len(parts) == 2:  # We expect "Generation X: Average Fitness: Y"
                try:
                    generation_part = parts[0].replace("Generation ", "").strip()
                    fitness_value = float(parts[1])
                    generations.append(int(generation_part))
                    mean_fitness.append(fitness_value)
                except ValueError as e:
                    print(f"Error parsing line '{line}' in file {file_path}: {e}")

    if not generations:
        print(f"No valid data in file: {file_path}")
        return np.array([]), np.array([])

    return np.array(generations), np.array(mean_fitness)

# Function to pad lists to the same length
def pad_sequences(seq_list, max_len):
    if len(seq_list) == 0:
        print("No sequences to pad.")
        return np.array([])
    return [np.pad(seq, (0, max_len - len(seq)), constant_values=np.nan) for seq in list(seq_list)]

# Function to plot mean and max fitness with standard deviation clouds for multiple experiments
def plot_evolution_with_std_cloud(num_runs, experiment_folder_base, label_prefix, color_mean, color_max, run_prefix):
    all_mean_fitness = []
    all_max_fitness = []
    max_generations = 0

    for run in range(1, num_runs + 1):
        # Use the correct run prefix for each experiment (EA 1 or EA 2)
        experiment_name = f"{experiment_folder_base}/{run_prefix}_run_{run}"
        results_file = os.path.join(experiment_name, 'average_fitness_per_generation.txt')

        print(f"Looking for file: {results_file}")
        generations, mean_fitness = read_results(results_file)

        if len(generations) > 0:
            all_mean_fitness.append(mean_fitness)
            max_generations = max(max_generations, len(generations))

            # Calculate the maximum fitness for the run by taking the highest value in mean_fitness
            max_fitness = np.max(mean_fitness)
            all_max_fitness.append([max_fitness])  # We store it as a single value for this run
        else:
            print(f"No data found for run {run}, skipping.")

    if len(all_mean_fitness) == 0:
        print(f"No valid data found for {label_prefix}, skipping plotting.")
        return

    all_mean_fitness = pad_sequences(all_mean_fitness, max_generations)
    all_mean_fitness = np.array(all_mean_fitness)

    mean_avg_fitness = np.nanmean(all_mean_fitness, axis=0)
    std_mean_fitness = np.nanstd(all_mean_fitness, axis=0)

    max_fitness_across_runs = np.array(all_max_fitness).flatten()  # We get the maximum fitness per run
    mean_max_fitness = np.mean(max_fitness_across_runs)
    std_max_fitness = np.std(max_fitness_across_runs)

    # Plot Mean Fitness
    plt.plot(np.arange(max_generations), mean_avg_fitness, label=f'{label_prefix} - Mean Fitness', color=color_mean, linewidth=2)
    plt.fill_between(np.arange(max_generations), mean_avg_fitness - std_mean_fitness, mean_avg_fitness + std_mean_fitness,
                     color=color_mean, alpha=0.2)

    # Plot Max Fitness as a horizontal line (since it is the maximum for the whole run)
    plt.hlines(mean_max_fitness, 0, max_generations - 1, label=f'{label_prefix} - Max Fitness', color=color_max, linewidth=2)
    plt.fill_between(np.arange(max_generations), mean_max_fitness - std_max_fitness, mean_max_fitness + std_max_fitness,
                     color=color_max, alpha=0.2)

# Main plot setup
num_runs = 10  # Adjust based on the number of runs

plt.figure(figsize=(10, 6))

# Plot for Evolutionary Algorithm 1 (No Island Evolution)
print("Processing EA 1 (extinction_event_ea_task2)")
plot_evolution_with_std_cloud(num_runs, experiment_folder_base='extinction_event_ea_task2/enemies_group_two',
                              label_prefix="Extinction Event EA", color_mean='skyblue', color_max='blue', run_prefix="extinction_event_ea_task2")

# Plot for Evolutionary Algorithm 2 (Island Evolution)
print("Processing EA 2 (island_ea_task2)")
plot_evolution_with_std_cloud(num_runs, experiment_folder_base='island_ea_task2/enemies_group_two',
                              label_prefix="Island EA", color_mean='red', color_max='darkred', run_prefix="island_ea_task2")

# Adding labels, title, and legend
plt.xlabel('Generation', fontsize=16, fontweight='bold')
plt.ylabel('Fitness', fontsize=16, fontweight='bold')
plt.title('Mean and Max Fitness Comparison for enemy subgroup [4,5,6,8]', fontsize=16)

# Add legend
plt.legend(loc='lower right', fontsize=12)

# Add grid
plt.grid(True)

# Show the plot once for both EAs
plt.tight_layout()
plt.show()
