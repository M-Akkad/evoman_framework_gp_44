################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os
import contextlib

# ATTENTION: To train change headless to true, visuals(within env) to false and run_mode to train job

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'Basic_EA_with_Islands'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for multiple enemies.
enemies = [1, 2, 3, 7]  # group 1
# enemies = [4, 5, 6, 8] #group 2

env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  playermode="ai",
                  multiplemode="yes",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  randomini="no",
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Genetic Algorithm   ###

ini = time.time()  # sets time marker

# Genetic algorithm params

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

run_mode = 'train'  # train or test
dom_u = 1
dom_l = -1
npop = 100
gens = 50
mutation_rate = 0.1
mutation_weight = 0.3
n_parents = 2
k = 3  # Tournament size
num_offspring = 50

# Island Model parameters
num_islands = 4  # Number of islands
migration_interval = 10  # Migrate every 10 generations
migration_size = 3  # Number of individuals to migrate


# Define a null output context to suppress prints and error outputs
class NullOutput:
    def write(self, _):
        pass

    def flush(self):
        pass


# Update parameter without logging messages
def update_parameter_silently(env, param, value):
    with contextlib.redirect_stdout(NullOutput()), contextlib.redirect_stderr(NullOutput()):
        env.update_parameter(param, value)


# Evaluate fitness
def evaluate_population(population):
    fitness_scores = []
    for individual in population:
        total_fitness = 0
        for enemy in enemies:
            update_parameter_silently(env, 'enemies', [enemy])  # Set the current enemy silently
            f, p, e, t = env.play(pcont=individual)
            total_fitness += f
        fitness_scores.append(total_fitness / len(enemies))
    return fitness_scores


# Tournament selection
def tournament_selection(population, fitness_scores, k):
    selected_parents = []
    for _ in range(0, len(population), 2):
        tournament_indices = np.random.choice(len(population), k, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_parents.append(population[winner_index])
    return selected_parents


# Selection
def select_parents(population, fitness_scores, k=3):
    selected_parents = tournament_selection(population, fitness_scores, k)
    return selected_parents


# Multi-parent recombination type 2
def multi_parent_recombination(parents):
    child = np.zeros(n_vars)
    for i in range(n_vars):
        child[i] = np.mean([parent[i] for parent in parents])
    return child


# Mutation
def mutate(child, mutation_rate, mutation_weight):
    for i in range(n_vars):
        if np.random.rand() < mutation_rate:
            child[i] += np.random.uniform((-1 * mutation_weight), mutation_weight)
    return child


def migrate(populations, fitness_scores_list, migration_size):
    for i in range(num_islands):
        source_island = i
        target_island = (i + 1) % num_islands

        combined = list(zip(populations[source_island], fitness_scores_list[source_island]))
        combined.sort(key=lambda x: x[1], reverse=True)
        migrants = [ind for ind, fit in combined[:migration_size]]

        combined_target = list(zip(populations[target_island], fitness_scores_list[target_island]))
        combined_target.sort(key=lambda x: x[1])
        for j in range(migration_size):
            populations[target_island][j] = migrants[j]
            fitness_scores_list[target_island][j] = fitness_scores_list[source_island][j]

    return populations, fitness_scores_list


# Evolution loop
def evolve_population(population, fitness_scores, mutation_rate, num_offspring=50, mutation_weight=0.1, k=3,
                      n_parents=2):
    offspring = []
    selected_parents = select_parents(population, fitness_scores, k)
    for _ in range(num_offspring):
        parent_indices = np.random.choice(len(selected_parents), n_parents, replace=False)
        parents = [selected_parents[i] for i in parent_indices]
        child = multi_parent_recombination(parents)
        mutated_child = mutate(child, mutation_rate, mutation_weight)
        offspring.append(mutated_child)

    new_population = population + offspring
    new_fitness_scores = evaluate_population(new_population)

    combined = list(zip(new_population, new_fitness_scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    population = [ind for ind, fitness in combined[:npop]]
    fitness_scores = [fitness for ind, fitness in combined[:npop]]

    return population, fitness_scores


# Main loop
if run_mode == 'train':

    # Number of runs
    num_runs = 10

    for run in range(1, num_runs + 1):
        print(f"\nStarting Run {run}...\n")

        # Create a unique folder for each run
        experiment_name = f'EA2_gr1_2_run_num_{run}'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        # Initialize environment for each run
        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)

        env.state_to_log()  # Log environment state

        # Initialize the genetic algorithm variables
        populations = [
            [np.random.uniform(dom_l, dom_u, n_vars) for _ in range(npop)]
            for _ in range(num_islands)
        ]
        fitness_scores_list = [evaluate_population(population) for population in populations]
        overall_best_individual = None
        overall_best_fitness = -float('inf')

        # Evolutionary process
        avg_fitness_per_generation = []  # Track average fitness per generation
        for generation in range(gens):
            print(f"\nEvolving Generation {generation} for Run {run}")

            for i in range(num_islands):
                populations[i], fitness_scores_list[i] = evolve_population(
                    populations[i],
                    fitness_scores_list[i],
                    mutation_rate,
                    num_offspring,
                    mutation_weight,
                    k,
                    n_parents
                )

            if generation % migration_interval == 0:
                print("  Migration between islands")
                populations, fitness_scores_list = migrate(populations, fitness_scores_list, migration_size)

            # Find best individual across all islands
            all_fitness = [fitness for fitness_scores in fitness_scores_list for fitness in fitness_scores]
            all_individuals = [ind for population in populations for ind in population]
            best_idx = np.argmax(all_fitness)
            best_individual = all_individuals[best_idx]
            best_fitness = all_fitness[best_idx]

            # Save the best individual of the current generation
            np.savetxt(f"{experiment_name}/best_individual_generation_{generation}.txt", best_individual)
            with open(f"{experiment_name}/best_fitness_generation.txt", 'a') as best_file:
                best_file.write(f"Generation {generation}: Best Fitness: {best_fitness}\n")

            # Calculate and save the average fitness for the current generation
            avg_fitness = np.mean(all_fitness)
            avg_fitness_per_generation.append(avg_fitness)
            with open(f"{experiment_name}/average_fitness_per_generation.txt", 'a') as avg_file:
                avg_file.write(f"Generation {generation}: Average Fitness: {avg_fitness}\n")

            # Evaluate best individual for each enemy and log results
            for enemy in enemies:
                update_parameter_silently(env, 'enemies', [enemy])  # Set the current enemy silently
                f_best, p_best, e_best, t_best = env.play(pcont=best_individual)
                gain_best = p_best - e_best

                # Save the best individual's evaluation metrics for each enemy
                with open(f"{experiment_name}/best_individual_enemy_{enemy}_results.txt", 'a') as file_aux:
                    print(
                        f' GENERATION {generation} Best for Enemy {enemy}: Fitness: {round(f_best, 6)}, Player Life: {p_best}, Enemy Life: {e_best}, Time: {t_best}, Gain: {gain_best}')
                    file_aux.write(
                        f'\nGeneration {generation} - Best Individual for Enemy {enemy}: Fitness: {round(f_best, 6)}, Player Life: {p_best}, Enemy Life: {e_best}, Time: {t_best}, Gain: {gain_best}')

            # Track the overall best individual and fitness for this run
            if best_fitness > overall_best_fitness:
                overall_best_fitness = best_fitness
                overall_best_individual = best_individual

        # After evolution is completed, save the all-time best individual
        if overall_best_individual is not None:
            np.savetxt(f"{experiment_name}/final_overall_best.txt", overall_best_individual)
            print(f"Overall best fitness for Run {run}: {overall_best_fitness}")

        print(f"\nRun {run} completed.\n")

    print(f"\nAll {num_runs} runs completed.")


# Test the best solution
elif run_mode == 'test':

    try:
        # Load the best solution from the file

        file_path = f"{experiment_name}/final_overall_best.txt"
        best_sol = np.loadtxt(file_path)

        print('\n RUNNING SAVED BEST SOLUTION \n')

        # Set the speed to normal for testing (you may adjust this)
        update_parameter_silently(env, 'speed', 'normal')

        # Loop through each enemy to evaluate the best solution separately
        for enemy in enemies:
            update_parameter_silently(env, 'enemies', [enemy])  # Set the current enemy silently

            # Evaluate the best solution multiple times (e.g., 5 times) for each enemy
            for i in range(5):
                f, p, e, t = env.play(pcont=best_sol)

                # Print the evaluation results (fitness, player life, enemy life, time taken)
                gain = p - e
                print(
                    f"Enemy {enemy} - Test Run {i + 1}: Fitness: {f}, Player Life: {p}, Enemy Life: {e}, Time: {t}, Gain: {gain}")

        sys.exit(0)

    except Exception as e:
        print(f"Error loading best solution: {e}")
        sys.exit(1)
