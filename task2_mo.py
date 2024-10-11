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
import glob, os
import contextlib

# ATTENTION: To train change headless to true, visuals(within env) to false and run_mode to train job

# choose this for not using visuals and thus making experiments faster
headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'task2_mo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for multiple enemies.
enemies = [1, 2, 3]  # Specify multiple enemies to train on

env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  playermode="ai",
                  # multiplemode="yes",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  randomini="no",
                  speed="fastest",
                  visuals=True)

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Genetic Algorithm   ###

ini = time.time()  # sets time marker

# Genetic algorithm params

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

run_mode = 'test'  # train or test
dom_u = 1
dom_l = -1
npop = 100
gens = 50
mutation_rate = 0.1
mutation_weight = 0.3
n_parents = 2
k = 3  # Tournament size
num_offspring = 50
last_best = 0

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


# Fitness functions for different islands
def fitness_main(f, p, e, t):
    return f


def fitness_player_life(f, p, e, t):
    return p


def fitness_enemy_life(f, p, e, t):
    return -e  # Negative because we want to minimize enemy life


def fitness_time(f, p, e, t):
    return -t  # Negative because we want to minimize time


# Evaluate fitness for a specific island
def evaluate_population_island(population, fitness_func):
    fitness_scores = []
    for individual in population:
        total_fitness = 0
        for enemy in enemies:
            update_parameter_silently(env, 'enemies', [enemy])
            f, p, e, t = env.play(pcont=individual)
            total_fitness += fitness_func(f, p, e, t)
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


# Evolution loop for a single island
def evolve_island(population, fitness_scores, fitness_func, num_offspring=50, mutation_rate=0.1, mutation_weight=0.1,
                  k=3, n_parents=2):
    offspring = []
    selected_parents = select_parents(population, fitness_scores, k)
    for _ in range(num_offspring):
        parent_indices = np.random.choice(len(selected_parents), n_parents, replace=False)
        parents = [selected_parents[i] for i in parent_indices]
        child = multi_parent_recombination(parents)
        mutated_child = mutate(child, mutation_rate, mutation_weight)
        offspring.append(mutated_child)

    new_population = population + offspring
    new_fitness_scores = evaluate_population_island(new_population, fitness_func)

    combined = list(zip(new_population, new_fitness_scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    population = [ind for ind, fitness in combined[:npop]]
    fitness_scores = [fitness for ind, fitness in combined[:npop]]

    return population, fitness_scores


# Migration between islands
def migrate(islands, fitness_scores):
    for i in range(num_islands):
        emigrants = []
        for _ in range(migration_size):
            idx = np.random.randint(len(islands[i]))
            emigrants.append(islands[i].pop(idx))
            fitness_scores[i].pop(idx)

        next_island = (i + 1) % num_islands
        islands[next_island].extend(emigrants)
        fitness_scores[next_island].extend(evaluate_population_island(emigrants, fitness_funcs[next_island]))

    return islands, fitness_scores


# Main loop
if run_mode == 'train':
    num_runs = 1

    for run in range(1, num_runs + 1):
        print(f"\nStarting Run {run}...\n")

        experiment_name = f'task2_mo_run_{run}'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)

        env.state_to_log()

        # Initialize islands
        islands = [
            [np.random.uniform(dom_l, dom_u, n_vars) for _ in range(npop)] for _ in range(num_islands)
        ]
        fitness_funcs = [fitness_main, fitness_player_life, fitness_enemy_life, fitness_time]
        fitness_scores = [
            evaluate_population_island(islands[i], fitness_funcs[i]) for i in range(num_islands)
        ]

        overall_best_individual = None
        overall_best_fitness = -float('inf')

        avg_fitness_per_generation = []

        for generation in range(gens):
            print(f"\nEvolving Generation {generation} for Run {run}")

            # Evolve each island
            for i in range(num_islands):
                islands[i], fitness_scores[i] = evolve_island(
                    islands[i], fitness_scores[i], fitness_funcs[i], num_offspring, mutation_rate, mutation_weight, k,
                    n_parents
                )

            # Migrate between islands
            if generation % migration_interval == 0 and generation > 0:
                islands, fitness_scores = migrate(islands, fitness_scores)

            # Evaluate best individual (from main island) for each enemy
            best_idx = np.argmax(fitness_scores[0])
            best_individual = islands[0][best_idx]
            best_fitness = fitness_scores[0][best_idx]

            np.savetxt(f"{experiment_name}/best_individual_generation_{generation}.txt", best_individual)
            with open(f"{experiment_name}/best_fitness_generation.txt", 'a') as best_file:
                best_file.write(f"Generation {generation}: Best Fitness: {best_fitness}\n")

            avg_fitness = np.mean(fitness_scores[0])
            avg_fitness_per_generation.append(avg_fitness)
            with open(f"{experiment_name}/average_fitness_per_generation.txt", 'a') as avg_file:
                avg_file.write(f"Generation {generation}: Average Fitness: {avg_fitness}\n")

            for enemy in enemies:
                update_parameter_silently(env, 'enemies', [enemy])
                f_best, p_best, e_best, t_best = env.play(pcont=best_individual)
                gain_best = p_best - e_best

                with open(f"{experiment_name}/best_individual_enemy_{enemy}_results.txt", 'a') as file_aux:
                    print(
                        f' GENERATION {generation} Best for Enemy {enemy}: Fitness: {round(f_best, 6)}, Player Life: {p_best}, Enemy Life: {e_best}, Time: {t_best}, Gain: {gain_best}')
                    file_aux.write(
                        f'\nGeneration {generation} - Best Individual for Enemy {enemy}: Fitness: {round(f_best, 6)}, Player Life: {p_best}, Enemy Life: {e_best}, Time: {t_best}, Gain: {gain_best}')

            if best_fitness > overall_best_fitness:
                overall_best_fitness = best_fitness
                overall_best_individual = best_individual

        if overall_best_individual is not None:
            np.savetxt(f"{experiment_name}/final_overall_best.txt", overall_best_individual)
            print(f"Overall best fitness for Run {run}: {overall_best_fitness}")

        print(f"\nRun {run} completed.\n")

    print(f"\nAll {num_runs} runs completed.")


# Test the best solution for each focus (p, e, t)
elif run_mode == 'test':
    try:
        print('\nTESTING BEST INDIVIDUALS FOR EACH FITNESS FOCUS\n')
        experiment_name = "task2_mo_run_1"

        # Load the final overall best individual
        final_best_path = f"{experiment_name}/final_overall_best.txt"
        if not os.path.exists(final_best_path):
            raise FileNotFoundError("Final overall best individual file not found.")

        final_best_individual = np.loadtxt(final_best_path)

        print('\nRUNNING TESTS ON FINAL OVERALL BEST INDIVIDUAL\n')

        # Loop through each fitness focus ('overall', 'p', 'e', 't') to evaluate
        for focus in ['overall', 'p', 'e', 't']:
            print(f"\nEvaluating fitness focus: '{focus}'\n")

            # Initialize list for storing test results per focus
            test_results = []

            # Evaluate each enemy separately
            for enemy in enemies:
                update_parameter_silently(env, 'enemies', [enemy])

                # Perform multiple test runs for consistency
                for i in range(1):
                    f, p, e, t = env.play(pcont=final_best_individual)

                    # Calculate relevant fitness metric based on the focus
                    if focus == 'overall':
                        fitness = f
                    elif focus == 'p':  # Focus on Player Health
                        fitness = p
                    elif focus == 'e':  # Focus on Enemy Health (minimization)
                        fitness = -e
                    elif focus == 't':  # Focus on Time (minimization)
                        fitness = -t

                    # Store and print test results
                    test_results.append((enemy, fitness, f, p, e, t))
                    print(
                        f"Enemy {enemy} - Test Run {i + 1}: Focus Fitness: {fitness}, Overall Fitness: {f}, Player Life: {p}, Enemy Life: {e}, Time: {t}")

            # Save test results for this focus
            test_results_path = f"{experiment_name}/test_results_final_best_focus_{focus}.txt"
            np.savetxt(test_results_path, test_results, fmt='%d %.2f %.2f %d %d %.2f',
                       header='Enemy FocusFitness OverallFitness PlayerLife EnemyLife Time')
            print(f"Test results saved for Final Best Individual, Focus {focus} at {test_results_path}")

        print('\nAll tests completed.')

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An error occurred: {e}")

# Created/Modified files during execution:
print("\nFiles created during testing:")
print("test_results_final_best_focus_[FocusType].txt")