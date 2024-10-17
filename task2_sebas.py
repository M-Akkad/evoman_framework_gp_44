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
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'EA_islands_with_advanced_features'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for multiple enemies.
# enemies = [1, 2, 3, 7]  # group 1
enemies = [4, 5, 6, 8] #group 2

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
G_blend = 5 # Generations to blend fitness scores


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
    return 0.45 * (100 - e) + 0.2 * p - 0.5 * np.log(t)


def fitness_enemy_life(f, p, e, t):
    return 1.8 * (100 - e) + 0.05 * p - 0.5 * np.log(t)


def fitness_time(f, p, e, t):
    return 0.45 * (100 - e) + 0.05 * p - 2 * np.log(t)


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

# Blended fitness function
def blended_fitness(old_fitness, new_fitness, generation, G_blend):
    blending_factor = max(0, 1 - generation / G_blend)  # Starts at 1, then gradually decreases to 0
    return blending_factor * old_fitness + (1 - blending_factor) * new_fitness

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
def evolve_island(population, fitness_scores, fitness_func, generation, num_offspring=50, mutation_rate=0.1, mutation_weight=0.1,
                  k=3, n_parents=2, G_blend=5):
    offspring = []
    selected_parents = select_parents(population, fitness_scores, k)
    for _ in range(int(num_offspring)):
        parent_indices = np.random.choice(len(selected_parents), n_parents, replace=False)
        parents = [selected_parents[i] for i in parent_indices]
        child = multi_parent_recombination(parents)
        mutated_child = mutate(child, mutation_rate, mutation_weight)
        offspring.append(mutated_child)

    new_population = population + offspring
    new_fitness_scores = evaluate_population_island(new_population, fitness_func)

    # Blend the fitness scores of migrated individuals over time
    blended_fitness_scores = [
        blended_fitness(old_fitness, new_fitness, generation, G_blend) 
        for old_fitness, new_fitness in zip(fitness_scores, new_fitness_scores)
    ]

    combined = list(zip(new_population, blended_fitness_scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    population = [ind for ind, fitness in combined[:npop]]
    fitness_scores = [fitness for ind, fitness in combined[:npop]]

    return population, fitness_scores



# Migration between islands
def migrate(islands, fitness_scores):
    main_island = islands[0]
    main_fitness = fitness_scores[0]

    for i in range(1, num_islands):
        # Migrate from focus island to main island
        emigrants_to_main = []
        previous_fitness_to_main = []  # To track fitness scores of emigrants
        for _ in range(migration_size):
            idx = np.random.randint(len(islands[i]))
            emigrants_to_main.append(islands[i].pop(idx))
            previous_fitness_to_main.append(fitness_scores[i].pop(idx))  # Track their old fitness

        main_island.extend(emigrants_to_main)
        main_fitness.extend(previous_fitness_to_main)  # Append old fitness scores instead of new evaluations

        # Migrate from main island to focus island
        emigrants_from_main = []
        previous_fitness_from_main = []
        for _ in range(migration_size):
            idx = np.random.randint(len(main_island))
            emigrants_from_main.append(main_island.pop(idx))
            previous_fitness_from_main.append(main_fitness.pop(idx))  # Track their fitness from the main island

        islands[i].extend(emigrants_from_main)
        fitness_scores[i].extend(previous_fitness_from_main)  # Append old fitness scores

    islands[0] = main_island
    fitness_scores[0] = main_fitness

    return islands, fitness_scores



# Main loop
if run_mode == 'train':
    num_runs = 10

    for run in range(1, num_runs + 1):
        print(f"\nStarting Run {run}...\n")

        experiment_name = f'EA1_gr2_run_num_{run}'
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


# Test the best solution for each focus (main, p, e, t)
elif run_mode == 'test':
    try:
        print('\nTESTING BEST INDIVIDUALS FOR EACH FITNESS FOCUS\n')
        experiment_name = "task2_mo_run_1"

        # Load the final overall best individuals for each focus
        focus_types = ['main', 'p', 'e', 't']
        best_individuals = {}

        for focus in focus_types:
            file_path = f"{experiment_name}/final_overall_best_{focus}.txt"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Final overall best individual file for {focus} focus not found.")
            best_individuals[focus] = np.loadtxt(file_path)

        # Initialize dictionary to store test results
        test_results = {focus: [] for focus in focus_types}

        # Test each best individual against all enemies
        for focus, individual in best_individuals.items():
            print(f"\nTesting best individual for {focus} focus")

            for enemy in enemies:
                update_parameter_silently(env, 'enemies', [enemy])

                # Perform multiple test runs for consistency
                for i in range(5):  # 5 test runs per enemy
                    f, p, e, t = env.play(pcont=individual)
                    gain = p - e

                    test_results[focus].append((enemy, f, p, e, t, gain))
                    print(
                        f"Enemy {enemy} - Test Run {i + 1}: Fitness: {f}, Player Life: {p}, Enemy Life: {e}, Time: {t}, Gain: {gain}")

        # Save test results for each focus
        for focus, results in test_results.items():
            test_results_path = f"{experiment_name}/test_results_final_best_{focus}.txt"
            np.savetxt(test_results_path, results, fmt='%d %.2f %d %d %.2f %.2f',
                       header='Enemy Fitness PlayerLife EnemyLife Time Gain')
            print(f"Test results saved for {focus} focus at {test_results_path}")

        # Calculate and print average gain for each focus
        print("\nAverage Gains:")
        for focus, results in test_results.items():
            gains = [r[5] for r in results]
            avg_gain = np.mean(gains)
            print(f"{focus.capitalize()} focus average gain: {avg_gain:.2f}")

        print('\nAll tests completed.')

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An error occurred: {e}")

# Created/Modified files during execution:
# print("\nFiles created during testing:")
# for focus in focus_types:
#     print(f"test_results_final_best_{focus}.txt")
