################################
# EvoMan Framework - V1.0 2016 #
# Author: Karine Miras          #
# karine.smiras@gmail.com       #
################################

import sys
from evoman.environment import Environment
from demo_controller import player_controller

import time
import numpy as np
import os
import contextlib
import matplotlib.pyplot as plt

# Disable visuals for faster training
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'extinction_event_ea_task2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# Specify multiple enemies to train on

enemies = [1, 2, 3, 7]  # group 1
# enemies = [4, 5, 6, 8]  # group 2

env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  randomini="no",
                  speed="fastest",
                  visuals=False)

env.state_to_log()

ini = time.time()

# Genetic algorithm parameters (default values)
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
dom_u = 1
dom_l = -1

# Fixed hyperparameters
initial_pop_size = 50
min_pop_size = 20
max_pop_size = 200
min_k = 2
max_k = 5
initial_mutation_weight = 0.5
final_mutation_weight = 0.1
disaster_exponent = 4
n_parents = 2  # Fixed number of parents for crossover
final_mutation_rate = 0.05  # Fixed final mutation rate
max_generations = 100  # Maximum number of generations

# Implemented hyperparameters
initial_mutation_rate = 0.1360
elitism_rate = 0.1131
growth_rate = 1.1362

# Number of runs
num_runs = 10

# Run mode: 'train' or 'test'
run_mode = 'train'


class NullOutput:
    def write(self, _):
        pass

    def flush(self):
        pass


def update_parameter_silently(env, param, value):
    with contextlib.redirect_stdout(NullOutput()), contextlib.redirect_stderr(NullOutput()):
        env.update_parameter(param, value)


def evaluate_population(population):
    fitness_scores = []
    per_enemy_fitness = []
    for individual in population:
        enemy_fitnesses = []
        for enemy in enemies:
            update_parameter_silently(env, 'enemies', [enemy])
            f, p, e_life, t = env.play(pcont=individual)
            enemy_fitnesses.append(f)
        # Store per-enemy fitnesses
        per_enemy_fitness.append(enemy_fitnesses)
        # Combine fitnesses as before
        total_fitness = sum(enemy_fitnesses)
        min_enemy_fitness = min(enemy_fitnesses)
        combined_fitness = (total_fitness / len(enemies)) + min_enemy_fitness
        fitness_scores.append(combined_fitness)
    return fitness_scores, per_enemy_fitness


def tournament_selection(population, fitness_scores, k):
    tournament_indices = np.random.choice(len(population), k, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index]


def uniform_crossover(parents):
    child = np.zeros(n_vars)
    for i in range(n_vars):
        selected_parent = np.random.choice(len(parents))
        child[i] = parents[selected_parent][i]
    return child


def mutate(child, mutation_rate, mutation_weight):
    for i in range(n_vars):
        if np.random.rand() < mutation_rate:
            child[i] += np.random.uniform(-mutation_weight, mutation_weight)
    return child


def calculate_disaster_probability(current_pop_size, min_pop_size, max_pop_size, exponent):
    normalized_pop = (current_pop_size - min_pop_size) / (max_pop_size - min_pop_size)
    normalized_pop = max(0, min(normalized_pop, 1))
    disaster_prob = normalized_pop ** exponent
    return disaster_prob


def adjust_selection_pressure(current_pop_size, min_pop_size, max_pop_size, min_k, max_k):
    normalized_pop = (current_pop_size - min_pop_size) / (max_pop_size - min_pop_size)
    normalized_pop = max(0, min(normalized_pop, 1))
    k = min_k + (max_k - min_k) * normalized_pop
    k = int(round(k))
    return k


def adjust_mutation_rate(generation, max_generations, initial_rate, final_rate):
    rate = initial_rate - ((initial_rate - final_rate) * (generation / max_generations))
    return max(rate, final_rate)


def adjust_mutation_weight(generation, max_generations, initial_weight, final_weight):
    weight = initial_weight - ((initial_weight - final_weight) * (generation / max_generations))
    return max(weight, final_weight)


def evolve_population(population, fitness_scores, num_offspring, mutation_rate, mutation_weight, k, n_parents,
                      elite_individuals):
    offspring = []
    for _ in range(num_offspring - len(elite_individuals)):
        parents = []
        for _ in range(n_parents):
            parent = tournament_selection(population, fitness_scores, k=k)
            parents.append(parent)
        child = uniform_crossover(parents)
        mutated_child = mutate(child, mutation_rate, mutation_weight)
        offspring.append(mutated_child)
    new_population = elite_individuals + offspring
    return new_population


if run_mode == 'train':

    for run in range(1, num_runs + 1):
        print(f"\nStarting Run {run}...\n")

        current_pop_size = initial_pop_size

        experiment_name_run = f'{experiment_name}_run_{run}'
        if not os.path.exists(experiment_name_run):
            os.makedirs(experiment_name_run)

        env = Environment(experiment_name=experiment_name_run,
                          enemies=enemies,
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)

        env.state_to_log()

        population = [np.random.uniform(dom_l, dom_u, n_vars) for _ in range(current_pop_size)]
        fitness_scores, per_enemy_fitness = evaluate_population(population)
        overall_best_individual = None
        overall_best_fitness = -float('inf')
        overall_best_per_enemy_fitness = None

        # Initialize log files for best individual stats and population size
        best_individuals_log = open(f"{experiment_name_run}/best_individuals_per_generation.txt", 'w')
        population_size_log = open(f"{experiment_name_run}/population_size_per_generation.txt", 'w')

        for generation in range(max_generations):
            print(f"\nGeneration {generation} for Run {run}")

            mutation_rate = adjust_mutation_rate(generation, max_generations, initial_mutation_rate,
                                                 final_mutation_rate)
            mutation_weight = adjust_mutation_weight(generation, max_generations, initial_mutation_weight,
                                                     final_mutation_weight)

            disaster_prob = calculate_disaster_probability(current_pop_size, min_pop_size, max_pop_size,
                                                           exponent=disaster_exponent)

            if np.random.rand() < disaster_prob:
                survivor_indices = np.random.choice(len(population), min_pop_size, replace=False)
                population = [population[i] for i in survivor_indices]
                fitness_scores = [fitness_scores[i] for i in survivor_indices]
                per_enemy_fitness = [per_enemy_fitness[i] for i in survivor_indices]
                current_pop_size = min_pop_size
                k = min_k
                population_size_log.write(
                    f"Generation {generation}: Extinction Event, Population Size: {current_pop_size}\n")
            else:
                current_pop_size = int(current_pop_size * growth_rate)
                if current_pop_size > max_pop_size:
                    current_pop_size = max_pop_size
                k = adjust_selection_pressure(current_pop_size, min_pop_size, max_pop_size, min_k, max_k)
                population_size_log.write(f"Generation {generation}: Population Size: {current_pop_size}\n")

            print(
                f"Population size: {current_pop_size}, Selection Pressure (k): {k}, Mutation Rate: {mutation_rate:.4f}, Mutation Weight: {mutation_weight:.4f}")

            elite_size = int(elitism_rate * len(population))
            combined = list(zip(population, fitness_scores, per_enemy_fitness))
            combined.sort(key=lambda x: x[1], reverse=True)
            elite_individuals = [ind for ind, fit, per_enemy in combined[:elite_size]]

            num_offspring = current_pop_size

            population = evolve_population(population, fitness_scores, num_offspring, mutation_rate, mutation_weight, k,
                                           n_parents, elite_individuals)
            fitness_scores, per_enemy_fitness = evaluate_population(population)

            best_idx = np.argmax(fitness_scores)
            best_individual = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            best_per_enemy_fitness = per_enemy_fitness[best_idx]

            # Log the best individual of this generation
            best_individuals_log.write(
                f"Generation {generation}: Best Fitness: {best_fitness}, Per-Enemy Fitness: {best_per_enemy_fitness}\n")

            if best_fitness > overall_best_fitness:
                overall_best_fitness = best_fitness
                overall_best_individual = best_individual
                overall_best_per_enemy_fitness = best_per_enemy_fitness

                # Save the overall best individual so far
                np.savetxt(f"{experiment_name_run}/overall_best_individual.txt", overall_best_individual)
                with open(f"{experiment_name_run}/overall_best_fitness.txt", 'w') as best_file:
                    best_file.write(f"Generation {generation}: Overall Best Fitness: {overall_best_fitness}\n")
                    best_file.write("Overall Best Per-Enemy Fitness:\n")
                    for enemy, fitness in zip(enemies, overall_best_per_enemy_fitness):
                        best_file.write(f"Enemy {enemy}: {fitness}\n")

            print(
                f"Best Fitness: {best_fitness}, Overall Best Fitness: {overall_best_fitness}")
            print("Best Individual Per-Enemy Fitness:")
            for enemy, fitness in zip(enemies, best_per_enemy_fitness):
                print(f"  Enemy {enemy}: {fitness}")

        # Close log files after training
        best_individuals_log.close()
        population_size_log.close()

        if overall_best_individual is not None:
            np.savetxt(f"{experiment_name_run}/final_overall_best.txt", overall_best_individual)
            with open(f"{experiment_name_run}/final_overall_best_fitness.txt", 'w') as final_best_file:
                final_best_file.write(f"Overall Best Fitness: {overall_best_fitness}\n")
                final_best_file.write("Overall Best Per-Enemy Fitness:\n")
                for enemy, fitness in zip(enemies, overall_best_per_enemy_fitness):
                    final_best_file.write(f"Enemy {enemy}: {fitness}\n")
            print(f"Overall best fitness for Run {run}: {overall_best_fitness}")
            print("Overall Best Individual Per-Enemy Fitness:")
            for enemy, fitness in zip(enemies, overall_best_per_enemy_fitness):
                print(f"  Enemy {enemy}: {fitness}")

        print(f"\nRun {run} completed in {max_generations} generations.\n")

    print(f"\nAll {num_runs} runs completed.")


elif run_mode == 'test':

    try:
        best_individuals = []

        # Iterate through each run
        for run in range(1, num_runs + 1):
            best_fitness_run = -float('inf')
            best_sol_run = None
            best_group_run = None

            for enemy_group in ['enemies_group_one', 'enemies_group_two']:
                run_fitness_file = f'{experiment_name}/{enemy_group}/{experiment_name}_run_{run}/final_overall_best_fitness.txt'
                if os.path.exists(run_fitness_file):
                    with open(run_fitness_file, 'r') as file:
                        first_line = file.readline()
                        try:
                            fitness_value = float(first_line.split(':')[-1].strip())
                        except ValueError:
                            continue
                        if fitness_value > best_fitness_run:
                            best_fitness_run = fitness_value
                            best_sol_run = np.loadtxt(
                                f'{experiment_name}/{enemy_group}/{experiment_name}_run_{run}/final_overall_best.txt')
                            best_group_run = enemy_group

            if best_sol_run is not None:
                best_individuals.append((best_group_run, run, best_fitness_run, best_sol_run))

        if len(best_individuals) == 0:
            raise ValueError("No valid best solutions found across runs.")

        results_file = f"{experiment_name}/all_best_individuals_gain_results_for_extinction.txt"
        with open(results_file, 'w') as res_file:
            res_file.write("Enemy, Run, Enemy_Group, Fitness, Player_Life, Enemy_Life, Time, Gain\n")

            for (enemy_group, run, fitness_value, best_sol) in best_individuals:
                print(f'\n RUNNING BEST INDIVIDUAL FROM {enemy_group} RUN {run} WITH FITNESS {fitness_value} \n')

                update_parameter_silently(env, 'speed', 'fastest')

                for enemy in range(1, 9):
                    update_parameter_silently(env, 'enemies', [enemy])
                    f, p, e, t = env.play(pcont=best_sol)
                    gain = p - e
                    print(
                        f"Enemy {enemy} - Run {run} from {enemy_group}: Fitness: {f}, Player Life: {p}, Enemy Life: {e}, Time: {t}, Gain: {gain}")

                    res_file.write(
                        f"Enemy {enemy}, Run {run}, Enemy Group {enemy_group}, Fitness: {f}, Player Life: {p}, Enemy Life: {e}, Time: {t}, Gain: {gain}\n")

        print(f"\nTesting completed. Results saved to {results_file}\n")
        sys.exit(0)

    except Exception as e:
        print(f"Error loading best solution: {e}")
        sys.exit(1)
