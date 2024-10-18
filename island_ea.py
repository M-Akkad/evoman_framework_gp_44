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

# Disable visuals for faster training
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'island_ea_task2_enemies_group_one'
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

# Genetic algorithm parameters
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
dom_u = 1
dom_l = -1

# Fixed hyperparameters
population_size = 68  # Total population size
n_parents = 2  # Number of parents for crossover
elitism_rate = 0.1  # Proportion of top individuals to carry over
tournament_k = 3  # Tournament size for selection
max_generations = 100

# Dynamic mutation parameters
initial_mutation_rate = 0.1560  # Starting mutation rate
final_mutation_rate = 0.05  # Minimum mutation rate at the end
initial_mutation_weight = 0.5  # Starting mutation weight
final_mutation_weight = 0.1  # Minimum mutation weight at the end

# Island model parameters
num_islands = 4  # Number of islands
migration_interval = 10  # Generations between migrations
migration_rate = 0.1  # Proportion of individuals to migrate

# Number of runs
num_runs = 10

# Run mode: 'train' or 'test'
run_mode = 'train'  # Change to 'test' as needed


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
        # Combine fitnesses
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


def evolve_population(population, fitness_scores, mutation_rate, mutation_weight, k, n_parents, elite_individuals):
    offspring = []
    num_offspring = len(population)
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


def migrate(islands, migration_rate):
    num_migrants = int(migration_rate * len(islands[0]))
    migrants = []
    # Collect migrants from each island
    for island in islands:
        np.random.shuffle(island)
        migrants.append(island[:num_migrants])
    # Send migrants to the next island
    for i in range(len(islands)):
        source_island = migrants[i]
        target_island_idx = (i + 1) % len(islands)
        islands[target_island_idx].extend(source_island)
        # Remove migrants from source island
        islands[i] = islands[i][num_migrants:]
    return islands


# Functions to adjust mutation rate and weight dynamically
def adjust_mutation_rate(generation, max_generations, initial_rate, final_rate):
    rate = initial_rate - ((initial_rate - final_rate) * (generation / max_generations))
    return max(rate, final_rate)


def adjust_mutation_weight(generation, max_generations, initial_weight, final_weight):
    weight = initial_weight - ((initial_weight - final_weight) * (generation / max_generations))
    return max(weight, final_weight)


if run_mode == 'train':

    for run in range(1, num_runs + 1):
        print(f"\nStarting Run {run}...\n")

        # Save hyperparameters to a config file
        hyperparameters = {
            'initial_mutation_rate': initial_mutation_rate,
            'final_mutation_rate': final_mutation_rate,
            'initial_mutation_weight': initial_mutation_weight,
            'final_mutation_weight': final_mutation_weight,
            'elitism_rate': elitism_rate,
            'tournament_k': tournament_k,
            'num_islands': num_islands,
            'migration_interval': migration_interval,
            'migration_rate': migration_rate,
            # Fixed hyperparameters
            'population_size': population_size,
            'n_parents': n_parents,
            'max_generations': max_generations
        }

        # Define the directory for the current run
        experiment_name_run = f'{experiment_name}_run_{run}'

        # Create the directory if it does not exist
        if not os.path.exists(experiment_name_run):
            os.makedirs(experiment_name_run)

        # Save hyperparameters to a text file in the run directory
        with open(f"{experiment_name_run}/hyperparameters.txt", 'w') as hp_file:
            for key, value in hyperparameters.items():
                hp_file.write(f"{key}: {value}\n")

        # Initialize log file for best individual stats in the run directory
        best_individuals_log = open(f"{experiment_name_run}/best_individuals_per_generation.txt", 'w')

        overall_best_individual = None
        overall_best_fitness = -float('inf')
        overall_best_per_enemy_fitness = None

        avg_fitness_per_generation = []
        avg_per_enemy_fitness_per_generation = []

        # Initialize islands
        islands = []
        for i in range(num_islands):
            island_population = [np.random.uniform(dom_l, dom_u, n_vars) for _ in range(population_size)]
            islands.append(island_population)

        for generation in range(max_generations):
            print(f"\nGeneration {generation} for Run {run}")

            # Adjust mutation rate and weight dynamically
            current_mutation_rate = adjust_mutation_rate(generation, max_generations, initial_mutation_rate,
                                                         final_mutation_rate)
            current_mutation_weight = adjust_mutation_weight(generation, max_generations, initial_mutation_weight,
                                                             final_mutation_weight)

            total_fitness_scores = []
            total_per_enemy_fitness = []

            # Evolve each island
            for idx, island_population in enumerate(islands):
                print(f"  Evolving Island {idx + 1}")
                fitness_scores, per_enemy_fitness = evaluate_population(island_population)

                # Elitism
                elite_size = int(elitism_rate * len(island_population))
                combined = list(zip(island_population, fitness_scores, per_enemy_fitness))
                combined.sort(key=lambda x: x[1], reverse=True)
                elite_individuals = [ind for ind, fit, per_enemy in combined[:elite_size]]

                # Evolve population with dynamic mutation parameters
                new_population = evolve_population(
                    population=island_population,
                    fitness_scores=fitness_scores,
                    mutation_rate=current_mutation_rate,
                    mutation_weight=current_mutation_weight,
                    k=tournament_k,
                    n_parents=n_parents,
                    elite_individuals=elite_individuals
                )
                islands[idx] = new_population

                # Update overall best individual
                best_idx = np.argmax(fitness_scores)
                best_individual = island_population[best_idx]
                best_fitness = fitness_scores[best_idx]
                best_per_enemy_fitness = per_enemy_fitness[best_idx]

                if best_fitness > overall_best_fitness:
                    overall_best_fitness = best_fitness
                    overall_best_individual = best_individual
                    overall_best_per_enemy_fitness = best_per_enemy_fitness

                    # Save the overall best individual so far in the run directory
                    np.savetxt(f"{experiment_name_run}/overall_best_individual.txt", overall_best_individual)
                    with open(f"{experiment_name_run}/overall_best_fitness.txt", 'w') as best_file:
                        best_file.write(f"Generation {generation}: Overall Best Fitness: {overall_best_fitness}\n")
                        best_file.write("Overall Best Per-Enemy Fitness:\n")
                        for enemy, fitness in zip(enemies, overall_best_per_enemy_fitness):
                            best_file.write(f"Enemy {enemy}: {fitness}\n")

                print(f"    Best Fitness on Island {idx + 1}: {best_fitness}")

                # Collect fitness scores for averaging
                total_fitness_scores.extend(fitness_scores)
                total_per_enemy_fitness.extend(per_enemy_fitness)

            # Migration
            if (generation + 1) % migration_interval == 0:
                print("  Migration occurring...")
                islands = migrate(islands, migration_rate)

            # Save average fitness across all islands in the run directory
            avg_fitness = np.mean(total_fitness_scores)
            avg_fitness_per_generation.append(avg_fitness)

            with open(f"{experiment_name_run}/average_fitness_per_generation.txt", 'a') as avg_file:
                avg_file.write(f"Generation {generation}: Average Fitness: {avg_fitness}\n")

            # Save per-enemy average fitness in the run directory
            avg_per_enemy_fitness = np.mean(total_per_enemy_fitness, axis=0)
            avg_per_enemy_fitness_per_generation.append(avg_per_enemy_fitness)
            with open(f"{experiment_name_run}/average_per_enemy_fitness_generation.txt", 'a') as avg_enemy_file:
                avg_enemy_file.write(f"Generation {generation}: Average Per-Enemy Fitness:\n")
                for enemy, fitness in zip(enemies, avg_per_enemy_fitness):
                    avg_enemy_file.write(f"Enemy {enemy}: {fitness}\n")

            # Log the best individual of this generation in the run directory
            best_individuals_log.write(
                f"Generation {generation}: Best Fitness: {overall_best_fitness}, Per-Enemy Fitness: {overall_best_per_enemy_fitness}\n")

            print(f"Overall Best Fitness so far: {overall_best_fitness}")
            print(f"Current Mutation Rate: {current_mutation_rate:.4f}, Mutation Weight: {current_mutation_weight:.4f}")

        # Close log file for best individuals
        best_individuals_log.close()

        # Save the final overall best individual in the run directory
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

        results_file = f"{experiment_name}/all_best_individuals_gain_results_for_island.txt"
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
