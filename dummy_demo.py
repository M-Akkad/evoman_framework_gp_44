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
from math import fabs,sqrt
import glob, os

# ATTENTION: To train change headless to true, visuals(within env) to false and run_mode to train job

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'team1_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10



# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  randomini="yes",
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

run_mode = 'test' # train or test
dom_u = 1
dom_l = -1
npop = 100
gens = 50
mutation_rate = 0.1
mutation_weight = 0.3
n_parents = 2
k = 3 # Tournament size
num_offspring = 50
last_best = 0


# Evaluate fitness
def evaluate_population(population):
    fitness_scores = []
    for individual in population:
        f, p, e, t = env.play(pcont=individual)
        fitness_scores.append(f)
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



# Evolution loop
def evolve_population(population, fitness_scores, num_offspring=50, mutation_rate=0.1, mutation_weight=0.1, k=3, n_parents=2):
    # Generate offspring using multi-parent recombination and mutation
    offspring = []
    selected_parents = select_parents(population, fitness_scores, k)
    for _ in range(num_offspring):
        # Randomly select n_parents without replacement
        parent_indices = np.random.choice(len(selected_parents), n_parents, replace=False)
        
        # Collect the selected parents based on the random indices
        parents = [selected_parents[i] for i in parent_indices]

        child = multi_parent_recombination(parents)
        mutated_child = mutate(child, mutation_rate, mutation_weight)
        offspring.append(mutated_child)
    
    # Combine population with offspring and re-evaluate
    new_population = population + offspring
    new_fitness_scores = evaluate_population(new_population)
    
    # Select the top `npop` individuals for the next generation
    combined = list(zip(new_population, new_fitness_scores))
    combined.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (higher is better)
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
        experiment_name = f'team1_test_run_enemy3{run}'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        
        # Initialize environment for each run
        env = Environment(experiment_name=experiment_name,
                        enemies=[3],
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)

        env.state_to_log()  # Log environment state

        # Initialize the genetic algorithm variables
        population = [np.random.uniform(dom_l, dom_u, n_vars) for _ in range(npop)]
        fitness_scores = evaluate_population(population)
        overall_best_individual = None
        overall_best_fitness = -float('inf')

        # Save results for the first population
        with open(experiment_name + '/results.txt', 'a') as file_aux:
            file_aux.write('\n\ngen best mean std')
            print(f'\n GENERATION 0 {round(fitness_scores[np.argmax(fitness_scores)], 6)} {round(np.mean(fitness_scores), 6)} {round(np.std(fitness_scores), 6)}')
            file_aux.write(f'\n0 {round(fitness_scores[np.argmax(fitness_scores)], 6)} {round(np.mean(fitness_scores), 6)} {round(np.std(fitness_scores), 6)}')

        # Evolutionary process
        best_old = 0
        for generation in range(gens):
            print(f"\nEvolving Generation {generation} for Run {run}")
            population, fitness_scores = evolve_population(population, fitness_scores, num_offspring, mutation_rate, mutation_weight, k, n_parents)

            # Update best, mean, std after evolving
            best = np.argmax(fitness_scores)
            mean = np.mean(fitness_scores)
            std = np.std(fitness_scores)

            # Track the overall best individual and fitness for this run
            if fitness_scores[best] > overall_best_fitness:
                overall_best_fitness = fitness_scores[best]
                overall_best_individual = population[best]
                np.savetxt(experiment_name + '/overall_best.txt', overall_best_individual)

            # Save the generation results
            with open(experiment_name + '/results.txt', 'a') as file_aux:
                print(f' GENERATION {generation} Best Fitness: {round(fitness_scores[best], 6)}, Mean Fitness: {round(mean, 6)}, Std: {round(std, 6)}')
                file_aux.write(f'\n{generation} {round(fitness_scores[best], 6)} {round(mean, 6)} {round(std, 6)}')

            # Save the best individual for this generation
            best_individual = population[best]
            np.save(f"{experiment_name}/best_individual_gen_{generation}.npy", best_individual)

            # Save the overall best individual so far
            if best_old < fitness_scores[best]:
                best_old = fitness_scores[best]
                np.savetxt(experiment_name + '/best.txt', best_individual)

            # Save the current state
            solutions = [population, fitness_scores]
            env.update_solutions(solutions)

            # Save the generation number
            with open(experiment_name + '/gen.txt', 'w') as file_aux:
                file_aux.write(str(generation))

        # After evolution is completed, save the overall best individual
        if overall_best_individual is not None:
            np.savetxt(f"{experiment_name}/final_overall_best.txt", overall_best_individual)
            print(f"Overall best fitness for Run {run}: {overall_best_fitness}")

        print(f"\nRun {run} completed.\n")

    print("\nAll 10 runs completed.")



# Test the best solution
elif run_mode == 'test':
   
    try:
        # Load the best solution from the file
        best_sol = np.loadtxt('/Users/s.broos/Documents/EVO/evoman_framework_gp_44/team1_test_run_enemy310/final_overall_best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')

        # Set the speed to normal for testing (you may adjust this)
        env.update_parameter('speed', 'fastest')
        for i in range(5):
            # Evaluate the best solution by passing it to the environment for testing
            f, p, e, t = env.play(pcont=best_sol)

            # Print the evaluation results (fitness, player life, enemy life, time taken)
            print(f"Fitness: {f}, Player Life: {p}, Enemy Life: {e}, Time: {t}, Gain: {p - e}")

        sys.exit(0)  # Exit after testing the best solution

    except Exception as e:
        print(f"Error loading best solution: {e}")
        sys.exit(1)