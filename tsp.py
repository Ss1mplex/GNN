import copy
import numpy as np
import tsplib95
import random
import math
import os
from collections import Counter

# Load the problem
problem = tsplib95.load('berlin52.tsp')
cities = list(problem.get_nodes())
distance = problem.get_weight
OPTIMAL_ROUTE_LENGTH = 7542



def read_optimal_tour(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    tour = []
    reading = False
    for line in lines:
        if 'TOUR_SECTION' in line:
            reading = True
            continue
        if 'EOF' in line:
            break
        if reading:
            city = int(line.strip())
            if city != -1:
                tour.append(city)
    return tour


def calculate_route_length(route):
    total_distance = 0
    for i in range(len(route)):
        city1 = route[i]
        city2 = route[(i + 1) % len(route)]
        total_distance += distance(city1, city2)
    return total_distance



def two_opt(route):
    n = len(route)
    if n < 2:
        return route

    ind = sorted(random.sample(range(n), 2))
    new_route = route[:ind[0]] + route[ind[0]:ind[1]][::-1] + route[ind[1]:]
    return new_route



def three_opt(route):
    n = len(route)
    if n < 3:
        return route


    ind = sorted(random.sample(range(n), 3))

    part1 = route[:ind[0]]
    part2 = route[ind[0]:ind[1]][::-1]
    part3 = route[ind[1]:ind[2]][::-1]
    part4 = route[ind[2]:]
    new_route = part1 + part2 + part3 + part4
    return new_route



def mutate(route, alpha, method="2-opt"):
    if method == "3-opt":
        return three_opt(route)
    return two_opt(route)


# Entropy calculation
def calculate_entropy(population):
    segment_counts = Counter()
    total_segments = 0
    for route in population:
        for i in range(len(route)):
            segment = (min(route[i], route[(i + 1) % len(route)]),
                       max(route[i], route[(i + 1) % len(route)]))
            segment_counts[segment] += 1
            total_segments += 1
    entropy = 0.0
    for count in segment_counts.values():
        p = count / total_segments
        entropy -= p * math.log(p)
    return entropy


def calculate_entropy_without_individual(population, index):
    temp_population = population[:index] + population[index + 1:]
    return calculate_entropy(temp_population)


def calculate_contributions(population):
    total_entropy = calculate_entropy(population)
    contributions = []
    for i in range(len(population)):
        entropy_without_i = calculate_entropy_without_individual(population, i)
        contribution = max(0, total_entropy - entropy_without_i)
        contributions.append(contribution)
    return contributions


# Initialize population
def initialize_population_from_optimal(mu, optimal_tour):
    return [optimal_tour.copy() for _ in range(mu)]


# Save snapshot function modified for separate folders
def save_population_snapshot(population, mu, alpha, iteration, mutation_method):

    directory = f"population_snapshots_{mutation_method}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/population_mu_{mu}_alpha_{alpha}_iter_{iteration}.txt"
    with open(filename, 'w') as f:
        contributions = calculate_contributions(population)
        for idx, route in enumerate(population):
            cost = calculate_route_length(route)
            contrib = contributions[idx]
            # Write full route details
            f.write(f"Individual {idx}:\n")
            f.write(f"  Route length: {cost}\n")
            f.write(f"  Contribution to Diversity: {contrib}\n")
            f.write(f"  Route: {route}\n\n")
        total_entropy = calculate_entropy(population)
        f.write(f"Total Entropy H: {total_entropy}\n")
    print(f"Saved population snapshot at iteration {iteration} to {filename}")


# Evolutionary algorithm
def evolutionary_algorithm(mu, alpha, iterations, snapshot_iterations, mutation_method="2-opt"):
    optimal_tour = read_optimal_tour('berlin52.opt.tour')
    population = initialize_population_from_optimal(mu, optimal_tour)

    for iteration in range(1, iterations + 1):
        I = random.choice(population)
        I_prime = mutate(I.copy(), alpha, method=mutation_method)
        cost = calculate_route_length(I_prime)

        if cost <= (1 + alpha) * OPTIMAL_ROUTE_LENGTH and I_prime not in population:
            population.append(I_prime)
            if len(population) > mu:
                contributions = calculate_contributions(population)
                min_contribution = min(contributions)
                index_to_remove = contributions.index(min_contribution)
                population.pop(index_to_remove)

        if iteration in snapshot_iterations:
            save_population_snapshot(population, mu, alpha, iteration, mutation_method)

    return population



if __name__ == "__main__":
    mu_values = [10, 20, 100, 150]
    alpha_values = [0.02, 0.05, 0.1, 0.25, 0.75, 1, 2]
    iterations = 10000
    snapshot_iterations = [1, 2, 10, 500, 1000, 2000, 5000, 6000, 7000, 8000, 9000, 10000]

    for mu in mu_values:
        for alpha in alpha_values:

            print(f"Running evolutionary algorithm with mu = {mu}, alpha = {alpha}, mutation = 3-opt")
            try:
                final_population = evolutionary_algorithm(mu, alpha, iterations, snapshot_iterations,
                                                          mutation_method="3-opt")
                print("Final population individuals and their route lengths:")
                for idx, route in enumerate(final_population):
                    cost = calculate_route_length(route)
                    print(f"Individual {idx}: Route length = {cost}")
                final_entropy = calculate_entropy(final_population)
                print(f"Final population entropy H: {final_entropy}")
                print("\n" + "-" * 50 + "\n")
            except Exception as e:
                print(f"Error occurred for mu = {mu}, alpha = {alpha}, mutation = 3-opt: {e}")
                print("\n" + "-" * 50 + "\n")
