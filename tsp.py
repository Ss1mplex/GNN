import tsplib95
import random
import math
import os
from collections import Counter


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
        city2 = route[(i + 1) % len(route)]  # 环路
        total_distance += distance(city1, city2)
    return total_distance

def two_opt(route, alpha):
    n = len(route)
    max_swap_length = max(2, int(alpha * n))  # according α to control the length
    i = random.randint(0, n - max_swap_length - 1)
    j = random.randint(i + 2, min(i + max_swap_length, n - 1))
    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
    return new_route

# mutate
def mutate(route, alpha):
    mutated_route = two_opt(route, alpha)
    return mutated_route

# calculate H，k=2
def calculate_entropy(population):
    segment_counts = Counter()
    total_segments = 0
    for route in population:
        for i in range(len(route)):
            segment = (route[i], route[(i + 1) % len(route)])
            segment_counts[segment] += 1
            total_segments += 1
    entropy = 0.0
    for count in segment_counts.values():
        p = count / total_segments
        entropy -= p * math.log(p)
    return entropy


def calculate_entropy_without_individual(population, index):
    temp_population = population[:index] + population[index+1:]
    entropy = calculate_entropy(temp_population)
    return entropy


def calculate_contributions(population):
    total_entropy = calculate_entropy(population)
    contributions = []
    for i in range(len(population)):
        entropy_without_i = calculate_entropy_without_individual(population, i)
        contribution = total_entropy - entropy_without_i

        if contribution < 0 and abs(contribution) < 1e-6:
            contribution = 0.0
        contributions.append(contribution)
    return contributions


def initialize_population_from_optimal(mu, optimal_tour):
    population = [optimal_tour.copy() for _ in range(mu)]
    return population


def save_population_snapshot(population, mu, alpha, iteration):
    directory = "population_snapshots"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/population_mu_{mu}_alpha_{alpha}_iter_{iteration}.txt"
    with open(filename, 'w') as f:
        # calculate contribution
        contributions = calculate_contributions(population)
        for idx, route in enumerate(population):
            cost = calculate_route_length(route)
            contrib = contributions[idx]
            f.write(f"Individual {idx}:\n")
            f.write(f"  Route length: {cost}\n")
            f.write(f"  Contribution to Diversity: {contrib}\n")
            f.write(f"  Route: {route}\n\n")
        total_entropy = calculate_entropy(population)
        f.write(f"Total Entropy H: {total_entropy}\n")
    print(f"Saved population snapshot at iteration {iteration} to {filename}")


def evolutionary_algorithm(mu, alpha, iterations, snapshot_iterations):

    optimal_tour = read_optimal_tour('berlin52.opt.tour')

    population = initialize_population_from_optimal(mu, optimal_tour)

    for iteration in range(1, iterations + 1):
        # randomly chose one to mutate
        I = random.choice(population)
        I_prime = mutate(I.copy(), alpha)
        cost = calculate_route_length(I_prime)

        if cost <= (1 + alpha) * OPTIMAL_ROUTE_LENGTH and I_prime not in population:

            population.append(I_prime)
            # remove
            if len(population) > mu:
                # contribution
                contributions = calculate_contributions(population)

                min_contribution = min(contributions)
                index_to_remove = contributions.index(min_contribution)
                # remove
                population.pop(index_to_remove)
        else:

            pass


        if iteration in snapshot_iterations:
            save_population_snapshot(population, mu, alpha, iteration)

    return population

# 主程序
if __name__ == "__main__":
    mu_values = [10, 20, 100, 150]
    alpha_values = [0.02, 0.05, 0.1, 0.25, 0.75, 1, 2]
    iterations = 10000  # 总迭代次数
    snapshot_iterations = [500, 1000, 5000, 10000]

    for mu in mu_values:
        for alpha in alpha_values:
            print(f"Running evolutionary algorithm with mu = {mu}, alpha = {alpha}")
            try:
                final_population = evolutionary_algorithm(mu, alpha, iterations, snapshot_iterations)
                # output
                print("Final population individuals and their route lengths:")
                for idx, route in enumerate(final_population):
                    cost = calculate_route_length(route)
                    print(f"Individual {idx}: Route length = {cost}")
                # calculate H
                final_entropy = calculate_entropy(final_population)
                print(f"Final population entropy H: {final_entropy}")
                print("\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"Error occurred for mu = {mu}, alpha = {alpha}: {e}")
                print("\n" + "-"*50 + "\n")
