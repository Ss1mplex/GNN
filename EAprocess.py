import os
import re
import pandas as pd
import torch
from collections import Counter
import ast
from torch_geometric.data import Data

def parse_ea_results(file_path):
    """
    Parse EA results file to extract routes, edge weights, and entropy.
    :param file_path: Path to EA result file.
    :return: edge_weights (dict), total_entropy (float)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    population = []
    total_entropy = None

    for line in lines:
        if line.startswith("Total Entropy H:"):
            try:
                total_entropy = float(line.split(":")[1].strip())
            except Exception as e:
                print(f"Error parsing entropy: {line}. Error: {e}")

        if "Route:" in line:
            try:
                route_str = line.split("Route:")[1].strip()
                route = ast.literal_eval(route_str)
                population.append(route)
            except Exception as e:
                print(f"Error parsing route: {line}. Error: {e}")

    edge_counts = Counter()
    total_routes = len(population)  # Total number of routes

    for route in population:
        for i in range(len(route)):
            edge = tuple(sorted((route[i], route[(i + 1) % len(route)])))  # Sort edges to ensure undirected edges
            edge_counts[edge] += 1

    edge_probs = {edge: count / total_routes for edge, count in edge_counts.items()}

    return edge_probs, total_entropy

def calculate_edge_lengths(city_coordinates, edge_index):
    """
    Calculate edge lengths based on node coordinates.
    :param city_coordinates: Coordinates of cities [num_nodes, 2]
    :param edge_index: Edge index [2, num_edges]
    :return: Edge lengths [num_edges, 1]
    """
    node_a, node_b = edge_index
    coords_a = city_coordinates[node_a]
    coords_b = city_coordinates[node_b]
    edge_lengths = torch.sqrt(torch.sum((coords_a - coords_b) ** 2, dim=1)).view(-1, 1)  # Euclidean distance
    return edge_lengths

def prepare_training_data(edge_weights, total_entropy, city_coordinates_file, output_file, alpha_value, mu_value):
    cities = pd.read_csv(city_coordinates_file)
    x = torch.tensor(cities[['X', 'Y']].values, dtype=torch.float)

    # Create edge_index and edge_attr
    edge_index = []
    edge_attr = []
    for edge, prob in edge_weights.items():
        edge_index.append(edge)
        edge_attr.append([prob])  # Edge length will be computed separately

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)  # [num_edges, 1] (edge lengths)

    # Calculate edge lengths based on city coordinates
    edge_lengths = calculate_edge_lengths(x, edge_index)

    # Combine edge length with the existing edge attribute (probability)
    edge_attr = torch.cat([edge_attr, edge_lengths], dim=1)  # [num_edges, 2] (probability and length)

    # Global features (alpha, mu) - these will be shared across all edges in a sample
    global_features = torch.tensor([alpha_value, mu_value], dtype=torch.float).view(1, -1)

    # Save PyG data format
    torch.save({
        'node_features': x,  # Node features
        'edge_index': edge_index,
        'edge_attr': edge_attr,  # Edge features [probability, edge length]
        'global_features': global_features,  # Global features [alpha, mu]
        'total_entropy': torch.tensor([total_entropy]),
        'alpha': torch.tensor([alpha_value], dtype=torch.float).view(1, -1),
        'mu': torch.tensor([mu_value], dtype=torch.float).view(1, -1)
    }, output_file)

def remove_self_loops(edge_index, edge_attr):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    return edge_index, edge_attr


def preprocess_population(folder, city_coordinates_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(folder):
        if "iter_10000" in file_name:  # Ensure it is an EA result file
            file_path = os.path.join(folder, file_name)
            print(f"Processing {file_name}...")

            alpha_match = re.search(r'alpha_([\d.]+)', file_name)
            mu_match = re.search(r'mu_([\d.]+)', file_name)
            if alpha_match:
                alpha_value = float(alpha_match.group(1))
            else:
                alpha_value = 0.1  # Default

            if mu_match:
                mu_value = float(mu_match.group(1))
            else:
                mu_value = 10.0  # Default

            edge_weights, total_entropy = parse_ea_results(file_path)

            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_processed.pt")

            prepare_training_data(edge_weights, total_entropy, city_coordinates_file, output_file, alpha_value, mu_value)
