import os
import re
import pandas as pd
import torch
from collections import Counter
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from attention_model import EnhancedAttentionLayer
from torch_geometric.data import Data


class GNN(nn.Module):
    """
    GNN model for TSP:
      1. Uses three GCNConv layers to compute node embeddings.
      2. For each edge, concatenates the embeddings of its two endpoints to form a 2*node_hidden_dim vector.
      3. Pads the edge attribute from 1D to 4D.
      5. Concatenates the edge embedding with the scaled padded edge attribute and passes it through the enhanced attention layer.
      6. The attention score modulates the edge embedding.
      7. Concatenates the modulated edge embedding, original edge attribute (1D), and global features (α and μ) to form a 259-dimensional vector.
      8. Fully connected layers predict the edge occurrence count, which is normalized by μ to yield the final predicted edge probability.

      If debug=True, returns (predicted_prob, att_score).
    """

    def __init__(self, node_input_dim, node_hidden_dim, edge_input_dim, edge_hidden_dim, output_dim, global_input_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(node_input_dim, node_hidden_dim)  # e.g., 2 -> 128
        self.conv2 = GCNConv(node_hidden_dim, node_hidden_dim)  # 128 -> 128
        self.conv3 = GCNConv(node_hidden_dim, node_hidden_dim)  # 128 -> 128

        # Pad edge_attr from 1D to 4D, so attention input dimension = 2*node_hidden_dim + 4 = 256 + 4 = 260.
        self.attention = EnhancedAttentionLayer(2 * node_hidden_dim + 4, hidden_dim=64, num_heads=4, temperature=0.1)

        # Fully connected layers:
        # Concatenate: modulated edge embedding (256) + original edge_attr (1) + global features (2) = 259.
        self.fc1 = nn.Linear(259, edge_hidden_dim)  # e.g., 259 -> 128
        self.fc2 = nn.Linear(edge_hidden_dim, output_dim)  # 128 -> 1

    def forward(self, x, edge_index, edge_attr, alpha, mu, debug=False):
        # x: [num_nodes, node_input_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_input_dim] (1D: normalized distance)
        # alpha, mu: [1, global_input_dim]

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=1)  # [num_edges, 256]

        # Pad edge_attr from 1D to 4D.
        padded_edge_attr = F.pad(edge_attr, (0, 3), mode='constant', value=0)  # [num_edges, 4]

        scale_edge_factor = 20.0  # adjust
        scaled_padded_edge_attr = padded_edge_attr * scale_edge_factor

        # Concatenate edge embeddings and scaled padded edge attribute -> [num_edges, 260]
        att_input = torch.cat([edge_embeddings, scaled_padded_edge_attr], dim=1)
        # send alpha to attention
        att_score = self.attention(att_input, row, alpha)
        modulated_edge_embeddings = att_score * edge_embeddings  # [num_edges, 256]

        global_features = torch.cat([alpha, mu], dim=1).expand(edge_embeddings.size(0), -1)  # [num_edges, 2]
        combined = torch.cat([modulated_edge_embeddings, edge_attr, global_features], dim=1)  # [num_edges, 259]

        hidden = F.relu(self.fc1(combined))
        predicted_count = F.softplus(self.fc2(hidden))

        predicted_prob = predicted_count / mu[0, 0]
        predicted_prob = predicted_prob.clamp(0, 1)

        if debug:
            return predicted_prob, att_score
        else:
            return predicted_prob
# Preprocessing and Data Loading


def parse_ea_results(file_path):
    """
    Parse EA result file and extract routes, edge weights, and entropy values.
    :param file_path: Path to the EA result file.
    :return: edge_weights (dict), total_entropy (float)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    population = []
    total_entropy = None

    for line in lines:
        # Parse entropy value
        if line.startswith("Total Entropy H:"):
            try:
                total_entropy = float(line.split(":")[1].strip())
            except Exception as e:
                print(f"Error parsing entropy: {line}. Error: {e}")

        # Parse route
        if "Route:" in line:
            try:
                route_str = line.split("Route:")[1].strip()
                route = ast.literal_eval(route_str)
                population.append(route)
                print(f"Parsed Route: {route}")
            except Exception as e:
                print(f"Failed strict parsing for line: {line}. Error: {e}")
                # Fallback parsing
                try:
                    route_str = line.split("Route:")[1].strip().strip("[]")
                    route = [int(x) for x in route_str.split(",")]
                    population.append(route)
                    print(f"Fallback Parsed Route: {route}")
                except Exception as e:
                    print(f"Failed fallback parsing for line: {line}. Error: {e}")

    print("Parsed Population:", population)

    # Count the occurrence of edges
    edge_counts = Counter()
    total_routes = len(population)  # Total number of routes

    for route in population:
        for i in range(len(route)):
            # Sort edges to ensure no distinction between undirected edges
            edge = tuple(sorted((route[i], route[(i + 1) % len(route)])))  # Undirected edge
            edge_counts[edge] += 1

    print("Edge Counts:", edge_counts)

    # Calculate edge probabilities: occurrence count divided by total_routes (mu)
    edge_probs = {edge: count / total_routes for edge, count in edge_counts.items()}
    print("Edge Probabilities:", edge_probs)

    # Output edge occurrences and probabilities
    print("\nEdge Occurrences and Probabilities:")
    for edge, count in edge_counts.items():
        probability = edge_probs[edge]
        print(f"Edge {edge} -> Occurrences: {count}, Probability: {probability:.4f}")

    return edge_probs, total_entropy

def remove_self_loops(edge_index, edge_attr):
    """
    Remove self-loops from edges.
    :param edge_index: Edge index [2, num_edges]
    :param edge_attr: Edge features [num_edges, edge_input_dim]
    :return: edge_index and edge_attr with self-loops removed
    """
    mask = edge_index[0] != edge_index[1]  # Find all non-self-loop edges
    edge_index = edge_index[:, mask]  # Remove self-loop edges
    edge_attr = edge_attr[mask]  # Remove self-loop edge features
    return edge_index, edge_attr


def prepare_training_data(edge_weights, total_entropy, city_coordinates_file, output_file, alpha_value, mu_value,
                          max_alpha=1.0, max_mu=100.0, max_distance=2000.0):
    """
    Prepare the training data.
    1. Read city coordinates (without adding global information), resulting in a shape of [num_nodes, 2].
    2. For each edge, compute the Euclidean distance based on city coordinates, normalize it, and use it as edge_attr (1D).
    3. Store global parameters α and μ separately, without adding them to node features.
    4. Compute edge_prob as the true edge probability (edge occurrence count divided by the population size).
    """
    import torch
    import pandas as pd

    cities = pd.read_csv(city_coordinates_file)
    x = torch.tensor(cities[['X', 'Y']].values, dtype=torch.float)  # shape: [num_nodes, 2]

    # Normalize the city coordinates to [0,1]
    x_min, _ = torch.min(x, dim=0, keepdim=True)
    x_max, _ = torch.max(x, dim=0, keepdim=True)
    x = (x - x_min) / (x_max - x_min)

    edge_index = []
    edge_attr = []
    for edge, prob in edge_weights.items():
        edge_index.append(edge)

        i, j = edge[0] - 1, edge[1] - 1
        coord_i = x[i]
        coord_j = x[j]
        distance = torch.dist(coord_i, coord_j, 2)
        distance_normalized = distance

        scale_factor = 10000.0
        scaled_distance = distance_normalized * scale_factor

        edge_attr.append([distance_normalized.item()])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # shape: [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)  # shape: [num_edges, 1]

    assert edge_index.shape[1] == edge_attr.shape[0], "edge_index and edge_attr edge count mismatch."

    # calculate edge_prob: times/μ
    edge_prob = []
    for edge in edge_index.t().tolist():
        sorted_edge = tuple(sorted(edge))
        prob = edge_weights.get(sorted_edge, 0.0)
        edge_prob.append(prob)
    edge_prob = torch.tensor(edge_prob, dtype=torch.float).view(-1, 1)

    torch.save({
        'node_features': x,             # Node features: city coordinates [num_nodes, 2]
        'edge_index': edge_index,       # Edge index [2, num_edges]
        'edge_attr': edge_attr,         # Edge attributes: [normalized_distance], shape: [num_edges, 1]
        'edge_prob': edge_prob,         # True edge probability (labels)
        'total_entropy': torch.tensor([total_entropy]),
        'alpha': torch.tensor([alpha_value], dtype=torch.float).view(1, -1),  # global α
        'mu': torch.tensor([mu_value], dtype=torch.float).view(1, -1)         # global μ
    }, output_file)


def preprocess_population(folder, city_coordinates_file, output_folder, max_alpha=1.0, max_mu=100.0):
    """
    Preprocess the EA results and generate data files suitable for GNN training.
    :param folder: Path to the EA result folder
    :param city_coordinates_file: Path to the city coordinates CSV file
    :param output_folder: Folder for the processed data
    :param max_alpha: Maximum alpha value for normalization
    :param max_mu: Maximum mu value for normalization
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(folder):
        if "iter_10000" in file_name:  # Ensure it's an EA result file
            file_path = os.path.join(folder, file_name)
            print(f"Processing {file_name}...")

            # Extract alpha and mu values from the filename
            alpha_match = re.search(r'alpha_([\d.]+)', file_name)
            mu_match = re.search(r'mu_([\d.]+)', file_name)
            if alpha_match:
                alpha_value = float(alpha_match.group(1))
            else:
                print(f"Warning: Unable to extract alpha from filename: {file_name}. Using default alpha=0.1.")
                alpha_value = 0.1

            if mu_match:
                mu_value = float(mu_match.group(1))
            else:
                print(f"Warning: Unable to extract mu from filename: {file_name}. Using default mu=10.")
                mu_value = 10.0

            edge_weights, total_entropy = parse_ea_results(file_path)

            # Construct output file path
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_processed.pt")

            # Call prepare_training_data function
            prepare_training_data(edge_weights, total_entropy, city_coordinates_file, output_file, alpha_value,
                                  mu_value, max_alpha, max_mu)
            print(f"Processed {file_name} -> {output_file}")

def load_processed_data(processed_folder):
    pyg_data_list = []

    file_list = sorted([f for f in os.listdir(processed_folder) if f.endswith('.pt')])
    for filename in file_list:
        alpha_match = re.search(r'alpha_([\d.]+)', filename)
        mu_match = re.search(r'mu_([\d.]+)', filename)
        if alpha_match:
            alpha_value = float(alpha_match.group(1))
        else:
            print(f"Warning: Unable to extract alpha from filename: {filename}. Skipping this file.")
            continue

        if mu_match:
            mu_value = float(mu_match.group(1))
        else:
            print(f"Warning: Unable to extract mu from filename: {filename}. Skipping this file.")
            continue

        file_path = os.path.join(processed_folder, filename)
        try:
            data = torch.load(file_path)
        except Exception as e:
            print(f"Error: Failed to load {file_path}. Error: {e}")
            continue

        # Filter out edge indices that are out of node range
        edge_index = data['edge_index']
        num_nodes = data['node_features'].size(0)
        valid_mask = (edge_index < num_nodes).all(dim=0)
        valid_edges = edge_index[:, valid_mask]
        valid_edge_attr = data['edge_attr'][valid_mask]  # Edge attribute: [distance_normalized] (1-dim)
        valid_edge_prob = data['edge_prob'][valid_mask]

        assert valid_edges.shape[1] == valid_edge_attr.shape[0], (
            f"Valid edges and valid edge attributes count mismatch: {valid_edges.shape[1]} vs {valid_edge_attr.shape[0]}"
        )

        pyg_data = Data(
            x=data['node_features'],  # Node features
            edge_index=valid_edges,
            edge_attr=valid_edge_attr,  # Edge attribute: [distance_normalized]
            edge_prob=valid_edge_prob,
            y=data['total_entropy'],
            alpha=data['alpha'],
            mu=data['mu']
        )
        pyg_data_list.append(pyg_data)

    if len(pyg_data_list) == 0:
        raise ValueError("No valid .pt files found in the specified folder.")

    return pyg_data_list



if __name__ == "__main__":
    # Example usage:
    # python GNN.py preprocess <input_folder> <city_coordinates_file> <output_folder>
    import sys

    if len(sys.argv) != 5:
        print("Usage: python GNN.py preprocess <population_snapshots> <berlin52_city_coordinates.csv> <process>")
    else:
        command = sys.argv[1]
        input_folder = sys.argv[2]
        city_coordinates_file = sys.argv[3]
        output_folder = sys.argv[4]
        if command == "preprocess":
            preprocess_population(input_folder, city_coordinates_file, output_folder)
        else:
            print("Unknown command. Use 'preprocess'.")
