import torch
from torch_geometric.data import Data
import tsplib95


def load_tsp_to_pyg(filename):
    problem = tsplib95.load(filename)
    nodes = list(problem.get_nodes())
    coords = torch.tensor([problem.node_coords[i] for i in nodes], dtype=torch.float)

    # Create edges and calculate distances
    edge_index = []
    edge_attr = []
    for i in nodes:
        for j in nodes:
            if i != j:
                edge_index.append([i - 1, j - 1])  # Adjust index to start from 0
                edge_attr.append(problem.get_weight(i, j))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)  # Single feature: distance

    # Create graph data
    data = Data(x=coords, edge_index=edge_index, edge_attr=edge_attr)
    return data


# Example usage
graph_data = load_tsp_to_pyg('berlin52.tsp')
print(graph_data)
