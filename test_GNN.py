import torch
from torch_geometric.data import DataLoader
from GNN import GNN, load_processed_data  # Import GNN model and data loading functions
import pandas as pd

# Test function
def test(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_ea_probs = []  # For storing EA probabilities
    alphas = []
    mus = []
    all_edge_indices = []
    path_lengths = []  # To store computed path lengths during testing

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index, edge_attr, edge_prob = data.x, data.edge_index, data.edge_attr, data.edge_prob

            alpha = data.alpha  # Global feature: alpha
            mu = data.mu  # Global feature: mu

            # Forward pass
            edge_logits = model(x, edge_index, edge_attr, alpha, mu)  # Pass alpha and mu to the model
            edge_probs = torch.sigmoid(edge_logits)  # Convert logits to probabilities

            all_predictions.append(edge_probs.cpu().numpy())
            all_ea_probs.append(edge_prob.cpu().numpy())  # Store EA probabilities
            alphas.append(alpha.cpu().numpy())
            mus.append(mu.cpu().numpy())
            all_edge_indices.append(edge_index.cpu().numpy())

            # Calculate predicted path length based on edge probabilities
            # We multiply each edge's length (edge_attr) by its predicted probability (edge_probs)
            predicted_path_length = torch.sum(edge_attr * edge_probs)  # Weighted sum of edge probabilities

            # Ensure the path length is in the range of the constraint
            predicted_path_length = torch.max(predicted_path_length, torch.tensor(7542).to(device))  # Path length should be >= 7542

            # If the predicted path length exceeds the constraint, we penalize the violation
            max_path_length = 7542 * (1 + alpha)
            predicted_path_length = torch.min(predicted_path_length, torch.tensor(max_path_length).to(device))  # Path length <= max_path_length

            path_lengths.append(predicted_path_length.cpu().numpy())

    return all_predictions, all_ea_probs, alphas, mus, all_edge_indices, path_lengths

def main_test():
    processed_folder = "process"
    model_path = "trained_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load processed data
    pyg_data_list = load_processed_data(processed_folder)
    test_loader = DataLoader(pyg_data_list, batch_size=1, shuffle=False)

    # Model configuration
    node_input_dim = 2  # X and Y coordinates as node features
    node_hidden_dim = 64  # GCN hidden dimension
    edge_input_dim = 2  # edge_attr: [alpha, mu]
    edge_hidden_dim = 64  # FC hidden layer dimension for edges
    output_dim = 1  # output dimension (probability for each edge)
    global_input_dim = 2  # alpha and mu as global input features

    # Initialize model
    model = GNN(node_input_dim, node_hidden_dim, edge_input_dim, edge_hidden_dim, output_dim, global_input_dim).to(device)

    # Load the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Test the model
    predictions, ea_probs, alphas, mus, all_edge_indices, path_lengths = test(model, test_loader, device)

    # Prepare results for saving
    results = []
    for i in range(len(predictions)):
        pred = predictions[i].flatten()
        ea_prob = ea_probs[i].flatten()
        alpha = alphas[i].flatten()[0]
        mu = mus[i].flatten()[0]
        edge_indices = all_edge_indices[i].T
        path_length = path_lengths[i][0]  # Assuming path_length is a single value per sample

        for edge_idx, prob, ea_p in zip(edge_indices, pred, ea_prob):
            edge = tuple(edge_idx.tolist())
            results.append({
                'Sample': i + 1,
                'Alpha': alpha,
                'Mu': mu,
                'Edge': edge,
                'EA_Prob': ea_p,  # EA algorithm probability
                'Predicted_Prob': prob,  # GNN predicted probability
                'Predicted_Path_Length': path_length  # Add predicted path length to results
            })

    # Save results to CSV
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(results)
    df.to_csv('predictions_comparison_with_path_length.csv', index=False)
    print("All predictions have been saved to 'predictions_comparison_with_path_length.csv'.")
    print("Example of all predictions:")
    print(df.head(100))  # Print first 100 rows for preview

if __name__ == "__main__":
    main_test()
