import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import pandas as pd
from GNN import GNN, load_processed_data
from torch_geometric.data import Data  # Ensure Data is imported


def compute_loss(predicted_prob, edge_prob, att_score, lambda_aux=0.1, att_threshold=0.1, lambda_att=0.1):
    """
    Total loss = MSELoss(predicted_prob, edge_prob)
                 + lambda_aux * (var(predicted_prob) - var(edge_prob))^2
                 + lambda_att * (max(0, att_threshold - var(att_score)))^2


    """
    mse_loss = torch.nn.MSELoss()(predicted_prob, edge_prob)
    pred_var = predicted_prob.var()
    true_var = edge_prob.var()
    aux_loss_prob = (pred_var - true_var) ** 2

    att_var = att_score.var()

    aux_loss_att = torch.relu(att_threshold - att_var) ** 2

    total_loss = mse_loss + lambda_aux * aux_loss_prob + lambda_att * aux_loss_att
    return total_loss, mse_loss, aux_loss_prob, aux_loss_att


def train(model, train_loader, optimizer, device, lambda_aux=0.1, att_threshold=0.1, lambda_att=0.1):
    model.train()
    total_loss = 0
    total_mse = 0
    total_aux_prob = 0
    total_aux_att = 0
    for data in train_loader:
        data = data.to(device)
        x, edge_index, edge_attr, edge_prob = data.x, data.edge_index, data.edge_attr, data.edge_prob
        alpha = data.alpha
        mu = data.mu


        predicted_prob, att_score = model(x, edge_index, edge_attr, alpha, mu, debug=True)
        loss, mse_loss, aux_loss_prob, aux_loss_att = compute_loss(
            predicted_prob, edge_prob, att_score, lambda_aux=lambda_aux,
            att_threshold=att_threshold, lambda_att=lambda_att)
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_aux_prob += aux_loss_prob.item()
        total_aux_att += aux_loss_att.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    n = len(train_loader)
    print(f"Train MSE Loss: {total_mse/n:.4f}, Aux Prob Loss: {total_aux_prob/n:.4f}, Aux Att Loss: {total_aux_att/n:.4f}")
    return total_loss / n


def test(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    results = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            x, edge_index, edge_attr, edge_prob = data.x, data.edge_index, data.edge_attr, data.edge_prob
            alpha = data.alpha
            mu = data.mu
            predicted_prob, att_score = model(x, edge_index, edge_attr, alpha, mu, debug=True)

            loss = criterion(predicted_prob, edge_prob)
            total_loss += loss.item()

            row, col = edge_index
            for j in range(predicted_prob.size(0)):
                results.append({
                    'Sample': i + 1,
                    'Edge': (int(row[j].item()), int(col[j].item())),
                    'EA_Prob': edge_prob[j].item(),
                    'Predicted_Prob': predicted_prob[j].item()
                })
            if i == 0:
                print("Attention scores for first 5 edges:", att_score[:5].detach().cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    return results, avg_loss


def print_tensor_stats(name, tensor):
    print(f"{name} statistics:")
    print("  Mean:", tensor.mean().item())
    print("  Variance:", tensor.var().item())
    print("  Min:", tensor.min().item())
    print("  Max:", tensor.max().item())
    print("=" * 40)


def print_all_samples_stats(all_data):
    for i, data in enumerate(all_data):
        print(f"Sample {i + 1}:")
        print_tensor_stats("Node features x", data.x)
        print_tensor_stats("Edge attributes", data.edge_attr)
        print_tensor_stats("Edge probability labels", data.edge_prob)


def main_train():
    processed_folder = "process"
    model_path = "trained_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load processed data
    all_data = load_processed_data(processed_folder)
    print_all_samples_stats(all_data)

    # Split data: first 14 for training, remaining for testing.
    train_data = all_data[:14]
    test_data = all_data[14:]
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Model parameters
    node_input_dim = 2  # City coordinates [X, Y]
    node_hidden_dim = 128
    edge_input_dim = 1  # Edge attribute: normalized distance
    edge_hidden_dim = 128
    output_dim = 1  # Predicted edge occurrence count (to be divided by μ)
    global_input_dim = 2  # Global parameters: α and μ

    model = GNN(node_input_dim, node_hidden_dim, edge_input_dim, edge_hidden_dim, output_dim, global_input_dim).to(
        device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Use MSELoss for evaluation.
    criterion = torch.nn.MSELoss()

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device, lambda_aux=0.1)
        _, test_loss = test(model, test_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Total Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    torch.save(model.state_dict(), model_path)
    print("Model training completed and saved to 'trained_model.pth'")

    test_results, _ = test(model, test_loader, criterion, device)
    df = pd.DataFrame(test_results)
    pd.set_option('display.max_rows', None)
    print("Predicted results (first 100 rows):")
    print(df.head(100))
    df.to_csv('predictions_table.csv', index=False)
    print("Prediction results saved to 'predictions_table.csv'")


if __name__ == "__main__":
    main_train()
