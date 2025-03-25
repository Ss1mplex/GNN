import torch
import os


def inspect_pt_files(folder_path):
    """
    遍历文件夹内的所有 .pt 文件，打印其中的 edge weights (边概率)。
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            print(f"\nInspecting file: {filename}")
            file_path = os.path.join(folder_path, filename)

            try:
                data = torch.load(file_path)

                # 假设边的概率保存在 edge_attr 字段
                if 'edge_attr' in data:
                    edge_attr = data['edge_attr']
                    print(f"Edge Weights (Probabilities) Shape: {edge_attr.shape}")
                    print(f"Edge Weights (First 10): {edge_attr[:10]}")  # 打印前10个边的权重
                    print(f"Sum of Edge Weights: {edge_attr.sum().item()}")
                else:
                    print("Warning: No 'edge_attr' field found in this file.")

            except Exception as e:
                print(f"Error loading file {filename}: {e}")


# 设置预处理文件夹路径
processed_folder = "process"  # 修改为你的实际文件夹路径
inspect_pt_files(processed_folder)
