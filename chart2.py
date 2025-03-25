import os
import re
import matplotlib.pyplot as plt

# 文件夹路径
snapshot_dir = "population_snapshots"

# 正则表达式从文件名中提取 μ、α 和迭代次数
pattern = re.compile(r"population_mu_(\d+)_alpha_(\d+\.\d+)_iter_(\d+)\.txt")

# 存储结果的字典
data = {}

# 遍历文件夹中的快照文件
for filename in os.listdir(snapshot_dir):
    match = pattern.match(filename)
    if match:
        mu = int(match.group(1))
        alpha = float(match.group(2))
        iteration = int(match.group(3))

        # 读取文件内容
        entropy = None
        route_length_sum = 0
        route_count = 0
        with open(os.path.join(snapshot_dir, filename), 'r') as file:
            for line in file:
                if line.startswith("Total Entropy H"):
                    entropy = float(line.split(":")[1].strip())
                elif line.startswith("  Route length"):
                    route_length_sum += int(line.split(":")[1].strip())
                    route_count += 1

        # 平均路径长度
        avg_route_length = route_length_sum / route_count if route_count > 0 else None

        # 存储结果
        if (mu, alpha) not in data:
            data[(mu, alpha)] = {'iterations': [], 'entropies': [], 'avg_route_lengths': []}

        data[(mu, alpha)]['iterations'].append(iteration)
        data[(mu, alpha)]['entropies'].append(entropy)
        data[(mu, alpha)]['avg_route_lengths'].append(avg_route_length)

# 绘制熵变化图和路径质量变化图
for (mu, alpha), results in data.items():
    plt.figure(figsize=(12, 5))

    # 熵变化图
    plt.subplot(1, 2, 1)
    plt.plot(results['iterations'], results['entropies'], marker='o', label=f"μ={mu}, α={alpha}")
    plt.xlabel("Iterations")
    plt.ylabel("Entropy")
    plt.title("Entropy over Iterations")
    plt.legend()

    # 路径质量变化图
    plt.subplot(1, 2, 2)
    plt.plot(results['iterations'], results['avg_route_lengths'], marker='o', label=f"μ={mu}, α={alpha}")
    plt.xlabel("Iterations")
    plt.ylabel("Average Route Length")
    plt.title("Average Route Length over Iterations")
    plt.legend()

    # 保存图像
    plt.suptitle(f"Results for μ={mu}, α={alpha}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"results_mu_{mu}_alpha_{alpha}.png")
    plt.show()
