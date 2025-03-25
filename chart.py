import os
import re
import matplotlib.pyplot as plt


# Function to extract entropy and iteration from each snapshot file
def extract_entropy(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        match_entropy = re.search(r'Total Entropy H: ([\d.]+)', content)
        match_iter = re.search(r'iter_(\d+)', file_path)

        if match_entropy and match_iter:
            entropy = float(match_entropy.group(1))
            iteration = int(match_iter.group(1))
            return iteration, entropy
    return None, None


# Folder where the snapshot files are located
snapshot_folder = "population_snapshots"
mu_values = [10, 20, 100, 150]
alpha_values = [0.02, 0.05, 0.1, 0.25, 0.75, 1, 2]

# Dictionary to store entropy values for each (mu, alpha) combination
entropy_data = {}

for mu in mu_values:
    for alpha in alpha_values:
        entropy_data[(mu, alpha)] = []

        # Read all snapshots for the current (mu, alpha) setting
        for filename in sorted(os.listdir(snapshot_folder)):
            if f"mu_{mu}_alpha_{alpha}" in filename:
                file_path = os.path.join(snapshot_folder, filename)
                iteration, entropy = extract_entropy(file_path)

                if iteration is not None and entropy is not None:
                    entropy_data[(mu, alpha)].append((iteration, entropy))

# Plot entropy changes for each (mu, alpha) combination
plt.figure(figsize=(14, 10))
for (mu, alpha), values in entropy_data.items():
    if values:
        iterations, entropies = zip(*sorted(values))
        plt.plot(iterations, entropies, label=f"μ={mu}, α={alpha}")

plt.xlabel("Iteration")
plt.ylabel("Entropy (H)")
plt.title("Entropy Changes Across Iterations for Different μ and α")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
