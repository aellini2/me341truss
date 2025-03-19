import numpy as np
import pandas as pd
import random
from scipy.stats import qmc
import matplotlib.pyplot as plt
import seaborn as sns
from doe.algorithms import parallel_tempering_sa, my_obj, my_is_feasible, my_nonlincon # Replace your_module

# 1. Define parameter ranges
param_ranges = {
    "cooling_rate": (0.8, 0.999),
    "t0": (1e3, 1e8),
    "swap_interval": (50, 500),
    "adapt_interval": (25, 200),
    "adapt_factor": (1.1, 2.0),
    "step_size_init": (0.01, 0.1),
    "min_temp": (1e-5, 1e-1)
}

# 2. Generate LHS design using scipy.stats.qmc
n_samples = 25  # Number of runs
sampler = qmc.LatinHypercube(d=len(param_ranges), seed=42)
lhs_samples = sampler.random(n_samples)

# Scale LHS samples to parameter ranges
doe_matrix = pd.DataFrame(lhs_samples, columns=param_ranges.keys())
for param, (low, high) in param_ranges.items():
    doe_matrix[param] = doe_matrix[param] * (high - low) + low

# 3. Run PTSA algorithm for each row
results = []
my_bounds = [(0.01, 0.5), (0.01, 0.5)]
for _, row in doe_matrix.iterrows():
    print(f"iteration {_}")
    config = {
        "x0_list": [[0.4, 0.4], [0.45, 0.15], [0.05, 0.35]],  # Example initial guesses
        "T0_list": [row["t0"], row["t0"] / 2, row["t0"] / 4],
        "bounds": my_bounds,
        "max_iter": 20000,
        "swap_interval": int(row["swap_interval"]),
        "adapt_interval": int(row["adapt_interval"]),
        "step_size_init": row["step_size_init"],
        "adapt_factor": row["adapt_factor"],
        "no_improve_limit": 1000,
        "min_temp": row["min_temp"],
        "cooling_rate": row["cooling_rate"]
        }
    best_r, best_E, logs = parallel_tempering_sa(config)
    results.append(my_obj(best_r))

doe_matrix["objective_weight"] = results

# Print the results (you can save to CSV or perform further analysis)
print(doe_matrix)

df = pd.DataFrame(doe_matrix)

# 1. Scatter Plots
features = ['cooling_rate', 't0', 'swap_interval', 'adapt_interval', 'adapt_factor']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    plt.scatter(df[feature], df['objective_weight'])
    plt.xlabel(feature)
    plt.ylabel('Objective Weight')
    plt.title(f'{feature} vs. Objective Weight')
plt.tight_layout()
plt.show()

# 2. Correlation Matrix Heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# 3. Histograms of Objective Weight
plt.figure(figsize=(8, 5))
plt.hist(df['objective_weight'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Objective Weight')
plt.ylabel('Frequency')
plt.title('Distribution of Objective Weight')
plt.show()

# 4. Histograms of Input Variables
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    plt.hist(df[feature], bins=10, color='lightgreen', edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()