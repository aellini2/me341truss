import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                           PROBLEM DEFINITION                                #
###############################################################################
def my_obj(x):
    """
    f(x) = x1 + x2
    """
    return x[0] + x[1]

def my_nonlincon(x):
    """
    Returns g = [g1, g2, g3].
    If g[i] > 0 => constraint i is violated.
    
    g1(x) = 20 - x1^2*x2       < 0
    g2(x) = 1 - ((x1 + x2 -5)^2/30) - ((x1 - x2 -12)^2/120) < 0
    g3(x) = x1^2 + 8*x2 - 75  < 0
    """
    x1, x2 = x[0], x[1]
    g1 = 20 - x1**2 * x2
    g2 = 1 - ((x1 + x2 - 5)**2 / 30.0) - ((x1 - x2 - 12)**2 / 120.0)
    g3 = x1**2 + 8*x2 - 75

    return np.array([g1, g2, g3])

def my_is_feasible(x):
    """
    Check if all constraints are satisfied (<= 0).
    """
    g = my_nonlincon(x)
    return np.all(g <= 0)

###############################################################################
#                     PLOT FEASIBILITY REGION + PATH                          #
###############################################################################
def plot_feasibility_and_path_2d(bounds, chain_paths, best_history, N=200, title="Feasibility & Path"):
    """
    For a 2D problem x=(x1,x2), plots the feasibility region:
      - Feasible region in white, infeasible in yellow
      - The chain paths in different colors
      - The best-so-far path in green
    """
    # Build the grid
    x1_lin = np.linspace(bounds[0][0], bounds[0][1], N)
    x2_lin = np.linspace(bounds[1][0], bounds[1][1], N)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    feasible_mask = np.zeros_like(X1, dtype=bool)

    # Evaluate constraints
    for i in range(N):
        for j in range(N):
            xx = [X1[i,j], X2[i,j]]
            g = my_nonlincon(xx)
            feasible_mask[i,j] = np.all(g <= 0)

    # Plot
    fig, ax = plt.subplots(figsize=(7,6))
    ax.contourf(X1, X2, feasible_mask, levels=[-0.5,0.5], colors=["yellow","white"], alpha=0.6)

    # Plot each chain's path
    colors = ["blue", "red", "orange", "purple", "cyan", "magenta"]
    for c_idx, path_c in enumerate(chain_paths):
        path_c = np.array(path_c)
        col = colors[c_idx % len(colors)]
        ax.plot(path_c[:,0], path_c[:,1], '-o', color=col, label=f"Chain {c_idx}", markersize=3, alpha=0.8)

    # Plot best-so-far path in green
    best_hist = np.array(best_history)
    ax.plot(best_hist[:,0], best_hist[:,1], '-o', color='green', linewidth=2, markersize=4, label='Best so far')
    ax.plot(best_hist[-1,0], best_hist[-1,1], 'o', color='green', markersize=10, label='Final Best')

    ax.set_xlim([bounds[0][0], bounds[0][1]])
    ax.set_ylim([bounds[1][0], bounds[1][1]])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()

###############################################################################
#                           4-SUBPLOT VISUALIZATION                           #
###############################################################################
def plot_4_subplots(logs, best_history, bounds):
    """
    Creates a 4-subplot visualization:
    1. Path visualization (feasibility + best path)
    2. Cost (objective) over iterations
    3. Acceptance probability over iterations
    4. Temperature over iterations
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Path visualization
    # Replicate the feasibility plot, but only plot the best path
    x1_lin = np.linspace(bounds[0][0], bounds[0][1], 200)
    x2_lin = np.linspace(bounds[1][0], bounds[1][1], 200)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    feasible_mask = np.zeros_like(X1, dtype=bool)
    for i in range(200):
        for j in range(200):
            xx = [X1[i,j], X2[i,j]]
            g = my_nonlincon(xx)
            feasible_mask[i,j] = np.all(g <= 0)
    axs[0, 0].contourf(X1, X2, feasible_mask, levels=[-0.5, 0.5], colors=["yellow", "white"], alpha=0.6)
    best_hist = np.array(best_history)
    axs[0, 0].plot(best_hist[:, 0], best_hist[:, 1], '-o', color='green', linewidth=2, markersize=4, label='Best so far')
    axs[0, 0].plot(best_hist[-1, 0], best_hist[-1, 1], 'o', color='green', markersize=10, label='Final Best')
    axs[0, 0].set_xlim([bounds[0][0], bounds[0][1]])
    axs[0, 0].set_ylim([bounds[1][0], bounds[1][1]])
    axs[0, 0].set_xlabel("x1")
    axs[0, 0].set_ylabel("x2")
    axs[0, 0].set_title("Feasibility & Best Path")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Cost (objective) over iterations
    costs = [my_obj(x) for x in best_history]
    axs[1, 0].plot(costs, label="Objective (Cost)")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Objective Value")
    axs[1, 0].set_title("Objective Value vs Iteration")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # 3. Acceptance probability over iterations
    axs[0, 1].plot(logs["accept_history"], label="Average Acceptance Rate")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Acceptance Rate")
    axs[0, 1].set_title("Acceptance Rate vs Iteration")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # 4. Temperature over iterations
    axs[1, 1].plot(logs["temp_history"], label="Average Temperature")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Temperature")
    axs[1, 1].set_title("Average Temperature vs Iteration")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()