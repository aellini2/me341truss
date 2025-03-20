import numpy as np
import random
import math
import matplotlib.pyplot as plt

###############################################################################
#                           PROBLEM DEFINITION                                #
###############################################################################
def my_obj(r):
    """
    Weight = sum of volumes * density
      6 bars with radius r1, length=9.14
      4 bars with radius r2, length=9.14*sqrt(2)
    """
    length = 9.14
    density = 7860
    # 6 bars of radius r[0], 4 of radius r[1]
    weight = (
        6 * np.pi * r[0]**2 * length
      + 4 * np.pi * r[1]**2 * length * np.sqrt(2)
    ) * density
    return weight

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
#         PARALLEL TEMPERING SA (Generic for 2-variable problem)              #
###############################################################################
def parallel_tempering_sa(pt_config):
    """
    Parallel Tempering SA reading parameters from 'pt_config'.
    
    Keys needed in pt_config:
      - x0_list (list of initial guesses, shape (n_chains, 2))
      - T0_list (list of initial temperatures)
      - bounds: [ (x1_min, x1_max), (x2_min, x2_max) ]
      - max_iter, swap_interval, adapt_interval,
      - penalty_alpha_init, penalty_growth,
      - step_size_init, adapt_factor,
      - no_improve_limit, min_temp
    """
    x0_list     = pt_config["x0_list"]     # list of initial x for each chain
    T0_list     = pt_config["T0_list"]     # list of initial temps
    bounds      = pt_config["bounds"]      # e.g. [(-10,10), (-10,10)]
    max_iter    = pt_config.get("max_iter", 3000)
    swap_int    = pt_config.get("swap_interval", 100)
    adapt_int   = pt_config.get("adapt_interval", 100)
    penalty_alpha_init = pt_config.get("penalty_alpha_init", 1e6)
    penalty_growth     = pt_config.get("penalty_growth", 2.0)
    step_size_init     = pt_config.get("step_size_init", 0.01)
    adapt_factor       = pt_config.get("adapt_factor", 1.2)
    no_improve_limit   = pt_config.get("no_improve_limit", 800)
    min_temp           = pt_config.get("min_temp", 1e-9)

    n_chains = len(x0_list)
    assert n_chains == len(T0_list), "Must have the same number of x0_list and T0_list entries!"

    # Data structures for each chain
    chain_x   = [list(x0_list[i]) for i in range(n_chains)]  # current solution
    chain_E   = [None]*n_chains                             # current energy
    chain_T   = [T0_list[i] for i in range(n_chains)]       # current temperature
    chain_step= [step_size_init]*n_chains
    chain_accept_count = [0]*n_chains

    # Keep entire path for each chain to plot later
    chain_paths = [[] for _ in range(n_chains)]

    # penalty alpha (for constraints)
    penalty_alpha = penalty_alpha_init

    def penalty_function(x):
        """
        Quadratic penalty for any constraint violation:
        sum of violation^2 * penalty_alpha
        """
        g = my_nonlincon(x)
        viol = g[g > 0]  # violation parts
        if len(viol) == 0:
            return 0.0
        return penalty_alpha * np.sum(viol**2)

    def energy(x):
        return my_obj(x) + penalty_function(x)

    def clamp(x):
        """
        Enforce bounds: x[i] in [bounds[i][0], bounds[i][1]].
        """
        return [
            min(max(x[i], bounds[i][0]), bounds[i][1]) for i in range(len(x))
        ]

    def get_neighbor(x, step):
        """
        Randomly perturb one dimension in [-step, +step].
        """
        y = x[:]
        idx = random.randint(0, len(x)-1)
        y[idx] += random.uniform(-step, step)
        return clamp(y)

    # Initialize
    for c in range(n_chains):
        chain_x[c] = clamp(chain_x[c])
        chain_E[c] = energy(chain_x[c])
        chain_paths[c].append(chain_x[c][:])

    # Best so far
    best_x = chain_x[0][:]
    best_E = chain_E[0]
    for i in range(1, n_chains):
        if chain_E[i] < best_E:
            best_E = chain_E[i]
            best_x = chain_x[i][:]

    # Logs
    temp_history  = []
    accept_history= []
    best_history  = []

    # For stopping
    last_improve_iter = 0
    feasible_in_interval = 0
    penalty_check_interval = 200

    # Main loop
    for iteration in range(max_iter):
        best_history.append(best_x[:])

        # Each chain proposes a move
        for c in range(n_chains):
            candidate = get_neighbor(chain_x[c], chain_step[c])
            cand_E = energy(candidate)

            if cand_E < chain_E[c]:
                # accept better
                chain_x[c] = candidate
                chain_E[c] = cand_E
                chain_accept_count[c] += 1
            else:
                # maybe accept worse
                deltaE = cand_E - chain_E[c]
                T_use  = max(chain_T[c], min_temp)
                accept_prob = math.exp(-deltaE / T_use) if deltaE>0 else 1.0
                if random.random() < accept_prob:
                    chain_x[c] = candidate
                    chain_E[c] = cand_E
                    chain_accept_count[c] += 1

            # store path
            chain_paths[c].append(chain_x[c][:])

            # update global best
            if chain_E[c] < best_E:
                best_E = chain_E[c]
                best_x = chain_x[c][:]
                last_improve_iter = iteration

        # Swap if needed
        if (iteration+1) % swap_int == 0 and n_chains > 1:
            for c in range(n_chains-1):
                E1, E2 = chain_E[c], chain_E[c+1]
                T1, T2 = max(chain_T[c], min_temp), max(chain_T[c+1], min_temp)
                arg = (E1 - E2)*(1/T1 - 1/T2)
                if arg > 700:
                    swap_prob = 1.0
                elif arg < -700:
                    swap_prob = 0.0
                else:
                    swap_prob = math.exp(arg)
                if random.random() < swap_prob:
                    # swap solutions
                    chain_x[c], chain_x[c+1] = chain_x[c+1], chain_x[c]
                    chain_E[c], chain_E[c+1] = chain_E[c+1], chain_E[c]
                    # keep chain_paths consistent
                    chain_paths[c][-1], chain_paths[c+1][-1] = chain_paths[c+1][-1], chain_paths[c][-1]

        # Logging
        avg_T = sum(chain_T)/n_chains
        total_accept = sum(chain_accept_count)
        avg_accept_rate = total_accept / ((iteration+1)*n_chains)
        temp_history.append(avg_T)
        accept_history.append(avg_accept_rate)

        # Penalty adaptation
        if my_is_feasible(best_x):
            feasible_in_interval += 1

        if (iteration+1) % penalty_check_interval == 0:
            if feasible_in_interval == 0:
                penalty_alpha *= penalty_growth
            feasible_in_interval = 0

        # Step size adaptation
        if (iteration+1) % adapt_int == 0:
            for c in range(n_chains):
                ratio = chain_accept_count[c]/adapt_int
                if ratio > 0.5:
                    chain_step[c] *= adapt_factor
                elif ratio < 0.2:
                    chain_step[c] /= adapt_factor
                chain_accept_count[c] = 0

        # Cooling
        for c in range(n_chains):
            chain_T[c] = max(chain_T[c]*0.99, min_temp)

        # Stopping if no improvement
        if (iteration - last_improve_iter) > no_improve_limit:
            print(f"No improvement for {no_improve_limit} iterations. Stopping.")
            break

    # Final logs
    logs = {
        "temp_history": temp_history,
        "accept_history": accept_history,
        "best_history": best_history,
        "chain_paths": chain_paths
    }
    return best_x, best_E, logs
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

###############################################################################
#                               MAIN DEMO                                     #
###############################################################################
if __name__=="__main__":
    # SA config
    pt_config = {
        "x0_list": [
            [0.0, 7.5],  # chain0 initial guess
            [-2, 4], # chain1 initial guess
            [-7.5, 0], # chain2 initial guess
        ],
        "T0_list": [100e3, 5e3, 1e3],     # initial temperatures
        "bounds": [(-10, 10), (-10, 10)],
        "max_iter": 2000,
        "swap_interval": 500,
        "adapt_interval": 100,
        "penalty_alpha_init": 1e6,
        "penalty_growth": 2.0,
        "step_size_init": 0.5,
        "adapt_factor": 1.2,
        "no_improve_limit": 300,
        "min_temp": 1e-9
    }

    best_x, best_E, logs = parallel_tempering_sa(pt_config)
# Print final result
    print("\n===============================")
    print("Parallel Tempering SA COMPLETE")
    print("Best solution found:", best_x)
    print("Best 'energy' (obj + penalty):", best_E)
    # Because penalty might be zero if feasible
    print("Objective alone (x1+x2):", my_obj(best_x))
    print("Constraints:", my_nonlincon(best_x))
    print("Feasible?", my_is_feasible(best_x))

    # Plot temperature vs iteration
    plt.figure()
    plt.plot(logs["temp_history"], label="Average Temperature")
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Average Temperature vs Iteration")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot acceptance rate vs iteration
    plt.figure()
    plt.plot(logs["accept_history"], label="Average Acceptance Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Acceptance Rate")
    plt.title("Acceptance Rate vs Iteration")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Feasibility region + path
    plot_feasibility_and_path_2d(
        pt_config["bounds"],
        logs["chain_paths"],
        logs["best_history"],
        N=200,
        title="Feasibility & Parallel Tempering SA Path"
    )

    # 4-subplot visualization
    plot_4_subplots(
        logs,
        logs["best_history"],
        pt_config["bounds"]
    )