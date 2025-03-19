import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

###############################################################################
# PROBLEM DEFINITION
###############################################################################
def my_obj(x):
    """
    Objective function: f(x) = x1 + x2
    """
    return x[0] + x[1]

def my_nonlincon(x):
    """
    Returns g = [g1, g2, g3].
    If g[i] > 0 => constraint i is violated.

    Constraints:
      g1(x) = 20 - x1^2 * x2 <= 0
      g2(x) = 1 - ((x1 + x2 - 5)^2 / 30) - ((x1 - x2 - 12)^2 / 120) <= 0
      g3(x) = x1^2 + 8*x2 - 75 <= 0
    """
    x1, x2 = x
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
# PARALLEL TEMPERING SIMULATED ANNEALING
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
    Returns:
      best_x, best_E, logs
    Where 'logs' contains keys:
      - "temp_history": list of average temperature per iteration
      - "accept_history": list of acceptance rate per iteration
      - "best_history": list of best-solution-so-far
      - "chain_paths": array of each chain's path through the search space
    """
    x0_list = pt_config["x0_list"]
    T0_list = pt_config["T0_list"]
    bounds = pt_config["bounds"]
    max_iter = pt_config.get("max_iter", 3000)
    swap_int = pt_config.get("swap_interval", 100)
    adapt_int = pt_config.get("adapt_interval", 100)
    penalty_alpha_init = pt_config.get("penalty_alpha_init", 1e6)
    penalty_growth = pt_config.get("penalty_growth", 2.0)
    step_size_init = pt_config.get("step_size_init", 0.01)
    adapt_factor = pt_config.get("adapt_factor", 1.2)
    no_improve_limit = pt_config.get("no_improve_limit", 800)
    min_temp = pt_config.get("min_temp", 1e-9)

    n_chains = len(x0_list)
    assert n_chains == len(T0_list), "Must have the same number of x0_list and T0_list entries!"

    # Data structures for each chain
    chain_x = [list(x0_list[i]) for i in range(n_chains)]  # current solution
    chain_E = [None]*n_chains                             # current energy
    chain_T = [T0_list[i] for i in range(n_chains)]       # current temperature
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
    temp_history = []
    accept_history= []
    best_history = []

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
                T_use = max(chain_T[c], min_temp)
                accept_prob = math.exp(-deltaE / T_use) if deltaE > 0 else 1.0
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
                # Handle extreme values of 'arg' to avoid overflow
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
                    # keep chain_paths consistent (swap last appended point)
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
            chain_T[c] = max(chain_T[c]*0.98, min_temp)

        # Stopping if no improvement
        if (iteration - last_improve_iter) > no_improve_limit:
            # print(f"No improvement for {no_improve_limit} iterations. Stopping.")
            break

    logs = {
        "temp_history": temp_history,
        "accept_history": accept_history,
        "best_history": best_history,
        "chain_paths": chain_paths
    }
    return best_x, best_E, logs

###############################################################################
# TRADITIONAL (SINGLE-CHAIN) SIMULATED ANNEALING
###############################################################################
def traditional_sa(sa_config):
    """
    Basic single-chain Simulated Annealing.
    Keys needed in sa_config:
      - x0 (initial guess, shape (2,))
      - T0 (initial temperature)
      - bounds: [ (x1_min, x1_max), (x2_min, x2_max) ]
      - max_iter, penalty_alpha_init, penalty_growth
      - step_size_init, adapt_factor, adapt_interval
      - no_improve_limit, min_temp
    Returns:
      best_x, best_E, logs
    Where 'logs' contains keys:
      - "temp_history": list of temperature per iteration
      - "accept_history": list of acceptance rate per iteration
      - "best_history": list of best-solution-so-far
      - "path": array of visited points
    """
    x0 = sa_config["x0"]
    T0 = sa_config["T0"]
    bounds = sa_config["bounds"]
    max_iter = sa_config.get("max_iter", 2000)
    penalty_alpha_init = sa_config.get("penalty_alpha_init", 1e6)
    penalty_growth = sa_config.get("penalty_growth", 2.0)
    step_size_init = sa_config.get("step_size_init", 0.5)
    adapt_factor = sa_config.get("adapt_factor", 1.2)
    adapt_interval = sa_config.get("adapt_interval", 100)
    no_improve_limit = sa_config.get("no_improve_limit", 300)
    min_temp = sa_config.get("min_temp", 1e-9)

    # Initialize
    current_x = [x0[0], x0[1]]
    def clamp(x):
        return [
            min(max(x[i], bounds[i][0]), bounds[i][1]) for i in range(len(x))
        ]

    current_x = clamp(current_x)
    penalty_alpha = penalty_alpha_init

    def penalty_function(x):
        g = my_nonlincon(x)
        viol = g[g > 0]  # violation
        if len(viol) == 0:
            return 0.0
        return penalty_alpha * np.sum(viol**2)

    def energy(x):
        return my_obj(x) + penalty_function(x)

    current_E = energy(current_x)
    best_x = current_x[:]
    best_E = current_E

    temperature = T0
    step_size = step_size_init
    last_improve_iter = 0
    feasible_in_interval = 0
    penalty_check_interval = 200

    # Logs
    temp_history = []
    accept_history= []
    best_history = []
    path = []
    accept_count = 0

    path.append(current_x[:])
    best_history.append(best_x[:])

    def get_neighbor(x, step):
        y = x[:]
        idx = random.randint(0, len(x)-1)
        y[idx] += random.uniform(-step, step)
        return clamp(y)

    for iteration in range(max_iter):
        candidate = get_neighbor(current_x, step_size)
        cand_E = energy(candidate)

        # Accept or reject
        if cand_E < current_E:
            current_x = candidate
            current_E = cand_E
            accept_count += 1
        else:
            deltaE = cand_E - current_E
            T_use = max(temperature, min_temp)
            accept_prob = math.exp(-deltaE / T_use) if deltaE > 0 else 1.0
            if random.random() < accept_prob:
                current_x = candidate
                current_E = cand_E
                accept_count += 1

        # Update best
        if current_E < best_E:
            best_E = current_E
            best_x = current_x[:]
            last_improve_iter = iteration

        path.append(current_x[:])
        best_history.append(best_x[:])

        # Penalty adaptation
        if my_is_feasible(best_x):
            feasible_in_interval += 1
        if (iteration+1) % penalty_check_interval == 0:
            if feasible_in_interval == 0:
                penalty_alpha *= penalty_growth
            feasible_in_interval = 0

        # Cooling
        temperature = max(temperature * 0.98, min_temp)

        # Step size adapt
        if (iteration+1) % adapt_interval == 0:
            accept_rate = accept_count / adapt_interval
            if accept_rate > 0.5:
                step_size *= adapt_factor
            elif accept_rate < 0.2:
                step_size /= adapt_factor
            accept_count = 0

        # Logs
        temp_history.append(temperature)
        if iteration == 0:
            accept_history.append(1.0)  # first step trivially "accepted"
        else:
            accept_history.append(accept_count / (iteration+1))
        
        # Stopping if no improvement
        if (iteration - last_improve_iter) > no_improve_limit:
            # print(f"No improvement for {no_improve_limit} iterations. Stopping.")
            break

    logs = {
        "temp_history": temp_history,
        "accept_history": accept_history,
        "best_history": best_history,
        "path": path
    }
    return best_x, best_E, logs

###############################################################################
# SCIPY OPTIMIZER (TRUST-CONSTR) WRAPPER
###############################################################################
def scipy_solve(bounds, x0):
    """
    Uses scipy.optimize.minimize with a nonlinear constraint approach
    to solve the same problem.
    """
    # Convert constraints g(x) <= 0 into a single function that returns an array
    # that must be <= 0. We'll pass it as a NonlinearConstraint object:
    def cons_f(x):
        return my_nonlincon(x)  # array of [g1, g2, g3]
    # Because we want g(x) <= 0, we pass  -∞  <  g_i(x)  <=  0
    nonlin_cons = NonlinearConstraint(cons_f, -np.inf, 0.0)

    # We'll keep track of the "path" by using a callback:
    path = []
    def callback_f(xk, *args):
        path.append(np.copy(xk))

    # Run minimize
    res = minimize(
        fun=my_obj,
        x0=np.array(x0),
        method='trust-constr',  # or 'SLSQP'
        constraints=[nonlin_cons],
        bounds=bounds,
        callback=callback_f,
        options={
            'maxiter': 1000,
            'verbose': 0
        }
    )

    # If you want guaranteed feasibility in the bounding box, ensure the bounds
    # are passed in the 'bounds' argument as well. The final 'path' includes
    # everything from each iteration if the solver chooses to store it.
    # Some methods might only call the callback at certain steps.

    return res, path

###############################################################################
# 4-SUBPLOT VISUALIZATION (OVERLAY ALL METHODS)
###############################################################################
def plot_comparison(bounds, pt_logs, sa_logs, scipy_path):
    """
    Creates a 4-subplot visualization that overlays:
      - Parallel Tempering SA
      - Traditional SA
      - Scipy's final solution
    on the same figure.

    Subplots:
      1) Feasibility region + final paths
      2) Objective value vs iteration
      3) Acceptance rate vs iteration
      4) Temperature vs iteration
    """
    # Build feasibility grid
    N = 200
    x1_lin = np.linspace(bounds[0][0], bounds[0][1], N)
    x2_lin = np.linspace(bounds[1][0], bounds[1][1], N)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    feasible_mask = np.zeros_like(X1, dtype=bool)
    for i in range(N):
        for j in range(N):
            xx = [X1[i, j], X2[i, j]]
            g = my_nonlincon(xx)
            feasible_mask[i, j] = np.all(g <= 0)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # === SUBPLOT 1: Feasibility region + final paths ===
    axs[0, 0].contourf(
        X1, X2, feasible_mask,
        levels=[-0.5, 0.5],
        colors=["yellow", "white"],
        alpha=0.6
    )

    # Plot best path from PT SA
    pt_best_hist = np.array(pt_logs["best_history"])
    axs[0, 0].plot(pt_best_hist[:, 0], pt_best_hist[:, 1],
                   '-o', markersize=3, label='PT SA (best path)')

    # Plot best path from traditional SA
    sa_best_hist = np.array(sa_logs["best_history"])
    axs[0, 0].plot(sa_best_hist[:, 0], sa_best_hist[:, 1],
                   '-o', markersize=3, label='Trad. SA (best path)')

    # Plot final path/points from Scipy
    # If we have a path, we can plot it. Otherwise, just final point.
    if len(scipy_path) > 0:
        scipy_path_arr = np.array(scipy_path)
        axs[0, 0].plot(scipy_path_arr[:,0], scipy_path_arr[:,1],
                       '-o', markersize=3, label='Scipy path')
        # Mark final
        final_x_scipy = scipy_path_arr[-1]
        axs[0, 0].plot(final_x_scipy[0], final_x_scipy[1],
                       'o', markersize=8, label='Scipy final')

    axs[0, 0].set_xlim(bounds[0])
    axs[0, 0].set_ylim(bounds[1])
    axs[0, 0].set_xlabel("x1")
    axs[0, 0].set_ylabel("x2")
    axs[0, 0].set_title("Feasibility Region + Paths")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # === SUBPLOT 2: Objective value vs iteration ===
    # PT SA
    pt_costs = [my_obj(x) for x in pt_logs["best_history"]]
    axs[1, 0].plot(pt_costs, label="PT SA")

    # Traditional SA
    sa_costs = [my_obj(x) for x in sa_logs["best_history"]]
    axs[1, 0].plot(sa_costs, label="Trad. SA")

    # For Scipy, we don't have iteration-by-iteration "objective" in the same sense;
    # we can just mark final or skip. Alternatively, if you want, you can generate
    # an array by evaluating the path. Let's do that:
    scipy_costs = [my_obj(x) for x in scipy_path]
    if len(scipy_costs) > 0:
        axs[1, 0].plot(scipy_costs, label="Scipy (along path)")

    axs[1, 0].set_xlabel("Iteration (not identical across methods)")
    axs[1, 0].set_ylabel("Objective Value (x1 + x2)")
    axs[1, 0].set_title("Objective vs Iteration")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # === SUBPLOT 3: Acceptance rate vs iteration ===
    # PT SA acceptance
    axs[0, 1].plot(pt_logs["accept_history"], label="PT SA Accept Rate")
    # Trad. SA acceptance
    axs[0, 1].plot(sa_logs["accept_history"], label="Trad. SA Accept Rate")
    # Scipy has no acceptance rate => skip
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Acceptance Rate")
    axs[0, 1].set_title("Acceptance Rate vs Iteration")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # === SUBPLOT 4: Temperature vs iteration ===
    # PT SA temperature
    axs[1, 1].plot(pt_logs["temp_history"], label="PT SA Avg Temp")
    # Trad. SA temperature
    axs[1, 1].plot(sa_logs["temp_history"], label="Trad. SA Temp")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Temperature")
    axs[1, 1].set_title("Temperature vs Iteration")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

###############################################################################
# MAIN DEMO
###############################################################################
if __name__=="__main__":
    random.seed(0)
    np.random.seed(0)

    # 1) PARALLEL TEMPERING SA CONFIG
    pt_config = {
        "x0_list": [
            [0.0, 7.5],
            [-7.5, 7.5],
            [7.5, 7.5],
        ],
        "T0_list": [1e4, 5e3, 1e3],
        "bounds": [(-10, 10), (-10, 10)],
        "max_iter": 1500,
        "swap_interval": 300,
        "adapt_interval": 100,
        "penalty_alpha_init": 1e6,
        "penalty_growth": 2.0,
        "step_size_init": 0.5,
        "adapt_factor": 1.2,
        "no_improve_limit": 300,
        "min_temp": 1e-9
    }
    pt_best_x, pt_best_E, pt_logs = parallel_tempering_sa(pt_config)

    # 2) TRADITIONAL SA CONFIG
    sa_config = {
        "x0": [0.0, 7.5],
        "T0": 1e4,
        "bounds": [(-10, 10), (-10, 10)],
        "max_iter": 1500,
        "penalty_alpha_init": 1e6,
        "penalty_growth": 2.0,
        "step_size_init": 0.5,
        "adapt_factor": 1.2,
        "adapt_interval": 100,
        "no_improve_limit": 300,
        "min_temp": 1e-9
    }
    sa_best_x, sa_best_E, sa_logs = traditional_sa(sa_config)

    # 3) SCIPY SOLUTION
    scipy_bounds = [(sa_config["bounds"][0][0], sa_config["bounds"][0][1]),
                    (sa_config["bounds"][1][0], sa_config["bounds"][1][1])]
    # Start from same x0
    scipy_res, scipy_path = scipy_solve(scipy_bounds, sa_config["x0"])
    scipy_best_x = scipy_res.x
    scipy_best_obj = scipy_res.fun
    scipy_feasible = my_is_feasible(scipy_best_x)

    # Print summary
    print("======== RESULTS ========")
    print("PT SA:")
    print(f"  Best x = {pt_best_x}, Energy = {pt_best_E}, Feasible? {my_is_feasible(pt_best_x)}")
    print(f"  Objective = {my_obj(pt_best_x)}, Constraints = {my_nonlincon(pt_best_x)}")

    print("\nTraditional SA:")
    print(f"  Best x = {sa_best_x}, Energy = {sa_best_E}, Feasible? {my_is_feasible(sa_best_x)}")
    print(f"  Objective = {my_obj(sa_best_x)}, Constraints = {my_nonlincon(sa_best_x)}")

    print("\nScipy (trust-constr):")
    print(f"  Success? {scipy_res.success}, Status: {scipy_res.message}")
    print(f"  Best x = {scipy_best_x}, Obj = {scipy_best_obj}, Feasible? {scipy_feasible}")
    print(f"  Constraints = {my_nonlincon(scipy_best_x)}")

    # COMPARISON PLOT (ALL METHODS)
    plot_comparison(pt_config["bounds"], pt_logs, sa_logs, scipy_path)
