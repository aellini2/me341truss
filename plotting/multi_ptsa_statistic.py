import numpy as np
import math
import random
import matplotlib.pyplot as plt

###############################################################################
# FINITE ELEMENT ANALYSIS (Truss)
###############################################################################
def truss_FEA(r):
    """
    Given r = [r1, r2], runs a 2D truss FEA with 10 elements.
    Returns:
      length_e: array of element lengths (size=10)
      stress:   array of element stresses (size=10)
      r_e:      array of element radii (size=10) => first 6 use r[0], last 4 use r[1]
      Q:        array of nodal displacements (size=12)
    """
    length = 9.14  # (m)
    E = 200e9      # (Pa)
    # External forces (N)
    F = np.zeros(12)
    F[3] = -1e7   # Node #2 in -y direction
    F[7] = -1e7   # Node #4 in -y direction

    # Node coordinates (6 nodes total)
    n = np.array([
        [2, 1], [2, 0], [1, 1],
        [1, 0], [0, 1], [0, 0]
    ]) * length

    # Element connectivity (10 elements), zero-based indexing
    ec = np.array([
        [2, 4], [0, 2], [3, 5], [1, 3], [2, 3],
        [0, 1], [3, 4], [2, 5], [1, 2], [0, 3]
    ])

    num_elements = ec.shape[0]
    length_e = np.zeros(num_elements)
    l = np.zeros(num_elements)
    m = np.zeros(num_elements)

    # Compute element lengths and direction cosines
    for i in range(num_elements):
        diff = n[ec[i, 1]] - n[ec[i, 0]]
        length_e[i] = np.linalg.norm(diff)
        l[i] = diff[0] / length_e[i]
        m[i] = diff[1] / length_e[i]

    # Radii assignment: first 6 elements use r[0], last 4 use r[1]
    r_e = np.concatenate([np.full(6, r[0]), np.full(4, r[1])])
    A = np.pi * r_e**2

    # Global stiffness matrix
    K = np.zeros((12, 12))
    for i in range(num_elements):
        k_small = (E * A[i] / length_e[i]) * np.array([
            [l[i]**2,    l[i]*m[i]],
            [l[i]*m[i],  m[i]**2 ]
        ])
        # Expand to global K
        idx = [
            ec[i, 0]*2, ec[i, 0]*2 + 1,
            ec[i, 1]*2, ec[i, 1]*2 + 1
        ]
        block = np.zeros((4, 4))
        block[:2, :2]   = k_small
        block[2:, 2:]   = k_small
        block[:2, 2:]   = -k_small
        block[2:, :2]   = -k_small
        # Assemble
        for ii in range(4):
            for jj in range(4):
                K[idx[ii], idx[jj]] += block[ii, jj]

    # DOF reduction => first 8 DOFs are unknown, last 4 are fixed
    K_reduced = K[:8, :8]
    F_reduced = F[:8]
    Q_reduced = np.linalg.solve(K_reduced, F_reduced)
    # Full displacement
    Q = np.concatenate([Q_reduced, np.zeros(4)])

    # Stress in each element
    stress = np.zeros(num_elements)
    for i in range(num_elements):
        idx = [
            ec[i, 0]*2, ec[i, 0]*2 + 1,
            ec[i, 1]*2, ec[i, 1]*2 + 1
        ]
        vec = np.array([-l[i], -m[i], l[i], m[i]])
        stress[i] = (E / length_e[i]) * np.dot(vec, Q[idx])

    return length_e, stress, r_e, Q

###############################################################################
# OBJECTIVE & CONSTRAINTS
###############################################################################
def my_obj(r):
    """
    Weight = sum of volumes * density
      6 bars with radius r[0], length=9.14
      4 bars with radius r[1], length=9.14*sqrt(2)
    """
    length = 9.14
    density = 7860
    # 6 bars of radius r[0], 4 bars of radius r[1]
    weight = (
        6 * np.pi * r[0]**2 * length
        + 4 * np.pi * r[1]**2 * length * np.sqrt(2)
    ) * density
    return weight

def my_nonlincon(r):
    """
    g_i(r) <= 0 => feasible
      - 10 buckling constraints
      - 10 stress constraints
      - 1 displacement constraint
    """
    length_e, stress, r_e, Q = truss_FEA(r)
    E = 200e9
    Y = 250e6
    I = (np.pi/4) * r_e**4
    dis_max = 0.02

    num_elems = len(stress)
    g_buckling = np.zeros(num_elems)
    g_stress   = np.zeros(num_elems)

    # Axial force in each element
    F_internal = np.pi*(r_e**2)*stress

    for i in range(num_elems):
        # Buckling if in compression => F_internal < 0
        if F_internal[i] < 0:
            # want: |F_comp| <= pi^2 E I / L^2
            F_comp = -F_internal[i]
            crit_buckling = (np.pi**2 * E * I[i]) / (length_e[i]**2)
            g_buckling[i] = F_comp - crit_buckling
        else:
            g_buckling[i] = -1e-6  # not active

        # Stress
        g_stress[i] = abs(stress[i]) - Y

    # Displacement constraint at node #2 => DOFs #2,3
    disp_constraint = (Q[2]**2 + Q[3]**2) - (dis_max**2)

    # Collect all
    return np.concatenate([g_buckling, g_stress, [disp_constraint]])

def my_is_feasible(r):
    return np.all(my_nonlincon(r) <= 0)

###############################################################################
# PARALLEL TEMPERING SA
###############################################################################
def penalty_function(r, alpha=1e16):
    g = my_nonlincon(r)
    viol = g[g > 0]
    return alpha * np.sum(viol**2) if len(viol) else 0.0

def energy(r):
    return my_obj(r) + penalty_function(r)

def clamp(r, bounds):
    return [
        min(max(r[i], bounds[i][0]), bounds[i][1])
        for i in range(len(r))
    ]

def get_neighbor(x, step, bounds):
    y = x[:]
    idx = random.randint(0, len(x)-1)
    y[idx] += random.uniform(-step, step)
    return clamp(y, bounds)

def parallel_tempering_sa(pt_config):
    """
    Parallel Tempering SA that strictly REJECTS any candidate violating constraints.
    This version does *not* use a penalty; only feasible points are even considered.

    Returns (best_r, best_obj, logs).

    pt_config keys (with example defaults):
        - x0_list : list of initial guesses for each chain, e.g. [[r1,r2], [r1,r2], ...]
        - bounds : [(r1_min, r1_max), (r2_min, r2_max)]
        - T0_list : list of initial temperatures, one per chain
        - max_iter : 10000  (larger default for more exploration)
        - swap_interval : 300
        - adapt_interval : 100
        - step_size_init : 0.02
        - adapt_factor : 1.2
        - no_improve_limit : 3000 (or None)
        - min_temp : 1e-9
        - cooling_rate : 0.98
        - seed : 0 or None
    """
    # Unpack config with fallback defaults
    x0_list = pt_config["x0_list"]
    bounds = pt_config["bounds"]
    T0_list = pt_config["T0_list"]

    max_iter = pt_config.get("max_iter", 10000)
    swap_int = pt_config.get("swap_interval", 300)
    adapt_int = pt_config.get("adapt_interval", 100)
    step_size_init = pt_config.get("step_size_init", 0.02)
    adapt_factor = pt_config.get("adapt_factor", 1.2)
    no_improve_limit = pt_config.get("no_improve_limit", None)
    min_temp = pt_config.get("min_temp", 1e-9)
    cooling_rate = pt_config.get("cooling_rate", 0.98)

    # For reproducibility
    random.seed(pt_config.get("seed", None))
    np.random.seed(pt_config.get("seed", None))

    n_chains = len(x0_list)
    assert len(T0_list) == n_chains, "Mismatch in chain count vs. T0_list length."

    # Helper to clamp a design within bounds
    def clamp_design(r):
        return [min(max(r_i, lb), ub) for r_i, (lb, ub) in zip(r, bounds)]

    # Prepare chain states
    chain_r = [clamp_design(x0_list[i]) for i in range(n_chains)]
    # If x0 is infeasible, try to find a random feasible point
    for i in range(n_chains):
        if not my_is_feasible(chain_r[i]):
            for _ in range(5000):
                trial = [random.uniform(*b) for b in bounds]
                if my_is_feasible(trial):
                    chain_r[i] = trial
                    break

    chain_obj = [my_obj(chain_r[i]) for i in range(n_chains)]
    chain_T = [T0_list[i] for i in range(n_chains)]
    chain_step = [step_size_init]*n_chains
    chain_accept_count = [0]*n_chains

    chain_paths = [[] for _ in range(n_chains)]
    for i in range(n_chains):
        chain_paths[i].append(chain_r[i][:])

    # Global best
    best_idx = np.argmin(chain_obj)
    best_r = chain_r[best_idx][:]
    best_obj = chain_obj[best_idx]
    last_improve_iter = 0

    # Logging
    logs = {
        "temp_history": [],
        "accept_history": [],
        "best_history": [],      # store best_r each iteration
        "chain_paths": chain_paths
    }

    def get_neighbor_local(x, step):
        y = x[:]
        idx = random.randint(0, len(x)-1)
        y[idx] += random.uniform(-step, step)
        return clamp_design(y)

    for iteration in range(max_iter):
        logs["best_history"].append(best_r[:])

        # Each chain tries a move
        for c in range(n_chains):
            # Propose a neighbor
            candidate = get_neighbor_local(chain_r[c], chain_step[c])
            # If candidate is infeasible, *reject* outright
            if not my_is_feasible(candidate):
                # do nothing, skip
                pass
            else:
                # Feasible => do standard SA acceptance
                cand_obj = my_obj(candidate)
                if cand_obj < chain_obj[c]:
                    # Accept improvement
                    chain_r[c] = candidate
                    chain_obj[c] = cand_obj
                    chain_accept_count[c] += 1
                else:
                    # Accept with probability exp(-(cand_obj - current)/T)
                    delta = cand_obj - chain_obj[c]
                    T_use = max(chain_T[c], min_temp)
                    accept_prob = math.exp(-delta / T_use)
                    if random.random() < accept_prob:
                        chain_r[c] = candidate
                        chain_obj[c] = cand_obj
                        chain_accept_count[c] += 1

                # Update global best
                if chain_obj[c] < best_obj:
                    best_obj = chain_obj[c]
                    best_r = chain_r[c][:]
                    last_improve_iter = iteration

            # Update chain path
            chain_paths[c].append(chain_r[c][:])

        # Attempt chain swaps every swap_int
        if (iteration+1) % swap_int == 0 and n_chains > 1:
            for c in range(n_chains - 1):
                E1, E2 = chain_obj[c], chain_obj[c+1]
                T1, T2 = max(chain_T[c], min_temp), max(chain_T[c+1], min_temp)
                arg = (E1 - E2)*(1.0/T1 - 1.0/T2)
                # Numerically safe check
                if arg > 700:
                    swap_prob = 1.0
                elif arg < -700:
                    swap_prob = 0.0
                else:
                    swap_prob = math.exp(arg)
                if random.random() < swap_prob:
                    # swap solutions
                    chain_r[c], chain_r[c+1] = chain_r[c+1], chain_r[c]
                    chain_obj[c], chain_obj[c+1] = chain_obj[c+1], chain_obj[c]
                    # also swap the last path entry to keep them consistent
                    chain_paths[c][-1], chain_paths[c+1][-1] = chain_paths[c+1][-1], chain_paths[c][-1]

        # Logging
        avg_T = sum(chain_T)/n_chains
        total_accept = sum(chain_accept_count)
        avg_accept_rate = total_accept / ((iteration+1)*n_chains)
        logs["temp_history"].append(avg_T)
        logs["accept_history"].append(avg_accept_rate)

        # Step-size adaptation
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
            chain_T[c] = max(chain_T[c] * cooling_rate, min_temp)

        # Early stopping if no improvement
        if (no_improve_limit is not None) and ((iteration - last_improve_iter) > no_improve_limit):
            break

    return best_r, best_obj, logs



def plot_multi_ptsa_4subplots(bounds, run_logs_list):
    """
    Plots the final 4-subplot figure:
      1) Feasibility region + PT-SA paths in (r1, r2) space
      2) Acceptance Rate vs Iteration
      3) Objective vs Iteration
      4) Average Temperature vs Iteration

    :param bounds: e.g., [(0.01,0.5),(0.01,0.5)]
    :param run_logs_list: list of (best_r, best_E, logs) for each run
                          logs is a dict with 'chain_paths', 'accept_history',
                          'temp_history', 'best_history', etc.
    """

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_path = axs[0, 0]  # top-left
    ax_acc  = axs[0, 1]  # top-right
    ax_obj  = axs[1, 0]  # bottom-left
    ax_temp = axs[1, 1]  # bottom-right

    # ----------------------------------
    # 1) Plot (r1, r2) paths (top-left)
    # ----------------------------------
    # Optionally shade the entire bounding box for r1,r2
    (r1_min, r1_max) = bounds[0]
    (r2_min, r2_max) = bounds[1]
    ax_path.fill(
        [r1_min, r1_max, r1_max, r1_min],
        [r2_min, r2_min, r2_max, r2_max],
        alpha=0.2, color="yellow", label="Bounds"
    )
    ax_path.set_xlim(r1_min, r1_max)
    ax_path.set_ylim(r2_min, r2_max)

    # Plot each run’s chain paths
    for i, (best_r, best_E, logs) in enumerate(run_logs_list, start=1):
        # Each run can have multiple chains => logs["chain_paths"] is a list of chain lists
        # We'll just connect all chain paths in iteration order
        # color or style can differ for each run
        for chain_idx, chain_pts in enumerate(logs["chain_paths"]):
            r1_vals = [pt[0] for pt in chain_pts]
            r2_vals = [pt[1] for pt in chain_pts]
            # We only label once per run (e.g. "Run i")
            label = f"Run {i}" if chain_idx == 0 else None
            ax_path.plot(r1_vals, r2_vals, marker="o", ms=3, label=label)

    ax_path.set_title("Feasibility + PT-SA Paths")
    ax_path.set_xlabel("r1")
    ax_path.set_ylabel("r2")
    ax_path.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))  # move legend outside

    # ----------------------------------
    # 2) Acceptance Rate vs Iteration (top-right)
    # ----------------------------------
    for i, (best_r, best_E, logs) in enumerate(run_logs_list, start=1):
        accept_hist = logs["accept_history"]  # list of acceptance rates
        iter_vals = range(len(accept_hist))
        ax_acc.plot(iter_vals, accept_hist, label=f"Run {i}")

    ax_acc.set_title("Acceptance Rate vs Iteration")
    ax_acc.set_xlabel("Iteration")
    ax_acc.set_ylabel("Acceptance Rate")
    ax_acc.legend(loc="upper right")

    # ----------------------------------
    # 3) Objective vs Iteration (bottom-left)
    # ----------------------------------
    # We assume logs["best_history"] is a list of r = [r1,r2], so we need my_obj(r) each iteration
    # Or some logs might store an "obj_history" directly. Adjust as needed.
    import numpy as np
    from math import inf

    def my_obj(r):
        length = 9.14
        density = 7860
        return (
            6 * np.pi * r[0]**2 * length
            + 4 * np.pi * r[1]**2 * length * np.sqrt(2)
        ) * density

    for i, (best_r, best_E, logs) in enumerate(run_logs_list, start=1):
        best_hist = logs.get("best_history", [])
        # Compute objective at each iteration
        obj_vals = [my_obj(rr) for rr in best_hist]
        iter_vals = range(len(obj_vals))
        ax_obj.plot(iter_vals, obj_vals, label=f"Run {i}")

    ax_obj.set_title("Objective vs Iteration")
    ax_obj.set_xlabel("Iteration")
    ax_obj.set_ylabel("Objective (Weight)")
    ax_obj.legend(loc="upper right")

    # ----------------------------------
    # 4) Average Temperature vs Iteration (bottom-right)
    # ----------------------------------
    for i, (best_r, best_E, logs) in enumerate(run_logs_list, start=1):
        temp_hist = logs["temp_history"]  # average temp each iteration
        iter_vals = range(len(temp_hist))
        ax_temp.plot(iter_vals, temp_hist, label=f"Run {i}")

    ax_temp.set_title("Average Temp vs Iteration")
    ax_temp.set_xlabel("Iteration")
    ax_temp.set_ylabel("Temperature")
    ax_temp.legend(loc="upper right")

    plt.tight_layout()
    return fig, axs

def plot_best_run_4subplots(bounds, best_r, best_obj, logs):
    """
    Plots the 4-subplot figure for a *single* run:
      1) (r1, r2) chain paths
      2) Acceptance rate vs iteration
      3) Objective vs iteration
      4) Average temperature vs iteration
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_path = axs[0, 0]  # top-left
    ax_acc  = axs[0, 1]  # top-right
    ax_obj  = axs[1, 0]  # bottom-left
    ax_temp = axs[1, 1]  # bottom-right
    
    (r1_min, r1_max) = bounds[0]
    (r2_min, r2_max) = bounds[1]
    
    # ------------------------------------------------
    # 1) (r1,r2) chain paths
    # ------------------------------------------------
    
    # chain_paths is a list of (iteration-wise) points for each chain
    chain_paths = logs.get("chain_paths", [])
    for chain_idx, chain_pts in enumerate(chain_paths):
        r1_vals = [pt[0] for pt in chain_pts]
        r2_vals = [pt[1] for pt in chain_pts]
        ax_path.plot(r1_vals, r2_vals, marker="o", ms=3)
    
    ax_path.set_title("Best Run: (r1, r2) Paths")
    ax_path.set_xlabel("r1")
    ax_path.set_ylabel("r2")
    
    # ------------------------------------------------
    # 2) Acceptance rate vs iteration
    # ------------------------------------------------
    accept_hist = logs.get("accept_history", [])
    ax_acc.plot(range(len(accept_hist)), accept_hist, label="Acceptance Rate")
    ax_acc.set_title("Acceptance Rate vs Iteration")
    ax_acc.set_xlabel("Iteration")
    ax_acc.set_ylabel("Acceptance Rate")
    
    # ------------------------------------------------
    # 3) Objective vs iteration
    # ------------------------------------------------
    # If logs has 'best_history' containing r = [r1, r2] per iteration, we can re-eval objective
    import numpy as np
    def my_obj(r):
        length = 9.14
        density = 7860
        return (
            6 * np.pi * r[0]**2 * length
            + 4 * np.pi * r[1]**2 * length * np.sqrt(2)
        ) * density
    
    best_hist = logs.get("best_history", [])
    obj_vals = [my_obj(r) for r in best_hist]
    ax_obj.plot(range(len(obj_vals)), obj_vals, label="Objective")
    ax_obj.set_title("Objective vs Iteration")
    ax_obj.set_xlabel("Iteration")
    ax_obj.set_ylabel("Objective (Weight)")
    
    # ------------------------------------------------
    # 4) Average Temperature vs iteration
    # ------------------------------------------------
    temp_hist = logs.get("temp_history", [])
    ax_temp.plot(range(len(temp_hist)), temp_hist, label="Average Temp")
    ax_temp.set_title("Average Temp vs Iteration")
    ax_temp.set_xlabel("Iteration")
    ax_temp.set_ylabel("Temperature")
    
    plt.tight_layout()
    return fig, axs

###############################################################################
# MAIN: RUN 30 TIMES WITH RANDOM INITIAL GUESSES, COLLECT & PLOT STATS
###############################################################################
if __name__ == "__main__":
    # Problem bounds
    my_bounds = [(0.01, 0.5), (0.01, 0.5)]

    # We'll do 3 chains per run in parallel tempering
    n_chains = 3

    # Common T0_list (must match number of chains)
    T0_list = [1e6, 5e4, 2e4]

    # We'll store final results here
    r1_values = []
    r2_values = []
    obj_values = []
    run_logs_list = []

    num_runs = 30

    for run_idx in range(num_runs):
        # Randomize initial guesses (x0_list) for each chain
        # All within the given bounds
        x0_list = []
        for _ in range(n_chains):
            r1 = random.uniform(my_bounds[0][0], my_bounds[0][1])
            r2 = random.uniform(my_bounds[1][0], my_bounds[1][1])
            x0_list.append([r1, r2])

        config = {
            "x0_list": x0_list,
            "T0_list": T0_list,
            "bounds": my_bounds,
            "max_iter": 5000,
            "swap_interval": 200,
            "adapt_interval": 100,
            "step_size_init": 0.02,
            "adapt_factor": 5,
            "no_improve_limit": 1000,
            "min_temp": 5e-3,
            "cooling_rate": 0.92,
            "seed": run_idx  # different seed per run
        }

        best_r, best_obj, logs = parallel_tempering_sa(config)
       
        # Record
        r1_values.append(best_r[0])
        r2_values.append(best_r[1])
        obj_values.append(best_obj)
        run_logs_list.append((best_r, best_obj, logs))

        print(f"Run {run_idx+1:2d} | Best radii = {best_r} | Objective = {best_obj:.3f} | Feasible = {my_is_feasible(best_r)}")

    # Compute statistics over the 30 runs
    mean_obj = np.mean(obj_values)
    std_obj  = np.std(obj_values)

    mean_r1 = np.mean(r1_values)
    std_r1  = np.std(r1_values)

    mean_r2 = np.mean(r2_values)
    std_r2  = np.std(r2_values)

    print("\n==========================")
    print("Summary of 30 runs:")
    print(f"Objective  => mean: {mean_obj:.3f}, std: {std_obj:.3f}")
    print(f"r1 (final) => mean: {mean_r1:.3f}, std: {std_r1:.3f}")
    print(f"r2 (final) => mean: {mean_r2:.3f}, std: {std_r2:.3f}")
    print("==========================")

    # ----------------------------------------------------
    # Plot a scatter of final solutions in (r1, r2) space,
    # colored by objective value, to see the distribution.
    # ----------------------------------------------------
    plt.figure()
    scatter = plt.scatter(r1_values, r2_values, c=obj_values)
    plt.colorbar(scatter, label='Final Objective')
    plt.xlabel('r1')
    plt.ylabel('r2')
    plt.title('Distribution of final solutions (30 runs)')
    plt.show()

    ## BEST RUN!
    # Find the index of the run with the minimum final objective:
    all_objectives = [item[1] for item in run_logs_list]  # gather all best_obj_i
    best_run_idx = np.argmin(all_objectives)

    # Extract that single best run’s data:
    best_r, best_obj, best_logs = run_logs_list[best_run_idx]
    print(f"The single best run is Run #{best_run_idx+1} with objective={best_obj:.3f}")

    fig, axs = plot_best_run_4subplots(my_bounds, best_r, best_obj, best_logs)
    plt.show()