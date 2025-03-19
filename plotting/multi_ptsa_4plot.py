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
        - no_improve_limit : 3000 (or larger / None)
        - min_temp : 1e-9
        - cooling_rate : 0.98
        - seed : 0 or None

    Make sure you have a "my_obj(r)" returning the objective (weight),
    and "my_is_feasible(r)" returning True/False, plus "my_nonlincon(r)" if you need it.
    """

    # Unpack config with fallback defaults
    x0_list = pt_config["x0_list"]
    bounds = pt_config["bounds"]
    T0_list = pt_config["T0_list"]

    max_iter = pt_config.get("max_iter", 10000)
    swap_int = pt_config.get("swap_interval", 50)
    adapt_int = pt_config.get("adapt_interval", 25)
    step_size_init = pt_config.get("step_size_init", 0.02)
    adapt_factor = pt_config.get("adapt_factor", 2)
    no_improve_limit = pt_config.get("no_improve_limit", 3000)
    min_temp = pt_config.get("min_temp", 1e-9)
    cooling_rate = pt_config.get("cooling_rate", 0.92)

    # For reproducibility
    random.seed(pt_config.get("seed", None))
    np.random.seed(pt_config.get("seed", None))

    n_chains = len(x0_list)
    assert len(T0_list) == n_chains, "Mismatch in chain count vs. T0_list length."

    # Helper to clamp a design within bounds
    def clamp(r):
        return [min(max(r_i, lb), ub)
                for r_i, (lb, ub) in zip(r, bounds)]

    # Prepare chain states
    chain_r = [clamp(x0_list[i]) for i in range(n_chains)]
    # If x0 is infeasible, you'll want to adjust or ensure it's feasible initially.
    # For safety, we do a quick check: if infeasible, we randomize inside the bounds until feasible.
    for i in range(n_chains):
        if not my_is_feasible(chain_r[i]):
            # Attempt random feasible
            # If it's tricky to find feasible, you might do more robust sampling or fallback
            for _ in range(10000):
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

    def get_neighbor(x, step):
        y = x[:]
        idx = random.randint(0, len(x)-1)
        y[idx] += random.uniform(-step, step)
        return clamp(y)

    for iteration in range(max_iter):
        logs["best_history"].append(best_r[:])

        # Each chain tries a move
        for c in range(n_chains):
            # Propose a neighbor
            candidate = get_neighbor(chain_r[c], chain_step[c])
            # If candidate is infeasible, *reject* outright
            if not my_is_feasible(candidate):
                # do nothing, we skip acceptance check
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

        # If we want to allow more time to explore, we can set this very high or remove it:
        if no_improve_limit is not None and (iteration - last_improve_iter) > no_improve_limit:
            # or just remove this condition if you don't want early stopping
            break

    return best_r, best_obj, logs



###############################################################################
# MULTI-RUN + 4-SUBPLOT COMPARISON
###############################################################################
def plot_multi_ptsa_4subplots(bounds, run_logs_list):
    """
    Plots:
      1) Feasibility region + final best path for each run
      2) Objective vs iteration for each run
      3) Acceptance rate vs iteration for each run
      4) Temperature vs iteration for each run

    run_logs_list is a list of (best_r, best_E, logs) from each run.
    We'll overlay them all on the same axes, using different colors/labels.
    """
    # Prepare feasibility region
    N = 100
    r1_vals = np.linspace(bounds[0][0], bounds[0][1], N)
    r2_vals = np.linspace(bounds[1][0], bounds[1][1], N)
    R1, R2 = np.meshgrid(r1_vals, r2_vals)
    feasible_mask = np.zeros_like(R1, dtype=bool)
    for i in range(N):
        for j in range(N):
            rr = [R1[i, j], R2[i, j]]
            feasible_mask[i, j] = np.all(my_nonlincon(rr) <= 0)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # === SUBPLOT 1: Feasibility + best path
    axs[0,0].contourf(R1, R2, feasible_mask,
                      levels=[-0.5,0.5], colors=["yellow","white"],
                      alpha=0.6)
    axs[0,0].set_xlabel("r1")
    axs[0,0].set_ylabel("r2")
    axs[0,0].set_title("Feasibility + PT-SA Paths")
    axs[0,0].grid(True)

    # We'll have a color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # For each run, we plot best_history on top
    for run_idx, (br, be, logs) in enumerate(run_logs_list):
        c = color_cycle[run_idx % len(color_cycle)]
        best_hist = np.array(logs["best_history"])
        axs[0,0].plot(best_hist[:,0], best_hist[:,1], '-o', color=c,
                      markersize=3, label=f"Run {run_idx+1}")

    axs[0,0].legend()

    # === SUBPLOT 2: Objective vs iteration
    for run_idx, (br, be, logs) in enumerate(run_logs_list):
        c = color_cycle[run_idx % len(color_cycle)]
        best_hist = logs["best_history"]
        obj_vals = [my_obj(x) for x in best_hist]
        axs[1,0].plot(obj_vals, color=c, label=f"Run {run_idx+1}")
    axs[1,0].set_xlabel("Iteration")
    axs[1,0].set_ylabel("Objective (Weight)")
    axs[1,0].set_title("Objective vs Iteration")
    axs[1,0].grid(True)

    # === SUBPLOT 3: Acceptance rate vs iteration
    for run_idx, (br, be, logs) in enumerate(run_logs_list):
        c = color_cycle[run_idx % len(color_cycle)]
        axs[0,1].plot(logs["accept_history"], color=c, label=f"Run {run_idx+1}")
    axs[0,1].set_xlabel("Iteration")
    axs[0,1].set_ylabel("Acceptance Rate")
    axs[0,1].set_title("Acceptance Rate vs Iteration")
    axs[0,1].grid(True)

    # === SUBPLOT 4: Temperature vs iteration
    for run_idx, (br, be, logs) in enumerate(run_logs_list):
        c = color_cycle[run_idx % len(color_cycle)]
        axs[1,1].plot(logs["temp_history"], color=c, label=f"Run {run_idx+1}")
    axs[1,1].set_xlabel("Iteration")
    axs[1,1].set_ylabel("Temperature")
    axs[1,1].set_title("Average Temp vs Iteration")
    axs[1,1].grid(True)

    # If you want a legend on each subplot, add it. We often just keep it on the first.
    axs[1,0].legend()
    axs[0,1].legend()
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()


###############################################################################
# MAIN DEMO: MULTIPLE PT-SA RUNS
###############################################################################
if __name__ == "__main__":
    # Problem bounds
    my_bounds = [(0.01, 0.5), (0.01, 0.5)]

    # We define multiple sets of chain initial conditions.
    # For instance, 3 chains per run, but we vary the chain starts from run to run.
    run_x0_sets = [
    # 1
    [
        [0.2, 0.3],
        [0.4, 0.05],
        [0.05, 0.4]
    ],
    # 2
    [
        [0.45, 0.45],
        [0.01, 0.01],
        [0.3, 0.2]
    ],
    # 3
    [
        [0.5, 0.2],
        [0.3, 0.5],
        [0.25, 0.25]
    ],
    # 4
    [
        [0.1, 0.45],
        [0.45, 0.1],
        [0.2, 0.2]
    ],
    # 5
    [
        [0.25, 0.45],
        [0.45, 0.35],
        [0.05, 0.3]
    ],
    # 6
    [
        [0.4, 0.4],
        [0.2, 0.2],
        [0.35, 0.25]
    ],
    # 7
    [
        [0.49, 0.49],
        [0.02, 0.02],
        [0.25, 0.33]
    ],
    # 8
    [
        [0.15, 0.15],
        [0.45, 0.25],
        [0.05, 0.45]
    ],
    # 9
    [
        [0.1, 0.5],
        [0.49, 0.05],
        [0.3, 0.3]
    ],
    # 10
    [
        [0.25, 0.05],
        [0.1, 0.4],
        [0.3, 0.45]
    ],
    # 11
    [
        [0.05, 0.05],
        [0.4, 0.4],
        [0.49, 0.3]
    ],
    # 12
    [
        [0.3, 0.1],
        [0.45, 0.4],
        [0.15, 0.25]
    ]
]


    # Common T0_list for each run
    T0_list = [1e6, 5e4, 2e4]

    # We'll store the results from each run in a list
    run_logs_list = []

    for run_idx, x0_list in enumerate(run_x0_sets):
        config = {
            "x0_list": x0_list,
            "T0_list": T0_list,
            "bounds": my_bounds,
            "max_iter": 10000,
            "swap_interval": 200,
            "adapt_interval": 100,
            "step_size_init": 0.02,
            "adapt_factor": 5,
            "no_improve_limit": 1000,
            "min_temp": 5e-3,
            "cooling_rate": 0.92,
            "seed": run_idx  # different seed per run
        }
        best_r, best_E, logs = parallel_tempering_sa(config)
        run_logs_list.append((best_r, best_E, logs))
        print(f"=== RUN {run_idx+1} ===")
        print("Best r:", best_r)
        print("Best E:", best_E)
        print("Objective =", my_obj(best_r))
        print("Feasible? =", my_is_feasible(best_r))
        print("Constraints:", my_nonlincon(best_r), "\n")

    # After all runs, do the big comparison plot
    plot_multi_ptsa_4subplots(my_bounds, run_logs_list)
