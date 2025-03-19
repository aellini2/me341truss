import numpy as np
import random
import math
import matplotlib.pyplot as plt

###############################################################################
#                           FINITE ELEMENT ANALYSIS                           #
###############################################################################
def FEA(r):
    """
    Simplified 2D finite element analysis for the 10-bar truss.
    r = [r1, r2]
    """
    length = 9.14  # m
    E = 200e9
    F = np.zeros(12)  # N
    F[3] = -1e7
    F[7] = -1e7

    # Node Table
    n = np.array([
        [2, 1], [2, 0], [1, 1], [1, 0],
        [0, 1], [0, 0]
    ]) * length

    # Element Connectivity (zero-based)
    ec = np.array([
        [3, 5], [1, 3], [4, 6], [2, 4], [3, 4],
        [1, 2], [4, 5], [3, 6], [2, 3], [1, 4]
    ]) - 1

    # Compute element lengths & direction cosines
    num_elems = 10
    length_e = np.zeros(num_elems)
    l = np.zeros(num_elems)
    m = np.zeros(num_elems)
    for i in range(num_elems):
        diff = n[ec[i, 1]] - n[ec[i, 0]]
        length_e[i] = np.linalg.norm(diff)
        l[i] = diff[0] / length_e[i]
        m[i] = diff[1] / length_e[i]

    # r1 for first 6 bars, r2 for last 4
    r_e = np.concatenate([np.full(6, r[0]), np.full(4, r[1])])
    A = np.pi * r_e**2

    # Global stiffness
    K = np.zeros((12, 12))
    for i in range(num_elems):
        k_small = (E * A[i] / length_e[i]) * np.array([
            [l[i]**2,    l[i]*m[i]],
            [l[i]*m[i],  m[i]**2]
        ])
        idx = [
            ec[i, 0]*2, ec[i, 0]*2 + 1,
            ec[i, 1]*2, ec[i, 1]*2 + 1
        ]
        k = np.zeros((12,12))
        k[np.ix_(idx[:2], idx[:2])] += k_small
        k[np.ix_(idx[2:], idx[2:])] += k_small
        k[np.ix_(idx[:2], idx[2:])] -= k_small
        k[np.ix_(idx[2:], idx[:2])] -= k_small
        K += k

    # Solve for displacements
    K_re = K[:8,:8]
    F_re = F[:8]
    Q_re = np.linalg.solve(K_re, F_re)
    Q = np.concatenate([Q_re, np.zeros(4)])

    # Compute stress
    stress = np.zeros(num_elems)
    for i in range(num_elems):
        idx = [
            ec[i, 0]*2, ec[i, 0]*2 + 1,
            ec[i, 1]*2, ec[i, 1]*2 + 1
        ]
        stress[i] = (E / length_e[i]) * np.dot(
            [-l[i], -m[i], l[i], m[i]],
            Q[idx]
        )

    return length_e, stress, r_e, Q


###############################################################################
#                           CONSTRAINT FUNCTION                               #
###############################################################################
def nonlincon(r):
    """
    Returns array of constraint values g_i(r).
    g_i>0 => constraint violation.
    (Buckling, yield stress, displacement constraints)
    """
    length_e, stress, r_e, Q = FEA(r)

    # Material
    E = 200e9
    Y = 250e6
    I = (np.pi/4)*r_e**4
    dis_max = 0.5

    num_elems = len(stress)
    g_buckling = np.zeros(num_elems)
    g_stress   = np.zeros(num_elems)

    F_internal = np.pi*(r_e**2)*stress
    for i in range(num_elems):
        if F_internal[i] < 0:  # compression
            F_comp = -F_internal[i]
            # Euler buckling: F_comp <= pi^2 E I / L^2
            g_buckling[i] = F_comp - (np.pi**2*E*I[i])/(length_e[i]**2)
        else:
            g_buckling[i] = -1e-6

        g_stress[i] = abs(stress[i]) - Y

    # Displacement at node #2 => sqrt(Q[2]^2 + Q[3]^2) <= dis_max
    disp_constraint = (Q[2]**2 + Q[3]**2) - dis_max**2

    return np.concatenate([g_buckling, g_stress, [disp_constraint]])


###############################################################################
#                           OBJECTIVE FUNCTION                                #
###############################################################################
def obj(r):
    """
    Weight: 6 bars at length=9.14 + 4 diagonals at length=9.14*sqrt(2).
    Cross-sectional area = pi*r^2; steel density=7860
    """
    length = 9.14
    density = 7860
    weight = (6 * np.pi * r[0]**2 * length
              + 4 * np.pi * r[1]**2 * length * np.sqrt(2)) * density
    return weight


###############################################################################
#                           FEASIBILITY CHECK                                 #
###############################################################################
def is_feasible(r):
    g = nonlincon(r)
    return np.all(g <= 0)


###############################################################################
#                     PARALLEL TEMPERING SA (Refactored)                      #
###############################################################################
def parallel_tempering_sa(pt_config):
    """
    Refactored parallel tempering SA that reads all parameters from 'pt_config'.

    pt_config must contain:
       - r0_list (list of lists)
       - T0_list (list of floats)
       - bounds
       - max_iter, swap_interval, adapt_interval
       - penalty_alpha_init, penalty_growth
       - step_size_init, adapt_factor
       - no_improve_limit, min_temp
    """

    # Unpack needed parameters
    r0_list = pt_config["r0_list"]
    T0_list = pt_config["T0_list"]
    bounds = pt_config["bounds"]
    max_iter = pt_config.get("max_iter", 3000)
    swap_interval = pt_config.get("swap_interval", 100)
    adapt_interval = pt_config.get("adapt_interval", 100)
    penalty_alpha_init = pt_config.get("penalty_alpha_init", 1e6)
    penalty_growth = pt_config.get("penalty_growth", 2.0)
    step_size_init = pt_config.get("step_size_init", 0.01)
    adapt_factor = pt_config.get("adapt_factor", 1.2)
    no_improve_limit = pt_config.get("no_improve_limit", 800)
    min_temp = pt_config.get("min_temp", 1e-9)

    n_chains = len(r0_list)
    assert n_chains == len(T0_list), "Length of r0_list and T0_list must match"

    # chain states
    chain_solutions = [list(r0_list[i]) for i in range(n_chains)]
    chain_energies  = [None]*n_chains
    chain_T         = [T0_list[i] for i in range(n_chains)]
    chain_step_size = [step_size_init]*n_chains
    chain_accept_count = [0]*n_chains

    # track entire path for each chain
    chain_paths = [[] for _ in range(n_chains)]  # chain_paths[c] = list of solutions

    # penalty alpha
    penalty_alpha = penalty_alpha_init

    def penalty_function(r):
        g = nonlincon(r)
        penalty_val = 0.0
        for val in g:
            if val > 0:
                penalty_val += val**2
        return penalty_alpha * penalty_val

    def energy(r):
        return obj(r) + penalty_function(r)

    def clamp(r):
        return [
            min(max(r[i], bounds[i][0]), bounds[i][1])
            for i in range(len(r))
        ]

    def get_neighbor(x, step):
        y = x[:]
        idx = random.randint(0, len(x)-1)
        y[idx] += random.uniform(-step, step)
        return clamp(y)

    # Initialize energies
    for i in range(n_chains):
        chain_solutions[i] = clamp(chain_solutions[i])
        chain_energies[i]  = energy(chain_solutions[i])
        chain_paths[i].append(chain_solutions[i][:])  # store first

    # Best across all chains
    best_solution = chain_solutions[0][:]
    best_energy   = chain_energies[0]
    for i in range(1, n_chains):
        if chain_energies[i] < best_energy:
            best_energy = chain_energies[i]
            best_solution = chain_solutions[i][:]

    # logs
    temp_history  = []
    accept_history= []
    best_history  = []  # track best solution so far each iteration

    last_improvement_iter = 0
    penalty_check_interval = 200
    feasible_in_interval = 0

    for iteration in range(max_iter):
        best_history.append(best_solution[:])  # record best so far

        # Each chain attempts a neighbor
        for c in range(n_chains):
            candidate = get_neighbor(chain_solutions[c], chain_step_size[c])
            candidate_energy = energy(candidate)

            if candidate_energy < chain_energies[c]:
                chain_solutions[c] = candidate
                chain_energies[c] = candidate_energy
                chain_accept_count[c] += 1
            else:
                deltaE = candidate_energy - chain_energies[c]
                T_use = max(chain_T[c], min_temp)
                if deltaE <= 0:
                    accept_prob = 1.0
                else:
                    accept_prob = math.exp(-deltaE / T_use)
                if random.random() < accept_prob:
                    chain_solutions[c] = candidate
                    chain_energies[c] = candidate_energy
                    chain_accept_count[c] += 1

            # update chain path
            chain_paths[c].append(chain_solutions[c][:])

            # update global best
            if chain_energies[c] < best_energy:
                best_energy = chain_energies[c]
                best_solution = chain_solutions[c][:]
                last_improvement_iter = iteration

        # swap interval
        if (iteration+1) % swap_interval == 0 and n_chains > 1:
            for c in range(n_chains-1):
                E1 = chain_energies[c]
                E2 = chain_energies[c+1]
                T1 = max(chain_T[c], min_temp)
                T2 = max(chain_T[c+1], min_temp)
                arg = (E1 - E2)*(1.0/T1 - 1.0/T2)
                if arg > 700:
                    swap_prob = 1.0
                elif arg < -700:
                    swap_prob = 0.0
                else:
                    swap_prob = math.exp(arg)

                if random.random() < swap_prob:
                    # swap solutions
                    chain_solutions[c], chain_solutions[c+1] = chain_solutions[c+1], chain_solutions[c]
                    chain_energies[c], chain_energies[c+1]   = chain_energies[c+1], chain_energies[c]
                    # also swap last point in chain_paths so they remain consistent
                    chain_paths[c][-1], chain_paths[c+1][-1] = chain_paths[c+1][-1], chain_paths[c][-1]

        # log average T, acceptance
        avg_T = sum(chain_T)/n_chains
        total_accept = sum(chain_accept_count)
        avg_accept_rate = total_accept / ((iteration+1)*n_chains)
        temp_history.append(avg_T)
        accept_history.append(avg_accept_rate)

        # penalty adaptation
        if is_feasible(best_solution):
            feasible_in_interval += 1

        if (iteration+1) % penalty_check_interval == 0:
            if feasible_in_interval == 0:
                penalty_alpha *= penalty_growth
            feasible_in_interval = 0

        # adapt step size
        if (iteration+1) % adapt_interval == 0:
            for c in range(n_chains):
                chain_ratio = chain_accept_count[c] / adapt_interval
                if chain_ratio > 0.5:
                    chain_step_size[c] *= adapt_factor
                elif chain_ratio < 0.2:
                    chain_step_size[c] /= adapt_factor
                chain_accept_count[c] = 0

        # cooling
        for c in range(n_chains):
            chain_T[c] = max(chain_T[c]*0.98, min_temp)

        # stopping
        if (iteration - last_improvement_iter) > pt_config["no_improve_limit"]:
            print(f"No improvement for {no_improve_limit} iterations. Stopping.")
            break

    logs = {
        "temp_history": temp_history,
        "accept_history": accept_history,
        "best_history": best_history,   # best solution so far at each iteration
        "chain_paths": chain_paths      # list of solutions for each chain
    }

    return best_solution, best_energy, logs

###############################################################################
#                       PLOT FEASIBILITY REGION + PATH                         #
###############################################################################
def plot_feasibility_and_path(bounds, chain_paths, best_history, N=150, title="Feasibility & Path"):
    """
    Plots the feasibility region (yellow=infeasible, white=feasible),
    plus the path of each chain, and the best so far in green.
    """
    import matplotlib.pyplot as plt

    # Evaluate feasibility on a grid
    r1_lin = np.linspace(bounds[0][0], bounds[0][1], N)
    r2_lin = np.linspace(bounds[1][0], bounds[1][1], N)
    R1, R2 = np.meshgrid(r1_lin, r2_lin)
    feasible_mask = np.zeros_like(R1, dtype=bool)
    for i in range(N):
        for j in range(N):
            test_r = [R1[i,j], R2[i,j]]
            g = nonlincon(test_r)
            feasible_mask[i,j] = np.all(g <= 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(7,6))
    ax.contourf(R1, R2, feasible_mask, levels=[-0.5,0.5], colors=["yellow","white"], alpha=0.6)

    # Plot each chain's path in different color
    colors = ["blue", "red", "orange", "purple", "cyan", "magenta"]
    for c_idx, path_c in enumerate(chain_paths):
        path_c = np.array(path_c)
        col = colors[c_idx % len(colors)]
        ax.plot(path_c[:, 0], path_c[:, 1], '-o', color=col, label=f"Chain {c_idx}", markersize=4, alpha=0.7)

    # Plot best history in green
    best_hist = np.array(best_history)
    ax.plot(best_hist[:, 0], best_hist[:, 1], '-o', color='green', linewidth=2, 
            markersize=4, label="Best so far")

    # Mark final best with a bigger green dot
    ax.plot(best_hist[-1,0], best_hist[-1,1], 'o', color='green', markersize=10, label="Final Best")

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel("r1")
    ax.set_ylabel("r2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()


###############################################################################
#                             MAIN (ONE INIT)                                 #
###############################################################################
if __name__ == "__main__":
    # 1) Initialize the configuration once
    
    # "r0_list": [
    #     [0.2, 0.3],
    #     [0.4, 0.45],
    #     [0.1, 0.2]
    # ],
    
    pt_config = {
        "r0_list": [
            [0.3, 0.8],
            [0.6, 0.8],
            [0.9, 0.8]
        ],
        "T0_list": [1e4, 5e3, 1e3],
        "bounds": [(0.01, 1), (0.01, 1)],
        "max_iter": 3000,
        "swap_interval": 25,
        "adapt_interval": 100,
        "penalty_alpha_init": 1e6,
        "penalty_growth": 2.0,
        "step_size_init": 0.02,
        "adapt_factor": 1.2,
        "no_improve_limit": 600,
        "min_temp": 1e-9
    }

    # 2) Run the parallel tempering SA with that config
    best_sol, best_val, logs = parallel_tempering_sa(pt_config)

    print("\nPARALLEL TEMPERING SA COMPLETE.")
    print("Best solution found:", best_sol)
    print("Best objective + penalty value:", best_val)
    # Check feasibility
    constr = nonlincon(best_sol)
    feasible = np.all(constr <= 0)
    print("Objective alone:", obj(best_sol))
    print("Constraints (should be <= 0):", constr)
    print("Feasible?", feasible)

    # 3) Additional Plots
    #   (A) Temperature vs Iteration
    plt.figure()
    plt.plot(logs["temp_history"], label="Avg Temperature")
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Average Temperature vs Iteration")
    plt.grid(True)
    plt.legend()
    plt.show()

    #   (B) Acceptance Rate vs Iteration
    plt.figure()
    plt.plot(logs["accept_history"], label="Avg Acceptance Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Acceptance Rate (0-1)")
    plt.title("Acceptance Rate vs Iteration (All Chains)")
    plt.grid(True)
    plt.legend()
    plt.show()

    #   (C) Feasibility Region + Paths (all chains + best)
    plot_feasibility_and_path(
        pt_config["bounds"],
        logs["chain_paths"],    # each chain's path
        logs["best_history"],   # best so far each iteration
        N=150,
        title="Feasibility & Parallel Tempering Paths"
    )
