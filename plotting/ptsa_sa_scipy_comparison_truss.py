import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

init_r1 = .2
init_r2 = .47

###############################################################################
# FINITE ELEMENT ANALYSIS (Truss) FOR A GIVEN (r1, r2)
###############################################################################
def truss_FEA(r):
    """
    Given r = [r1, r2], runs a 2D truss FEA with 10 elements.
    Returns:
      length_e: array of element lengths (size 10)
      stress:   array of element stresses (size 10)
      r_e:      array of element radii (size 10) => first 6 use r1, last 4 use r2
      Q:        array of nodal displacements (size 12)
    """

    length = 9.14  # (m)
    E = 200e9      # (Pa)

    # External forces (N)
    F = np.zeros(12)
    F[3] = -1e7   # Node #2 in -y direction
    F[7] = -1e7   # Node #4 in -y direction

    # Node coordinates (6 nodes total)
    #   n[i] = [x, y]
    #   scaled by length = 9.14
    n = np.array([
        [2, 1], [2, 0], [1, 1], [1, 0], [0, 1], [0, 0]
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

    # Compute element lengths & direction cosines
    for i in range(num_elements):
        diff = n[ec[i, 1]] - n[ec[i, 0]]
        length_e[i] = np.linalg.norm(diff)
        l[i] = diff[0] / length_e[i]
        m[i] = diff[1] / length_e[i]

    # Radii assignment: first 6 elements use r[0], last 4 use r[1]
    # => r_e is a length-10 array: r[0] repeated 6 times, r[1] repeated 4 times
    r_e = np.concatenate([np.full(6, r[0]), np.full(4, r[1])])
    A = np.pi * r_e**2  # cross-sectional area

    # Construct global stiffness matrix
    K = np.zeros((12, 12))
    for i in range(num_elements):
        k_small = (E * A[i] / length_e[i]) * np.array([
            [l[i]**2, l[i]*m[i]],
            [l[i]*m[i], m[i]**2]
        ])
        # Build into the global K
        # Each node i has 2 DOFs: (x, y)
        # ec[i,0] -> DOFs (..., ...), ec[i,1] -> DOFs (..., ...)
        idx = [
            ec[i, 0]*2, ec[i, 0]*2 + 1,
            ec[i, 1]*2, ec[i, 1]*2 + 1
        ]
        # Local assembly
        block = np.zeros((4, 4))
        block[:2, :2] = k_small
        block[2:, 2:] = k_small
        block[:2, 2:] = -k_small
        block[2:, :2] = -k_small

        # Expand block into K
        for ii in range(4):
            for jj in range(4):
                K[idx[ii], idx[jj]] += block[ii, jj]

    # DOF reduction => first 8 DOFs are unknown, last 4 are fixed
    K_reduced = K[:8, :8]
    F_reduced = F[:8]

    # Solve for unknown DOFs
    Q_reduced = np.linalg.solve(K_reduced, F_reduced)
    Q = np.concatenate([Q_reduced, np.zeros(4)])  # total 12 DOFs

    # Compute stress in each element
    stress = np.zeros(num_elements)
    for i in range(num_elements):
        # Vector of relevant DOFs
        idx = [
            ec[i, 0]*2, ec[i, 0]*2 + 1,
            ec[i, 1]*2, ec[i, 1]*2 + 1
        ]
        # Stress = (E/length_e) * [-l, -m, l, m]* Q_elem
        # Q_elem = [ux_i, uy_i, ux_j, uy_j]
        vec = np.array([-l[i], -m[i], l[i], m[i]])
        stress[i] = (E / length_e[i]) * np.dot(vec, Q[idx])

    # Reaction forces (not strictly needed for the constraints, but included for completeness)
    # last 4 DOFs = fixed => reaction = K[8:, :] * Q
    R = np.dot(K[8:, :], Q)

    return length_e, stress, r_e, Q


###############################################################################
# PROBLEM FUNCTIONS: OBJECTIVE AND CONSTRAINTS
###############################################################################
def my_obj(r):
    """
    Weight of the truss:
    - 6 elements of length 9.14, radius r1
    - 4 elements of length 9.14*sqrt(2), radius r2
    - material density = 7860
    """
    length = 9.14
    density = 7860

    # 6 elements of radius r[0], 4 of radius r[1], approximate lengths
    # 6 of length=9.14, 4 of length=9.14*sqrt(2)
    weight = (6 * np.pi * r[0]**2 * length
              + 4 * np.pi * r[1]**2 * length * np.sqrt(2)) * density
    return weight

def my_nonlincon(r):
    """
    Returns array g of constraints, each <= 0 if feasible.
      - 10 buckling constraints (one per element)
      - 10 stress constraints (|sigma| - Yield)
      - 1 displacement constraint at node #2 (Q[2], Q[3])

    If g[i] > 0 => that constraint is violated.
    """
    length_e, stress, r_e, Q = truss_FEA(r)

    E = 200e9
    Y = 250e6       # yield stress
    I = (np.pi/4) * r_e**4
    dis_max = 0.02  # maximum displacement

    num_elems = len(stress)
    g_buckling = np.zeros(num_elems)
    g_stress   = np.zeros(num_elems)

    # Axial force in each element:
    #   F_internal = A_i * stress_i = pi * r_e[i]^2 * stress[i]
    F_internal = np.pi * (r_e**2) * stress

    for i in range(num_elems):
        # If in compression (F_internal < 0), check buckling
        if F_internal[i] < 0:
            # we want F_comp <= pi^2 * E * I / L^2
            F_comp = -F_internal[i]
            crit_buckling = (np.pi**2 * E * I[i]) / (length_e[i]**2)
            g_buckling[i] = F_comp - crit_buckling
        else:
            # not in compression => not violating buckling
            g_buckling[i] = -1e-6

        # Stress limit: |sigma| <= Y
        g_stress[i] = abs(stress[i]) - Y

    # Displacement constraint at node #2 => DOFs #2,3
    #   sqrt(Q[2]^2 + Q[3]^2) <= dis_max
    #   => Q[2]^2 + Q[3]^2 - dis_max^2 <= 0
    disp_constraint = (Q[2]**2 + Q[3]**2) - (dis_max**2)

    # Combine all constraints:
    #   g_buckling (10), g_stress (10), disp_constraint (1)
    g = np.concatenate([g_buckling, g_stress, [disp_constraint]])
    return g

def my_is_feasible(r):
    """
    Return True if all constraints are <= 0.
    """
    g = my_nonlincon(r)
    return np.all(g <= 0)


###############################################################################
# PENALTY + ENERGY
###############################################################################
def penalty_function(r, alpha=1e16):
    """
    Quadratic penalty for constraints:
      sum_{g_i>0} (g_i^2) * alpha
    """
    g = my_nonlincon(r)
    viol = g[g > 0]
    return alpha * np.sum(viol**2) if len(viol) else 0.0

def energy(r):
    """
    'Energy' = objective + penalty
    """
    return my_obj(r) + penalty_function(r)

def clamp(r, bounds):
    """
    Enforce bounds on each component of r.
    """
    r_c = []
    for val, (low, high) in zip(r, bounds):
        r_c.append(min(max(val, low), high))
    return r_c


###############################################################################
# SINGLE-CHAIN SIMULATED ANNEALING
###############################################################################
def single_chain_sa(sa_config):
    """
    Basic single-chain Simulated Annealing with constraints by penalty.
    Returns (best_r, best_energy, logs).
    logs = {
      'temp_history': [...],
      'accept_history': [...],
      'best_history': [...],    # the best solution at each iteration
      'path': [...],            # the path of the current solution
    }
    """
    # Unpack config
    r0 = sa_config["r0"]
    T0 = sa_config.get("T0", 1e5)
    bounds = sa_config["bounds"]
    max_iter = sa_config.get("max_iter", 2000)
    min_temp = sa_config.get("min_temp", 1e-2)
    cooling_rate = sa_config.get("cooling_rate", 0.98)
    step_size = sa_config.get("step_size", 0.01)
    no_improve_limit = sa_config.get("no_improve_limit", 300)

    random.seed(sa_config.get("seed", 0))
    np.random.seed(sa_config.get("seed", 0))

    def get_neighbor(x, step):
        y = x[:]
        idx = random.randint(0, len(x)-1)
        y[idx] += random.uniform(-step, step)
        return clamp(y, bounds)

    current = clamp(r0, bounds)
    current_E = energy(current)
    best = current[:]
    best_E = current_E

    temp_history = []
    accept_history = []
    best_history = []
    path = []

    temperature = T0
    accept_count = 0
    total_count = 0
    last_improve_iter = 0

    for iteration in range(max_iter):
        path.append(current[:])
        best_history.append(best[:])
        temp_history.append(temperature)
        if total_count == 0:
            accept_history.append(1.0)  # first acceptance rate is trivial
        else:
            accept_history.append(accept_count / total_count)

        # Stopping if temperature is too low
        if temperature < min_temp:
            break

        # Generate neighbor
        candidate = get_neighbor(current, step_size)
        candidate_E = energy(candidate)

        # Accept or reject
        if candidate_E < current_E:
            current = candidate
            current_E = candidate_E
            accept_count += 1
        else:
            deltaE = candidate_E - current_E
            accept_prob = math.exp(-deltaE / temperature)
            if random.random() < accept_prob:
                current = candidate
                current_E = candidate_E
                accept_count += 1

        total_count += 1

        # Update best
        if current_E < best_E:
            best_E = current_E
            best = current[:]
            last_improve_iter = iteration

        # Cooling
        temperature *= cooling_rate

        # Stop if no improvement for a long time
        if (iteration - last_improve_iter) > no_improve_limit:
            break

    logs = {
        "temp_history": temp_history,
        "accept_history": accept_history,
        "best_history": best_history,
        "path": path
    }
    return best, best_E, logs


###############################################################################
# PARALLEL TEMPERING SIMULATED ANNEALING
###############################################################################
def parallel_tempering_sa(pt_config):
    """
    Parallel Tempering SA reading parameters from 'pt_config'.
    We reuse the same approach as the "clean" example:
      - multiple chains
      - swap interval
      - adapt interval
      - penalty approach for constraints
    Returns (best_r, best_energy, logs).
    logs keys:
      - temp_history
      - accept_history
      - best_history
      - chain_paths
    """
    x0_list = pt_config["x0_list"]
    T0_list = pt_config["T0_list"]
    bounds  = pt_config["bounds"]
    max_iter = pt_config.get("max_iter", 3000)
    swap_int = pt_config.get("swap_interval", 100)
    adapt_int = pt_config.get("adapt_interval", 100)
    step_size_init = pt_config.get("step_size_init", 0.01)
    adapt_factor = pt_config.get("adapt_factor", 1.2)
    no_improve_limit = pt_config.get("no_improve_limit", 800)
    min_temp = pt_config.get("min_temp", 1e-9)
    cooling_rate = pt_config.get("cooling_rate", 0.98)

    random.seed(pt_config.get("seed", 0))
    np.random.seed(pt_config.get("seed", 0))

    n_chains = len(x0_list)
    assert n_chains == len(T0_list), "Mismatched chain count vs. temperature count!"

    # Initialize per-chain
    chain_r = [clamp(x0_list[i], bounds) for i in range(n_chains)]
    chain_E = [energy(chain_r[i]) for i in range(n_chains)]
    chain_T = [T0_list[i] for i in range(n_chains)]
    chain_step = [step_size_init]*n_chains
    chain_accept_count = [0]*n_chains

    chain_paths = [[] for _ in range(n_chains)]
    for i in range(n_chains):
        chain_paths[i].append(chain_r[i][:])

    # Global best
    best_idx = np.argmin(chain_E)
    best_r = chain_r[best_idx][:]
    best_E = chain_E[best_idx]

    # Logs
    temp_history = []
    accept_history = []
    best_history = []
    last_improve_iter = 0

    def get_neighbor(x, step):
        y = x[:]
        idx = random.randint(0, len(x)-1)
        y[idx] += random.uniform(-step, step)
        return clamp(y, bounds)

    for iteration in range(max_iter):
        best_history.append(best_r[:])

        # Each chain tries a move
        for c in range(n_chains):
            candidate = get_neighbor(chain_r[c], chain_step[c])
            cand_E = energy(candidate)
            if cand_E < chain_E[c]:
                chain_r[c] = candidate
                chain_E[c] = cand_E
                chain_accept_count[c] += 1
            else:
                deltaE = cand_E - chain_E[c]
                T_use = max(chain_T[c], min_temp)
                accept_prob = math.exp(-deltaE / T_use)
                if random.random() < accept_prob:
                    chain_r[c] = candidate
                    chain_E[c] = cand_E
                    chain_accept_count[c] += 1

            # Update chain path
            chain_paths[c].append(chain_r[c][:])

            # Update global best
            if chain_E[c] < best_E:
                best_E = chain_E[c]
                best_r = chain_r[c][:]
                last_improve_iter = iteration

        # Swap attempt
        if (iteration+1) % swap_int == 0 and n_chains > 1:
            for c in range(n_chains - 1):
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
                    chain_r[c], chain_r[c+1] = chain_r[c+1], chain_r[c]
                    chain_E[c], chain_E[c+1] = chain_E[c+1], chain_E[c]
                    # Also swap last appended path point
                    chain_paths[c][-1], chain_paths[c+1][-1] = chain_paths[c+1][-1], chain_paths[c][-1]

        # Logging
        avg_T = sum(chain_T)/n_chains
        total_accept = sum(chain_accept_count)
        avg_accept_rate = total_accept / ((iteration+1)*n_chains)
        temp_history.append(avg_T)
        accept_history.append(avg_accept_rate)

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
            chain_T[c] = max(chain_T[c] * cooling_rate, min_temp)

        # Stopping if no improvement
        if (iteration - last_improve_iter) > no_improve_limit:
            break

    logs = {
        "temp_history": temp_history,
        "accept_history": accept_history,
        "best_history": best_history,
        "chain_paths": chain_paths
    }
    return best_r, best_E, logs


###############################################################################
# SCIPY OPTIMIZATION (trust-constr)
###############################################################################
def scipy_solve(r0, bounds):
    """
    Solve the same problem using scipy's 'trust-constr' with NonlinearConstraint.
    We want: my_nonlincon(r) <= 0.  
    => NonlinearConstraint( func, -inf, 0 ).
    """
    def cons_f(x):
        return my_nonlincon(x)

    # Create the constraint object
    nlc = NonlinearConstraint(cons_f, -np.inf, 0.0)

    # Bounds must be in the form: [ (lb1, ub1), (lb2, ub2) ]
    # trust-constr can accept these directly as 'bounds'.
    res = minimize(
        fun=my_obj,
        x0=r0,
        method='trust-constr',
        constraints=[nlc],
        bounds=bounds,
        options={'maxiter': 1000, 'verbose': 0}
    )
    return res


###############################################################################
# PLOTTING: FEASIBILITY + PATH, ETC.
###############################################################################
def plot_feasibility_and_paths(
    bounds,
    pt_logs=None,
    sa_logs=None,
    scipy_sol=None,
    gridN=100
):
    """
    Plots the 2D feasibility region (r1 vs r2) + any of these:
      - PT SA best path
      - Single-chain SA best path
      - SciPy final solution
    """
    # Build a mesh over [r1_min, r1_max] x [r2_min, r2_max]
    r1_vals = np.linspace(bounds[0][0], bounds[0][1], gridN)
    r2_vals = np.linspace(bounds[1][0], bounds[1][1], gridN)
    R1, R2 = np.meshgrid(r1_vals, r2_vals)
    feasible_mask = np.zeros_like(R1, dtype=bool)

    for i in range(gridN):
        for j in range(gridN):
            rr = [R1[i, j], R2[i, j]]
            g = my_nonlincon(rr)
            feasible_mask[i, j] = np.all(g <= 0)

    plt.figure(figsize=(7,6))
    # contourf to show feasible (white) vs infeasible (yellow)
    plt.contourf(R1, R2, feasible_mask, levels=[-0.5, 0.5],
                 colors=["yellow","white"], alpha=0.6)
    plt.xlabel("r1")
    plt.ylabel("r2")
    plt.title("Feasibility Region + Paths")
    plt.grid(True)

    # If PT logs given, plot best-history
    if pt_logs is not None:
        pt_best_hist = np.array(pt_logs["best_history"])
        plt.plot(pt_best_hist[:,0], pt_best_hist[:,1],
                 '-o', markersize=3, label="PT SA (best path)")

    # If single-chain logs given, plot best-history
    if sa_logs is not None:
        sa_best_hist = np.array(sa_logs["best_history"])
        plt.plot(sa_best_hist[:,0], sa_best_hist[:,1],
                 '-o', markersize=3, label="Single SA (best path)")

    # If scipy solution is given
    if scipy_sol is not None:
        plt.plot(scipy_sol[0], scipy_sol[1],
                 'r*', markersize=12, label="Scipy final")

    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.legend()
    plt.show()

def plot_4_subplots(bounds, pt_logs=None, sa_logs=None, scipy_res=None):
    """
    Makes 4 subplots:
      1) Feasibility region + final paths
      2) Objective vs iteration
      3) Acceptance rate vs iteration (SA only)
      4) Temperature vs iteration (SA only)
    Overlays PT SA and single-chain SA, plus the final point from SciPy.
    """

    # We'll gather data carefully. SciPy doesn't have acceptance rate or temperature.
    fig, axs = plt.subplots(2, 2, figsize=(14,10))

    # SUBPLOT 1: Feasibility + final paths
    gridN = 100
    r1_vals = np.linspace(bounds[0][0], bounds[0][1], gridN)
    r2_vals = np.linspace(bounds[1][0], bounds[1][1], gridN)
    R1, R2 = np.meshgrid(r1_vals, r2_vals)
    feasible_mask = np.zeros_like(R1, dtype=bool)
    for i in range(gridN):
        for j in range(gridN):
            rr = [R1[i, j], R2[i, j]]
            feasible_mask[i, j] = np.all(my_nonlincon(rr) <= 0)
    axs[0,0].contourf(R1, R2, feasible_mask, levels=[-0.5,0.5],
                      colors=["yellow","white"], alpha=0.6)
    axs[0,0].set_xlabel("r1")
    axs[0,0].set_ylabel("r2")
    axs[0,0].set_title("Feasibility + Paths")
    axs[0,0].grid(True)

    # PT SA path
    if pt_logs:
        pt_best_hist = np.array(pt_logs["best_history"])
        axs[0,0].plot(pt_best_hist[:,0], pt_best_hist[:,1], '-o',
                      markersize=3, label="PT SA path")

    # Single SA path
    if sa_logs:
        sa_best_hist = np.array(sa_logs["best_history"])
        axs[0,0].plot(sa_best_hist[:,0], sa_best_hist[:,1], '-o',
                      markersize=3, label="Single SA path")

    # SciPy final
    if scipy_res is not None:
        axs[0,0].plot(scipy_res.x[0], scipy_res.x[1], 'r*', markersize=12,
                      label="Scipy final")
    axs[0,0].legend()

    # SUBPLOT 2: Objective vs iteration
    if pt_logs:
        pt_obj = [my_obj(x) for x in pt_logs["best_history"]]
        axs[1,0].plot(pt_obj, label="PT SA obj")
    if sa_logs:
        sa_obj = [my_obj(x) for x in sa_logs["best_history"]]
        axs[1,0].plot(sa_obj, label="Single SA obj")
    # SciPy => just final or we can skip
    if scipy_res is not None:
        axs[1,0].axhline(scipy_res.fun, color='r', linestyle='--',
                         label="Scipy final obj")
    axs[1,0].set_xlabel("Iteration")
    axs[1,0].set_ylabel("Objective (Weight)")
    axs[1,0].set_title("Objective vs Iteration")
    axs[1,0].grid(True)
    axs[1,0].legend()

    # SUBPLOT 3: Acceptance rate vs iteration
    if pt_logs:
        axs[0,1].plot(pt_logs["accept_history"], label="PT SA accept")
    if sa_logs:
        axs[0,1].plot(sa_logs["accept_history"], label="Single SA accept")
    axs[0,1].set_xlabel("Iteration")
    axs[0,1].set_ylabel("Acceptance Rate")
    axs[0,1].set_title("Acceptance Rate (SA)")
    axs[0,1].grid(True)
    axs[0,1].legend()

    # SUBPLOT 4: Temperature vs iteration
    if pt_logs:
        axs[1,1].plot(pt_logs["temp_history"], label="PT SA Temp")
    if sa_logs:
        axs[1,1].plot(sa_logs["temp_history"], label="Single SA Temp")
    axs[1,1].set_xlabel("Iteration")
    axs[1,1].set_ylabel("Temperature")
    axs[1,1].set_title("Temperature (SA)")
    axs[1,1].grid(True)
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()


###############################################################################
# MAIN: DEMO COMPARISON
###############################################################################
if __name__ == "__main__":

    # Bounds for [r1, r2]
    my_bounds = [(0.01, 0.5), (0.01, 0.5)]

    # 1) PARALLEL TEMPERING SA
    pt_config = {
        "x0_list": [
            [init_r1, init_r2],
            [0.4, 0.05],
            [0.05, 0.4]
        ],
        "T0_list": [1e5, 5e4, 2e4],
        "bounds": my_bounds,
        "max_iter": 3000,
        "swap_interval": 300,
        "adapt_interval": 100,
        "step_size_init": 0.02,
        "adapt_factor": 1.2,
        "no_improve_limit": 500,
        "min_temp": 1e-3,
        "cooling_rate": 0.95,
        "seed": 0
    }
    pt_best_r, pt_best_E, pt_logs = parallel_tempering_sa(pt_config)

    # 2) SINGLE-CHAIN SA
    sa_config = {
        "r0": [init_r1, init_r2],
        "T0": 1e5,
        "bounds": my_bounds,
        "max_iter": 2000,
        "min_temp": 1e-3,
        "cooling_rate": 0.95,
        "step_size": 0.02,
        "no_improve_limit": 500,
        "seed": 0
    }
    sa_best_r, sa_best_E, sa_logs = single_chain_sa(sa_config)

    # 3) SCIPY "trust-constr"
    scipy_x0 = [init_r1, init_r2]
    scipy_res = scipy_solve(scipy_x0, my_bounds)

    # Print results
    print("============ RESULTS ============")
    print("\nParallel Tempering SA:")
    print(f" Best r = {pt_best_r}, Energy = {pt_best_E}")
    print(f" Objective = {my_obj(pt_best_r):.3f}, Feasible? {my_is_feasible(pt_best_r)}")
    print(f" Constraints = {my_nonlincon(pt_best_r)}")

    print("\nSingle-Chain SA:")
    print(f" Best r = {sa_best_r}, Energy = {sa_best_E}")
    print(f" Objective = {my_obj(sa_best_r):.3f}, Feasible? {my_is_feasible(sa_best_r)}")
    print(f" Constraints = {my_nonlincon(sa_best_r)}")

    print("\nScipy trust-constr:")
    print(" success =", scipy_res.success, ", status =", scipy_res.message)
    print(" x =", scipy_res.x)
    print(" Objective =", scipy_res.fun)
    print(" Feasible? =", my_is_feasible(scipy_res.x))
    print(" Constraints =", my_nonlincon(scipy_res.x))

    # Plot side-by-side comparison (4 subplots)
    plot_4_subplots(bounds=my_bounds,
                    pt_logs=pt_logs,
                    sa_logs=sa_logs,
                    scipy_res=scipy_res)

    # Also show a single large feasibility plot with final paths
    plot_feasibility_and_paths(
        bounds=my_bounds,
        pt_logs=pt_logs,
        sa_logs=sa_logs,
        scipy_sol=scipy_res.x,
        gridN=150
    )
