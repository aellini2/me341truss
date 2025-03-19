import numpy as np
import random
import math
import matplotlib.pyplot as plt

class TenBarTrussOptimizer:
    """
    optimizes the design of a ten-bar truss using simulated annealing.
    """

    def __init__(self, initial_solution, initial_temperature, cooling_rate, min_temperature, max_iterations):
        """
        initializes the optimizer with initial parameters.

        Args:
            initial_solution (list): initial design variables (radii).
            initial_temperature (float): starting temperature for simulated annealing.
            cooling_rate (float): rate at which the temperature decreases.
            min_temperature (float): minimum temperature to stop the algorithm.
            max_iterations (int): maximum number of iterations.
        """
        self.current_solution = initial_solution
        self.best_solution = initial_solution.copy()
        self.current_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.current_cost = self.objective_function(self.current_solution)
        self.best_cost = self.current_cost
        self.cost_history = []
        self.solution_history = []
        self.temperature_history = []

    def objective_function(self, solution):
        """
        calculates the objective function (weight) for a given solution.

        Args:
            solution (list): design variables (radii).

        Returns:
            float: weight of the truss.
        """
        length = 9.14
        density = 7860
        weight = (6 * np.pi * solution[0]**2 * length + 4 * np.pi * solution[1]**2 * length * np.sqrt(2)) * density
        return weight

    def non_linear_constraints(self, solution):
        """
        calculates the nonlinear constraints for a given solution.

        Args:
            solution (list): design variables (radii).

        Returns:
            numpy.ndarray: array of constraint values.
        """
        length_e, stress, r_d, Q = self.fea(solution)
        Y = 250e6  # Yield stress in Pa
        dis_max = 0.02  # Maximum displacement in meters
        I = np.pi / 4 * r_d**4  # Moment of inertia for each element
        E = 200e9  # Young's modulus in Pa

        # inequality constraints
        g_1 = np.zeros(10)
        g_2 = np.zeros(10)
        F = np.zeros(10)

        for i in range(10):
            F[i] = np.pi * r_d[i]**2 * stress[i]
            g_1[i] = -(-F[i] - (np.pi**2 * E * I[i]) / length_e[i]**2)
            g_2[i] = -(abs(stress[i]) - Y)

        g_3 = -(Q[2]**2 + Q[3]**2 - dis_max**2)
        g = np.concatenate([g_1, g_2, [g_3]])
        return g

    def fea(self, r):
        """
        performs finite element analysis on the truss.

        Args:
            r (list): design variables (radii).

        Returns:
            tuple: element lengths, stresses, radii, and displacements.
        """
        length = 9.14  # unit: m
        E = 200e9
        F = np.zeros(12)  # unit: N
        F[3] = -1e7
        F[7] = -1e7

        # Node Table
        n = np.array([
            [2, 1], [2, 0], [1, 1], [1, 0], [0, 1], [0, 0]
        ]) * length

        # Element Connectivity Table
        ec = np.array([
            [3, 5], [1, 3], [4, 6], [2, 4], [3, 4], 
            [1, 2], [4, 5], [3, 6], [2, 3], [1, 4]
        ]) - 1  # Convert to zero-based indexing

        # Initialize element properties
        num_elements = 10
        length_e = np.zeros(num_elements)
        l = np.zeros(num_elements)
        m = np.zeros(num_elements)

        # Compute element lengths and direction cosines
        for i in range(num_elements):
            diff = n[ec[i, 1]] - n[ec[i, 0]]
            length_e[i] = np.linalg.norm(diff)
            l[i] = diff[0] / length_e[i]
            m[i] = diff[1] / length_e[i]

        # Element radius and area
        r = np.concatenate([np.full(6, r[0]), np.full(4, r[1])])
        A = np.pi * r**2

        # Global stiffness matrix
        K = np.zeros((12, 12))
        for i in range(num_elements):
            k_small = (E * A[i] / length_e[i]) * np.array([
                [l[i]**2, l[i]*m[i]], 
                [l[i]*m[i], m[i]**2]
            ])
            k = np.zeros((12, 12))
            idx = np.array([
                ec[i, 0] * 2, ec[i, 0] * 2 + 1, 
                ec[i, 1] * 2, ec[i, 1] * 2 + 1
            ])
            k[np.ix_(idx[:2], idx[:2])] += k_small
            k[np.ix_(idx[2:], idx[2:])] += k_small
            k[np.ix_(idx[:2], idx[2:])] -= k_small
            k[np.ix_(idx[2:], idx[:2])] -= k_small
            K += k

        # DOF Reduction
        K_re = K[:8, :8]
        F_re = F[:8]

        # Displacement Calculation
        Q_re = np.linalg.solve(K_re, F_re)
        Q = np.concatenate([Q_re, np.zeros(4)])

        # Stress Calculation
        stress = np.zeros(num_elements)
        for i in range(num_elements):
            idx = np.array([
                ec[i, 0] * 2, ec[i, 0] * 2 + 1, 
                ec[i, 1] * 2, ec[i, 1] * 2 + 1
            ])
            stress[i] = (E / length_e[i]) * np.dot(
                np.array([-l[i], -m[i], l[i], m[i]]), Q[idx]
            )

        # Reaction Force Calculation
        K_R = K[8:, :]
        R = np.dot(K_R, Q)
        R = np.concatenate([np.zeros(8), R])

        return length_e, stress, r, Q

    def generate_neighbor(self, solution, step_size=0.1):
        """
        generates a neighboring solution by randomly perturbing the current solution.

        Args:
            solution (list): current solution.
            step_size (float): maximum perturbation size.

        Returns:
            list: neighboring solution.
        """
        neighbor = solution.copy()
        index = random.randint(0, len(neighbor) - 1)
        neighbor[index] += random.uniform(-step_size, step_size)
        neighbor[index] = max(0.001, neighbor[index]) # ensure radius is positive and non zero
        return neighbor

    def acceptance_probability(self, current_cost, new_cost, temperature):
        """
        calculates the acceptance probability for a new solution.

        Args:
            current_cost (float): cost of the current solution.
            new_cost (float): cost of the new solution.
            temperature (float): current temperature.

        Returns:
            float: acceptance probability.
        """
        if new_cost < current_cost:
            return 1.0
        else:
            return math.exp((current_cost))
        
if __name__ == "__main__":
    # Adjusted parameters
    initial_solution = [0.15, 0.15] # new initial solution.
    initial_temperature = 500.0 # increased initial temperature.
    cooling_rate = 0.98 # adjusted cooling rate.
    min_temperature = 1.0
    max_iterations = 2000 # increased max iterations.
    step_size = 0.02 # increased step size.

    # Create an instance of the optimizer
    optimizer = TenBarTrussOptimizer(initial_solution, initial_temperature, cooling_rate, min_temperature, max_iterations)

    # Run the simulated annealing algorithm
    while optimizer.current_temperature > optimizer.min_temperature and optimizer.max_iterations > 0:
        new_solution = optimizer.generate_neighbor(optimizer.current_solution, step_size=step_size)
        new_cost = optimizer.objective_function(new_solution)

        #Check feasibility
        if np.all(optimizer.non_linear_constraints(new_solution) <= 0): #check if all constraints are satisfied
            acceptance_prob = optimizer.acceptance_probability(optimizer.current_cost, new_cost, optimizer.current_temperature)

            if random.random() < acceptance_prob:
                optimizer.current_solution = new_solution
                optimizer.current_cost = new_cost

                if new_cost < optimizer.best_cost:
                    optimizer.best_solution = new_solution
                    optimizer.best_cost = new_cost
        else:
            acceptance_prob = 0 #always reject if constraints are violated

        optimizer.current_temperature *= optimizer.cooling_rate
        optimizer.max_iterations -= 1

        # Store history
        optimizer.cost_history.append(optimizer.current_cost)
        optimizer.solution_history.append(optimizer.current_solution.copy()) # crucial fix. copy the list.
        optimizer.temperature_history.append(optimizer.current_temperature)

    # Print the results
    print("Best Solution (Radii):", optimizer.best_solution)
    print("Best Cost (Weight):", optimizer.best_cost)
    print("Final Temperature:", optimizer.current_temperature)
    print("Final Solution Constraints:", optimizer.non_linear_constraints(optimizer.best_solution))

    # Plotting
    # Cost vs. Iterations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(optimizer.cost_history)
    plt.title("Cost (Weight) vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost (Weight)")

    # Contour plot of solution space
    r1_values = np.linspace(0.01, 0.2, 50)
    r2_values = np.linspace(0.01, 0.2, 50)
    cost_matrix = np.zeros((50, 50))
    for i, r1 in enumerate(r1_values):
        for j, r2 in enumerate(r2_values):
            cost_matrix[i, j] = optimizer.objective_function([r1, r2])

    plt.subplot(1, 2, 2)
    contour = plt.contourf(r1_values, r2_values, cost_matrix.T, levels=50, cmap='viridis')
    plt.colorbar(contour, label="Cost (Weight)")
    plt.plot([s[0] for s in optimizer.solution_history], [s[1] for s in optimizer.solution_history], 'r.-', markersize=3, label="SA Path")
    plt.plot(optimizer.best_solution[0], optimizer.best_solution[1], 'ro', markersize=8, label="Best Solution")
    plt.title("Solution Space Contour Plot")
    plt.xlabel("Radius 1")
    plt.ylabel("Radius 2")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Temperature vs. Iterations plot
    plt.figure(figsize=(8, 6))
    plt.plot(optimizer.temperature_history)
    plt.title("Temperature vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Temperature")
    plt.show()

    # Validation:
    print("\nValidation:")
    best_radii = optimizer.best_solution
    lengths, stresses, radii, displacements = optimizer.fea(best_radii)
    constraints = optimizer.non_linear_constraints(best_radii)

    print("Best Radii:", best_radii)
    print("Element Lengths:", lengths)
    print("Element Stresses:", stresses)
    print("Node Displacements:", displacements)
    print("Constraints:", constraints)

    # Constraint Validation Check
    if np.all(constraints <= 0):
        print("All constraints are satisfied. Solution is valid.")
    else:
        print("Constraints are violated. Solution is invalid.")

    # Stress Validation
    Y = 250e6
    if np.all(np.abs(stresses) <= Y):
        print("Stress constraints satisfied.")
    else:
        print("Stress constraints violated.")

    # Displacement Validation
    dis_max = 0.02
    displacement_magnitude = np.sqrt(displacements[2]**2 + displacements[3]**2)
    if displacement_magnitude <= dis_max:
        print("Displacement constraint satisfied.")
    else:
        print("Displacement constraint violated.")