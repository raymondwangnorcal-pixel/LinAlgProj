import numpy as np
import matplotlib.pyplot as plt

def get_payoff_matrix(resource_level):
    value = 6 + 4 * resource_level
    cost = 8 - 2 * resource_level

    CC = value / 2              # cooperator vc cooperator
    CCh = 1                     # cooperator vs cheater
    ChC = value                 # cheater vs cooperator
    ChCh = (value - cost) / 2   # cheater vs cheater

    return np.array([
        [CC, CCh],
        [ChC, ChCh]
    ], dtype=float)

def get_fitness(payoff_matrix, x):
    population_vector = np.array([x, 1 - x], dtype=float)
    fitness_vector = payoff_matrix @ population_vector
    return fitness_vector[0], fitness_vector[1]

def get_average_fitness(payoff_matrix, x):
    population_vector = np.array([x, 1 - x], dtype=float)
    return population_vector @ (payoff_matrix @ population_vector)

def update_population(payoff_matrix, x, dt):
    fitness_cooperator, fitness_cheater = get_fitness(payoff_matrix, x)
    average_fitness = get_average_fitness(payoff_matrix, x)

    x_next = x + dt * x * (fitness_cooperator - average_fitness)

    if x_next < 0:
        x_next = 0
    elif x_next > 1:
        x_next = 1

    return x_next

def run_simulation(resource_level, x0, total_time, dt):
    payoff_matrix = get_payoff_matrix(resource_level)
    steps = int(total_time / dt)

    times = [0]
    cooperator_frequencies = [x0]
    cheater_frequencies = [1 - x0]

    x = x0

    for step in range(steps):
        x = update_population(payoff_matrix, x, dt)

        times.append((step + 1) * dt)
        cooperator_frequencies.append(x)
        cheater_frequencies.append(1 - x)

    return times, cooperator_frequencies, cheater_frequencies, payoff_matrix

def find_equilibria(payoff_matrix):
    equilibria = [0.0, 1.0]

    a, b = payoff_matrix[0, 0], payoff_matrix[0, 1]
    c, d = payoff_matrix[1, 0], payoff_matrix[1, 1]

    denominator = a - b - c + d
    numerator = d - b

    if abs(denominator) > 1e-12:
        x_star = numerator / denominator
        if 0 <= x_star <= 1:
            equilibria.append(x_star)

    return sorted(set(round(eq, 10) for eq in equilibria))

def classify_stability(payoff_matrix, x_star, eps=1e-4):
    dt = 0.01

    if x_star <= eps:
        left = None
        right = x_star + eps
    elif x_star >= 1 - eps:
        left = x_star - eps
        right = None
    else:
        left = x_star - eps
        right = x_star + eps

    left_direction = None
    right_direction = None

    if left is not None:
        next_left = update_population(payoff_matrix, left, dt)
        left_direction = np.sign(next_left - left)

    if right is not None:
        next_right = update_population(payoff_matrix, right, dt)
        right_direction = np.sign(next_right - right)

    if left_direction is not None and right_direction is not None:
        if left_direction > 0 and right_direction < 0:
            return "stable"
        elif left_direction < 0 and right_direction > 0:
            return "unstable"
        else:
            return "semi-stable or neutral"

    if x_star == 0.0 and right_direction is not None:
        if right_direction < 0:
            return "stable"
        elif right_direction > 0:
            return "unstable"
        else:
            return "neutral"

    if x_star == 1.0 and left_direction is not None:
        if left_direction > 0:
            return "stable"
        elif left_direction < 0:
            return "unstable"
        else:
            return "neutral"

    return "undetermined"

def print_summary(resource_level, payoff_matrix, final_x):
    print("=" * 60)
    print(f"Resource level: {resource_level}")
    print("Payoff matrix:")
    print(payoff_matrix)
    print(f"Final cooperator proportion: {final_x:.4f}")
    print(f"Final cheater proportion: {1 - final_x:.4f}")

    if final_x > 0.99:
        print("Result: cooperators dominate")
    elif final_x < 0.01:
        print("Result: cheaters dominate")
    else:
        print("Result: both strategies coexist")

    equilibria = find_equilibria(payoff_matrix)
    print("Candidate equilibria:")
    for eq in equilibria:
        stability = classify_stability(payoff_matrix, eq)
        print(f"  x* = {eq:.4f} -> {stability}")

def test_resource_level(resource_level):
    starting_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    plt.figure(figsize=(8, 5))

    for x0 in starting_values:
        times, cooperators, cheaters, payoff_matrix = run_simulation(
            resource_level=resource_level,
            x0=x0,
            total_time=50,
            dt=0.01
        )

    plt.plot(times, cooperators, label=f"initial cooperator = {x0}")
    print_summary(resource_level, payoff_matrix, cooperators[-1])

    plt.title(f"Microbial Strategy Simulation (resource level = {resource_level})")
    plt.xlabel("Time")
    plt.ylabel("Proportion of Cooperators")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    resource_levels = [0.2, 0.5, 0.8]

    for resource_level in resource_levels:
        test_resource_level(resource_level)

if __name__ == "__main__":
    main()
