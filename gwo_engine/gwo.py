import random


def initialize_population(all_nodes, num_controllers, population_size):
    """Each wolf is a placement: a random sample of unique node IDs."""
    population = []
    for _ in range(population_size):
        wolf = random.sample(all_nodes, num_controllers)
        population.append(wolf)
    return population


def fitness(placement, G, score_fn, weights=None):
    """
    Turns a metrics dict into a single number to MINIMIZE.
    weights lets you tune how much each metric matters.
    """
    if weights is None:
        weights = {"avg_delay": 1.0, "max_delay": 0.5, "load_balance": 0.5}

    metrics = score_fn(G, placement)
    return (
        weights["avg_delay"] * metrics["avg_delay"]
        + weights["max_delay"] * metrics["max_delay"]
        + weights["load_balance"] * metrics["load_balance"]
    )


def repair(wolf, all_nodes, num_controllers):
    """Ensures a wolf is a valid placement: right size, no duplicates,
    only real node IDs."""
    wolf = list(dict.fromkeys(wolf))  # dedupe, preserve order
    wolf = [n for n in wolf if n in all_nodes]  # drop invalid nodes

    # If too few nodes (due to dedup/removal), fill with random new ones
    while len(wolf) < num_controllers:
        candidate = random.choice(all_nodes)
        if candidate not in wolf:
            wolf.append(candidate)

    # If too many, trim
    wolf = wolf[:num_controllers]
    return wolf


def update_wolf(wolf, alpha, beta, delta, all_nodes, a, mutation_rate=0.1):
    new_wolf = []
    for i in range(len(wolf)):
        r = random.random()
        if r < mutation_rate:
            # Random mutation: force exploration regardless of convergence
            new_wolf.append(random.choice(all_nodes))
        elif r < mutation_rate + (1 - a) / 3:
            new_wolf.append(alpha[i % len(alpha)])
        elif r < mutation_rate + 2 * (1 - a) / 3:
            new_wolf.append(beta[i % len(beta)])
        elif r < mutation_rate + (1 - a):
            new_wolf.append(delta[i % len(delta)])
        else:
            new_wolf.append(wolf[i])
    return new_wolf

def run_gwo(G, score_fn, num_controllers, population_size=15, generations=30, weights=None):
    all_nodes = list(G.nodes())
    population = initialize_population(all_nodes, num_controllers, population_size)

    best_wolf = None
    best_score = float("inf")

    for gen in range(generations):
        scored = [(w, fitness(w, G, score_fn, weights)) for w in population]
        scored.sort(key=lambda x: x[1])

        alpha, alpha_score = scored[0]
        beta, _ = scored[1] if len(scored) > 1 else scored[0]
        delta, _ = scored[2] if len(scored) > 2 else scored[0]

        if alpha_score < best_score:
            best_score = alpha_score
            best_wolf = alpha

        a = 2 - gen * (2 / generations)  # linearly decreases from 2 to 0
        a = max(0, min(1, a / 2))  # normalize roughly into [0,1]

        new_population = []
        for wolf in population:
            updated = update_wolf(wolf, alpha, beta, delta, all_nodes, a, mutation_rate=0.2)
            updated = repair(updated, all_nodes, num_controllers)
            new_population.append(updated)

        population = new_population

        print(f"Gen {gen+1}/{generations}: best_score so far = {best_score:.2f}, current alpha = {alpha}")

    return best_wolf, best_score


if __name__ == "__main__":
    import os
    import sys
    import networkx as nx

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "network_model"))
    from edge_weights import add_edge_weights
    from scoring import score_placement

    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(BASE, "btnorthamerica.graphml")

    G = nx.read_graphml(path)
    G = add_edge_weights(G)

    best_placement, best_score = run_gwo(
        G, score_placement, num_controllers=3, population_size=15, generations=20
    )

    print(f"\nBest placement found: {best_placement}")
    print(f"Best fitness score: {best_score:.2f}")