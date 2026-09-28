import os
import random
import csv

import networkx as nx
import matplotlib.pyplot as plt

from network_model.edge_weights import add_edge_weights
from network_model.scoring import score_placement


random.seed(15)


class DiscreteGWO:

    def __init__(
        self,
        iterations,
        pack_size,
        all_nodes,
        num_controllers,
        G,
        weights=None,
        min_range=-10,
        max_range=10,
        mutation_rate=0.1
    ):

        self.iterations = iterations
        self.pack_size = pack_size
        self.all_nodes = all_nodes
        self.vector_size = len(all_nodes)
        self.num_controllers = num_controllers
        self.G = G

        self.weights = weights or {
            "avg_delay": 1.0,
            "max_delay": 0.5,
            "load_balance": 0.5
        }

        self.min_range = min_range
        self.max_range = max_range
        self.mutation_rate = mutation_rate

        # Stores convergence information
        self.history = []

    def wolf(self):

        return [
            (
                (self.max_range - self.min_range)
                * random.random()
                + self.min_range
            )
            for _ in range(self.vector_size)
        ]

    def pack(self):

        return [
            self.wolf()
            for _ in range(self.pack_size)
        ]

    def decode(self, vector):

        scored_nodes = list(
            zip(self.all_nodes, vector)
        )

        scored_nodes.sort(
            key=lambda x: x[1],
            reverse=True
        )

        placement = [
            node
            for node, _ in scored_nodes[
                :self.num_controllers
            ]
        ]

        return placement

    def fitness(self, vector):

        placement = self.decode(vector)

        metrics = score_placement(
            self.G,
            placement
        )

        return (
            self.weights["avg_delay"]
            * metrics["avg_delay"]

            + self.weights["max_delay"]
            * metrics["max_delay"]

            + self.weights["load_balance"]
            * metrics["load_balance"]
        )

    def mutate(self, vector):

        new_vector = list(vector)

        for j in range(len(new_vector)):

            if random.random() < self.mutation_rate:

                new_vector[j] = (
                    (self.max_range - self.min_range)
                    * random.random()
                    + self.min_range
                )

        return new_vector

    def hunt(self):

        # --------------------------------------------------
        # Initial population
        # --------------------------------------------------

        wolf_pack = self.pack()

        pack_fit = sorted(
            [
                (self.fitness(w), w)
                for w in wolf_pack
            ],
            key=lambda x: x[0]
        )

        # Initial global best
        best_overall_score = pack_fit[0][0]

        best_overall_placement = self.decode(
            pack_fit[0][1]
        )

        # --------------------------------------------------
        # GWO iterations
        # --------------------------------------------------

        for k in range(self.iterations):

            alpha = pack_fit[0][1]
            beta = pack_fit[1][1]
            delta = pack_fit[2][1]

            alpha_fitness = pack_fit[0][0]
            beta_fitness = pack_fit[1][0]
            delta_fitness = pack_fit[2][0]

            alpha_placement = self.decode(alpha)
            beta_placement = self.decode(beta)
            delta_placement = self.decode(delta)

            # GWO control parameter
            a = 2 * (
                1 - k / self.iterations
            )

            # --------------------------------------------------
            # Print current leadership
            # --------------------------------------------------

            print(
                f"\nIteration {k}"
            )

            print(
                f"  a = {a:.4f}"
            )

            print(
                f"  ALPHA | fitness = {alpha_fitness:.2f} "
                f"| placement = {alpha_placement}"
            )

            print(
                f"  BETA  | fitness = {beta_fitness:.2f} "
                f"| placement = {beta_placement}"
            )

            print(
                f"  DELTA | fitness = {delta_fitness:.2f} "
                f"| placement = {delta_placement}"
            )

            print(
                f"  GLOBAL BEST = {best_overall_score:.2f}"
            )

            # --------------------------------------------------
            # Store history BEFORE generating next population
            # --------------------------------------------------

            self.history.append({
                "iteration": k,
                "a": a,
                "alpha_fitness": alpha_fitness,
                "beta_fitness": beta_fitness,
                "delta_fitness": delta_fitness,
                "global_best": best_overall_score,
                "alpha_placement": list(alpha_placement),
                "beta_placement": list(beta_placement),
                "delta_placement": list(delta_placement)
            })

            # --------------------------------------------------
            # Generate new population
            # --------------------------------------------------

            new_fits = []

            for old_fit, wolf in pack_fit:

                A1, A2, A3 = (
                    a * (2 * random.random() - 1)
                    for _ in range(3)
                )

                C1, C2, C3 = (
                    2 * random.random()
                    for _ in range(3)
                )

                new_position = [
                    0.0
                    for _ in range(self.vector_size)
                ]

                for j in range(self.vector_size):

                    X1 = (
                        alpha[j]
                        - A1
                        * abs(
                            C1 * alpha[j]
                            - wolf[j]
                        )
                    )

                    X2 = (
                        beta[j]
                        - A2
                        * abs(
                            C2 * beta[j]
                            - wolf[j]
                        )
                    )

                    X3 = (
                        delta[j]
                        - A3
                        * abs(
                            C3 * delta[j]
                            - wolf[j]
                        )
                    )

                    new_position[j] = (
                        X1 + X2 + X3
                    ) / 3.0

                # Mutation
                new_position = self.mutate(
                    new_position
                )

                new_fitness = self.fitness(
                    new_position
                )

                # Greedy replacement
                if new_fitness < old_fit:

                    new_fits.append(
                        (
                            new_fitness,
                            new_position
                        )
                    )

                else:

                    new_fits.append(
                        (
                            old_fit,
                            wolf
                        )
                    )

            # Sort new population
            pack_fit = sorted(
                new_fits,
                key=lambda x: x[0]
            )

            # --------------------------------------------------
            # Update global best
            # --------------------------------------------------

            if pack_fit[0][0] < best_overall_score:

                best_overall_score = pack_fit[0][0]

                best_overall_placement = (
                    self.decode(
                        pack_fit[0][1]
                    )
                )

        return (
            best_overall_placement,
            best_overall_score
        )


# ==========================================================
# Experiment utilities
# ==========================================================

def save_history_csv(history, output_file):

    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "iteration",
            "a",
            "alpha_fitness",
            "beta_fitness",
            "delta_fitness",
            "global_best",
            "alpha_placement",
            "beta_placement",
            "delta_placement"
        ])

        for row in history:

            writer.writerow([
                row["iteration"],
                row["a"],
                row["alpha_fitness"],
                row["beta_fitness"],
                row["delta_fitness"],
                row["global_best"],
                "|".join(row["alpha_placement"]),
                "|".join(row["beta_placement"]),
                "|".join(row["delta_placement"])
            ])


def plot_convergence(history, output_file):

    iterations = [
        row["iteration"]
        for row in history
    ]

    alpha = [
        row["alpha_fitness"]
        for row in history
    ]

    beta = [
        row["beta_fitness"]
        for row in history
    ]

    delta = [
        row["delta_fitness"]
        for row in history
    ]

    global_best = [
        row["global_best"]
        for row in history
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        iterations,
        alpha,
        label="Alpha"
    )

    plt.plot(
        iterations,
        beta,
        label="Beta"
    )

    plt.plot(
        iterations,
        delta,
        label="Delta"
    )

    plt.plot(
        iterations,
        global_best,
        label="Global Best",
        linewidth=2
    )

    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Discrete GWO Variant 1 - OS3E Convergence")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.show()


# ==========================================================
# Main experiment
# ==========================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # Project paths
    # --------------------------------------------------

    PROJECT_ROOT = os.path.join(
        os.path.dirname(__file__),
        ".."
    )

    graph_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "OS3E.graphml"
    )

    results_dir = os.path.join(
        PROJECT_ROOT,
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # Load OS3E
    # --------------------------------------------------

    print("Loading OS3E topology...")

    G = nx.read_graphml(
        graph_file
    )

    print(
        f"Nodes: {G.number_of_nodes()}"
    )

    print(
        f"Edges: {G.number_of_edges()}"
    )

    print(
        f"Connected: {nx.is_connected(G)}"
    )

    # --------------------------------------------------
    # Add geographic-distance weights
    # --------------------------------------------------

    G = add_edge_weights(G)

    all_nodes = list(
        G.nodes()
    )

    # --------------------------------------------------
    # GWO parameters
    # --------------------------------------------------

    iterations = 60
    pack_size = 25
    num_controllers = 3
    mutation_rate = 0.15

    weights = {
        "avg_delay": 1.0,
        "max_delay": 0.5,
        "load_balance": 0.5
    }

    print("\nGWO Variant 1")
    print("----------------")
    print(f"Iterations       : {iterations}")
    print(f"Pack size        : {pack_size}")
    print(f"Controllers      : {num_controllers}")
    print(f"Mutation rate    : {mutation_rate}")
    print(f"Random seed      : 0")
    print(f"Weights          : {weights}")

    # --------------------------------------------------
    # Run GWO
    # --------------------------------------------------

    model = DiscreteGWO(
        iterations=iterations,
        pack_size=pack_size,
        all_nodes=all_nodes,
        num_controllers=num_controllers,
        G=G,
        weights=weights,
        mutation_rate=mutation_rate
    )

    best_placement, best_score = model.hunt()

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------

    metrics = score_placement(
        G,
        best_placement
    )

    print("\n================================")
    print("FINAL GWO RESULTS")
    print("================================")

    print(
        "Best placement:",
        best_placement
    )

    print(
        f"Best fitness: {best_score:.6f}"
    )

    print(
        f"Average distance: "
        f"{metrics['avg_delay']:.6f} km"
    )

    print(
        f"Maximum distance: "
        f"{metrics['max_delay']:.6f} km"
    )

    print(
        f"Load balance (std): "
        f"{metrics['load_balance']:.6f}"
    )

    print(
        "Controller loads:",
        metrics["controller_loads"]
    )

    # --------------------------------------------------
    # Save convergence history
    # --------------------------------------------------

    csv_file = os.path.join(
        results_dir,
        "gwo_variant1_os3e_history.csv"
    )

    save_history_csv(
        model.history,
        csv_file
    )

    # --------------------------------------------------
    # Save convergence plot
    # --------------------------------------------------

    plot_file = os.path.join(
        results_dir,
        "gwo_variant1_os3e_convergence.png"
    )

    plot_convergence(
        model.history,
        plot_file
    )

    print("\nResults saved:")
    print(csv_file)
    print(plot_file)