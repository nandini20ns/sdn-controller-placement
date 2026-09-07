import random
import networkx as nx

from network_model.scoring import score_placement

random.seed(0)


class DiscreteGWO:
    def __init__(self, iterations, pack_size, all_nodes, num_controllers,
                 G, weights=None, min_range=-10, max_range=10, mutation_rate=0.1):
        self.iterations = iterations
        self.pack_size = pack_size
        self.all_nodes = all_nodes
        self.vector_size = len(all_nodes)
        self.num_controllers = num_controllers
        self.G = G
        self.weights = weights or {"avg_delay": 1.0, "max_delay": 0.5, "load_balance": 0.5}
        self.min_range = min_range
        self.max_range = max_range
        self.mutation_rate = mutation_rate  # chance per position to get randomly reset

    def wolf(self):
        return [((self.max_range - self.min_range) * random.random() + self.min_range)
                for _ in range(self.vector_size)]

    def pack(self):
        return [self.wolf() for _ in range(self.pack_size)]

    def decode(self, vector):
        scored_nodes = list(zip(self.all_nodes, vector))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        placement = [node for node, _ in scored_nodes[:self.num_controllers]]
        return placement

    def fitness(self, vector):
        placement = self.decode(vector)
        metrics = score_placement(self.G, placement)
        return (
            self.weights["avg_delay"] * metrics["avg_delay"]
            + self.weights["max_delay"] * metrics["max_delay"]
            + self.weights["load_balance"] * metrics["load_balance"]
        )

    def mutate(self, vector):
        """With probability mutation_rate, reset each position to a
        brand new random value -- forces exploration even after the
        pack has mostly converged."""
        new_vector = list(vector)
        for j in range(len(new_vector)):
            if random.random() < self.mutation_rate:
                new_vector[j] = (self.max_range - self.min_range) * random.random() + self.min_range
        return new_vector

    def hunt(self):
        wolf_pack = self.pack()
        pack_fit = sorted([(self.fitness(w), w) for w in wolf_pack], key=lambda x: x[0])

        best_overall_score = pack_fit[0][0]
        best_overall_placement = self.decode(pack_fit[0][1])

        for k in range(self.iterations):
            alpha, beta, delta = pack_fit[0][1], pack_fit[1][1], pack_fit[2][1]

            if pack_fit[0][0] < best_overall_score:
                best_overall_score = pack_fit[0][0]
                best_overall_placement = self.decode(alpha)

            print(f"iteration: {k}, best_score: {pack_fit[0][0]:.2f}, "
                  f"placement: {self.decode(alpha)}")

            a = 2 * (1 - k / self.iterations)

            new_fits = []
            for idx, (old_fit, wolf) in enumerate(pack_fit):
                A1, A2, A3 = (a * (2 * random.random() - 1) for _ in range(3))
                C1, C2, C3 = (2 * random.random() for _ in range(3))

                new_position = [0.0 for _ in range(self.vector_size)]
                for j in range(self.vector_size):
                    X1 = alpha[j] - A1 * abs(C1 * alpha[j] - wolf[j])
                    X2 = beta[j] - A2 * abs(C2 * beta[j] - wolf[j])
                    X3 = delta[j] - A3 * abs(C3 * delta[j] - wolf[j])
                    new_position[j] = (X1 + X2 + X3) / 3.0

                # mutation step -- force some exploration regardless of convergence
                new_position = self.mutate(new_position)

                new_fitness = self.fitness(new_position)
                if new_fitness < old_fit:
                    new_fits.append((new_fitness, new_position))
                else:
                    new_fits.append((old_fit, wolf))

            pack_fit = sorted(new_fits, key=lambda x: x[0])

        return best_overall_placement, best_overall_score


if __name__ == "__main__":
    import os
    from network_model.edge_weights import add_edge_weights

    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(BASE, "sprint.graphml")

    G = nx.read_graphml(path)
    G = add_edge_weights(G)

    all_nodes = list(G.nodes())

    model = DiscreteGWO(
        iterations=60,
        pack_size=25,
        all_nodes=all_nodes,
        num_controllers=3,
        G=G,
        mutation_rate=0.15,
    )

    best_placement, best_score = model.hunt()

    print(f"\nBest placement found: {best_placement}")
    print(f"Best fitness score: {best_score:.2f}")