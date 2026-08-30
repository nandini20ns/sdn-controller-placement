import os
import networkx as nx

from edge_weights import add_edge_weights


def assign_nodes_to_controllers(G, controllers):
    """
    For every node in G, find the nearest controller (by shortest
    weighted path) and the distance to it.

    Returns:
        assignment: dict {node: nearest_controller}
        distances: dict {node: distance_to_nearest_controller}
    """
    assignment = {}
    distances = {}

    # For each controller, compute shortest path length to every node
    # (Dijkstra, since weights are non-negative distances)
    dist_from_each_controller = {
        c: nx.single_source_dijkstra_path_length(G, c, weight="weight")
        for c in controllers
    }

    for node in G.nodes():
        best_controller = None
        best_dist = float("inf")

        for c in controllers:
            d = dist_from_each_controller[c].get(node, float("inf"))
            if d < best_dist:
                best_dist = d
                best_controller = c

        assignment[node] = best_controller
        distances[node] = best_dist

    return assignment, distances


if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(BASE, "btnorthamerica.graphml")

    G = nx.read_graphml(path)
    G = add_edge_weights(G)

    # Manual test: pick 3 arbitrary nodes as controllers
    test_controllers = ["0", "10", "20"]

    assignment, distances = assign_nodes_to_controllers(G, test_controllers)

    print(f"Controllers: {test_controllers}\n")
    for node in G.nodes():
        print(f"Node {node} -> Controller {assignment[node]} (distance: {distances[node]:.2f} km)")