import os
import networkx as nx
from collections import Counter
from network_model.edge_weights import add_edge_weights

from network_model.assignment import assign_nodes_to_controllers


def score_placement(G, controllers):
    """
    Given a graph and a controller placement (list of node IDs),
    returns a metrics dictionary:
        - avg_delay: mean shortest-path distance from each node to its
          assigned controller (km, using edge weights as proxy for delay)
        - max_delay: worst-case single-node distance
        - load_balance: standard deviation of node-count per controller
          (lower = more balanced)
        - controller_loads: dict {controller: number of nodes assigned}
    """
    assignment, distances = assign_nodes_to_controllers(G, controllers)

    dist_values = list(distances.values())
    avg_delay = sum(dist_values) / len(dist_values)
    max_delay = max(dist_values)

    controller_loads = Counter(assignment.values())
    # Ensure every controller appears even if it got 0 nodes (edge case)
    for c in controllers:
        controller_loads.setdefault(c, 0)

    load_values = list(controller_loads.values())
    mean_load = sum(load_values) / len(load_values)
    variance = sum((x - mean_load) ** 2 for x in load_values) / len(load_values)
    load_balance = variance ** 0.5  # standard deviation

    return {
        "avg_delay": avg_delay,
        "max_delay": max_delay,
        "load_balance": load_balance,
        "controller_loads": dict(controller_loads),
    }


if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(BASE, "btnorthamerica.graphml")

    G = nx.read_graphml(path)
    G = add_edge_weights(G)

    test_controllers = ["0", "10", "20"]
    metrics = score_placement(G, test_controllers)

    print(f"Controllers: {test_controllers}")
    for k, v in metrics.items():
        print(f"{k}: {v}")