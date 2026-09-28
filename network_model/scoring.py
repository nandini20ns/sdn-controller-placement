from collections import Counter

from network_model.assignment import (
    assign_nodes_to_controllers
)


def score_placement(G, controllers):
    """
    Evaluate a controller placement.

    Metrics:
        avg_delay:
            Average shortest-path distance from every node
            to its assigned controller.

        max_delay:
            Maximum shortest-path distance.

        load_balance:
            Standard deviation of nodes assigned per controller.

        controller_loads:
            Number of nodes assigned to each controller.
    """

    assignment, distances = assign_nodes_to_controllers(
        G,
        controllers
    )

    distance_values = list(distances.values())

    avg_delay = (
        sum(distance_values)
        / len(distance_values)
    )

    max_delay = max(distance_values)

    # Count nodes assigned to each controller
    controller_loads = Counter(
        assignment.values()
    )

    # Make sure every controller appears
    for controller in controllers:
        controller_loads.setdefault(
            controller,
            0
        )

    load_values = list(
        controller_loads.values()
    )

    mean_load = (
        sum(load_values)
        / len(load_values)
    )

    variance = (
        sum(
            (load - mean_load) ** 2
            for load in load_values
        )
        / len(load_values)
    )

    load_balance = variance ** 0.5

    return {
        "avg_delay": avg_delay,
        "max_delay": max_delay,
        "load_balance": load_balance,
        "controller_loads": dict(
            controller_loads
        ),
    }