import networkx as nx


def assign_nodes_to_controllers(G, controllers):
    """
    Assign every network node to its nearest controller
    using weighted shortest-path distance.

    Returns:
        assignment:
            {node: assigned_controller}

        distances:
            {node: distance_to_assigned_controller}
    """

    assignment = {}
    distances = {}

    # Shortest-path distances from each controller
    controller_distances = {
        controller: nx.single_source_dijkstra_path_length(
            G,
            controller,
            weight="weight"
        )
        for controller in controllers
    }

    # Assign each node to its closest controller
    for node in G.nodes():

        best_controller = None
        best_distance = float("inf")

        for controller in controllers:

            distance = controller_distances[
                controller
            ].get(node, float("inf"))

            if distance < best_distance:
                best_distance = distance
                best_controller = controller

        assignment[node] = best_controller
        distances[node] = best_distance

    return assignment, distances