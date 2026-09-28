import math


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two
    latitude/longitude points in kilometres.
    """

    R = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


def add_edge_weights(G):
    """
    Add geographic distance as the NetworkX edge weight.

    The OS3E GraphML stores node coordinates as:
        latitude
        longitude
    """

    missing = []

    for u, v, data in G.edges(data=True):

        lat1 = G.nodes[u].get("latitude")
        lon1 = G.nodes[u].get("longitude")

        lat2 = G.nodes[v].get("latitude")
        lon2 = G.nodes[v].get("longitude")

        if None in (lat1, lon1, lat2, lon2):
            missing.append((u, v))
            continue

        data["weight"] = haversine_distance(
            lat1,
            lon1,
            lat2,
            lon2
        )

    # Average known weight for missing coordinates
    known_weights = [
        data["weight"]
        for _, _, data in G.edges(data=True)
        if "weight" in data
    ]

    placeholder = (
        sum(known_weights) / len(known_weights)
        if known_weights
        else 0.0
    )

    for u, v in missing:
        G[u][v]["weight"] = placeholder
        G[u][v]["weight_estimated"] = True

    if missing:
        print(
            f"WARNING: {len(missing)} edges have missing "
            f"coordinates. Using {placeholder:.2f} km."
        )

    return G