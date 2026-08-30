import os
import math
import networkx as nx

def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def add_edge_weights(G):
    """Adds a 'weight' attribute (km) to every edge. Edges where either
    endpoint lacks coordinates get a placeholder weight (average of all
    known edge weights), since there's no way to estimate a distance for
    a node with no location data at all."""
    missing = []

    # First pass: compute real weights where possible
    for u, v, data in G.edges(data=True):
        lat1, lon1 = G.nodes[u].get("Latitude"), G.nodes[u].get("Longitude")
        lat2, lon2 = G.nodes[v].get("Latitude"), G.nodes[v].get("Longitude")

        if None in (lat1, lon1, lat2, lon2):
            missing.append((u, v))
            continue

        data["weight"] = haversine_distance(lat1, lon1, lat2, lon2)

    # Compute placeholder as the average of known weights
    known_weights = [d["weight"] for _, _, d in G.edges(data=True) if "weight" in d]
    placeholder = sum(known_weights) / len(known_weights) if known_weights else 0

    # Second pass: fill in placeholders
    for u, v in missing:
        G[u][v]["weight"] = placeholder
        G[u][v]["weight_estimated"] = True  # flag so we know it's not real

    if missing:
        print(f"WARNING: {len(missing)} edges missing lat/long. "
              f"Assigned placeholder weight = {placeholder:.2f} km: {missing}")

    return G

if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(BASE, "btnorthamerica.graphml")

    G = nx.read_graphml(path)
    G = add_edge_weights(G)

    print("\nFirst 5 edges:")
    for u, v, data in list(G.edges(data=True))[:5]:
        print(u, v, data)

    print("\nEdges with estimated (placeholder) weight:")
    for u, v, data in G.edges(data=True):
        if data.get("weight_estimated"):
            print(u, v, data)