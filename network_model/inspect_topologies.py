import os
import networkx as nx

def inspect_topology(filepath):
    G = nx.read_graphml(filepath)

    print(f"--- {filepath} ---")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Connected: {nx.is_connected(G)}")

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"WARNING: {len(components)} components, sizes: {[len(c) for c in components]}")

    sample_node = list(G.nodes(data=True))[0]
    print(f"Sample node: {sample_node}")

    sample_edge = list(G.edges(data=True))[0]
    print(f"Sample edge: {sample_edge}")

    return G

BASE = os.path.join(os.path.dirname(__file__), "..", "data")

topologies = {
    "OS3E": os.path.join(BASE, "os3e.graphml"),
    "BTNorthAmerica": os.path.join(BASE, "btnorthamerica.graphml"),
    "Sprint": os.path.join(BASE, "sprint.graphml"),
}

if __name__ == "__main__":
    graphs = {}
    for name, path in topologies.items():
        graphs[name] = inspect_topology(path)
        print()