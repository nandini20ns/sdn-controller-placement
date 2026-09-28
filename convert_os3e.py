import os
import networkx as nx


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(
    BASE,
    "network_model",
    "topologies",
    "OS3E.txt"
)

output_file = os.path.join(
    BASE,
    "data",
    "OS3E.graphml"
)


# --------------------------------------------------
# Create graph
# --------------------------------------------------

G = nx.Graph()

with open(input_file, "r") as file:
    lines = file.readlines()


section = None


# --------------------------------------------------
# Parse OS3E.txt
# --------------------------------------------------

for line in lines:

    line = line.strip()

    # Ignore blank lines and comments
    if not line or line.startswith("#"):
        continue

    # Section headers
    if line == "router":
        section = "router"
        continue

    if line == "link":
        section = "link"
        continue

    # --------------------------------------------------
    # Router information
    # --------------------------------------------------

    if section == "router":

        parts = line.split()

        node = parts[0]
        longitude = float(parts[1])
        latitude = float(parts[2])

        G.add_node(
            node,
            longitude=longitude,
            latitude=latitude
        )

    # --------------------------------------------------
    # Link information
    # --------------------------------------------------

    elif section == "link":

        parts = line.split()

        source = parts[0]
        destination = parts[1]
        capacity = parts[2]
        metric = int(parts[3])
        delay = float(parts[4].replace("s", ""))
        queue = int(parts[5])

        G.add_edge(
            source,
            destination,
            capacity=capacity,
            metric=metric,
            delay=delay,
            queue=queue
        )


# --------------------------------------------------
# Save GraphML
# --------------------------------------------------

nx.write_graphml(G, output_file)


# --------------------------------------------------
# Verification
# --------------------------------------------------

print("OS3E conversion completed.")
print("--------------------------")
print("Input :", input_file)
print("Output:", output_file)
print()
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected:", nx.is_connected(G))