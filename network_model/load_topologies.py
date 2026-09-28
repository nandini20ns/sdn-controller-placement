import networkx as nx

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TOPOLOGY_FILE = "network_model/topologies/OS3E.txt"


# --------------------------------------------------
# Load OS3E topology
# --------------------------------------------------

G = nx.Graph()

with open(TOPOLOGY_FILE, "r") as file:
    lines = file.readlines()

section = None

for line in lines:
    line = line.strip()

    # Ignore blank lines and comments
    if not line or line.startswith("#"):
        continue

    # Identify sections
    if line == "router":
        section = "router"
        continue

    if line == "link":
        section = "link"
        continue

    # Read router information
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

    # Read link information
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
# Basic information
# --------------------------------------------------

print("Topology: OS3E")
print("----------------")

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

print("\nFirst 5 nodes:")
for node, data in list(G.nodes(data=True))[:5]:
    print(node, data)

print("\nFirst 5 edges:")
for source, destination, data in list(G.edges(data=True))[:5]:
    print(source, "->", destination, data)

# --------------------------------------------------
# Network characteristics
# --------------------------------------------------

print("\nNetwork characteristics:")
print("------------------------")

# Check whether the network is connected
print("Is connected:", nx.is_connected(G))

# Average node degree
degrees = [degree for node, degree in G.degree()]
average_degree = sum(degrees) / len(degrees)

print("Average node degree:", average_degree)

# Minimum and maximum node degree
print("Minimum node degree:", min(degrees))
print("Maximum node degree:", max(degrees))

# Node with highest degree
highest_degree_node = max(G.degree, key=lambda x: x[1])

print(
    "Highest-degree node:",
    highest_degree_node[0],
    "with degree",
    highest_degree_node[1]
)
# --------------------------------------------------
# Link delay statistics
# --------------------------------------------------

delays = [
    data["delay"]
    for source, destination, data in G.edges(data=True)
]

print("\nLink delay:")
print("-----------")
print("Minimum delay:", min(delays), "seconds")
print("Maximum delay:", max(delays), "seconds")
print("Average delay:", sum(delays) / len(delays), "seconds")

# --------------------------------------------------
# Link capacity statistics
# --------------------------------------------------

capacities = [
    data["capacity"]
    for source, destination, data in G.edges(data=True)
]

print("\nLink capacities:")
print("----------------")
print("Unique capacities:", set(capacities))