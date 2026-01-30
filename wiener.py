# wiener_tracker.py
#
# Incremental exact Wiener index tracker for growing unweighted undirected graphs.
#
# Policy:
#   - add_leaf(u, attach_to): exact fast update using one BFS from the new node
#   - add_edge(a, b): exact update by full recomputation via nx.wiener_index


from typing import Any
import networkx as nx


class WienerGrowingUnweighted:
    def __init__(self) -> None:
        self.G = nx.Graph()
        self.W = 0.0

    def add_initial_node(self, u: Any) -> None:
        self.G.add_node(u)
        self.W = 0.0

    def add_leaf(self, u: Any, attach_to: Any) -> float:
        self.G.add_node(u)
        self.G.add_edge(u, attach_to)

        dist = nx.single_source_shortest_path_length(self.G, u)
        self.W += sum(dist.values())
        return self.W

    def add_edge(self, a: Any, b: Any) -> float:
        if a == b:
            return self.W

        self.G.add_edge(a, b)
        self.W = float(nx.wiener_index(self.G))
        return self.W
