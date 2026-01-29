# wiener_tracker.py
#
# Incremental exact Wiener index tracker for growing unweighted undirected graphs.
#
# Policy:
#   - add_leaf(u, attach_to): exact fast update using one BFS from the new node
#   - add_edge(a, b): exact update by full recomputation via nx.wiener_index
#
# This module is intended to be imported by main.py.

from typing import Any
import networkx as nx


class WienerGrowingUnweightedExact:
    """
    Exact Wiener index tracker for a growing unweighted undirected graph.

    Maintains:
      - self.G : nx.Graph   (current graph)
      - self.W : float      (current Wiener index)

    Definitions:
      Wiener index W(G) = sum_{unordered pairs {i,j}} d(i,j),
      where d(i,j) is the shortest-path distance in an unweighted graph.

    Update rules implemented:

    1) add_leaf(u, attach_to):
       Adds a new node u and one edge (u, attach_to). In this case, distances among old nodes
       do not change. New contribution to Wiener index comes only from pairs (u, x) for all
       existing nodes x. Therefore:
         W_new = W_old + sum_x d(u, x)
       This sum is computed exactly by BFS:
         dist = nx.single_source_shortest_path_length(G, u)
         sum(dist.values()) gives sum_x d(u, x) including d(u,u)=0.

    2) add_edge(a, b):
       Adds an edge between two existing nodes. This can reduce shortest-path distances
       for many pairs of nodes; exact incremental update in general graphs is non-trivial.
       We use an exact fallback:
         W = nx.wiener_index(G)
    """

    def __init__(self) -> None:
        self.G = nx.Graph()
        self.W = 0.0

    def add_initial_node(self, u: Any) -> None:
        """
        Initialize tracker with a single node u.
        """
        self.G.add_node(u)
        self.W = 0.0

    def add_leaf(self, u: Any, attach_to: Any) -> float:
        """
        Add a new node u and connect it with exactly one edge to existing node attach_to.
        Exact update in O(V+E) via one BFS from u.

        Parameters:
          u:        new node id (must not exist in self.G)
          attach_to: existing node id (must exist in self.G)

        Returns:
          Updated Wiener index self.W.
        """
        if u in self.G:
            raise ValueError("add_leaf: node u already exists")
        if attach_to not in self.G:
            raise ValueError("add_leaf: attach_to must exist")

        self.G.add_node(u)
        self.G.add_edge(u, attach_to)

        dist = nx.single_source_shortest_path_length(self.G, u)
        self.W += sum(dist.values())
        return self.W

    def add_edge(self, a: Any, b: Any) -> float:
        """
        Add an edge (a, b) between two existing nodes and recompute exact Wiener index.

        Parameters:
          a, b: existing node ids in self.G

        Returns:
          Updated Wiener index self.W.
        """
        if a not in self.G or b not in self.G:
            raise ValueError("add_edge: both endpoints must exist")
        if a == b:
            return self.W

        self.G.add_edge(a, b)
        self.W = float(nx.wiener_index(self.G))
        return self.W
