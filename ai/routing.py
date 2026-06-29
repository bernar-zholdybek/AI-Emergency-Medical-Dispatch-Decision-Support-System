"""
ai/routing.py
--------------
Step 7 of the AI pipeline: compute the route an ambulance (or patient
transfer) should take across the real Cambridge road network.

Both algorithms are implemented here directly (not just called from
networkx) since they are graded AI algorithms for the course:

  - astar_route   : A* search using the straight-line (haversine) distance
                    to the target as an admissible heuristic.
  - dijkstra_route: Dijkstra's algorithm (equivalent to A* with h(n) = 0),
                    used as the baseline "uninformed search" comparison.

The graph `G` is expected to be a networkx (Multi)DiGraph in the OSMnx
convention: each node has 'y' (latitude) and 'x' (longitude) attributes,
and each edge has a 'length' attribute in metres.
"""

import heapq
from dataclasses import dataclass

from utils.metrics import timer, haversine_distance_m


@dataclass
class RouteResult:
    algorithm: str
    path: list           # list of node ids, source -> target
    distance_m: float
    runtime_s: float
    visited_nodes: int    # number of nodes popped from the open set (search effort)
    visited_pct: float = 0.0   # visited_nodes as a % of total graph nodes


def _edge_length(G, u, v) -> float:
    """Shortest-length edge between u and v (graphs may be multigraphs
    with several parallel edges, e.g. a dual carriageway)."""
    edge_data = G.get_edge_data(u, v)
    if edge_data is None:
        return float("inf")
    # MultiDiGraph: edge_data is {key: {attrs}}; DiGraph: edge_data is {attrs}
    if isinstance(edge_data, dict) and all(isinstance(v_, dict) for v_ in edge_data.values()) and "length" not in edge_data:
        return min(d.get("length", float("inf")) for d in edge_data.values())
    return edge_data.get("length", float("inf"))


def _heuristic(G, node, target) -> float:
    """Admissible heuristic for A*: straight-line distance in metres,
    which is always <= true road distance, so A* stays optimal.
    """
    y1, x1 = G.nodes[node]["y"], G.nodes[node]["x"]
    y2, x2 = G.nodes[target]["y"], G.nodes[target]["x"]
    return haversine_distance_m(y1, x1, y2, x2)


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_route(G, source, target) -> RouteResult:
    """A* search from source to target, weighted by edge 'length' (m)."""
    with timer() as t:
        open_heap = [(0.0, source)]
        g_score = {source: 0.0}
        came_from = {}
        visited = set()

        found = False
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == target:
                found = True
                break

            for neighbor in G.neighbors(current):
                tentative_g = g_score[current] + _edge_length(G, current, neighbor)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + _heuristic(G, neighbor, target)
                    heapq.heappush(open_heap, (f_score, neighbor))

    n_nodes = G.number_of_nodes()
    if not found:
        return RouteResult("A*", [], float("inf"), t.elapsed, len(visited),
                            visited_pct=len(visited) / n_nodes * 100 if n_nodes else 0.0)

    path = _reconstruct_path(came_from, target)
    return RouteResult("A*", path, g_score[target], t.elapsed, len(visited),
                        visited_pct=len(visited) / n_nodes * 100 if n_nodes else 0.0)


def dijkstra_route(G, source, target) -> RouteResult:
    """Dijkstra's algorithm: identical to the A* loop above but with a
    heuristic of zero, i.e. uninformed search. Implemented separately
    (rather than just calling astar with h=0) to keep it transparent for
    the course presentation.
    """
    with timer() as t:
        open_heap = [(0.0, source)]
        g_score = {source: 0.0}
        came_from = {}
        visited = set()

        found = False
        while open_heap:
            current_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == target:
                found = True
                break

            for neighbor in G.neighbors(current):
                tentative_g = g_score[current] + _edge_length(G, current, neighbor)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    heapq.heappush(open_heap, (tentative_g, neighbor))

    n_nodes = G.number_of_nodes()
    if not found:
        return RouteResult("Dijkstra", [], float("inf"), t.elapsed, len(visited),
                            visited_pct=len(visited) / n_nodes * 100 if n_nodes else 0.0)

    path = _reconstruct_path(came_from, target)
    return RouteResult("Dijkstra", path, g_score[target], t.elapsed, len(visited),
                        visited_pct=len(visited) / n_nodes * 100 if n_nodes else 0.0)


def compare_routing_algorithms(G, source, target) -> dict:
    """Convenience wrapper used by the Analytics page."""
    return {
        "A*": astar_route(G, source, target),
        "Dijkstra": dijkstra_route(G, source, target),
    }


def estimate_distance_factory(G):
    """Returns a distance_fn(node_a, node_b) -> metres usable by
    ai/assignment.py's build_cost_matrix, backed by real A* routing on
    graph G rather than straight-line distance.
    """
    def distance_fn(node_a, node_b):
        return astar_route(G, node_a, node_b).distance_m
    return distance_fn


if __name__ == "__main__":
    # Self-test on a small synthetic grid graph (no OSM/network needed),
    # just to validate correctness and that A* visits <= Dijkstra nodes.
    import networkx as nx

    G = nx.grid_2d_graph(10, 10)
    G = nx.convert_node_labels_to_integers(G, label_attribute="grid_pos")
    for n, data in G.nodes(data=True):
        gx, gy = data["grid_pos"]
        data["x"], data["y"] = gx * 0.001, gy * 0.001  # fake lon/lat degrees
    for u, v in G.edges():
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        G[u][v]["length"] = haversine_distance_m(uy, ux, vy, vx)

    source, target = 0, 99
    results = compare_routing_algorithms(G, source, target)
    for name, r in results.items():
        print(f"{name}: distance={r.distance_m:.1f}m runtime={r.runtime_s*1000:.3f}ms "
              f"visited_nodes={r.visited_nodes} path_len={len(r.path)}")
