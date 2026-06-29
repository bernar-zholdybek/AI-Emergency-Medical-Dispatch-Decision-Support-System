
import os
import warnings

import networkx as nx

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data_files", "cambridge_network.graphml")
PLACE_NAME = "Cambridge, Cambridgeshire, England, United Kingdom"
NETWORK_TYPE = "drive"

# Roughly the bounding box of central + greater Cambridge, used only by
# the offline synthetic fallback graph below.
_FALLBACK_BBOX = dict(north=52.232, south=52.166, east=0.165, west=0.085)


def _load_from_cache():
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        import osmnx as ox
        return ox.load_graphml(CACHE_PATH)
    except ImportError:
        # osmnx not installed: still try to read the graphml with plain
        # networkx and coerce the numeric attributes OSMnx normally fixes.
        G = nx.read_graphml(CACHE_PATH)
        for _, data in G.nodes(data=True):
            data["x"], data["y"] = float(data["x"]), float(data["y"])
        for _, _, data in G.edges(data=True):
            if "length" in data:
                data["length"] = float(data["length"])
        return G


def _download_fresh():
    """Download the Cambridge drive network from OSM. Requires internet."""
    import osmnx as ox
    graph = ox.graph_from_place(PLACE_NAME, network_type=NETWORK_TYPE)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    ox.save_graphml(graph, CACHE_PATH)
    return graph


def _build_synthetic_fallback_graph():
    """Offline-only stand-in: a grid graph spanning the Cambridge bounding
    box, with realistic-looking 'length' edge weights computed from real
    lat/lon spacing. Used ONLY when there is no cached graph and no
    internet access. Logged loudly so it's never mistaken for real data.
    """
    warnings.warn(
        "No cached Cambridge road network found and OSMnx/internet is "
        "unavailable. Falling back to a SYNTHETIC grid network for "
        "development/testing only. Run `python setup_data.py` with "
        "internet access to fetch the real OpenStreetMap network.",
        RuntimeWarning,
    )
    from utils.metrics import haversine_distance_m

    rows, cols = 25, 25
    G = nx.grid_2d_graph(rows, cols)
    G = nx.convert_node_labels_to_integers(G, label_attribute="grid_pos")

    lat_step = (_FALLBACK_BBOX["north"] - _FALLBACK_BBOX["south"]) / rows
    lon_step = (_FALLBACK_BBOX["east"] - _FALLBACK_BBOX["west"]) / cols

    for _, data in G.nodes(data=True):
        gx, gy = data["grid_pos"]
        data["x"] = _FALLBACK_BBOX["west"] + gx * lon_step
        data["y"] = _FALLBACK_BBOX["south"] + gy * lat_step

    for u, v in G.edges():
        uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
        vy, vx = G.nodes[v]["y"], G.nodes[v]["x"]
        length = haversine_distance_m(uy, ux, vy, vx)
        G[u][v]["length"] = length

    return G.to_directed()


def load_cambridge_graph(force_refresh: bool = False):
    """Load the Cambridge road network, preferring the local cache.

    Returns a networkx (Multi)DiGraph where every node has 'x' (lon) and
    'y' (lat) attributes and every edge has a 'length' attribute (metres).
    """
    if not force_refresh:
        cached = _load_from_cache()
        if cached is not None:
            return cached

    try:
        return _download_fresh()
    except Exception as exc:  # ImportError (no osmnx) or network failure
        warnings.warn(f"Could not download live OSM data ({exc}).")
        return _build_synthetic_fallback_graph()


def nearest_node(G, lat: float, lon: float):
    """Find the graph node closest to a given lat/lon point."""
    try:
        import osmnx as ox
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except ImportError:
        from utils.metrics import haversine_distance_m
        best_node, best_dist = None, float("inf")
        for node, data in G.nodes(data=True):
            d = haversine_distance_m(lat, lon, data["y"], data["x"])
            if d < best_dist:
                best_node, best_dist = node, d
        return best_node


if __name__ == "__main__":
    G = load_cambridge_graph()
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    n = nearest_node(G, 52.1750, 0.1410)  # near Addenbrooke's
    print(f"Nearest node to Addenbrooke's coords: {n}")
