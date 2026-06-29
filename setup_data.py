"""
setup_data.py
---------------
Run this ONCE, on a machine with internet access, before using the app:

    python setup_data.py

It downloads the real Cambridge, UK drive-network graph from OpenStreetMap
(via OSMnx) and caches it to data_files/cambridge_network.graphml.  After
that, app.py and every AI module load the cached file.

NOTE: the Interactive Map page still requires an internet connection to
load Leaflet JS/CSS and OpenStreetMap map tiles each time it is displayed.
Only the road-network graph used by the AI algorithms is cached locally.

Optional: re-geocode the hospital list in data/hospitals.py against OSM
Nominatim (see geocode_hospitals() below).  Results are printed for manual
review — not auto-written — so you can sanity-check them first.
"""

import sys

from data.network_loader import load_cambridge_graph, CACHE_PATH


def fetch_road_network():
    print("Downloading Cambridge road network from OpenStreetMap...")
    graph = load_cambridge_graph(force_refresh=True)
    print(
        f"Done. Cached {graph.number_of_nodes()} nodes / "
        f"{graph.number_of_edges()} edges to {CACHE_PATH}"
    )


def _ox_geocode(ox, query: str):
    """Geocode `query` using whichever osmnx API is available.

    osmnx 1.x exposed ox.geocode(query) -> (lat, lon).
    osmnx 2.x removed that; use urllib + Nominatim directly instead,
    which has no additional dependencies.
    """
    if hasattr(ox, "geocode"):
        # osmnx 1.x
        return ox.geocode(query)

    # osmnx 2.x fallback: call Nominatim directly via stdlib urllib
    import json
    import urllib.parse
    import urllib.request

    url = (
        "https://nominatim.openstreetmap.org/search"
        f"?q={urllib.parse.quote(query)}&format=json&limit=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ai-dispatch-project/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        results = json.loads(resp.read())
    if not results:
        raise ValueError(f"No geocoding result for {query!r}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def geocode_hospitals():
    """Optional: look up hospital coordinates via OSM Nominatim.
    Results are printed for manual review — nothing is auto-written.
    """
    import osmnx as ox
    from data.hospitals import HOSPITALS

    print("\nGeocoding hospitals via OSM Nominatim:")
    for h in HOSPITALS:
        query = f"{h['name']}, Cambridge, UK"
        try:
            lat, lon = _ox_geocode(ox, query)
            print(
                f"  {h['name']}: lat={lat:.4f}, lon={lon:.4f} "
                f"(was lat={h['lat']}, lon={h['lon']})"
            )
        except Exception as exc:
            print(f"  {h['name']}: lookup failed ({exc}), keeping existing coordinates")


if __name__ == "__main__":
    fetch_road_network()
    if "--geocode-hospitals" in sys.argv:
        geocode_hospitals()
    print(
        "\nSetup complete. You can now run `python app.py`.\n"
        "Note: the Interactive Map page requires internet access for map tiles."
    )
