from dataclasses import dataclass

import numpy as np

from data.medical_data import sample_call_vitals
from data.hospitals import get_hospitals
from data.ambulance_stations import get_ambulance_stations
from data.network_loader import nearest_node

LOAD_TIERS = {
    10: {"label": "Usual Load",          "n_ambulances": 6},
    20: {"label": "Above-Average Calls", "n_ambulances": 12},
    30: {"label": "High Load",           "n_ambulances": 18},
}
ALLOWED_CALL_COUNTS = tuple(LOAD_TIERS.keys())


@dataclass
class Scenario:
    graph: object
    calls: list
    ambulances: list
    hospitals: list


def _snap_point(G, lat, lon, name):
    node = nearest_node(G, lat, lon)
    data = G.nodes[node]
    return dict(name=name, node=node, lat=data["y"], lon=data["x"])


def _place_hospitals(G):
    hospitals = get_hospitals()
    for h in hospitals:
        snapped = _snap_point(G, h["lat"], h["lon"], h["name"])
        h["node"] = snapped["node"]
        h["lat"], h["lon"] = snapped["lat"], snapped["lon"]
    return hospitals


def _place_ambulances(G, n_calls: int):
    """Place ambulances on the road network.

    Ambulance count scales with the selected load tier:
    - 10 calls -> 6 ambulances
    - 20 calls -> 12 ambulances
    - 30 calls -> 18 ambulances

    The 6 station locations are reused as needed, allowing up to
    3 ambulances per station.
    """
    n_ambulances = LOAD_TIERS[n_calls]["n_ambulances"]

    stations = get_ambulance_stations()
    ambulances = []

    for i in range(n_ambulances):
        s = stations[i % len(stations)]

        snapped = _snap_point(G, s["lat"], s["lon"], s["name"])

        ambulances.append(
            dict(
                id=i,
                station_name=s["name"],
                node=snapped["node"],
                lat=snapped["lat"],
                lon=snapped["lon"],
            )
        )

    return ambulances

def _generate_calls(G, n_calls, rng):
    """Pick n_calls distinct, valid road-network nodes and attach real
    or synthetically sampled medical attributes to each.
    """
    all_nodes = list(G.nodes())
    chosen_nodes = rng.choice(len(all_nodes), size=min(n_calls, len(all_nodes)), replace=False)

    calls = []
    for i, idx in enumerate(chosen_nodes):
        node = all_nodes[idx]
        data = G.nodes[node]
        vitals = sample_call_vitals(rng)
        calls.append(dict(
            id=i,
            node=node,
            lat=data["y"],
            lon=data["x"],
            symptoms=_symptom_summary(vitals),
            **vitals,
        ))
    return calls


def _symptom_summary(vitals: dict) -> str:
    """Convert real-data vitals into a short human-readable string for map
    popups and the dashboard.
    """
    parts = []

    pain_flag  = vitals.get("pain", 0)
    pain_scale = vitals.get("pain_scale", 0) or 0
    if pain_flag:
        if pain_scale >= 7:
            parts.append(f"severe pain (NRS {pain_scale:.0f}/10)")
        elif pain_scale >= 4:
            parts.append(f"moderate pain (NRS {pain_scale:.0f}/10)")
        else:
            parts.append("mild pain")

    if vitals.get("injury", 0):
        parts.append("injury")

    mental = vitals.get("mental_status", 1)
    if mental == 4:
        parts.append("unresponsive")
    elif mental == 3:
        parts.append("responds to pain only")
    elif mental == 2:
        parts.append("confused / verbal response only")

    hr  = vitals.get("heart_rate")
    rr  = vitals.get("respiratory_rate")
    spo2 = vitals.get("spo2")
    if hr is not None and hr > 120:
        parts.append(f"tachycardia HR {hr:.0f}")
    if rr is not None and rr > 22:
        parts.append(f"tachypnoea RR {rr:.0f}")
    if spo2 is not None and spo2 < 92:
        parts.append(f"low SpO2 {spo2:.0f}%")

    return ", ".join(parts) if parts else "no acute symptoms reported"


def generate_scenario(G, n_calls: int, random_state: int = None) -> Scenario:
    """Build a full scenario: hospitals + ambulances snapped to the real
    road network, plus n_calls emergency calls at valid road locations.
    """
    if n_calls not in ALLOWED_CALL_COUNTS:
        raise ValueError(f"n_calls must be one of {ALLOWED_CALL_COUNTS}, got {n_calls}")

    rng = np.random.default_rng(random_state)
    hospitals  = _place_hospitals(G)
    ambulances = _place_ambulances(G, n_calls)
    calls      = _generate_calls(G, n_calls, rng)

    return Scenario(graph=G, calls=calls, ambulances=ambulances, hospitals=hospitals)



