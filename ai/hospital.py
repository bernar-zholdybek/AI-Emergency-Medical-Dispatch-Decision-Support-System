"""
ai/hospital.py
--------------
Step 8 of the AI pipeline: recommend which hospital should receive each
patient, based on:
  - travel distance (from the ambulance's location, via routing)
  - hospital capacity (available-bed ratio)
  - specialty match (e.g. Critical patients need a trauma centre)

FIX (vs original): a hard constraint is applied first — if any hospital
in the list has the required specialty for the patient's priority, only
those hospitals are scored.  Previously the 0.2 specialty weight could be
overwhelmed by a large distance gap, routing Critical patients to general
hospitals that lack trauma capability.

Baseline: always send the patient to the nearest hospital, ignoring
capacity and specialty — a common naive approach this project improves on.
"""

from dataclasses import dataclass

from utils.metrics import timer

# Which specialty each priority level ideally requires.
PRIORITY_REQUIRED_SPECIALTY = {
    "Critical": "trauma_centre",
    "High":     "emergency",
    "Medium":   "emergency",
    "Low":      "general",
}

# Weights applied after the hard-constraint filter.
# All three criteria are on a 0–1 scale, so these are directly comparable.
HOSPITAL_SCORE_WEIGHTS = dict(distance=0.5, beds=0.3, specialty=0.2)


@dataclass
class HospitalChoice:
    call_index: int
    hospital_name: str
    distance_m: float
    score: float
    specialty_matched: bool


@dataclass
class HospitalRecommendationResult:
    method_name: str
    choices: list          # list[HospitalChoice]
    average_distance_m: float
    specialty_match_rate: float
    runtime_s: float


def _normalise(value, min_v, max_v):
    if max_v - min_v < 1e-9:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def _score_hospital(distance_m, hospital, priority, candidate_distances):
    """Combine distance, available-bed ratio and specialty match into a
    single 0–1 score (higher is better).

    `candidate_distances` is the list of distances for the hospitals that
    passed the hard-constraint filter, so normalisation is done only across
    viable options.
    """
    distance_score = 1.0 - _normalise(distance_m,
                                       min(candidate_distances),
                                       max(candidate_distances))
    bed_ratio = hospital["available_beds"] / max(hospital["total_beds"], 1)
    required = PRIORITY_REQUIRED_SPECIALTY[priority]
    specialty_matched = required in hospital["specialties"]
    specialty_score = 1.0 if specialty_matched else 0.0

    score = (
        HOSPITAL_SCORE_WEIGHTS["distance"]  * distance_score
        + HOSPITAL_SCORE_WEIGHTS["beds"]    * bed_ratio
        + HOSPITAL_SCORE_WEIGHTS["specialty"] * specialty_score
    )
    return score, specialty_matched


def recommend_hospitals(calls, hospitals, priorities, distance_fn) -> HospitalRecommendationResult:
    """For each call, select only hospitals that have the required specialty
    (hard constraint), then pick the best-scoring one among those.
    If no hospital has the required specialty, fall back to all hospitals.

    calls       : list of dicts with a 'node' key
    hospitals   : list of dicts with 'name', 'node', 'total_beds',
                  'available_beds', 'specialties' (set/list of strings)
    priorities  : predicted priority string per call, same order
    distance_fn : distance_fn(node_a, node_b) -> metres
    """
    choices = []
    with timer() as t:
        for i, call in enumerate(calls):
            all_distances = [distance_fn(call["node"], h["node"]) for h in hospitals]
            priority = priorities[i]
            required = PRIORITY_REQUIRED_SPECIALTY[priority]

            # --- Hard constraint: prefer hospitals with the required specialty ---
            candidates = [
                (h, d) for h, d in zip(hospitals, all_distances)
                if required in h["specialties"]
            ]
            if not candidates:
                # No specialist hospital available — fall back to full list
                candidates = list(zip(hospitals, all_distances))

            candidate_distances = [d for _, d in candidates]
            scored = [
                (h, d, *_score_hospital(d, h, priority, candidate_distances))
                for h, d in candidates
            ]
            best_h, best_d, best_score, best_matched = max(scored, key=lambda x: x[2])
            choices.append(HospitalChoice(i, best_h["name"], best_d, best_score, best_matched))

    avg_distance = sum(c.distance_m for c in choices) / len(choices) if choices else 0.0
    match_rate = sum(c.specialty_matched for c in choices) / len(choices) if choices else 0.0
    return HospitalRecommendationResult(
        method_name="Multi-criteria Recommendation",
        choices=choices,
        average_distance_m=avg_distance,
        specialty_match_rate=match_rate,
        runtime_s=t.elapsed,
    )


def recommend_nearest_only(calls, hospitals, priorities, distance_fn) -> HospitalRecommendationResult:
    """Baseline: ignore capacity/specialty, always pick the closest hospital."""
    choices = []
    with timer() as t:
        for i, call in enumerate(calls):
            distances = [(h, distance_fn(call["node"], h["node"])) for h in hospitals]
            best_h, best_d = min(distances, key=lambda x: x[1])
            required = PRIORITY_REQUIRED_SPECIALTY[priorities[i]]
            matched = required in best_h["specialties"]
            choices.append(HospitalChoice(i, best_h["name"], best_d,
                                          score=0.0, specialty_matched=matched))

    avg_distance = sum(c.distance_m for c in choices) / len(choices) if choices else 0.0
    match_rate = sum(c.specialty_matched for c in choices) / len(choices) if choices else 0.0
    return HospitalRecommendationResult(
        method_name="Nearest Hospital (baseline)",
        choices=choices,
        average_distance_m=avg_distance,
        specialty_match_rate=match_rate,
        runtime_s=t.elapsed,
    )


def compare_hospital_methods(calls, hospitals, priorities, distance_fn) -> dict:
    return {
        "Multi-criteria Recommendation": recommend_hospitals(calls, hospitals, priorities, distance_fn),
        "Nearest Hospital (baseline)":   recommend_nearest_only(calls, hospitals, priorities, distance_fn),
    }


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(1)
    fake_calls = [{"node": (rng.uniform(0, 5000), rng.uniform(0, 5000))} for _ in range(6)]
    fake_priorities = rng.choice(["Critical", "High", "Medium", "Low"], size=6).tolist()
    fake_hospitals = [
        dict(name="Addenbrooke's (Trauma)", node=(2500, 2500), total_beds=900,
             available_beds=120, specialties={"trauma_centre", "emergency", "general"}),
        dict(name="Community Hospital",     node=(1000, 4000), total_beds=150,
             available_beds=10,  specialties={"general"}),
        dict(name="The Evelyn (Private)",   node=(3500, 1000), total_beds=80,
             available_beds=40,  specialties={"emergency", "general"}),
    ]

    def straight_line(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    results = compare_hospital_methods(fake_calls, fake_hospitals, fake_priorities, straight_line)
    for name, r in results.items():
        print(f"{name}: avg_dist={r.average_distance_m:.1f}m "
              f"specialty_match={r.specialty_match_rate:.2f} "
              f"runtime={r.runtime_s*1000:.3f}ms")
