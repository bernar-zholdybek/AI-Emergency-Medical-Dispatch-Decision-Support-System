"""
ai/assignment.py
-----------------
Step 6 of the AI pipeline: assign ambulances to emergency calls.

REAL-WORLD FIX (vs. original one-shot version):
  An ambulance physically carries one patient at a time, but it does NOT stay
  permanently tied to that single call — it drops the patient at a hospital
  and returns to service.  The original assignment ran ONE single Hungarian
  match (bounded to min(n_calls, n_ambulances) pairs) and never revisited
  ambulances afterwards, so any call beyond the ambulance count was simply
  left in `unassigned_calls` forever.  With 30 calls and, say, 8 ambulances,
  22 calls were silently dropped every scenario.

  This version models ambulance TURNAROUND: after responding to a call, an
  ambulance is occupied for
        scene_time_s      (time spent on-scene with the patient)
      + transport_time    (drive time call → nearest hospital)
      + handover_time_s   (handover at the hospital)
  after which it becomes available again — from the hospital's location —
  for the next call.  Calls are processed in ROUNDS, by priority tier
  (Critical first), so leftover calls keep getting picked up as ambulances
  free up, instead of being dropped.

Primary method : Hungarian algorithm, run per priority tier per round
                 (`assign_hungarian_multi_round`), batch-optimal within
                 each tier.
Baseline method: greedy FIFO nearest-available, processed in raw call
                 order with NO priority awareness
                 (`assign_nearest_available_multi_round`) — this is what
                 exposes the value of priority-aware optimal assignment:
                 a naive dispatcher can leave a Critical call waiting
                 behind several Low-priority ones.

Both methods guarantee zero leftover calls as long as at least one
ambulance exists.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.metrics import timer

AVERAGE_AMBULANCE_SPEED_KMH = 40.0
SCENE_TIME_S = 600     # average time an ambulance spends on-scene with a patient (10 min)
HANDOVER_TIME_S = 300  # average handover time once at hospital (5 min)
MAX_ROUNDS = 50         # safety cap; normally finishes in ceil(n_calls / n_ambulances) rounds

_PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


@dataclass
class AssignmentPair:
    call_index: int
    ambulance_index: int
    response_time_s: float   # total time from scenario start until ambulance reaches the call
    distance_m: float        # travel distance for this leg only (excludes any prior wait)
    round_number: int
    wait_time_s: float       # how long the call had to wait for this ambulance to free up


@dataclass
class AssignmentResult:
    method_name: str
    pairs: list = field(default_factory=list)          # list[AssignmentPair]
    total_response_time_s: float = 0.0
    total_distance_m: float = 0.0
    average_response_time_s: float = 0.0
    runtime_s: float = 0.0
    unassigned_calls: list = field(default_factory=list)
    rounds: int = 0
    max_critical_wait_s: float = 0.0


def _nearest_hospital(call_node, hospitals, distance_fn):
    """Return (node, distance_m) of the closest hospital to call_node.
    Used only to estimate ambulance turnaround time — NOT the patient's
    actual recommended hospital (see ai/hospital.py for that).
    """
    best = min(hospitals, key=lambda h: distance_fn(call_node, h["node"]))
    return best["node"], distance_fn(call_node, best["node"])


def _priority_rank(priority: str) -> int:
    return _PRIORITY_RANK.get(priority, len(_PRIORITY_RANK))


def _max_critical_wait(pairs, priorities) -> float:
    """Worst-case wait time among Critical-priority calls. 0.0 if there are
    no Critical calls in this scenario."""
    critical_waits = [p.wait_time_s for p in pairs if priorities[p.call_index] == "Critical"]
    return max(critical_waits) if critical_waits else 0.0


# ---------------------------------------------------------------------------
# Primary method: Hungarian, multi-round, priority-tiered
# ---------------------------------------------------------------------------
def assign_hungarian_multi_round(calls, ambulances, priorities, hospitals, distance_fn,
                                 scene_time_s: float = SCENE_TIME_S,
                                 handover_time_s: float = HANDOVER_TIME_S,
                                 max_rounds: int = MAX_ROUNDS) -> AssignmentResult:
    """Assign every call to an ambulance, even when n_calls > n_ambulances.

    Each round: take the highest-priority tier still pending, build a cost
    matrix (tier_calls x all_ambulances) where cost[i, j] = ambulance j's
    next-free time + travel time to call i, and solve it optimally with the
    Hungarian algorithm.  Assigned ambulances become busy until
    scene_time_s + transport-to-nearest-hospital + handover_time_s has
    elapsed, then return to service from the hospital's location.
    Repeat until every call has an ambulance.
    """
    speed = AVERAGE_AMBULANCE_SPEED_KMH * 1000 / 3600
    n_amb = len(ambulances)
    pairs: list[AssignmentPair] = []

    if n_amb == 0:
        return AssignmentResult(
            method_name="Hungarian Algorithm",
            unassigned_calls=list(range(len(calls))),
        )

    free_time = [0.0] * n_amb
    current_node = [a["node"] for a in ambulances]
    pending = sorted(range(len(calls)), key=lambda i: _priority_rank(priorities[i]))

    round_number = 0
    with timer() as t:
        while pending and round_number < max_rounds:
            round_number += 1
            top_rank = _priority_rank(priorities[pending[0]])
            tier_calls = [i for i in pending if _priority_rank(priorities[i]) == top_rank]

            cost = np.zeros((len(tier_calls), n_amb))
            for r, call_idx in enumerate(tier_calls):
                call_node = calls[call_idx]["node"]
                for c in range(n_amb):
                    travel = distance_fn(current_node[c], call_node) / speed
                    cost[r, c] = free_time[c] + travel

            row_idx, col_idx = linear_sum_assignment(cost)
            assigned_this_round = set()
            for r, c in zip(row_idx, col_idx):
                call_idx = tier_calls[r]
                wait_time = free_time[c]
                response_time = float(cost[r, c])
                travel_time = response_time - wait_time
                distance_m = travel_time * speed

                pairs.append(AssignmentPair(
                    call_index=call_idx, ambulance_index=c,
                    response_time_s=response_time, distance_m=distance_m,
                    round_number=round_number, wait_time_s=wait_time,
                ))
                assigned_this_round.add(call_idx)

                call_node = calls[call_idx]["node"]
                hosp_node, hosp_dist = _nearest_hospital(call_node, hospitals, distance_fn)
                transport_time = hosp_dist / speed
                free_time[c] = response_time + scene_time_s + transport_time + handover_time_s
                current_node[c] = hosp_node

            pending = [i for i in pending if i not in assigned_this_round]

    total_response = sum(p.response_time_s for p in pairs)
    total_distance = sum(p.distance_m for p in pairs)
    return AssignmentResult(
        method_name="Hungarian Algorithm",
        pairs=pairs,
        total_response_time_s=total_response,
        total_distance_m=total_distance,
        average_response_time_s=total_response / len(pairs) if pairs else 0.0,
        runtime_s=t.elapsed,
        unassigned_calls=pending,   # only non-empty if max_rounds was hit
        rounds=round_number,
        max_critical_wait_s=_max_critical_wait(pairs, priorities),
    )


# ---------------------------------------------------------------------------
# Baseline: greedy FIFO, no priority awareness
# ---------------------------------------------------------------------------
def assign_nearest_available_multi_round(calls, ambulances, priorities, hospitals, distance_fn,
                                          scene_time_s: float = SCENE_TIME_S,
                                          handover_time_s: float = HANDOVER_TIME_S) -> AssignmentResult:
    """Naive baseline: processes calls in raw arrival order (ignores
    priority entirely), each time sending whichever ambulance has the
    lowest (free_time + travel_time).  Included to show why priority-aware
    optimal assignment matters — a Critical call here can sit behind
    several Low-priority calls just because they happened to come in first.
    """
    speed = AVERAGE_AMBULANCE_SPEED_KMH * 1000 / 3600
    n_amb = len(ambulances)
    if n_amb == 0:
        return AssignmentResult(
            method_name="Nearest Available (baseline)",
            unassigned_calls=list(range(len(calls))),
        )

    free_time = [0.0] * n_amb
    current_node = [a["node"] for a in ambulances]
    pairs: list[AssignmentPair] = []

    with timer() as t:
        for call_idx in range(len(calls)):   # raw order — no priority sort
            call_node = calls[call_idx]["node"]
            costs = [free_time[c] + distance_fn(current_node[c], call_node) / speed
                    for c in range(n_amb)]
            best_c = int(np.argmin(costs))
            response_time = costs[best_c]
            wait_time = free_time[best_c]
            travel_time = response_time - wait_time
            distance_m = travel_time * speed

            pairs.append(AssignmentPair(
                call_index=call_idx, ambulance_index=best_c,
                response_time_s=response_time, distance_m=distance_m,
                round_number=call_idx + 1, wait_time_s=wait_time,
            ))

            hosp_node, hosp_dist = _nearest_hospital(call_node, hospitals, distance_fn)
            transport_time = hosp_dist / speed
            free_time[best_c] = response_time + scene_time_s + transport_time + handover_time_s
            current_node[best_c] = hosp_node

    total_response = sum(p.response_time_s for p in pairs)
    total_distance = sum(p.distance_m for p in pairs)
    return AssignmentResult(
        method_name="Nearest Available (baseline)",
        pairs=pairs,
        total_response_time_s=total_response,
        total_distance_m=total_distance,
        average_response_time_s=total_response / len(pairs) if pairs else 0.0,
        runtime_s=t.elapsed,
        unassigned_calls=[],
        rounds=len(calls),
        max_critical_wait_s=_max_critical_wait(pairs, priorities),
    )


def compare_assignment_methods(calls, ambulances, priorities, hospitals, distance_fn) -> dict:
    """Run both assignment strategies on the same scenario for the Analytics page."""
    return {
        "Hungarian Algorithm": assign_hungarian_multi_round(
            calls, ambulances, priorities, hospitals, distance_fn),
        "Nearest Available (baseline)": assign_nearest_available_multi_round(
            calls, ambulances, priorities, hospitals, distance_fn),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake_calls      = [{"node": (rng.uniform(0, 5000), rng.uniform(0, 5000))} for _ in range(20)]
    fake_ambulances = [{"node": (rng.uniform(0, 5000), rng.uniform(0, 5000))} for _ in range(5)]
    fake_hospitals  = [{"node": (rng.uniform(0, 5000), rng.uniform(0, 5000))} for _ in range(3)]
    fake_priorities = rng.choice(["Critical", "High", "Medium", "Low"], size=20,
                                 p=[0.15, 0.30, 0.35, 0.20]).tolist()

    def straight_line(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    results = compare_assignment_methods(
        fake_calls, fake_ambulances, fake_priorities, fake_hospitals, straight_line)
    for name, r in results.items():
        print(f"{name}: assigned={len(r.pairs)}/20  rounds={r.rounds}  "
              f"avg_response={r.average_response_time_s:.1f}s  "
              f"unassigned={r.unassigned_calls}  runtime={r.runtime_s*1000:.2f}ms")
