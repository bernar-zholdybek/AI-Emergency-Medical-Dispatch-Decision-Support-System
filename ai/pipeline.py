"""
ai/pipeline.py
--------------
Orchestrates the full AI pipeline:

  1. Load Cambridge map            -> data.network_loader
  2. Load hospitals                -> data.scenario_generator
  3. Load ambulance stations       -> data.scenario_generator
  4. Generate emergency calls      -> data.scenario_generator
  5. Predict patient priority      -> ai.triage
  6. Assign ambulances (multi-round, leftover-free) -> ai.assignment
  7. Compute routes                -> ai.routing
  8. Recommend best hospital       -> ai.hospital

ROUTE DISPLAY RULE:
  Drawing a route for every single call→hospital leg makes the map
  unreadable and isn't always operationally meaningful (Medium/Low priority
  patients are often treated on-scene or transported later, without the
  same urgency).  A second "patient → hospital" route is only computed and
  shown for calls predicted as one of SHOW_HOSPITAL_ROUTE_FOR (Critical by
  default) — these are the cases where getting to the right hospital fast
  is itself part of the emergency response.  Edit SHOW_HOSPITAL_ROUTE_FOR
  to also include "High" if you want that tier shown too.

No GUI code lives here — the module is fully testable headless.
"""

from dataclasses import dataclass

from data.network_loader import load_cambridge_graph
from data.scenario_generator import generate_scenario
from ai.triage import TriageClassifier
from ai.assignment import compare_assignment_methods
from ai.routing import estimate_distance_factory, astar_route, dijkstra_route
from ai.hospital import compare_hospital_methods

# Priorities for which a second call→hospital route is computed and shown.
SHOW_HOSPITAL_ROUTE_FOR = {"Critical"}


@dataclass
class PipelineResult:
    scenario: object               # Scenario (graph, calls, ambulances, hospitals)
    predicted_priorities: list     # one per call, same order as scenario.calls
    triage_eval: dict              # {"Random Forest": TriageEvalResult, ...}
    assignment_results: dict       # {"Hungarian Algorithm": ..., "Nearest Available (baseline)": ...}
    routes: list                   # list of dicts, see _build_routes()
    routing_comparison: dict       # {"A*": RouteResult, "Dijkstra": RouteResult} for one representative pair
    hospital_results: dict         # {"Multi-criteria Recommendation": ..., "Nearest Hospital (baseline)": ...}


def _build_routes(graph, scenario, hungarian_pairs, predicted_priorities, hospital_choices):
    """Compute ambulance->call routes for every assigned pair, and
    additionally a call->hospital route for calls in SHOW_HOSPITAL_ROUTE_FOR.

    hospital_choices: {call_index: HospitalChoice} from the multi-criteria
    recommendation, used to resolve which hospital node to route to.
    """
    hospital_node_by_name = {h["name"]: h["node"] for h in scenario.hospitals}

    routes = []
    for pair in hungarian_pairs:
        amb_node  = scenario.ambulances[pair.ambulance_index]["node"]
        call_node = scenario.calls[pair.call_index]["node"]
        call_route = astar_route(graph, amb_node, call_node)

        entry = dict(
            call_index=pair.call_index,
            ambulance_index=pair.ambulance_index,
            route=call_route,
            hospital_route=None,
            hospital_name=None,
        )

        priority = predicted_priorities[pair.call_index]
        if priority in SHOW_HOSPITAL_ROUTE_FOR:
            choice = hospital_choices.get(pair.call_index)
            if choice is not None:
                hosp_node = hospital_node_by_name.get(choice.hospital_name)
                if hosp_node is not None:
                    entry["hospital_route"] = astar_route(graph, call_node, hosp_node)
                    entry["hospital_name"]  = choice.hospital_name

        routes.append(entry)
    return routes


def run_full_pipeline(n_calls: int = None, random_state: int = None,
                      graph=None, scenario=None) -> PipelineResult:
    """Run steps 1-8 end to end and return everything the GUI needs.

    Pass an existing `scenario` to run AI on those exact calls/ambulances/hospitals.
    If `scenario` is None a fresh one is generated from `n_calls`.
    """
    # Step 1
    if graph is None:
        graph = load_cambridge_graph()

    # Steps 2-4
    if scenario is None:
        if n_calls is None:
            raise ValueError("Either `scenario` or `n_calls` must be provided.")
        scenario = generate_scenario(graph, n_calls=n_calls, random_state=random_state)

    # Step 5: train (or load cached model) on real KTAS data, predict this scenario's calls
    triage = TriageClassifier()
    triage_eval = triage.evaluate_all()   # loads cached model if present, else trains + saves
    predicted_priorities = triage.predict_batch(scenario.calls)

    # Step 6: Hungarian (multi-round, priority-tiered) vs naive FIFO baseline.
    # Every call gets an ambulance eventually — no more leftover/unassigned calls.
    distance_fn = estimate_distance_factory(graph)
    assignment_results = compare_assignment_methods(
        scenario.calls, scenario.ambulances, predicted_priorities, scenario.hospitals, distance_fn,
    )

    # Step 8 runs before route-building so hospital choices are available for
    # the optional call->hospital route (Step 7 needs them for Critical calls).
    hospital_results = compare_hospital_methods(
        scenario.calls, scenario.hospitals, predicted_priorities, distance_fn,
    )
    hospital_choices = {
        c.call_index: c for c in hospital_results["Multi-criteria Recommendation"].choices
    }

    # Step 7: ambulance->call routes for every assigned pair, plus
    # call->hospital routes for Critical (see SHOW_HOSPITAL_ROUTE_FOR).
    hungarian_pairs = assignment_results["Hungarian Algorithm"].pairs
    routes = _build_routes(graph, scenario, hungarian_pairs, predicted_priorities, hospital_choices)

    # A* vs Dijkstra head-to-head for Analytics — reuse the already-computed
    # A* result for the first pair instead of running A* a second time.
    routing_comparison: dict = {}
    if hungarian_pairs:
        first_pair = hungarian_pairs[0]
        amb_node  = scenario.ambulances[first_pair.ambulance_index]["node"]
        call_node = scenario.calls[first_pair.call_index]["node"]
        dijkstra_result = dijkstra_route(graph, amb_node, call_node)
        routing_comparison = {
            "A*":       routes[0]["route"],   # already computed above
            "Dijkstra": dijkstra_result,
        }

    return PipelineResult(
        scenario=scenario,
        predicted_priorities=list(predicted_priorities),
        triage_eval=triage_eval,
        assignment_results=assignment_results,
        routes=routes,
        routing_comparison=routing_comparison,
        hospital_results=hospital_results,
    )


if __name__ == "__main__":
    result = run_full_pipeline(n_calls=30, random_state=1)
    print(f"Calls: {len(result.scenario.calls)}  Ambulances: {len(result.scenario.ambulances)}")
    for c, p in zip(result.scenario.calls, result.predicted_priorities):
        print(f"  call {c['id']}: predicted={p:>8}  true={c['true_priority']}")

    rf = result.triage_eval["Random Forest"]
    print(f"\nTriage (RF) accuracy={rf.accuracy:.2f}  f1={rf.f1:.2f}")

    hung    = result.assignment_results["Hungarian Algorithm"]
    nearest = result.assignment_results["Nearest Available (baseline)"]
    print(f"\nAssignment: Hungarian {len(hung.pairs)}/{len(result.scenario.calls)} assigned "
          f"in {hung.rounds} rounds, avg={hung.average_response_time_s:.1f}s, "
          f"unassigned={hung.unassigned_calls}")
    print(f"            Nearest   {len(nearest.pairs)}/{len(result.scenario.calls)} assigned "
          f"in {nearest.rounds} rounds, avg={nearest.average_response_time_s:.1f}s")

    print(f"\nComputed {len(result.routes)} ambulance routes; "
          f"{sum(1 for r in result.routes if r['hospital_route'] is not None)} "
          f"also have a call->hospital route (Critical only).")

    hosp = result.hospital_results["Multi-criteria Recommendation"]
    print(f"Hospital: avg_dist={hosp.average_distance_m:.0f}m  "
          f"specialty_match={hosp.specialty_match_rate:.2f}")
