# AI Emergency Medical Dispatch Decision Support System

A desktop decision-support tool for emergency medical dispatch, using real-world
patient triage data and the Cambridge, UK road network.

## What it does

1. **Triage** — Predicts patient priority (Critical / High / Medium / Low) from
   age, vital signs and observations using a Random Forest classifier trained on
   the KTAS (Korean Triage and Acuity Scale) ED dataset.
2. **Ambulance assignment** — Assigns available ambulances to calls to minimise
   total response time via the Hungarian algorithm.
3. **Routing** — Computes optimal ambulance routes on the real Cambridge road
   network using A* (vs Dijkstra baseline).
4. **Hospital recommendation** — Recommends the best hospital per patient using
   a hard specialty-match constraint followed by distance/capacity scoring.
5. **Analytics** — Confusion matrix, accuracy/F1/recall, response-time charts,
   and algorithm comparisons all displayed on the Analytics page.

## Setup (run once, requires internet)

```bash
pip install -r requirements.txt
python setup_data.py
```

This downloads and caches the Cambridge road network graph locally.

**Note:** The Interactive Map page always requires an internet connection to load
Leaflet JS/CSS and OpenStreetMap tiles.  Only the road-network graph used by the
AI algorithms is cached locally.

## Running the app

```bash
python app.py
```

## Data

`data_files/ktas_data.csv` — real Korean ED triage dataset (1,267 records,
KTAS levels 1–5 mapped to Critical/High/Medium/Low).  The triage classifier is
trained on this data with `class_weight='balanced'` to handle the real-world
class imbalance.

## Project structure

```
app.py                      Entry point
setup_data.py               One-time network download
data_files/
    ktas_data.csv           Real KTAS triage dataset
    cambridge_network.graphml  Cached after setup_data.py runs
data/
    medical_data.py         Real-data loader + synthetic fallback
    scenario_generator.py   Emergency-call and ambulance placement
    hospitals.py            Cambridge hospital definitions
    ambulance_stations.py   Ambulance station definitions
    network_loader.py       OSMnx graph loader / cache
ai/
    triage.py               Random Forest triage classifier
    assignment.py           Hungarian algorithm assignment
    routing.py              A* and Dijkstra routing
    hospital.py             Multi-criteria hospital recommendation
    pipeline.py             Full pipeline orchestration
gui/
    main_window.py          Main window + page navigation
    dashboard.py            Scenario controls and status
    map_widget.py           Leaflet interactive map
    analytics.py            Analytics charts and metrics
utils/
    metrics.py              Timing utilities
```

## Requirements

- Python 3.12+
- See `requirements.txt` for library versions
