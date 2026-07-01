# AI Emergency Medical Dispatch Decision Support System

A desktop decision-support tool for emergency medical dispatch, using real-world
patient triage data and the Cambridge, UK road network.

## What it does

* **Triage**: Predicts patient priority (Critical / High / Medium / Low) from age, vital signs and observations using a Random Forest classifier trained on the KTAS dataset
* **Ambulance assignment**: Assigns available ambulances to calls to minimise total response time via the Hungarian algorithm
* **Routing**: Computes optimal ambulance routes on the real Cambridge road network using A* (vs Dijkstra baseline)
* **Hospital recommendation**: Recommends the best hospital per patient using a hard specialty-match constraint followed by distance/capacity scoring
* **Analytics**: Confusion matrix, accuracy/F1/recall, response-time charts, and algorithm comparisons displayed on the Analytics page



## Project structure

```
app.py                         # main code
setup_data.py                  # one-time network download
data_files/
    ktas_data.csv              # KTAS triage dataset
    cambridge_network.graphml  # cached after setup_data.py runs
data/
    medical_data.py            # data loader, synthetic fallback
    scenario_generator.py      # emergency-call and ambulance placement
    hospitals.py               # Cambridge hospital definitions
    ambulance_stations.py      # ambulance station definitions
    network_loader.py          # OSMnx graph loader / cache
ai/
    triage.py                  # random forest triage classifier
    assignment.py              # hungarian algorithm assignment
    routing.py                 # A* and Dijkstra routing
    hospital.py                # multi-criteria hospital recommendation
    pipeline.py                # full pipeline orchestration
gui/
    main_window.py             # main window, page navigation
    dashboard.py               # scenario controls and status
    map_widget.py              # leaflet interactive map
    analytics.py               # analytics charts and metrics
utils/
    metrics.py                 # timing utilities
```


## Prerequisites
* Python 3.12+
* Internet connection for first-time setup (downloads Cambridge road network)
* Internet connection required at runtime for the Interactive Map page (loads Leaflet JS/CSS and OpenStreetMap tiles)


## Required Libraries
```
osmnx
networkx
scikit-learn
pandas
numpy
PyQt5
folium
scipy
```

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/bernar-zholdybek/AI-Emergency-Medical-Dispatch-Decision-Support-System.git
cd AI-Emergency-Medical-Dispatch-Decision-Support-System
```


### 2. Dependency Installation
Install all required packages
```bash
pip install -r requirements.txt
```


### 3. Download Road Network Data
Run once to cache the Cambridge road network locally
```bash
python setup_data.py
```
(note: requires an internet connection. Only needs to be run once — the graph is saved to data_files/cambridge_network.graphml for all future runs.)


### 4. How to Run
```bash
python app.py
```


## Tech Stack

* **UI:** PyQt5
* **Mapping:** Leaflet JS + OpenStreetMap (via Folium)
* **Triage AI:** Scikit-learn (Random Forest classifier)
* **Assignment:** Hungarian algorithm (SciPy)
* **Routing:** A* and Dijkstra on OSMnx road graph
* **Road Network:** OSMnx + NetworkX


## Data
`data_files/ktas_data.csv` — real Korean ED triage dataset (1,267 records,
KTAS levels 1–5 mapped to Critical/High/Medium/Low).  The triage classifier is
trained on this data with `class_weight='balanced'` to handle the real-world
class imbalance.


## About System Limits

* Road network and hospital data are scoped to Cambridge, UK only
* The Interactive Map page requires an internet connection at all times
* Triage predictions are based on the KTAS dataset and should not be used for real clinical decisions
