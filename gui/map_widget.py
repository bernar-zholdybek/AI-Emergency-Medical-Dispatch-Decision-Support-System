"""
gui/map_widget.py
--------------------
GUI Page 2: Interactive Map.

Renders hospitals, ambulances, emergency calls (coloured by predicted
priority) and the optimal A* routes on a Leaflet map, embedded in the
desktop app via QWebEngineView.

INTERNET REQUIREMENT (Bug 2 fix):
  Leaflet JS/CSS and OpenStreetMap tiles are fetched from the internet
  each time the map renders.  The app therefore requires an active internet
  connection to display the map page.  The README previously claimed "no
  internet needed after setup" — that was incorrect.  The road-network
  GRAPH used by the AI pipeline is cached locally (setup_data.py), but
  the map rendering itself always contacts the CDN and tile servers.
  TODO for production: bundle Leaflet locally (run setup_data.py once to
  download it into data_files/leaflet/) and point the HTML at the local
  copy.

POPUP SAFETY:
  All user-facing strings inserted into popup HTML are escaped via
  html.escape() so that special characters in symptom text can't break
  the HTML structure.

ROUTE STYLES:
  Each entry in `routes` (see ai/pipeline.py) carries:
    route           : ambulance -> call leg, always drawn (solid blue)
    hospital_route  : call -> hospital leg, present only for Critical
                       calls (see SHOW_HOSPITAL_ROUTE_FOR in pipeline.py),
                       drawn dashed in red so it reads as the urgent
                       "still needs to get to hospital" leg.
"""

import html
import json

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtWebEngineWidgets import QWebEngineView

CAMBRIDGE_CENTER = (52.1951, 0.1313)

PRIORITY_COLORS = {
    "Critical": "#e63946",
    "High":     "#f4a261",
    "Medium":   "#e9c46a",
    "Low":      "#2a9d8b",
    "Unknown":  "#999999",  # pre-AI / unpredicted
}
HOSPITAL_COLOR  = "#3a86ff"
AMBULANCE_COLOR = "#222222"


class MapPage(QWidget):
    """Page 2 of the GUI: the interactive map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel("Interactive Map")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")
        layout.addWidget(self.title_label)

        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        self.show_placeholder()

    def show_placeholder(self):
        self.web_view.setHtml(_build_map_html(
            center=CAMBRIDGE_CENTER, hospitals=[], ambulances=[], calls=[],
            ambulance_routes_latlon=[], hospital_routes_latlon=[],
        ))

    def update_map(self, scenario, predicted_priorities, routes, graph):
        """Rebuild the map from the latest pipeline run.

        scenario             : Scenario dataclass (calls, ambulances, hospitals)
        predicted_priorities : list, one per scenario.calls, same order.
                               Use "Unknown" before AI has run.
        routes               : list of {call_index, ambulance_index, route} dicts
        graph                : road-network graph (to resolve node → lat/lon)
        """
        calls_payload = []
        for call, priority in zip(scenario.calls, predicted_priorities):
            age = call.get("age")
            hr  = call.get("heart_rate")
            spo2 = call.get("spo2")
            age_str  = f"{age:.0f}"  if age  is not None else "?"
            hr_str   = f"{hr:.0f}"   if hr   is not None else "?"
            spo2_str = f"{spo2:.0f}%" if spo2 is not None else "n/a"
            # Escape all user-facing strings before embedding in HTML
            popup = (
                f"Call {call['id']} — {html.escape(priority)}<br>"
                f"Age {age_str}  HR {hr_str}  SpO₂ {spo2_str}<br>"
                f"Symptoms: {html.escape(str(call.get('symptoms', '')))}"
            )
            calls_payload.append(dict(
                lat=call["lat"], lon=call["lon"],
                priority=priority, popup=popup,
            ))

        hospitals_payload = [
            dict(lat=h["lat"], lon=h["lon"],
                 popup=f"{html.escape(h['name'])}<br>"
                       f"Beds available: {h['available_beds']}/{h['total_beds']}")
            for h in scenario.hospitals
        ]

        ambulances_payload = [
            dict(lat=a["lat"], lon=a["lon"],
                 popup=f"Ambulance {a['id']} ({html.escape(a['station_name'])})")
            for a in scenario.ambulances
        ]

        ambulance_routes_latlon = []
        hospital_routes_latlon = []
        for r in routes:
            path_nodes = r["route"].path
            coords = [[graph.nodes[n]["y"], graph.nodes[n]["x"]] for n in path_nodes]
            ambulance_routes_latlon.append(coords)

            if r.get("hospital_route") is not None:
                hosp_path = r["hospital_route"].path
                hosp_coords = [[graph.nodes[n]["y"], graph.nodes[n]["x"]] for n in hosp_path]
                hospital_routes_latlon.append(hosp_coords)

        html_content = _build_map_html(
            center=CAMBRIDGE_CENTER,
            hospitals=hospitals_payload,
            ambulances=ambulances_payload,
            calls=calls_payload,
            ambulance_routes_latlon=ambulance_routes_latlon,
            hospital_routes_latlon=hospital_routes_latlon,
        )
        self.web_view.setHtml(html_content)


def _build_map_html(center, hospitals, ambulances, calls,
                    ambulance_routes_latlon, hospital_routes_latlon) -> str:
    """Build a self-contained Leaflet HTML page.

    Leaflet is loaded from unpkg CDN — an internet connection is required.
    """
    hospitals_json        = json.dumps(hospitals)
    ambulances_json       = json.dumps(ambulances)
    calls_json             = json.dumps(calls)
    ambulance_routes_json = json.dumps(ambulance_routes_latlon)
    hospital_routes_json  = json.dumps(hospital_routes_latlon)
    colors_json            = json.dumps(PRIORITY_COLORS)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
  .legend {{ background: white; padding: 6px 10px; font: 13px sans-serif; line-height: 22px; }}
  .legend span {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; }}
</style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const map = L.map('map').setView([{center[0]}, {center[1]}], 13);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);

  const priorityColors = {colors_json};

  function circleMarker(lat, lon, color, popup, radius) {{
      const m = L.circleMarker([lat, lon], {{
          radius: radius, color: "#000", weight: 1,
          fillColor: color, fillOpacity: 0.9
      }}).addTo(map);
      if (popup) m.bindPopup(popup);
      return m;
  }}

  for (const h of {hospitals_json})  {{ circleMarker(h.lat, h.lon, "{HOSPITAL_COLOR}",  h.popup, 9); }}
  for (const a of {ambulances_json}) {{ circleMarker(a.lat, a.lon, "{AMBULANCE_COLOR}", a.popup, 6); }}
  for (const c of {calls_json})      {{ circleMarker(c.lat, c.lon, priorityColors[c.priority] || "#999", c.popup, 7); }}

  for (const coords of {ambulance_routes_json}) {{
      L.polyline(coords, {{ color: "#1d3557", weight: 3, opacity: 0.7 }}).addTo(map);
  }}
  for (const coords of {hospital_routes_json}) {{
      L.polyline(coords, {{ color: "#e63946", weight: 3, opacity: 0.85, dashArray: "8 6" }}).addTo(map);
  }}

  const legend = L.control({{position: 'bottomright'}});
  legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML =
          '<span style="background:#e63946"></span>Critical<br>' +
          '<span style="background:#f4a261"></span>High<br>' +
          '<span style="background:#e9c46a"></span>Medium<br>' +
          '<span style="background:#2a9d8b"></span>Low<br>' +
          '<span style="background:#999999"></span>Unknown (pre-AI)<br>' +
          '<span style="background:{HOSPITAL_COLOR}"></span>Hospital<br>' +
          '<span style="background:{AMBULANCE_COLOR}"></span>Ambulance<br>' +
          '<hr style="margin:4px 0">' +
          '<span style="background:#1d3557"></span>Ambulance route<br>' +
          '<span style="background:#e63946"></span>Hospital route (Critical)';
      return div;
  }};
  legend.addTo(map);
</script>
</body>
</html>"""
