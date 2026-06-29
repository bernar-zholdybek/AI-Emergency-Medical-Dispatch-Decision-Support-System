"""
gui/analytics.py
-------------------
GUI Page 3: Analytics.

Displays everything the project brief asks for: confusion matrix,
accuracy/precision/recall/F1, average response time, average travel
distance, algorithm runtime, and baseline comparisons for every AI
component (triage, assignment, routing, hospital recommendation).

Charts are drawn with matplotlib and embedded via FigureCanvasQTAgg.
Each section lives on its own tab (instead of one long stacked scroll)
so charts get real room and nothing gets clipped, with its own scroll
area in case the window is small.

NOTE: the data source for triage is now the real KTAS ED dataset.  Metrics
reflect real patient records, not a balanced synthetic sample, so class
imbalance is visible in the confusion matrix — this is intentional and
accurately represents real-world performance.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QScrollArea,
    QGridLayout, QTabWidget,
)

CHART_MIN_HEIGHT = 380


def _metric_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 13px;")
    return lbl


def _canvas(fig) -> FigureCanvas:
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(CHART_MIN_HEIGHT)
    return canvas


def _scrollable(box: QGroupBox) -> QScrollArea:
    """Wrap a section's QGroupBox in its own scroll area so its charts and
    table can't get squeezed by the tab widget's available space."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(box)
    return scroll


class AnalyticsPage(QWidget):
    """Page 3 of the GUI: Analytics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Analytics")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        outer_layout.addWidget(title)

        self.placeholder = QLabel("Run AI from the Dashboard to see analytics here.")
        self.placeholder.setStyleSheet("color: #555; font-style: italic; padding: 20px;")
        outer_layout.addWidget(self.placeholder)

        self.tabs = QTabWidget()
        self.tabs.hide()
        outer_layout.addWidget(self.tabs)

    # ------------------------------------------------------------------
    def update_results(self, result):
        """Rebuild the whole analytics page from a fresh PipelineResult."""
        self.placeholder.hide()
        self.tabs.show()
        self.tabs.clear()

        self.tabs.addTab(_scrollable(self._build_triage_section(result.triage_eval)), "Triage")
        self.tabs.addTab(_scrollable(self._build_assignment_section(result.assignment_results)), "Assignment")
        self.tabs.addTab(_scrollable(self._build_routing_section(result.routing_comparison)), "Routing")
        self.tabs.addTab(_scrollable(self._build_hospital_section(result.hospital_results)), "Hospital")

    # ------------------------------------------------------------------
    # Section 1: Triage
    # ------------------------------------------------------------------
    def _build_triage_section(self, triage_eval: dict) -> QGroupBox:
        box = QGroupBox("Patient Priority Classification (Random Forest vs Decision Tree baseline)")
        layout = QVBoxLayout(box)

        rf = triage_eval["Random Forest"]
        dt = triage_eval["Decision Tree (baseline)"]

        fig = Figure(figsize=(9, 4.5))
        ax_counts = fig.add_subplot(121)
        ax_pct = fig.add_subplot(122)

        # Raw counts.
        ax_counts.imshow(rf.confusion, cmap="Blues")
        ax_counts.set_xticks(range(len(rf.labels)))
        ax_counts.set_yticks(range(len(rf.labels)))
        ax_counts.set_xticklabels(rf.labels, rotation=45, ha="right")
        ax_counts.set_yticklabels(rf.labels)
        ax_counts.set_xlabel("Predicted")
        ax_counts.set_ylabel("True")
        ax_counts.set_title("Random Forest — Raw Counts")
        for i in range(rf.confusion.shape[0]):
            for j in range(rf.confusion.shape[1]):
                ax_counts.text(j, i, int(rf.confusion[i, j]), ha="center", va="center",
                                color="white" if rf.confusion[i, j] > rf.confusion.max() / 2 else "black")

        # Row-normalized percentages — keeps class imbalance from
        # swamping the visual on real (unbalanced) KTAS data.
        row_sums = rf.confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized = rf.confusion / row_sums * 100
        im = ax_pct.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
        ax_pct.set_xticks(range(len(rf.labels)))
        ax_pct.set_yticks(range(len(rf.labels)))
        ax_pct.set_xticklabels(rf.labels, rotation=45, ha="right")
        ax_pct.set_yticklabels(rf.labels)
        ax_pct.set_xlabel("Predicted")
        ax_pct.set_ylabel("True")
        ax_pct.set_title("Random Forest — Row-Normalized (%)")
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                ax_pct.text(j, i, f"{normalized[i, j]:.0f}%", ha="center", va="center",
                            color="white" if normalized[i, j] > 50 else "black")
        fig.colorbar(im, ax=ax_pct, fraction=0.046, pad=0.04)
        fig.tight_layout()
        layout.addWidget(_canvas(fig))

        metrics_layout = QGridLayout()
        headers = ["Metric", "Random Forest", "Decision Tree (baseline)"]
        for col, h in enumerate(headers):
            metrics_layout.addWidget(QLabel(f"<b>{h}</b>"), 0, col)

        rows = [
            ("Accuracy",            rf.accuracy,          dt.accuracy),
            ("Precision (macro)",   rf.precision,         dt.precision),
            ("Recall (macro)",      rf.recall,            dt.recall),
            ("F1-score (macro)",    rf.f1,                dt.f1),
            ("Train time (ms)",     rf.train_seconds * 1000, dt.train_seconds * 1000),
            ("Predict time (ms)",   rf.predict_seconds * 1000, dt.predict_seconds * 1000),
        ]
        for r, (name, rf_val, dt_val) in enumerate(rows, start=1):
            metrics_layout.addWidget(_metric_label(name),         r, 0)
            metrics_layout.addWidget(_metric_label(f"{rf_val:.3f}"), r, 1)
            metrics_layout.addWidget(_metric_label(f"{dt_val:.3f}"), r, 2)

        metrics_widget = QWidget()
        metrics_widget.setLayout(metrics_layout)
        layout.addWidget(metrics_widget)
        return box

    # ------------------------------------------------------------------
    # Section 2: Assignment
    # ------------------------------------------------------------------
    def _build_assignment_section(self, assignment_results: dict) -> QGroupBox:
        box = QGroupBox("Ambulance Assignment (Hungarian Algorithm vs Nearest Available baseline)")
        layout = QVBoxLayout(box)

        hung    = assignment_results["Hungarian Algorithm"]
        nearest = assignment_results["Nearest Available (baseline)"]

        fig = Figure(figsize=(9, 4.5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        names = ["Hungarian", "Nearest"]
        bars1 = ax1.bar(names, [hung.average_response_time_s, nearest.average_response_time_s],
                         color=["#2a9d8b", "#e76f51"])
        ax1.bar_label(bars1, fmt="%.1f", padding=3)
        ax1.set_title("Avg. Response Time (s)")
        bars2 = ax2.bar(names, [hung.total_distance_m / 1000, nearest.total_distance_m / 1000],
                         color=["#2a9d8b", "#e76f51"])
        ax2.bar_label(bars2, fmt="%.1f", padding=3)
        ax2.set_title("Total Distance (km)")
        fig.tight_layout()
        layout.addWidget(_canvas(fig))

        info_layout = QGridLayout()
        info_layout.addWidget(_metric_label("<b>Metric</b>"),              0, 0)
        info_layout.addWidget(_metric_label("<b>Hungarian</b>"),           0, 1)
        info_layout.addWidget(_metric_label("<b>Nearest (baseline)</b>"), 0, 2)
        rows = [
            ("Avg. response time (s)",          hung.average_response_time_s, nearest.average_response_time_s),
            ("Total distance (m)",              hung.total_distance_m,        nearest.total_distance_m),
            ("Calls assigned",                  len(hung.pairs),              len(nearest.pairs)),
            ("Unassigned calls",                len(hung.unassigned_calls),   len(nearest.unassigned_calls)),
            ("Dispatch rounds",                 hung.rounds,                  nearest.rounds),
            ("Max wait — Critical call (s)",    hung.max_critical_wait_s,     nearest.max_critical_wait_s),
            ("Runtime (ms)",                    hung.runtime_s * 1000,        nearest.runtime_s * 1000),
        ]
        for r, (name, a, b) in enumerate(rows, start=1):
            info_layout.addWidget(_metric_label(name),        r, 0)
            info_layout.addWidget(_metric_label(f"{a:.2f}"),  r, 1)
            info_layout.addWidget(_metric_label(f"{b:.2f}"),  r, 2)
        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        layout.addWidget(info_widget)
        return box

    # ------------------------------------------------------------------
    # Section 3: Routing
    # ------------------------------------------------------------------
    def _build_routing_section(self, routing_comparison: dict) -> QGroupBox:
        box = QGroupBox("Routing (A* vs Dijkstra) — one representative ambulance-to-call route")
        layout = QVBoxLayout(box)

        if not routing_comparison:
            layout.addWidget(QLabel("No route available for this scenario."))
            return box

        astar    = routing_comparison["A*"]
        dijkstra = routing_comparison["Dijkstra"]

        fig = Figure(figsize=(9, 4.5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        names = ["A*", "Dijkstra"]
        bars1 = ax1.bar(names, [astar.visited_nodes, dijkstra.visited_nodes], color=["#264653", "#e9c46a"])
        ax1.bar_label(bars1, fmt="%.0f", padding=3)
        ax1.set_title("Nodes Visited")
        bars2 = ax2.bar(names, [astar.runtime_s * 1000, dijkstra.runtime_s * 1000], color=["#264653", "#e9c46a"])
        ax2.bar_label(bars2, fmt="%.2f", padding=3)
        ax2.set_title("Runtime (ms)")
        fig.tight_layout()
        layout.addWidget(_canvas(fig))

        info_layout = QGridLayout()
        info_layout.addWidget(_metric_label("<b>Metric</b>"), 0, 0)
        info_layout.addWidget(_metric_label("<b>A*</b>"),     0, 1)
        info_layout.addWidget(_metric_label("<b>Dijkstra</b>"), 0, 2)
        rows = [
            ("Distance (m)",               astar.distance_m,       dijkstra.distance_m),
            ("Nodes visited",               astar.visited_nodes,    dijkstra.visited_nodes),
            ("Nodes visited (% of graph)",  astar.visited_pct,      dijkstra.visited_pct),
            ("Runtime (ms)",                astar.runtime_s * 1000, dijkstra.runtime_s * 1000),
        ]
        for r, (name, a, b) in enumerate(rows, start=1):
            info_layout.addWidget(_metric_label(name),       r, 0)
            info_layout.addWidget(_metric_label(f"{a:.2f}"), r, 1)
            info_layout.addWidget(_metric_label(f"{b:.2f}"), r, 2)
        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        layout.addWidget(info_widget)
        return box

    # ------------------------------------------------------------------
    # Section 4: Hospital recommendation
    # ------------------------------------------------------------------
    def _build_hospital_section(self, hospital_results: dict) -> QGroupBox:
        box = QGroupBox("Hospital Recommendation (Multi-criteria vs Nearest-only baseline)")
        layout = QVBoxLayout(box)

        multi   = hospital_results["Multi-criteria Recommendation"]
        nearest = hospital_results["Nearest Hospital (baseline)"]

        fig = Figure(figsize=(9, 4.5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        names = ["Multi-criteria", "Nearest"]
        bars1 = ax1.bar(names, [multi.average_distance_m, nearest.average_distance_m], color=["#6a4c93", "#e76f51"])
        ax1.bar_label(bars1, fmt="%.0f", padding=3)
        ax1.set_title("Avg. Distance (m)")
        bars2 = ax2.bar(names, [multi.specialty_match_rate, nearest.specialty_match_rate], color=["#6a4c93", "#e76f51"])
        ax2.bar_label(bars2, fmt="%.2f", padding=3)
        ax2.set_title("Specialty Match Rate")
        ax2.set_ylim(0, 1)
        fig.tight_layout()
        layout.addWidget(_canvas(fig))

        info_layout = QGridLayout()
        info_layout.addWidget(_metric_label("<b>Metric</b>"),              0, 0)
        info_layout.addWidget(_metric_label("<b>Multi-criteria</b>"),      0, 1)
        info_layout.addWidget(_metric_label("<b>Nearest (baseline)</b>"), 0, 2)
        rows = [
            ("Avg. distance (m)",     multi.average_distance_m,   nearest.average_distance_m),
            ("Specialty match rate",  multi.specialty_match_rate, nearest.specialty_match_rate),
            ("Runtime (ms)",          multi.runtime_s * 1000,     nearest.runtime_s * 1000),
        ]
        for r, (name, a, b) in enumerate(rows, start=1):
            info_layout.addWidget(_metric_label(name),       r, 0)
            info_layout.addWidget(_metric_label(f"{a:.2f}"), r, 1)
            info_layout.addWidget(_metric_label(f"{b:.2f}"), r, 2)
        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        layout.addWidget(info_widget)
        return box
