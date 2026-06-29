"""
gui/dashboard.py
-------------------
GUI Page 1: Dashboard.

Shows current scenario stats (number of calls, ambulances, hospitals),
and the two controls that drive the whole pipeline: "Generate Scenario"
(steps 2-4) and "Run AI" (steps 5-9, via ai.pipeline.run_full_pipeline).

This page only emits signals; MainWindow owns the actual pipeline calls
so the GUI pages stay simple and independently testable.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QGridLayout, QFrame,
)

from data.scenario_generator import LOAD_TIERS


class StatCard(QFrame):
    """A small labelled number, e.g. '10  Emergency Calls'."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)

        self.value_label = QLabel("0")
        self.value_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #555;")

        layout.addWidget(self.value_label)
        layout.addWidget(title_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


class DashboardPage(QWidget):
    """Page 1 of the GUI: the Dashboard."""

    generate_scenario_requested = Signal(int)   # emits chosen call count
    run_ai_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("AI Emergency Medical Dispatch - Dashboard")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # --- stat cards -------------------------------------------------
        stats_box = QGroupBox("Current Scenario")
        stats_layout = QGridLayout(stats_box)
        self.calls_card = StatCard("Emergency Calls")
        self.ambulances_card = StatCard("Available Ambulances")
        self.hospitals_card = StatCard("Hospitals")
        stats_layout.addWidget(self.calls_card, 0, 0)
        stats_layout.addWidget(self.ambulances_card, 0, 1)
        stats_layout.addWidget(self.hospitals_card, 0, 2)
        layout.addWidget(stats_box)

        # --- controls -----------------------------------------------------
        controls_box = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_box)

        controls_layout.addWidget(QLabel("Call volume:"))
        self.call_count_combo = QComboBox()
        for n_calls, tier in LOAD_TIERS.items():
            self.call_count_combo.addItem(f"{tier['label']} ({n_calls} calls)", userData=n_calls)
        controls_layout.addWidget(self.call_count_combo)

        self.generate_button = QPushButton("Generate Scenario")
        self.generate_button.clicked.connect(self._on_generate_clicked)
        controls_layout.addWidget(self.generate_button)

        self.run_ai_button = QPushButton("Run AI")
        self.run_ai_button.setEnabled(False)
        self.run_ai_button.clicked.connect(self.run_ai_requested.emit)
        controls_layout.addWidget(self.run_ai_button)

        controls_layout.addStretch()
        layout.addWidget(controls_box)

        self.status_label = QLabel("Generate a scenario to begin.")
        self.status_label.setStyleSheet("color: #555; font-style: italic;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _on_generate_clicked(self):
        n_calls = self.call_count_combo.currentData()
        self.generate_scenario_requested.emit(n_calls)

    # -- called by MainWindow after each pipeline step -------------------
    def set_scenario_stats(self, n_calls: int, n_ambulances: int, n_hospitals: int):
        self.calls_card.set_value(n_calls)
        self.ambulances_card.set_value(n_ambulances)
        self.hospitals_card.set_value(n_hospitals)
        self.run_ai_button.setEnabled(True)
        self.status_label.setText("Scenario ready. Press 'Run AI' to compute priorities, "
                                   "assignments, routes and hospital recommendations.")

    def set_status(self, text: str):
        self.status_label.setText(text)
