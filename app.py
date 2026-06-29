"""
app.py
-------
Entry point for the AI Emergency Medical Dispatch Decision Support System.

Run with:
    python app.py

First-time setup (run once, with internet access, to cache the real
Cambridge road network so the app itself never needs internet):
    python setup_data.py
"""

import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Emergency Medical Dispatch Decision Support System")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
