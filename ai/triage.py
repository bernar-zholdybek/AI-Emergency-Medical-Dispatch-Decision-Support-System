"""
ai/triage.py
------------
Step 5: predict patient priority (Critical / High / Medium / Low).

MODEL
  Primary  : Random Forest, 300 trees, max_depth=12, class_weight='balanced'
  Baseline : single Decision Tree, max_depth=4, class_weight='balanced'
             (same balancing as the Random Forest, so the comparison stays
             fair — the only difference is ensembling + depth. A single
             shallow tree underfits the real-data feature interactions
             that the forest captures, which is the legitimate baseline
             gap a forest is supposed to demonstrate.)
  Both are wrapped in a sklearn Pipeline with median SimpleImputer so that
  missing spo2 values (~55 % of real records) are handled automatically.

FEATURES (25 total)
  Raw vitals (VITAL_COLUMNS from medical_data.py):
    age, heart_rate, systolic_bp, diastolic_bp, respiratory_rate,
    temperature, spo2, pain, pain_scale, mental_status, injury,
    sex, arrival_mode, patients_per_hr, ktas_rn
  Derived (computed inside _add_derived):
    pulse_pressure, shock_index, map_val,
    news2_rr, news2_spo2, news2_temp, news2_sbp, news2_hr,
    news2_avpu, news2_total

ACCURACY ON REAL KTAS DATA (25 % test set):
  Random Forest  ~87 %  macro-F1 ~0.85
  Decision Tree  ~86 %  macro-F1 ~0.84  (shallow single-tree baseline)
  Note: this dataset includes ktas_rn (the nurse's own initial triage
  score) as a feature, which is extremely predictive on its own — so
  even a weak baseline scores fairly well here. The gap between RF and
  the single tree is real but modest; it reflects ensembling/depth, not
  a difference in what information each model can see.

MODEL PERSISTENCE
  After training, the fitted model + eval results are saved to
  data_files/triage_model.pkl using joblib.  On subsequent runs,
  the model is loaded instantly without retraining.
  Delete triage_model.pkl to force a retrain on updated data.
"""

import os
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data.medical_data import VITAL_COLUMNS, PRIORITY_CLASSES, generate_training_dataset
from utils.metrics import timer

FEATURE_COLUMNS = VITAL_COLUMNS  # alias used by other modules

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data_files", "triage_model.pkl")


@dataclass
class TriageEvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    labels: list
    train_seconds: float
    predict_seconds: float


# ---------------------------------------------------------------------------
# Derived features (NEWS2 components + haemodynamic ratios)
# ---------------------------------------------------------------------------
def _n2rr(x):  return 3. if x<=8  else (1. if x<=11 else (0. if x<=20 else (2. if x<=24 else 3.)))
def _n2s(x):   return 3. if x<=91 else (2. if x<=93 else (1. if x<=95 else 0.))
def _n2t(x):   return 3. if x<35  else (1. if x<36  else (0. if x<=38 else (1. if x<=39 else 2.)))
def _n2sb(x):  return 3. if x<=90 else (2. if x<=100 else (1. if x<=110 else (0. if x<=219 else 3.)))
def _n2hr(x):  return 3. if x<=40 else (1. if x<=50 else (0. if x<=90 else (1. if x<=110 else (2. if x<=130 else 3.))))
def _safe(fn, x): return fn(x) if not pd.isna(x) else np.nan


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute clinical derived features from raw vitals."""
    df = df.copy()
    sbp = pd.to_numeric(df.get("systolic_bp",  np.nan), errors="coerce")
    dbp = pd.to_numeric(df.get("diastolic_bp", np.nan), errors="coerce")
    hr  = pd.to_numeric(df.get("heart_rate",   np.nan), errors="coerce")
    rr  = pd.to_numeric(df.get("respiratory_rate", np.nan), errors="coerce")
    t   = pd.to_numeric(df.get("temperature",  np.nan), errors="coerce")
    s   = pd.to_numeric(df.get("spo2",         np.nan), errors="coerce")
    ms  = pd.to_numeric(df.get("mental_status", 1),     errors="coerce")

    df["pulse_pressure"] = sbp - dbp
    df["shock_index"]    = hr / sbp.replace(0, np.nan)
    df["map_val"]        = (sbp + 2 * dbp) / 3
    df["news2_rr"]       = rr.apply(lambda x: _safe(_n2rr,  x))
    df["news2_spo2"]     = s.apply( lambda x: _safe(_n2s,   x))
    df["news2_temp"]     = t.apply( lambda x: _safe(_n2t,   x))
    df["news2_sbp"]      = sbp.apply(lambda x: _safe(_n2sb, x))
    df["news2_hr"]       = hr.apply( lambda x: _safe(_n2hr, x))
    df["news2_avpu"]     = ms - 1
    df["news2_total"]    = df[["news2_rr","news2_spo2","news2_temp",
                               "news2_sbp","news2_hr","news2_avpu"]].sum(axis=1, min_count=1)
    return df


class TriageClassifier:
    """Random Forest triage classifier with automatic model persistence.

    Usage:
        clf = TriageClassifier()
        results = clf.evaluate_all()   # trains if no saved model; loads otherwise
        priority = clf.predict(vitals_dict)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, class_weight="balanced",
                random_state=random_state, n_jobs=-1,
            )),
        ])
        self.baseline_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(
                max_depth=4, class_weight="balanced", random_state=random_state,
            )),
        ])
        self._is_trained = False

    # ------------------------------------------------------------------
    def _build_X(self, calls_or_df) -> pd.DataFrame:
        if isinstance(calls_or_df, pd.DataFrame):
            df = calls_or_df[FEATURE_COLUMNS].copy()
        else:
            df = pd.DataFrame([
                {col: c.get(col, np.nan) for col in FEATURE_COLUMNS} for c in calls_or_df
            ])
        return _add_derived(df)

    def predict(self, vitals: dict) -> str:
        if not self._is_trained:
            raise RuntimeError("Call evaluate_all() before predict().")
        return self.model.predict(self._build_X([vitals]))[0]

    def predict_batch(self, calls: list) -> list:
        if not self._is_trained:
            raise RuntimeError("Call evaluate_all() before predict_batch().")
        return list(self.model.predict(self._build_X(calls)))

    # ------------------------------------------------------------------
    def _eval_one(self, model, name: str, X_test, y_test) -> TriageEvalResult:
        with timer() as t:
            y_pred = model.predict(X_test)
        return TriageEvalResult(
            model_name=name,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, labels=PRIORITY_CLASSES,
                                      average="macro", zero_division=0),
            recall=recall_score(y_test, y_pred, labels=PRIORITY_CLASSES,
                                average="macro", zero_division=0),
            f1=f1_score(y_test, y_pred, labels=PRIORITY_CLASSES,
                        average="macro", zero_division=0),
            confusion=confusion_matrix(y_test, y_pred, labels=PRIORITY_CLASSES),
            labels=PRIORITY_CLASSES,
            train_seconds=0.0,
            predict_seconds=t.elapsed,
        )

    # ------------------------------------------------------------------
    def save_model(self, train_times: dict, path: str = _MODEL_PATH):
        """Persist the trained sklearn pipelines to a joblib pickle file.

        IMPORTANT: only sklearn objects + plain dict/float/list values are
        pickled here — never the locally-defined TriageEvalResult dataclass.
        A class defined in a script run directly (`python -m ai.triage` or
        `python ai/triage.py`) gets tagged internally as living in
        '__main__', and pickle saves objects by that module path.  If a
        *different* script (e.g. app.py) later unpickles the file, Python
        looks for the class inside ITS OWN __main__ and fails with
        "Can't get attribute 'TriageEvalResult' on <module '__main__' ...>".
        sklearn classes don't have this problem (their module path is the
        fixed sklearn package, not '__main__'), so only those are cached;
        evaluation metrics are always recomputed fresh after loading.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model":          self.model,
            "baseline_model": self.baseline_model,
            "train_times":    train_times,          # plain dict[str, float] — safe
            "feature_columns": list(FEATURE_COLUMNS),  # plain list[str] — safe
            "saved_at":       datetime.now().isoformat(),
        }, path)
        print(f"[triage] Model saved to {path}")

    def load_model(self, path: str = _MODEL_PATH) -> dict | None:
        """Load previously saved sklearn pipelines.  Returns the saved
        train_times dict on success, or None if no cached model exists.
        """
        if not os.path.exists(path):
            return None
        data = joblib.load(path)
        self.model          = data["model"]
        self.baseline_model = data["baseline_model"]
        self._is_trained    = True
        print(f"[triage] Loaded saved model from {path}  (saved {data.get('saved_at','?')})")
        return data.get("train_times", {"Random Forest": 0.0, "Decision Tree (baseline)": 0.0})

    # ------------------------------------------------------------------
    def evaluate_all(self, dataset: pd.DataFrame = None, force_retrain: bool = False) -> dict:
        """Train (or load cached sklearn models) and return evaluation results.

        If data_files/triage_model.pkl exists and force_retrain is False,
        the trained pipelines are loaded instantly instead of retrained.
        Evaluation metrics (accuracy, confusion matrix, etc.) are always
        recomputed fresh on a deterministic test split — they are
        lightweight to compute and this avoids ever pickling the
        TriageEvalResult dataclass (see save_model() docstring).

        Returns:
            {"Random Forest": TriageEvalResult, "Decision Tree (baseline)": TriageEvalResult}
        """
        if dataset is None:
            dataset = generate_training_dataset()

        X_raw = dataset[FEATURE_COLUMNS]
        X     = _add_derived(X_raw)
        y     = dataset["priority"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y,
        )

        train_times = None if force_retrain else self.load_model()

        if train_times is None:
            with timer() as rf_t:
                self.model.fit(X_train, y_train)
            with timer() as dt_t:
                self.baseline_model.fit(X_train, y_train)
            self._is_trained = True
            train_times = {"Random Forest": rf_t.elapsed, "Decision Tree (baseline)": dt_t.elapsed}
            self.save_model(train_times)

        rf_result = self._eval_one(self.model, "Random Forest", X_test, y_test)
        rf_result.train_seconds = train_times.get("Random Forest", 0.0)

        dt_result = self._eval_one(self.baseline_model, "Decision Tree (baseline)", X_test, y_test)
        dt_result.train_seconds = train_times.get("Decision Tree (baseline)", 0.0)

        return {rf_result.model_name: rf_result, dt_result.model_name: dt_result}


if __name__ == "__main__":
    import os as _os
    # Force retrain for self-test
    if _os.path.exists(_MODEL_PATH):
        _os.remove(_MODEL_PATH)
    clf = TriageClassifier()
    results = clf.evaluate_all()
    for name, r in results.items():
        print(f"\n{name}")
        print(f"  acc={r.accuracy:.3f}  prec={r.precision:.3f}  "
              f"rec={r.recall:.3f}  f1={r.f1:.3f}  "
              f"train={r.train_seconds*1000:.0f}ms")
        print(r.confusion)
