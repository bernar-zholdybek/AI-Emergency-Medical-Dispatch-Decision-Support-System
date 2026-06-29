"""
data/medical_data.py
--------------------
Real KTAS emergency-department dataset loader + synthetic fallback.

VITAL_COLUMNS (raw features, fed into triage.py before derived features):
  age, heart_rate, systolic_bp, diastolic_bp, respiratory_rate,
  temperature, spo2, pain, pain_scale, mental_status, injury,
  sex, arrival_mode, patients_per_hr, ktas_rn

ktas_rn is the nurse's initial triage score (1-5).  It is the single
strongest predictor and represents "first-responder clinical judgment"
at scene — a realistic feature for a dispatch support system.

KTAS → project priority mapping:
  1 (Resuscitation) + 2 (Emergency) → Critical
  3 (Urgent)                         → High
  4 (Semi-urgent)                    → Medium
  5 (Non-urgent)                     → Low
"""

import os
import warnings

import numpy as np
import pandas as pd

PRIORITY_CLASSES = ["Critical", "High", "Medium", "Low"]

VITAL_COLUMNS = [
    "age",              # years
    "heart_rate",       # bpm
    "systolic_bp",      # mmHg
    "diastolic_bp",     # mmHg
    "respiratory_rate", # breaths/min
    "temperature",      # °C
    "spo2",             # % (often missing → imputed)
    "pain",             # 0/1
    "pain_scale",       # NRS 0–10
    "mental_status",    # 1=alert  2=verbal  3=pain  4=unresponsive
    "injury",           # 0=non-injury  1=injury
    "sex",              # 1=male  2=female
    "arrival_mode",     # 1=walk-in … 7=helicopter
    "patients_per_hr",  # ED crowding
    "ktas_rn",          # nurse's initial KTAS level 1-5
]

_KTAS_TO_PRIORITY = {1: "Critical", 2: "Critical", 3: "High", 4: "Medium", 5: "Low"}

PRIORITY_PRIOR = {"Critical": 0.194, "High": 0.384, "Medium": 0.362, "Low": 0.059}

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data_files", "ktas_data.csv")
_REAL_DATA_CACHE: pd.DataFrame | None = None


def _clean_nrs(val) -> float:
    try:
        return float(str(val).replace(",", "."))
    except (ValueError, TypeError):
        return 0.0


def _load_and_clean_csv(path: str = _CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df["priority"] = df["KTAS_expert"].map(_KTAS_TO_PRIORITY)
    df = df.dropna(subset=["priority"]).copy()

    c = pd.DataFrame()
    for col, src in [("age", "Age"), ("heart_rate", "HR"), ("systolic_bp", "SBP"),
                     ("diastolic_bp", "DBP"), ("respiratory_rate", "RR"), ("temperature", "BT")]:
        c[col] = pd.to_numeric(df[src], errors="coerce")
    c["spo2"]            = pd.to_numeric(df["Saturation"], errors="coerce")
    c["pain"]            = pd.to_numeric(df["Pain"], errors="coerce").fillna(0).astype(int)
    c["pain_scale"]      = df["NRS_pain"].apply(_clean_nrs)
    c["mental_status"]   = pd.to_numeric(df["Mental"], errors="coerce").fillna(1).astype(int)
    c["injury"]          = (pd.to_numeric(df["Injury"], errors="coerce") == 2).astype(int)
    c["sex"]             = pd.to_numeric(df["Sex"], errors="coerce")
    c["arrival_mode"]    = pd.to_numeric(df["Arrival mode"], errors="coerce")
    c["patients_per_hr"] = pd.to_numeric(df["Patients number per hour"], errors="coerce")
    c["ktas_rn"]         = pd.to_numeric(df["KTAS_RN"], errors="coerce")
    c["priority"]        = df["priority"].values
    return c.reset_index(drop=True)


def _get_real_data() -> pd.DataFrame:
    global _REAL_DATA_CACHE
    if _REAL_DATA_CACHE is None:
        _REAL_DATA_CACHE = _load_and_clean_csv()
    return _REAL_DATA_CACHE


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
_PROFILES = {
    "Critical": dict(age=(70,15),hr=(135,20),sbp=(80,15),dbp=(50,10),rr=(26,4),
                     spo2=(85,5),temp=(39.,1.),pain_p=0.8,ps_mean=7.,mental_mean=2.5,inj_p=0.3,ktas_rn_choices=[1,2]),
    "High":     dict(age=(60,18),hr=(115,15),sbp=(95,12), dbp=(65,10),rr=(22,3),
                     spo2=(91,3),temp=(38.3,.8),pain_p=0.6,ps_mean=5.,mental_mean=1.5,inj_p=0.2,ktas_rn_choices=[2,3]),
    "Medium":   dict(age=(50,20),hr=(95,12), sbp=(110,12),dbp=(72,8), rr=(19,2),
                     spo2=(95,2),temp=(37.6,.6),pain_p=0.4,ps_mean=3.,mental_mean=1.1,inj_p=0.15,ktas_rn_choices=[3,4]),
    "Low":      dict(age=(40,18),hr=(78,10), sbp=(120,10),dbp=(78,7), rr=(17,2),
                     spo2=(98,1),temp=(36.9,.4),pain_p=0.2,ps_mean=1.5,mental_mean=1.,inj_p=0.05,ktas_rn_choices=[4,5]),
}


def _sample_one_synthetic(priority: str, rng: np.random.Generator) -> dict:
    p = _PROFILES[priority]
    pain_flag  = int(rng.random() < p["pain_p"])
    pain_scale = float(np.clip(rng.normal(p["ps_mean"], 1.5), 0, 10)) if pain_flag else 0.0
    mental     = int(np.clip(round(rng.normal(p["mental_mean"], 0.5)), 1, 4))
    # Simulate nurse assessment with occasional error
    ktas_rn    = int(rng.choice(p["ktas_rn_choices"]))
    return dict(
        age=float(np.clip(rng.normal(*p["age"]), 16, 100)),
        heart_rate=float(np.clip(rng.normal(*p["hr"]), 30, 220)),
        systolic_bp=float(np.clip(rng.normal(*p["sbp"]), 50, 275)),
        diastolic_bp=float(np.clip(rng.normal(*p["dbp"]), 31, 160)),
        respiratory_rate=float(np.clip(rng.normal(*p["rr"]), 8, 40)),
        temperature=float(np.clip(rng.normal(*p["temp"]), 33, 42)),
        spo2=float(np.clip(rng.normal(*p["spo2"]), 60, 100)),
        pain=pain_flag, pain_scale=pain_scale, mental_status=mental,
        injury=int(rng.random() < p["inj_p"]),
        sex=int(rng.integers(1, 3)),
        arrival_mode=int(rng.choice([1,2,3,4], p=[0.06,0.21,0.59,0.12])),
        patients_per_hr=float(np.clip(rng.normal(8, 4), 1, 30)),
        ktas_rn=ktas_rn,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_training_dataset(n_samples: int = None, random_state: int = 42) -> pd.DataFrame:
    """Return VITAL_COLUMNS + 'priority' DataFrame for training.
    Uses real KTAS CSV when present; synthetic fallback otherwise.
    """
    if os.path.exists(_CSV_PATH):
        return _get_real_data().sample(frac=1, random_state=random_state).reset_index(drop=True)

    warnings.warn(
        f"KTAS CSV not found at {_CSV_PATH!r}. Using synthetic fallback.", RuntimeWarning, stacklevel=2,
    )
    if n_samples is None:
        n_samples = 2000
    rng = np.random.default_rng(random_state)
    per_class = n_samples // len(PRIORITY_CLASSES)
    rows = [_sample_one_synthetic(p, rng) for p in PRIORITY_CLASSES for _ in range(per_class)]
    return pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)


def sample_call_vitals(rng: np.random.Generator) -> dict:
    """Sample one patient's vitals for scenario generation.
    Draws from the real dataset when available.
    """
    if os.path.exists(_CSV_PATH):
        df = _get_real_data()
        row = df.iloc[int(rng.integers(0, len(df)))]
        record = {col: (None if pd.isna(row[col]) else row[col]) for col in VITAL_COLUMNS}
        record["true_priority"] = row["priority"]
        return record

    priority = rng.choice(PRIORITY_CLASSES, p=[PRIORITY_PRIOR[c] for c in PRIORITY_CLASSES])
    record = _sample_one_synthetic(priority, rng)
    record["true_priority"] = record.pop("priority")
    return record


def load_real_dataset(csv_path: str) -> pd.DataFrame:
    return _load_and_clean_csv(csv_path)


if __name__ == "__main__":
    data = generate_training_dataset()
    print(f"Shape: {data.shape}")
    print(data["priority"].value_counts())
