HOSPITALS = [
    dict(
        name="Addenbrooke's Hospital",
        lat=52.1750, lon=0.1410,
        total_beds=1000, available_beds=180,
        specialties={"trauma_centre", "emergency", "general", "cardiac"},
    ),

    dict(
        name="Royal Papworth Hospital",
        lat=52.1737, lon=0.1448,
        total_beds=300, available_beds=40,
        specialties={"cardiac"},
    ),

    dict(
        name="Peterborough City Hospital",
        lat=52.5844, lon=-0.2503,
        total_beds=700, available_beds=120,
        specialties={"trauma_centre", "emergency", "general"},
    ),

    dict(
        name="Hinchingbrooke Hospital",
        lat=52.3307, lon=-0.1835,
        total_beds=350, available_beds=70,
        specialties={"emergency", "general"},
    ),

    dict(
        name="West Suffolk Hospital",
        lat=52.2355, lon=0.7278,
        total_beds=430, available_beds=90,
        specialties={"emergency", "general"},
    ),

    dict(
        name="Nuffield Health Cambridge Hospital",
        lat=52.1944, lon=0.1140,
        total_beds=50, available_beds=12,
        specialties={"general"},
    ),
]


def get_hospitals():
    """Return a fresh copy of the hospital list (callers may mutate
    available_beds during a scenario run, so always copy)."""
    return [dict(h, specialties=set(h["specialties"])) for h in HOSPITALS]
