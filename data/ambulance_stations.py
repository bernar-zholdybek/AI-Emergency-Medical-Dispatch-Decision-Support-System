

AMBULANCE_STATIONS = [
    dict(name="Station - City Centre", lat=52.2053, lon=0.1218),
    dict(name="Station - East Cambridge (Cherry Hinton)", lat=52.1889, lon=0.1572),
    dict(name="Station - North Cambridge (Arbury)", lat=52.2238, lon=0.1295),
    dict(name="Station - South Cambridge (Trumpington)", lat=52.1789, lon=0.1066),
    dict(name="Station - West Cambridge (Newnham)", lat=52.1958, lon=0.1041),
    dict(name="Station - Biomedical Campus", lat=52.1745, lon=0.1432),
]


def get_ambulance_stations():
    """Return a fresh copy of the ambulance station list."""
    return [dict(s) for s in AMBULANCE_STATIONS]
