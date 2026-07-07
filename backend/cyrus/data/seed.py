import json

from core.database import get_session
from core.models import Satellite, FlightRoute, GridNode


def seed_satellites():
    with open("json/satellites.json") as f:
        data = json.load(f)

    with get_session() as session:
        for item in data:
            satellite = Satellite(**item)
            session.merge(satellite)

        session.commit()

def seed_flight():
    with open("json/flight_routes.json") as f:
        data = json.load(f)

    with get_session() as session:
        for item in data:
            flight = FlightRoute(**item)
            session.merge(flight)

def seed_grid_nodes():
    with open("json/grid_nodes.py") as f:
        data = json.load(f)
    
    with get_session() as session:
        for item in data:
            node = GridNode(**item)
            session.merge(node)


if __name__ == "__main__":
    seed_satellites()
    seed_flight()
    seed_grid_nodes()
    print("Successfully seeded satellite, flight and grid node data.")