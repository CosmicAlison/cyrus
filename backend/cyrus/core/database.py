import logging
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from core.config import settings
from core.models import Base, Satellite, GridNode, FlightRoute

log = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables and seed reference data."""
    Base.metadata.create_all(bind=engine)
    log.info("Database tables created")
    with get_session() as session:
        _seed_satellites(session)
        _seed_grid_nodes(session)
        _seed_flight_routes(session)


@contextmanager
def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Seed Data ─────────────────────────────────────────────────────────────────

def _seed_satellites(session: Session) -> None:
    if session.query(Satellite).count() > 0:
        return

    satellites = [
        Satellite(id="SAT-A", name="Cyrus-Obs-1",    orbit_type="LEO", altitude_km=550,  inclination_deg=53.0),
        Satellite(id="SAT-B", name="Cyrus-Obs-2",    orbit_type="LEO", altitude_km=560,  inclination_deg=97.6),
        Satellite(id="SAT-C", name="WeatherEye-3",   orbit_type="LEO", altitude_km=705,  inclination_deg=98.2),
        Satellite(id="SAT-D", name="CommLink-7",      orbit_type="MEO", altitude_km=20200, inclination_deg=55.0),
        Satellite(id="SAT-E", name="NavStar-12",      orbit_type="MEO", altitude_km=20200, inclination_deg=55.0),
        Satellite(id="SAT-F", name="GeoRelay-Alpha",  orbit_type="GEO", altitude_km=35786, inclination_deg=0.0),
    ]
    session.add_all(satellites)
    log.info("Seeded %d satellites", len(satellites))


def _seed_grid_nodes(session: Session) -> None:
    if session.query(GridNode).count() > 0:
        return

    nodes = [
        GridNode(id="SUB-NE-01", name="Northeast Hub Alpha",    region="Northeast US", node_type="substation",  capacity_mw=2400, gic_vulnerability=0.85),
        GridNode(id="SUB-NE-02", name="Northeast Hub Beta",     region="Northeast US", node_type="substation",  capacity_mw=1800, gic_vulnerability=0.72),
        GridNode(id="TRF-MN-01", name="Manitoba Transformer 1", region="Canada Plains", node_type="transformer", capacity_mw=800,  gic_vulnerability=0.91),
        GridNode(id="TRF-MN-02", name="Manitoba Transformer 2", region="Canada Plains", node_type="transformer", capacity_mw=750,  gic_vulnerability=0.88),
        GridNode(id="SUB-SC-01", name="Scandinavia North Sub",  region="Scandinavia",  node_type="substation",  capacity_mw=1200, gic_vulnerability=0.78),
        GridNode(id="GEN-HL-01", name="Highland Generator Hub", region="Scotland",     node_type="generator",   capacity_mw=600,  gic_vulnerability=0.45),
        GridNode(id="SUB-TX-01", name="Texas Coastal Sub",      region="Texas Coast",  node_type="substation",  capacity_mw=3200, gic_vulnerability=0.62),
    ]
    session.add_all(nodes)
    log.info("Seeded %d grid nodes", len(nodes))


def _seed_flight_routes(session: Session) -> None:
    if session.query(FlightRoute).count() > 0:
        return

    routes = [
        FlightRoute(id="POLAR-01", flight_number="AA 100",  origin="KJFK", destination="EGLL", route_type="polar",     hf_dependency="high"),
        FlightRoute(id="POLAR-02", flight_number="UA 901",  origin="KORD", destination="RJAA", route_type="polar",     hf_dependency="high"),
        FlightRoute(id="POLAR-03", flight_number="DL 402",  origin="KATL", destination="RKSI", route_type="polar",     hf_dependency="high"),
        FlightRoute(id="POLAR-04", flight_number="LH 400",  origin="EDDF", destination="KLAX", route_type="polar",     hf_dependency="medium"),
        FlightRoute(id="OCN-01",   flight_number="BA 177",  origin="EGLL", destination="KLAX", route_type="oceanic",   hf_dependency="medium"),
        FlightRoute(id="OCN-02",   flight_number="QF 12",   origin="YSSY", destination="KDFW", route_type="oceanic",   hf_dependency="low"),
        FlightRoute(id="STD-01",   flight_number="AA 300",  origin="KJFK", destination="KDFW", route_type="non-polar", hf_dependency="low"),
    ]
    session.add_all(routes)
    log.info("Seeded %d flight routes", len(routes))