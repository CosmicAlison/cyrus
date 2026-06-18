from models.base import Base
from db.session import engine
from models import frame, flare_event  # noqa

Base.metadata.create_all(bind=engine)