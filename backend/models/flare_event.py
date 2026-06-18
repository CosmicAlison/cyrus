from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base


class FlareEvent(Base):
    __tablename__ = "flare_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, index=True)

    class_type = Column(String)
    region = Column(String)

    raw = Column(JSON)
