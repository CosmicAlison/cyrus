from sqlalchemy import Column, String, DateTime, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .base import Base


class Frame(Base):
    __tablename__ = "frames"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, index=True)

    s3_url = Column(String, nullable=False)
    s3_key = Column(String, nullable=False)

    instrument = Column(String)
    wavelength = Column(Integer)

    metadata = Column(JSON) 
