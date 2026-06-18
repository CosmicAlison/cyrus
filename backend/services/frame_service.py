  
from db.session import SessionLocal
from models.frame import Frame


class FrameService:

    def create_frame(self, data):
        db = SessionLocal()

        frame = Frame(**data)

        db.add(frame)
        db.commit()
        db.refresh(frame)

        db.close()
        return frame

    def get_latest_frame(self):
        db = SessionLocal()

        frame = (
            db.query(Frame)
            .order_by(Frame.timestamp.desc())
            .first()
        )

        db.close()
        return frame