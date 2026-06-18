from flask import Blueprint, jsonify
from services.frame_service import FrameService

routes = Blueprint("routes", __name__)
service = FrameService()


@routes.route("/latest-frame")
def latest_frame():
    frame = service.get_latest()

    if not frame:
        return jsonify({"error": "no data"}), 404

    return jsonify({
        "id": str(frame.id),
        "timestamp": frame.timestamp,
        "image_url": frame.s3_url,
        "instrument": frame.instrument,
        "wavelength": frame.wavelength,
        "metadata": frame.metadata
    })
