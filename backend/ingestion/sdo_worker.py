import time
import requests
from datetime import datetime

from storage.s3 import upload_image
from services.frame_service import FrameService


service = FrameService()


def fetch_sdo_image():
    url = "https://example-sdo-image.jpg"
    return requests.get(url).content


def run():
    while True:
        image = fetch_sdo_image()

        timestamp = datetime.utcnow().isoformat()

        s3_key, s3_url = upload_image(image, timestamp)

        service.create({
            "timestamp": datetime.utcnow(),
            "s3_url": s3_url,
            "s3_key": s3_key,
            "instrument": "AIA",
            "wavelength": 171,
            "metadata": {"source": "SDO"}
        })

        time.sleep(300)


if __name__ == "__main__":
    run()