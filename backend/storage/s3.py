import boto3
import uuid
from config.settings import AWS_BUCKET

s3 = boto3.client("s3")


def upload_image(image_bytes, timestamp):
    key = f"sdo/{timestamp[:10]}/{uuid.uuid4()}.jpg"

    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType="image/jpeg"
    )

    url = f"https://{AWS_BUCKET}.s3.amazonaws.com/{key}"

    return key, url 
