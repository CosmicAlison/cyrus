import logging

from flask import Flask

from cyrus.core.config import settings
from cyrus.core.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s %(message)s",
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.SECRET_KEY
    app.config["JSON_SORT_KEYS"] = False

    with app.app_context():
        init_db()

    # Register blueprints
    from cyrus.api.routes.forecast import bp as forecast_bp
    from cyrus.api.routes.threats import bp as threats_bp
    from cyrus.api.routes.actions import bp as actions_bp
    from cyrus.api.routes.stream import bp as stream_bp
    from cyrus.api.routes.status import bp as status_bp

    app.register_blueprint(forecast_bp, url_prefix="/api")
    app.register_blueprint(threats_bp, url_prefix="/api")
    app.register_blueprint(actions_bp, url_prefix="/api")
    app.register_blueprint(stream_bp, url_prefix="/api")
    app.register_blueprint(status_bp, url_prefix="/api")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "cyrus-api"}

    return app