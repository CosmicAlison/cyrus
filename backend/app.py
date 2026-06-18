from flask import Flask
from api.routes import routes

app = Flask(__name__)
app.register_blueprint(routes)


if __name__ == "__main__":
    app.run(port=5000, debug=True)