from flask import Flask
from .controller.main_controller import predict

def create_app():
    app = Flask(__name__)
    app.register_blueprint(predict)
    return app
