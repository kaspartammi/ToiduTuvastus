# api/server.py
from fastapi import FastAPI
from api import routes
from pipeline.analyze import Analyzer

# TODO: load your class names and classifier weights
CLASS_NAMES = [...]  # list of 101 food names
CLF_WEIGHTS = "weights/efficientnet_food101.pth"

def create_app() -> FastAPI:
    app = FastAPI(title="Food Calorie Estimation Service")

    routes.analyzer = Analyzer(CLF_WEIGHTS, CLASS_NAMES)
    app.include_router(routes.router)

    return app
