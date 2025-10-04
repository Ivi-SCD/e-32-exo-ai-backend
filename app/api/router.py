from fastapi import APIRouter
from app.api.v1.endpoints import (
    prediction,
    explain,
    compare,
    batch,
    dashboard,
    model_training
)

api_router = APIRouter(prefix="/v1")

api_router.include_router(
    prediction.router,
    prefix="/predict",
    tags=["Prediction"]
)

api_router.include_router(
    explain.router,
    prefix="/explain",
    tags=["Explainability"]
)

api_router.include_router(
    compare.router,
    prefix="/compare",
    tags=["Comparison"]
)

api_router.include_router(
    batch.router,
    prefix="/batch",
    tags=["Batch Processing"]
)

api_router.include_router(
    dashboard.router,
    prefix="/dashboard",
    tags=["Scientific Dashboard"]
)

api_router.include_router(
    model_training.router,
    prefix="/model",
    tags=["Model Training & Management"]
)
