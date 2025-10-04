from fastapi import APIRouter, HTTPException, Query, Depends
from app.schemas import (
    PredictionUserRequest, PredictionScientistRequest, ComparisonResponse
)
from app.services import ModelService
from app.database import get_db
from app.services.database_service import DatabaseService
from sqlalchemy.orm import Session

router = APIRouter()
model_service = ModelService()

@router.post("/user", response_model=ComparisonResponse)
async def compare_user_candidate(
    request: PredictionUserRequest,
    top_k: int = Query(default=5, ge=1, le=20, description="Número de exoplanetas similares para retornar"),
    db: Session = Depends(get_db)
):
    """
    Compares the prediction with the most similar confirmed exoplanets.
    
    This endpoint finds the confirmed exoplanets most similar to the candidate
    based on the provided features and returns:
    - A list of similar exoplanets with similarity scores
    - Comparative characteristics (radius, orbital period, temperature, etc.)
    - Habitability score for each similar exoplanet
    - Comparison summary and scientific interest level
    - Uniqueness score of the candidate
    """
    try:
        # Get or create session
        db_service = DatabaseService(db)
        session = db_service.get_or_create_anonymous_session()
        
        # Find similar exoplanets
        comparison = model_service.find_similar_exoplanets(request.model_dump(), top_k=top_k)
        
        # Note: We could save the comparison to database here if we had a prediction_id
        # For standalone comparison requests, we'll just return the result
        
        return comparison
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scientist", response_model=ComparisonResponse)
async def compare_scientist_candidate(
    request: PredictionScientistRequest,
    top_k: int = Query(default=5, ge=1, le=20, description="Número de exoplanetas similares para retornar"),
    db: Session = Depends(get_db)
):
    """
    Compares the prediction with the most similar confirmed exoplanets using advanced features.
    
    For scientists with more complete data, this endpoint provides:
    - More accurate similarity analysis
    - Comparison with exoplanets of different types
    - Detailed habitability analysis
    - Classification of the planet type
    - Assessment of the scientific interest of the candidate
    """
    try:
        # Get or create session
        db_service = DatabaseService(db)
        session = db_service.get_or_create_anonymous_session()
        
        # Find similar exoplanets
        comparison = model_service.find_similar_exoplanets(request.model_dump(), top_k=top_k)
        
        return comparison
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
