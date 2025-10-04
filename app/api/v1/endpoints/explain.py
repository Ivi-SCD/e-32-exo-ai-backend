from fastapi import APIRouter, HTTPException, Depends
from app.schemas import (
    PredictionUserRequest, PredictionScientistRequest, ExplainabilityResponse
)
from app.services import ModelService
from app.database import get_db
from app.services.database_service import DatabaseService
from sqlalchemy.orm import Session

router = APIRouter()
model_service = ModelService()

@router.post("/user", response_model=ExplainabilityResponse)
async def explain_user_prediction(
    request: PredictionUserRequest, 
    db: Session = Depends(get_db)
):
    """
    Explain WHY an exoplanet was classified as Candidate/Confirmed/False Positive
    usando feature importance dos modelos.
    
    This endpoint analyzes the characteristics of the candidate and returns:
    - Feature importance of each model (RF, HGB, Ensemble)
    - Textual summary of the decision
    - Confidence score in the explanation
    - Principais fatores que influenciaram a classificação
    """
    try:
        # Get or create session
        db_service = DatabaseService(db)
        session = db_service.get_or_create_anonymous_session()
        
        # Generate explanation
        explanation = model_service.explain_prediction(request.model_dump())
        
        # Save explanation to database (optional - for tracking)
        # This would require a prediction_id, which we don't have in explain-only requests
        # For now, we'll just return the explanation
        
        return explanation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scientist", response_model=ExplainabilityResponse)
async def explain_scientist_prediction(
    request: PredictionScientistRequest,
    db: Session = Depends(get_db)
):
    """
    Explain WHY an exoplanet was classified as Candidate/Confirmed/False Positive
    usando features avançadas e análise detalhada.
    
    For scientists with more complete data, this endpoint provides:
    - Detailed analysis of all features
    - Comparison with validation thresholds
    - Explanation based on astronomical knowledge
    - Scientific confidence score
    """
    try:
        # Get or create session
        db_service = DatabaseService(db)
        session = db_service.get_or_create_anonymous_session()
        
        # Generate explanation
        explanation = model_service.explain_prediction(request.model_dump())
        
        return explanation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
