from fastapi import APIRouter, HTTPException, Depends
from app.schemas import PredictionUserRequest, PredictionScientistRequest, PredictionGeneralResponse
from app.services import ModelService
from app.database import get_db
from app.services.database_service import DatabaseService
from sqlalchemy.orm import Session
import time

router = APIRouter()
model_service = ModelService()

@router.post("/user", response_model=PredictionGeneralResponse)
async def predict_user(request: PredictionUserRequest, db: Session = Depends(get_db)):
    """
    Predict if a celestial object is a planet candidate based on user-level features.
    """
    try:
        start_time = time.time()
        
        # Make prediction
        prediction = model_service.predict(request.model_dump())
        
        # Calculate processing time and data quality
        processing_time = time.time() - start_time
        data_quality_score = calculate_data_quality_score(prediction)
        
        # Save to database
        db_service = DatabaseService(db)
        session = db_service.get_or_create_anonymous_session()
        
        db_prediction = db_service.save_prediction(
            session_id=session.id,
            input_data=request.model_dump(),
            prediction=prediction,
            processing_time=processing_time,
            data_quality_score=data_quality_score
        )
        
        # Add prediction ID to response (we'll need to modify the schema)
        prediction.prediction_id = db_prediction.id
        
        return prediction
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/scientist", response_model=PredictionGeneralResponse)
async def predict_scientist(request: PredictionScientistRequest, db: Session = Depends(get_db)):
    """
    Predict if a celestial object is a planet candidate based on scientist-level features.
    """
    try:
        start_time = time.time()
        
        # Make prediction
        prediction = model_service.predict(request.model_dump())
        
        # Calculate processing time and data quality
        processing_time = time.time() - start_time
        data_quality_score = calculate_data_quality_score(prediction)
        
        # Save to database
        db_service = DatabaseService(db)
        session = db_service.get_or_create_anonymous_session()
        
        db_prediction = db_service.save_prediction(
            session_id=session.id,
            input_data=request.model_dump(),
            prediction=prediction,
            processing_time=processing_time,
            data_quality_score=data_quality_score
        )
        
        # Add prediction ID to response
        prediction.prediction_id = db_prediction.id
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_data_quality_score(prediction: PredictionGeneralResponse) -> float:
    """Calculate data quality score based on imputation flags"""
    if not prediction.quality_flags:
        return 1.0
    
    imputed_count = sum([
        prediction.quality_flags.planet_mass_imputed,
        prediction.quality_flags.stellar_mass_imputed,
        prediction.quality_flags.planet_radius_imputed
    ])
    
    return max(0, 1 - imputed_count * 0.33)  # 0.33 penalty per imputed value

