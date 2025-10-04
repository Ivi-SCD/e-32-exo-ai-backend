from sqlalchemy.orm import Session
from app.database.models import (
    User, Session as DBSession, Prediction, Explanation, Comparison,
    BatchJob, BatchCandidate, ModelMetrics
)
from app.schemas import PredictionGeneralResponse, ExplainabilityResponse, ComparisonResponse
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

class DatabaseService:
    """Service for database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # User operations
    def create_user(self, name: str, email: str) -> User:
        """Create a new user"""
        user = User(name=name, email=email)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_or_create_anonymous_session(self) -> DBSession:
        """Get or create anonymous session"""
        # For now, create a new session for each request
        # In production, you'd want to use cookies/tokens
        session = DBSession()
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
    
    # Prediction operations
    def save_prediction(
        self, 
        session_id: str, 
        input_data: Dict[str, Any], 
        prediction: PredictionGeneralResponse,
        processing_time: float,
        data_quality_score: float,
        user_id: Optional[int] = None
    ) -> Prediction:
        """Save prediction to database"""
        
        # Extract planet data
        planet_data = {
            "planet": prediction.planet.model_dump() if prediction.planet else None,
            "star": prediction.star.model_dump() if prediction.star else None,
            "transit": prediction.transit.model_dump() if prediction.transit else None,
            "derived_features": prediction.derived_features.model_dump() if prediction.derived_features else None,
            "comparison_to_earth": prediction.comparison_to_earth.model_dump() if prediction.comparison_to_earth else None
        }
        
        # Extract quality flags
        quality_flags = prediction.quality_flags.model_dump() if prediction.quality_flags else None
        
        db_prediction = Prediction(
            session_id=session_id,
            user_id=user_id,
            input_data=input_data,
            prob_rf=prediction.prob_rf,
            prob_hgb=prediction.prob_hgb,
            prob_ens=prediction.prob_ens,
            label_rf=prediction.label_rf,
            label_hgb=prediction.label_hgb,
            label_ens=prediction.label_ens,
            planet_data=planet_data,
            quality_flags=quality_flags,
            processing_time=processing_time,
            data_quality_score=data_quality_score
        )
        
        self.db.add(db_prediction)
        self.db.commit()
        self.db.refresh(db_prediction)
        return db_prediction
    
    def save_explanation(self, prediction_id: str, explanation: ExplainabilityResponse) -> Explanation:
        """Save explanation to database"""
        
        # Extract feature importances for each model
        feature_importances = []
        for exp in explanation.explanations:
            feature_importances.append({
                "model_name": exp.model_name,
                "base_value": exp.base_value,
                "prediction_probability": exp.prediction_probability,
                "feature_importances": [f.model_dump() for f in exp.feature_importances],
                "explanation_summary": exp.explanation_summary
            })
        
        db_explanation = Explanation(
            prediction_id=prediction_id,
            model_name="Ensemble",  # We'll save the overall explanation
            base_value=0.5,  # Default
            prediction_probability=explanation.explanations[0].prediction_probability if explanation.explanations else 0.0,
            feature_importances=feature_importances,
            explanation_summary=explanation.overall_summary,
            confidence_score=explanation.confidence_score,
            key_factors=explanation.key_factors,
            overall_summary=explanation.overall_summary
        )
        
        self.db.add(db_explanation)
        self.db.commit()
        self.db.refresh(db_explanation)
        return db_explanation
    
    def save_comparison(self, prediction_id: str, comparison: ComparisonResponse) -> Comparison:
        """Save comparison to database"""
        
        # Extract similar exoplanets data
        similar_exoplanets = [planet.model_dump() for planet in comparison.similar_exoplanets]
        
        db_comparison = Comparison(
            prediction_id=prediction_id,
            similar_exoplanets=similar_exoplanets,
            comparison_summary=comparison.comparison_summary,
            uniqueness_score=comparison.uniqueness_score,
            scientific_interest=comparison.scientific_interest
        )
        
        self.db.add(db_comparison)
        self.db.commit()
        self.db.refresh(db_comparison)
        return db_comparison
    
    def get_prediction_by_id(self, prediction_id: str) -> Optional[Prediction]:
        """Get prediction by ID"""
        return self.db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    def get_latest_prediction(self, session_id: str) -> Optional[Prediction]:
        """Get latest prediction for session"""
        return self.db.query(Prediction).filter(
            Prediction.session_id == session_id
        ).order_by(Prediction.created_at.desc()).first()
    
    def get_predictions_by_session(self, session_id: str, limit: int = 10) -> List[Prediction]:
        """Get predictions for session"""
        return self.db.query(Prediction).filter(
            Prediction.session_id == session_id
        ).order_by(Prediction.created_at.desc()).limit(limit).all()
    
    # Batch operations
    def save_batch_job(
        self, 
        session_id: str, 
        filename: Optional[str], 
        dataset_type: Optional[str],
        total_candidates: int,
        user_id: Optional[int] = None
    ) -> BatchJob:
        """Save batch job to database"""
        
        batch_job = BatchJob(
            session_id=session_id,
            user_id=user_id,
            filename=filename,
            dataset_type=dataset_type,
            total_candidates=total_candidates,
            processed_candidates=0,
            status="pending"
        )
        
        self.db.add(batch_job)
        self.db.commit()
        self.db.refresh(batch_job)
        return batch_job
    
    def update_batch_job(
        self, 
        batch_job_id: str, 
        status: str, 
        processed_candidates: int,
        results: Optional[Dict[str, Any]] = None,
        summary_statistics: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> BatchJob:
        """Update batch job status and results"""
        
        batch_job = self.db.query(BatchJob).filter(BatchJob.id == batch_job_id).first()
        if not batch_job:
            raise ValueError(f"Batch job {batch_job_id} not found")
        
        batch_job.status = status
        batch_job.processed_candidates = processed_candidates
        
        if results:
            batch_job.results = results
        if summary_statistics:
            batch_job.summary_statistics = summary_statistics
        if processing_time:
            batch_job.processing_time = processing_time
        if error_message:
            batch_job.error_message = error_message
        
        if status == "completed":
            batch_job.completed_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(batch_job)
        return batch_job
    
    def save_batch_candidates(self, batch_job_id: str, candidates: List[Dict[str, Any]]) -> List[BatchCandidate]:
        """Save batch candidates to database"""
        
        db_candidates = []
        for candidate in candidates:
            db_candidate = BatchCandidate(
                batch_job_id=batch_job_id,
                candidate_id=candidate["candidate_id"],
                input_data=candidate.get("input_data", {}),
                prediction_results=candidate.get("prediction_results", {}),
                interestingness_score=candidate.get("interestingness_score", 0.0),
                ranking_position=candidate.get("ranking_position", 0),
                key_highlights=candidate.get("key_highlights", [])
            )
            db_candidates.append(db_candidate)
            self.db.add(db_candidate)
        
        self.db.commit()
        return db_candidates
    
    def get_batch_job(self, batch_job_id: str) -> Optional[BatchJob]:
        """Get batch job by ID"""
        return self.db.query(BatchJob).filter(BatchJob.id == batch_job_id).first()
    
    def get_batch_candidates(self, batch_job_id: str) -> List[BatchCandidate]:
        """Get candidates for batch job"""
        return self.db.query(BatchCandidate).filter(
            BatchCandidate.batch_job_id == batch_job_id
        ).order_by(BatchCandidate.ranking_position).all()
    
    # Model metrics
    def get_active_model_metrics(self) -> Optional[ModelMetrics]:
        """Get active model metrics"""
        return self.db.query(ModelMetrics).filter(ModelMetrics.is_active == True).first()
    
    def get_model_metrics_history(self, limit: int = 10) -> List[ModelMetrics]:
        """Get model metrics history"""
        return self.db.query(ModelMetrics).order_by(
            ModelMetrics.created_at.desc()
        ).limit(limit).all()
    
    # Statistics
    def get_prediction_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get prediction statistics"""
        query = self.db.query(Prediction)
        
        if session_id:
            query = query.filter(Prediction.session_id == session_id)
        
        total_predictions = query.count()
        
        if total_predictions == 0:
            return {
                "total_predictions": 0,
                "avg_processing_time": 0.0,
                "avg_data_quality": 0.0,
                "prediction_distribution": {}
            }
        
        # Calculate statistics
        avg_processing_time = query.with_entities(
            self.db.func.avg(Prediction.processing_time)
        ).scalar() or 0.0
        
        avg_data_quality = query.with_entities(
            self.db.func.avg(Prediction.data_quality_score)
        ).scalar() or 0.0
        
        # Prediction distribution
        distribution = {}
        for pred in query.all():
            label = pred.label_ens
            distribution[label] = distribution.get(label, 0) + 1
        
        return {
            "total_predictions": total_predictions,
            "avg_processing_time": float(avg_processing_time),
            "avg_data_quality": float(avg_data_quality),
            "prediction_distribution": distribution
        }
