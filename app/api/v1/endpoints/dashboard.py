from fastapi import APIRouter, HTTPException, Query, Depends
from app.schemas.dashboard import (
    ScientificDashboardResponse, 
    ModelMetrics, 
    DashboardChartsResponse,
    FeatureImportanceChart,
    ProbabilityComparisonChart,
    HabitabilityRadarChart,
    ChartData,
    ExportDataResponse
)
from app.services import ModelService
from app.services.database_service import DatabaseService
from app.database import get_db
from app.database.models import Explanation, Comparison, Prediction
from app.schemas import PredictionScientistRequest
from sqlalchemy.orm import Session
from typing import List, Optional
import time
import json

router = APIRouter()
model_service = ModelService()

@router.get("/scientific/{prediction_id}", response_model=ScientificDashboardResponse)
async def get_scientific_dashboard(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """
    Get complete scientific dashboard data for a saved prediction.
    
    Uses the prediction_id to fetch all data from the database instead
    of recalculating everything.
    """
    try:
        start_time = time.time()
        
        # Get data from database
        db_service = DatabaseService(db)
        
        # Get prediction from database
        db_prediction = db_service.get_prediction_by_id(prediction_id)
        if not db_prediction:
            raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
        
        from app.schemas.prediction import PredictionGeneralResponse
        from app.schemas.physics import Planet, Star, Transit, DerivedFeatures, QualityFlags
        from app.schemas.comparison import ComparisonToEarth
        
        # Reconstruct planet data
        planet_data = db_prediction.planet_data or {}
        planet = Planet(**planet_data.get("planet", {})) if planet_data.get("planet") else None
        star = Star(**planet_data.get("star", {})) if planet_data.get("star") else None
        transit = Transit(**planet_data.get("transit", {})) if planet_data.get("transit") else None
        derived_features = DerivedFeatures(**planet_data.get("derived_features", {})) if planet_data.get("derived_features") else None
        comparison_to_earth = ComparisonToEarth(**planet_data.get("comparison_to_earth", {})) if planet_data.get("comparison_to_earth") else None
        
        # Reconstruct quality flags
        quality_flags_data = db_prediction.quality_flags or {}
        quality_flags = QualityFlags(**quality_flags_data) if quality_flags_data else None
        
        # Create prediction response
        prediction = PredictionGeneralResponse(
            prob_rf=db_prediction.prob_rf,
            label_rf=db_prediction.label_rf,
            prob_hgb=db_prediction.prob_hgb,
            label_hgb=db_prediction.label_hgb,
            prob_ens=db_prediction.prob_ens,
            label_ens=db_prediction.label_ens,
            planet=planet,
            star=star,
            transit=transit,
            derived_features=derived_features,
            quality_flags=quality_flags,
            comparison_to_earth=comparison_to_earth,
            prediction_id=prediction_id
        )
        
        # Get explainability from database or generate if not exists
        db_explanation = db.query(Explanation).filter(Explanation.prediction_id == prediction_id).first()
        if db_explanation:
            # Reconstruct explainability from database
            from app.schemas.explainability import ExplainabilityResponse, ModelExplanation, FeatureImportance
            
            explanations = []
            for exp_data in db_explanation.feature_importances:
                feature_importances = [FeatureImportance(**f) for f in exp_data.get("feature_importances", [])]
                explanations.append(ModelExplanation(
                    model_name=exp_data["model_name"],
                    base_value=exp_data["base_value"],
                    prediction_probability=exp_data["prediction_probability"],
                    feature_importances=feature_importances,
                    explanation_summary=exp_data["explanation_summary"]
                ))
            
            explainability = ExplainabilityResponse(
                explanations=explanations,
                overall_summary=db_explanation.overall_summary,
                confidence_score=db_explanation.confidence_score,
                key_factors=db_explanation.key_factors
            )
        else:
            # Generate explainability if not in database
            explainability = model_service.explain_prediction(db_prediction.input_data)
            # Save to database for future use
            db_service.save_explanation(prediction_id, explainability)
        
        # Get comparison from database or generate if not exists
        db_comparison = db.query(Comparison).filter(Comparison.prediction_id == prediction_id).first()
        if db_comparison:
            # Reconstruct comparison from database
            from app.schemas.comparison import ComparisonResponse, SimilarExoplanet
            
            similar_exoplanets = [SimilarExoplanet(**planet) for planet in db_comparison.similar_exoplanets]
            comparison = ComparisonResponse(
                similar_exoplanets=similar_exoplanets,
                comparison_summary=db_comparison.comparison_summary,
                uniqueness_score=db_comparison.uniqueness_score,
                scientific_interest=db_comparison.scientific_interest
            )
        else:
            # Generate comparison if not in database
            comparison = model_service.find_similar_exoplanets(db_prediction.input_data, top_k=5)
            # Save to database for future use
            db_service.save_comparison(prediction_id, comparison)
        
        # Get model metrics from database
        model_metrics_db = db_service.get_active_model_metrics()
        if model_metrics_db:
            model_metrics = ModelMetrics(
                roc_auc=model_metrics_db.roc_auc,
                pr_auc=model_metrics_db.pr_auc,
                balanced_accuracy=model_metrics_db.balanced_accuracy,
                brier_score=model_metrics_db.brier_score,
                f1_score=model_metrics_db.f1_score,
                recall=model_metrics_db.recall,
                precision=model_metrics_db.precision,
                accuracy=model_metrics_db.accuracy,
                training_samples=model_metrics_db.training_samples,
                missions=model_metrics_db.missions,
                features_used=model_metrics_db.features_used,
                threshold=model_metrics_db.threshold,
                last_updated=model_metrics_db.created_at.strftime("%Y-%m-%d")
            )
        else:
            # Fallback to default metrics
            model_metrics = ModelMetrics(
                roc_auc=0.983, pr_auc=0.973, balanced_accuracy=0.929, brier_score=0.051,
                f1_score=0.888, recall=0.969, precision=0.830, accuracy=0.960,
                training_samples=21225, missions=["K2", "Kepler", "TESS"],
                features_used=67, threshold=0.363, last_updated="2025-01-02"
            )
        
        # Prepare chart data
        charts = prepare_chart_data(prediction, explainability, comparison)
        
        # Calculate additional metrics
        processing_time = time.time() - start_time
        data_quality_score = db_prediction.data_quality_score
        scientific_interest_level = assess_scientific_interest(prediction, comparison)
        
        return ScientificDashboardResponse(
            prediction=prediction,
            explainability=explainability,
            comparison=comparison,
            model_metrics=model_metrics,
            charts=charts,
            candidate_id=prediction_id,
            processing_time=processing_time,
            data_quality_score=data_quality_score,
            scientific_interest_level=scientific_interest_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")

@router.get("/predictions", response_model=List[dict])
async def list_predictions(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(10, description="Number of predictions to return"),
    db: Session = Depends(get_db)
):
    """
    List recent predictions with basic info.
    """
    try:
        db_service = DatabaseService(db)
        
        if session_id:
            predictions = db_service.get_predictions_by_session(session_id, limit)
        else:
            # Get latest predictions from any session
            predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()
        
        result = []
        for pred in predictions:
            result.append({
                "prediction_id": pred.id,
                "prob_ens": pred.prob_ens,
                "label_ens": pred.label_ens,
                "created_at": pred.created_at.isoformat(),
                "processing_time": pred.processing_time,
                "data_quality_score": pred.data_quality_score,
                "session_id": pred.session_id
            })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing predictions: {str(e)}")

@router.get("/latest", response_model=ScientificDashboardResponse)
async def get_latest_dashboard(
    session_id: Optional[str] = Query(None, description="Session ID"),
    db: Session = Depends(get_db)
):
    """
    Get dashboard for the latest prediction.
    """
    try:
        db_service = DatabaseService(db)
        
        if session_id:
            prediction = db_service.get_latest_prediction(session_id)
        else:
            # Get latest prediction from any session
            prediction = db.query(Prediction).order_by(Prediction.created_at.desc()).first()
        
        if not prediction:
            raise HTTPException(status_code=404, detail="No predictions found")
        
        # Use the main dashboard endpoint
        return await get_scientific_dashboard(prediction.id, db)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting latest dashboard: {str(e)}")

@router.get("/model/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """
    Get static model performance metrics.
    """
    return ModelMetrics(
        roc_auc=0.983,
        pr_auc=0.973,
        balanced_accuracy=0.929,
        brier_score=0.051,
        f1_score=0.888,
        recall=0.969,
        precision=0.830,
        accuracy=0.960,
        training_samples=21225,
        missions=["K2", "Kepler", "TESS"],
        features_used=67,
        threshold=0.363,
        last_updated="2025-01-02"
    )

@router.get("/charts/feature-importance/{candidate_id}", response_model=List[FeatureImportanceChart])
async def get_feature_importance_charts(
    candidate_id: str,
    orbital_period_days: float = Query(...),
    transit_depth_ppm: float = Query(...),
    planet_radius_re: float = Query(...),
    planet_mass_me: float = Query(...),
    stellar_teff_k: float = Query(...),
    stellar_radius_rsun: float = Query(...),
    stellar_mass_msun: float = Query(...)
):
    """
    Get feature importance chart data for all models.
    """
    try:
        input_data = {
            "orbital_period_days": orbital_period_days,
            "transit_depth_ppm": transit_depth_ppm,
            "planet_radius_re": planet_radius_re,
            "planet_mass_me": planet_mass_me,
            "stellar_teff_k": stellar_teff_k,
            "stellar_radius_rsun": stellar_radius_rsun,
            "stellar_mass_msun": stellar_mass_msun
        }
        
        explainability = model_service.explain_prediction(input_data)
        
        charts = []
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
        
        for i, explanation in enumerate(explainability.explanations):
            # Get top 10 features
            top_features = explanation.feature_importances[:10]
            
            features = [f.feature_name for f in top_features]
            importance_values = [f.importance_value for f in top_features]
            
            charts.append(FeatureImportanceChart(
                model_name=explanation.model_name,
                features=features,
                importance_values=importance_values,
                colors=colors[:len(features)]
            ))
        
        return charts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating feature importance charts: {str(e)}")

@router.get("/charts/probability-comparison/{candidate_id}", response_model=ProbabilityComparisonChart)
async def get_probability_comparison_chart(
    candidate_id: str,
    orbital_period_days: float = Query(...),
    transit_depth_ppm: float = Query(...),
    planet_radius_re: float = Query(...),
    planet_mass_me: float = Query(...),
    stellar_teff_k: float = Query(...),
    stellar_radius_rsun: float = Query(...),
    stellar_mass_msun: float = Query(...)
):
    """
    Get probability comparison chart data.
    """
    try:
        input_data = {
            "orbital_period_days": orbital_period_days,
            "transit_depth_ppm": transit_depth_ppm,
            "planet_radius_re": planet_radius_re,
            "planet_mass_me": planet_mass_me,
            "stellar_teff_k": stellar_teff_k,
            "stellar_radius_rsun": stellar_radius_rsun,
            "stellar_mass_msun": stellar_mass_msun
        }
        
        prediction = model_service.predict(input_data)
        
        return ProbabilityComparisonChart(
            models=["Random Forest", "HistGradientBoosting", "Ensemble"],
            probabilities=[prediction.prob_rf, prediction.prob_hgb, prediction.prob_ens],
            colors=["#FF6B6B", "#4ECDC4", "#45B7D1"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating probability comparison chart: {str(e)}")

@router.get("/charts/habitability-radar/{candidate_id}", response_model=HabitabilityRadarChart)
async def get_habitability_radar_chart(
    candidate_id: str,
    orbital_period_days: float = Query(...),
    transit_depth_ppm: float = Query(...),
    planet_radius_re: float = Query(...),
    planet_mass_me: float = Query(...),
    stellar_teff_k: float = Query(...),
    stellar_radius_rsun: float = Query(...),
    stellar_mass_msun: float = Query(...)
):
    """
    Get habitability radar chart data.
    """
    try:
        input_data = {
            "orbital_period_days": orbital_period_days,
            "transit_depth_ppm": transit_depth_ppm,
            "planet_radius_re": planet_radius_re,
            "planet_mass_me": planet_mass_me,
            "stellar_teff_k": stellar_teff_k,
            "stellar_radius_rsun": stellar_radius_rsun,
            "stellar_mass_msun": stellar_mass_msun
        }
        
        comparison = model_service.find_similar_exoplanets(input_data, top_k=1)
        
        # Calculate habitability metrics
        radius = input_data.get("planet_radius_re", 1.0)
        period = input_data.get("orbital_period_days", 365)
        teff = input_data.get("stellar_teff_k", 5778)
        
        # Habitability categories
        categories = ["Size", "Temperature", "Orbital Period", "Stellar Type", "Habitability Zone"]
        
        # Calculate values (0-1 scale)
        size_score = 1.0 if 0.8 <= radius <= 1.4 else 0.5 if 0.5 <= radius <= 2.0 else 0.0
        temp_score = 0.8  # Placeholder - would need actual temperature calculation
        period_score = 1.0 if 50 <= period <= 500 else 0.5 if 20 <= period <= 1000 else 0.0
        stellar_score = 1.0 if 5000 <= teff <= 6000 else 0.7 if 4000 <= teff <= 7000 else 0.3
        zone_score = 0.8  # Placeholder - would need actual habitable zone calculation
        
        values = [size_score, temp_score, period_score, stellar_score, zone_score]
        max_values = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        return HabitabilityRadarChart(
            categories=categories,
            values=values,
            max_values=max_values
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating habitability radar chart: {str(e)}")

@router.get("/export/{candidate_id}", response_model=ExportDataResponse)
async def export_candidate_data(
    candidate_id: str,
    export_type: str = Query("json", description="Export type: json, csv, pdf"),
    orbital_period_days: float = Query(...),
    transit_depth_ppm: float = Query(...),
    planet_radius_re: float = Query(...),
    planet_mass_me: float = Query(...),
    stellar_teff_k: float = Query(...),
    stellar_radius_rsun: float = Query(...),
    stellar_mass_msun: float = Query(...)
):
    """
    Export candidate data in various formats.
    """
    try:
        input_data = {
            "orbital_period_days": orbital_period_days,
            "transit_depth_ppm": transit_depth_ppm,
            "planet_radius_re": planet_radius_re,
            "planet_mass_me": planet_mass_me,
            "stellar_teff_k": stellar_teff_k,
            "stellar_radius_rsun": stellar_radius_rsun,
            "stellar_mass_msun": stellar_mass_msun
        }
        
        # Get all data
        prediction = model_service.predict(input_data)
        explainability = model_service.explain_prediction(input_data)
        comparison = model_service.find_similar_exoplanets(input_data, top_k=5)
        
        # Prepare export data
        export_data = {
            "candidate_id": candidate_id,
            "prediction": prediction.model_dump(),
            "explainability": explainability.model_dump(),
            "comparison": comparison.model_dump(),
            "input_parameters": input_data,
            "export_metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "export_type": export_type
            }
        }
        
        # Calculate file size
        json_str = json.dumps(export_data, default=str)
        file_size = len(json_str.encode('utf-8'))
        
        return ExportDataResponse(
            candidate_id=candidate_id,
            export_type=export_type,
            data=export_data,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            file_size=file_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

def prepare_chart_data(prediction, explainability, comparison):
    """Prepare chart data for dashboard"""
    charts = {
        "feature_importance": [],
        "probability_comparison": {
            "models": ["Random Forest", "HistGradientBoosting", "Ensemble"],
            "probabilities": [prediction.prob_rf, prediction.prob_hgb, prediction.prob_ens]
        },
        "habitability_radar": {
            "categories": ["Size", "Temperature", "Period", "Stellar", "Zone"],
            "values": [0.8, 0.7, 0.9, 0.8, 0.6]  # Placeholder values
        },
        "similarity_comparison": {
            "planets": [p.planet_name for p in comparison.similar_exoplanets],
            "similarities": [p.similarity_score for p in comparison.similar_exoplanets]
        }
    }
    
    # Add feature importance for each model
    for explanation in explainability.explanations:
        top_features = explanation.feature_importances[:5]
        charts["feature_importance"].append({
            "model": explanation.model_name,
            "features": [f.feature_name for f in top_features],
            "values": [f.importance_value for f in top_features]
        })
    
    return charts

def calculate_data_quality_score(prediction):
    """Calculate data quality score based on imputation flags"""
    quality_flags = prediction.quality_flags
    imputed_count = sum([
        quality_flags.planet_mass_imputed,
        quality_flags.stellar_mass_imputed,
        quality_flags.planet_radius_imputed
    ])
    return max(0, 1 - imputed_count * 0.33)  # 0.33 penalty per imputed value

def assess_scientific_interest(prediction, comparison):
    """Assess scientific interest level"""
    prob = prediction.prob_ens
    
    if prob > 0.8:
        return "VERY HIGH"
    elif prob > 0.6:
        return "HIGH"
    elif prob > 0.4:
        return "MODERATE"
    else:
        return "LOW"
