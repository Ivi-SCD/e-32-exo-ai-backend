from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.model_training_service import ModelTrainingService
from app.schemas.model_training import (
    TrainingRequest, TrainingResponse, ModelListResponse, 
    ModelDeploymentRequest, ModelDeploymentResponse,
    ModelHyperparameters, ModelVersion as ModelVersionSchema
)
from typing import List
import asyncio

router = APIRouter()

@router.post("/train", response_model=TrainingResponse)
async def train_new_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Train a new model with custom hyperparameters.
    
    This endpoint allows scientists to:
    - Adjust hyperparameters for Random Forest and HistGradientBoosting
    - Set ensemble weights and classification threshold
    - Compare performance with the current active model
    - Get detailed training logs and metrics
    """
    try:
        training_service = ModelTrainingService(db)
        
        # Start training in background
        result = training_service.train_new_model(request)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/versions", response_model=ModelListResponse)
async def list_model_versions(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    List all model versions with their performance metrics.
    
    Returns:
    - All model versions ordered by creation date
    - Currently active model
    - Performance comparison data
    """
    try:
        training_service = ModelTrainingService(db)
        return training_service.list_model_versions(limit)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.post("/deploy", response_model=ModelDeploymentResponse)
async def deploy_model(
    request: ModelDeploymentRequest,
    db: Session = Depends(get_db)
):
    """
    Deploy a trained model version as the active model.
    
    The system will:
    - Check if the new model performs better than the current one
    - Deactivate the current model
    - Activate the new model
    - Return deployment status and comparison
    """
    try:
        training_service = ModelTrainingService(db)
        result = training_service.deploy_model(
            model_version_id=request.model_version_id,
            force_deploy=request.force_deploy
        )
        
        if result["success"]:
            return ModelDeploymentResponse(
                success=True,
                previous_model=result.get("previous_model"),
                new_model=result["new_model"],
                message=result["message"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

@router.get("/hyperparameters/default", response_model=ModelHyperparameters)
async def get_default_hyperparameters():
    """
    Get default hyperparameters for model training.
    
    Useful for:
    - Pre-filling forms in the frontend
    - Understanding parameter ranges
    - Starting point for experimentation
    """
    return ModelHyperparameters()

@router.get("/versions/{version_id}", response_model=ModelVersionSchema)
async def get_model_version(
    version_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific model version.
    
    Returns:
    - Complete hyperparameters used
    - Performance metrics
    - Training metadata
    - Status information
    """
    try:
        from app.database.models import ModelVersion
        
        model = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model version not found")
        
        return ModelVersionSchema(
            id=model.id,
            version=model.version,
            hyperparameters=ModelHyperparameters(**model.hyperparameters),
            performance_metrics=model.performance_metrics,
            is_active=model.is_active,
            created_at=model.created_at,
            training_time_seconds=model.training_time_seconds,
            dataset_size=model.dataset_size,
            features_used=model.features_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model version: {str(e)}")

@router.get("/compare/{version_id_1}/{version_id_2}")
async def compare_model_versions(
    version_id_1: str,
    version_id_2: str,
    db: Session = Depends(get_db)
):
    """
    Compare two model versions side by side.
    
    Returns detailed comparison including:
    - Performance metrics comparison
    - Hyperparameter differences
    - Training metadata
    - Recommendation on which to use
    """
    try:
        from app.database.models import ModelVersion
        
        model1 = db.query(ModelVersion).filter(ModelVersion.id == version_id_1).first()
        model2 = db.query(ModelVersion).filter(ModelVersion.id == version_id_2).first()
        
        if not model1 or not model2:
            raise HTTPException(status_code=404, detail="One or both model versions not found")
        
        # Create comparison
        comparison = {
            "model_1": {
                "id": model1.id,
                "version": model1.version,
                "performance": model1.performance_metrics,
                "hyperparameters": model1.hyperparameters,
                "training_time": model1.training_time_seconds,
                "is_active": model1.is_active
            },
            "model_2": {
                "id": model2.id,
                "version": model2.version,
                "performance": model2.performance_metrics,
                "hyperparameters": model2.hyperparameters,
                "training_time": model2.training_time_seconds,
                "is_active": model2.is_active
            },
            "comparison": {}
        }
        
        # Calculate performance differences
        metrics_to_compare = ["ensemble_roc_auc", "ensemble_f1", "ensemble_accuracy", "ensemble_precision", "ensemble_recall"]
        
        for metric in metrics_to_compare:
            val1 = model1.performance_metrics.get(metric, 0)
            val2 = model2.performance_metrics.get(metric, 0)
            comparison["comparison"][f"{metric}_delta"] = val2 - val1
            comparison["comparison"][f"{metric}_improvement"] = val2 > val1
        
        # Overall recommendation
        improvements = sum(1 for key, value in comparison["comparison"].items() if key.endswith("_improvement") and value)
        total_metrics = len(metrics_to_compare)
        
        if improvements > total_metrics // 2:
            recommendation = "Model 2 is better"
        elif improvements < total_metrics // 2:
            recommendation = "Model 1 is better"
        else:
            recommendation = "Models are comparable"
        
        comparison["recommendation"] = recommendation
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.delete("/versions/{version_id}")
async def delete_model_version(
    version_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a model version (only if not active).
    
    This will:
    - Remove the model version from database
    - Delete associated model files
    - Clean up training logs
    """
    try:
        from app.database.models import ModelVersion, TrainingLog
        import os
        
        model = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model version not found")
        
        if model.is_active:
            raise HTTPException(status_code=400, detail="Cannot delete active model. Deploy another model first.")
        
        # Delete training logs
        db.query(TrainingLog).filter(TrainingLog.model_version_id == version_id).delete()
        
        # Delete model files
        if model.model_files:
            for file_path in model.model_files.values():
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Delete model version
        db.delete(model)
        db.commit()
        
        return {"message": f"Model version {model.version} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/status")
async def get_training_status(db: Session = Depends(get_db)):
    """
    Get current training status and system information.
    
    Returns:
    - Currently active model
    - Any models currently training
    - System statistics
    """
    try:
        from app.database.models import ModelVersion
        
        active_model = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        training_models = db.query(ModelVersion).filter(ModelVersion.training_status == "training").all()
        
        total_models = db.query(ModelVersion).count()
        
        return {
            "active_model": {
                "version": active_model.version if active_model else None,
                "created_at": active_model.created_at if active_model else None,
                "performance": active_model.performance_metrics if active_model else None
            },
            "training_models": [
                {
                    "id": model.id,
                    "version": model.version,
                    "status": model.training_status,
                    "started_at": model.training_started_at
                } for model in training_models
            ],
            "statistics": {
                "total_models": total_models,
                "active_models": 1 if active_model else 0,
                "training_models": len(training_models)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")
