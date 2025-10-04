from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ModelHyperparameters(BaseModel):
    """Hyperparameters for model training"""
    
    # Random Forest parameters
    rf_n_estimators: int = Field(200, ge=1, le=1000, description="Number of trees in the forest")
    rf_max_depth: int = Field(15, ge=1, le=50, description="Maximum depth of the tree")
    rf_min_samples_split: int = Field(2, ge=2, le=20, description="Minimum samples to split a node")
    rf_min_samples_leaf: int = Field(1, ge=1, le=10, description="Minimum samples in a leaf")
    rf_max_features: str = Field("sqrt", description="Number of features to consider for best split")
    rf_random_state: int = Field(42, description="Random state for reproducibility")
    
    # HistGradientBoosting parameters
    hgb_max_iter: int = Field(100, ge=1, le=1000, description="Maximum number of iterations")
    hgb_learning_rate: float = Field(0.1, ge=0.01, le=1.0, description="Learning rate")
    hgb_max_depth: int = Field(31, ge=1, le=50, description="Maximum depth of the trees")
    hgb_min_samples_leaf: int = Field(20, ge=1, le=100, description="Minimum samples in a leaf")
    hgb_random_state: int = Field(42, description="Random state for reproducibility")
    
    # Ensemble parameters
    ensemble_weights: List[float] = Field([1, 2, 3], description="Weights for ensemble voting")
    threshold: float = Field(0.363, ge=0.1, le=0.9, description="Classification threshold")
    
    # Training parameters
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set size")
    random_state: int = Field(42, description="Random state for train/test split")
    cv_folds: int = Field(5, ge=3, le=10, description="Number of cross-validation folds")

class ModelVersion(BaseModel):
    """Model version information"""
    id: str
    version: str
    hyperparameters: ModelHyperparameters
    performance_metrics: Dict[str, Any]
    is_active: bool
    created_at: datetime
    training_time_seconds: float
    dataset_size: int
    features_used: int

class TrainingRequest(BaseModel):
    """Request to train a new model"""
    hyperparameters: ModelHyperparameters
    model_name: str = Field("Custom Model", description="Name for the new model")
    description: Optional[str] = Field(None, description="Description of the model")

class TrainingResponse(BaseModel):
    """Response from model training"""
    model_version: ModelVersion
    training_logs: List[str]
    performance_comparison: Optional[Dict[str, Any]] = None
    status: str  # "success", "failed", "in_progress"

class ModelComparison(BaseModel):
    """Comparison between model versions"""
    current_version: ModelVersion
    new_version: ModelVersion
    performance_delta: Dict[str, float]  # Difference in metrics
    improvement: bool  # Whether new version is better
    recommendation: str  # "keep_current", "deploy_new", "needs_review"

class ModelListResponse(BaseModel):
    """List of all model versions"""
    models: List[ModelVersion]
    active_model: Optional[ModelVersion]
    total_models: int

class ModelDeploymentRequest(BaseModel):
    """Request to deploy a model version"""
    model_version_id: str
    force_deploy: bool = Field(False, description="Force deployment even if performance is worse")

class ModelDeploymentResponse(BaseModel):
    """Response from model deployment"""
    success: bool
    previous_model: Optional[ModelVersion]
    new_model: ModelVersion
    message: str
