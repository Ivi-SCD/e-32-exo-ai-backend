from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .prediction import PredictionGeneralResponse
from .explainability import ExplainabilityResponse
from .comparison import ComparisonResponse
from .batch import BatchProcessingResponse

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    brier_score: float
    f1_score: float
    recall: float
    precision: float
    accuracy: float
    training_samples: int
    missions: List[str]
    features_used: int
    threshold: float
    last_updated: str

class ChartData(BaseModel):
    """Data structure for charts"""
    labels: List[str]
    datasets: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = None

class FeatureImportanceChart(BaseModel):
    """Feature importance chart data"""
    model_name: str
    features: List[str]
    importance_values: List[float]
    colors: List[str]
    chart_type: str = "bar"

class ProbabilityComparisonChart(BaseModel):
    """Probability comparison chart data"""
    models: List[str]
    probabilities: List[float]
    colors: List[str]
    chart_type: str = "bar"

class HabitabilityRadarChart(BaseModel):
    """Habitability radar chart data"""
    categories: List[str]
    values: List[float]
    max_values: List[float]
    chart_type: str = "radar"

class ScientificDashboardResponse(BaseModel):
    """Complete scientific dashboard data"""
    # Section 1: Prediction
    prediction: PredictionGeneralResponse
    
    # Section 2: Explainability
    explainability: ExplainabilityResponse
    
    # Section 3: Comparison
    comparison: ComparisonResponse
    
    # Section 4: Model metrics
    model_metrics: ModelMetrics
    
    # Section 5: Chart data
    charts: Dict[str, Any]
    
    # Section 6: Additional metadata
    candidate_id: str
    processing_time: float
    data_quality_score: float
    scientific_interest_level: str

class DashboardChartsResponse(BaseModel):
    """Charts data for dashboard"""
    feature_importance: List[FeatureImportanceChart]
    probability_comparison: ProbabilityComparisonChart
    habitability_radar: HabitabilityRadarChart
    similarity_comparison: Optional[ChartData] = None
    batch_processing: Optional[ChartData] = None

class ExportDataResponse(BaseModel):
    """Export data structure"""
    candidate_id: str
    export_type: str  # "pdf", "csv", "json"
    data: Dict[str, Any]
    generated_at: str
    file_size: Optional[int] = None
