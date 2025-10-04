from pydantic import BaseModel
from typing import List
from pydantic import Field

class FeatureImportance(BaseModel):
    feature_name: str = Field(..., description="Feature name")
    importance_value: float = Field(..., description="Importance value SHAP")
    feature_category: str = Field(..., description="Feature category (orbital, stellar, transit, etc.)")
    description: str = Field(..., description="Description of what the feature represents")

class ModelExplanation(BaseModel):
    model_name: str = Field(..., description="Model name (rf, hgb, ensemble)")
    base_value: float = Field(..., description="Base value of the model")
    prediction_probability: float = Field(..., description="Prediction probability")
    feature_importances: List[FeatureImportance] = Field(..., description="List of feature importances")
    explanation_summary: str = Field(..., description="Textual explanation summary")

class ExplainabilityResponse(BaseModel):
    explanations: List[ModelExplanation] = Field(..., description="Explanations of each model")
    overall_summary: str = Field(..., description="Overall decision summary")
    confidence_score: float = Field(..., description="Confidence score in the explanation")
    key_factors: List[str] = Field(..., description="Key factors that influenced the decision")