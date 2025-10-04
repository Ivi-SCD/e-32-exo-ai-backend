from pydantic import BaseModel
from pydantic import Field
from typing import Optional
from .physics import Planet, Star, Transit, DerivedFeatures, QualityFlags
from .comparison import  ComparisonToEarth


class PredictionUserRequest(BaseModel):
    orbital_period_days: float = Field(..., example=12.3, description="Orbital Period (days)")
    transit_depth_ppm: float = Field(..., example=1500, description="Transit Depth (ppm)")
    planet_radius_re: float = Field(..., example=1.05, description="Planet Radius (R⊕)")
    planet_mass_me: float = Field(..., example=1.0, description="Planet Mass (M⊕)")
    stellar_teff_k: float = Field(..., example=5700, description="Stellar Temperature (K)")
    stellar_radius_rsun: float = Field(..., example=1.0, description="Stellar Radius (R☉)")
    stellar_mass_msun: float  = Field(..., example=1.0, description="Stellar Mass (M☉)")

class PredictionScientistRequest(BaseModel):
    orbital_period_days: float = Field(..., example=12.3, description="Orbital Period (days)")
    transit_depth: float = Field(..., example=1500, description="Transit Depth (ppm)")
    transit_duration: float = Field(..., example=3.2, description="Transit Duration (hours)")
    planet_radius_re: float = Field(..., example=1.05, description="Planet Radius (R⊕)")
    planet_mass_me: float = Field(..., example=1.0, description="Planet Mass (M⊕)")
    stellar_teff_k: float = Field(..., example=5700, description="Stellar Temperature (K)")
    stellar_radius_rsun: float = Field(..., example=1.0, description="Stellar Radius (R☉)")
    stellar_mass_msun: float = Field(..., example=1.0, description="Stellar Mass (M☉)")
    radius_ratio: float = Field(..., example=0.01, description="Radius Ratio (Rp/Rs)")
    semi_major_axis_au: float = Field(..., example=0.1, description="Semi-Major Axis (AU)")
    equilibrium_temp_recalc_k: float = Field(..., example=300, description="Recalculated Equilibrium Temperature (K)")
    log_orbital_period: float = Field(..., example=1.09, description="Log Orbital Period")
    period_mass_interaction: float = Field(..., example=12.3, description="Period-Mass Interaction")
    stellar_teff_bin: float = Field(..., example=5700, description="Binned Stellar Temperature")

class PredictionGeneralResponse(BaseModel):
    prob_rf: float = Field(..., example=0.78, description="Probability of being a Planet Candidate (Random Forest Model)")
    label_rf: str = Field(..., example="Planet Candidate", description="Label based on Random Forest Model Prediction")
    prob_hgb: float = Field(..., example=0.82, description="Probability of being a Planet Candidate (HistGradieentBoostingClassifier Model)")
    label_hgb: str = Field(..., example="Planet Candidate", description="Label based on HistGradieentBoostingClassifier Model Prediction")
    prob_ens: float = Field(..., example=0.85, description="Probability of being a Planet Candidate (Ensemble Model)")
    label_ens: str = Field(..., example="Planet Candidate", description="Label based on Ensemble Model Prediction")
    planet: Planet = Field(..., description="Planetary Characteristics")
    star: Star = Field(..., description="Stellar Characteristics")
    transit: Transit = Field(..., description="Transit Characteristics")
    derived_features: DerivedFeatures = Field(..., description="Derived Features")
    quality_flags: QualityFlags = Field(..., description="Quality Flags indicating imputed values")
    comparison_to_earth: ComparisonToEarth = Field(..., description="Comparison of Planet to Earth")
    prediction_id: Optional[str] = Field(None, description="Database ID of the prediction")