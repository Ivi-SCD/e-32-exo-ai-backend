from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field

class SimilarExoplanet(BaseModel):
    planet_name: str = Field(..., description="Similar exoplanet name")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    distance: float = Field(..., description="Distance of the exoplanet (pc)")
    stellar_type: str = Field(..., description="Stellar type")
    planet_type: str = Field(..., description="Planet type")
    orbital_period_days: float = Field(..., description="Orbital period (days)")
    planet_radius_re: float = Field(..., description="Planet radius (R⊕)")
    equilibrium_temp_k: float = Field(..., description="Equilibrium temperature (K)")
    habitability_score: float = Field(..., description="Habitability score")
    discovery_year: int = Field(..., description="Discovery year")

class ComparisonResponse(BaseModel):
    similar_exoplanets: List[SimilarExoplanet] = Field(..., description="Similar exoplanets")
    comparison_summary: str = Field(..., description="Comparison summary")
    uniqueness_score: float = Field(..., description="Uniqueness score (0-1)")
    scientific_interest: str = Field(..., description="Scientific interest level")

class ComparisonToEarth(BaseModel):
    radius_ratio_earth: Optional[float]
    insolation_ratio_earth: Optional[float]