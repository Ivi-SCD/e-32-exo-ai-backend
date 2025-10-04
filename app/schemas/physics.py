from pydantic import BaseModel
from typing import Optional

class Planet(BaseModel):
    radius_re: Optional[float]
    mass_me: Optional[float]
    density_gcm3: Optional[float]
    orbital_period_days: Optional[float]
    semi_major_axis_au: Optional[float]
    equilibrium_temp_k: Optional[float]
    insolation_earth: Optional[float]

class Star(BaseModel):
    teff_k: Optional[float]
    radius_rsun: Optional[float]
    mass_msun: Optional[float]
    luminosity_lsun: Optional[float]
    teff_bin: Optional[str]
    teff_label: Optional[str]

class Transit(BaseModel):
    depth_ppm: Optional[float]
    duration_hours: Optional[float]
    radius_ratio: Optional[float]

class DerivedFeatures(BaseModel):
    period_mass_interaction: Optional[float]
    log_orbital_period: Optional[float]

class QualityFlags(BaseModel):
    planet_mass_imputed: bool
    stellar_mass_imputed: bool
    planet_radius_imputed: bool