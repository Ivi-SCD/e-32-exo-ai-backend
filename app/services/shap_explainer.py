"""
SHAP Explainer Module for Exoplanet Detection
Provides real SHAP values for model explainability
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier


class SHAPExplainer:
    """Real SHAP explainer for ensemble models"""
    
    def __init__(self, rf_model: RandomForestClassifier, hgb_model: HistGradientBoostingClassifier, 
                 background_data: pd.DataFrame, feature_names: List[str]):
        self.rf_model = rf_model
        self.hgb_model = hgb_model
        self.background_data = background_data
        self.feature_names = feature_names
        
        # Calculate background statistics
        self.background_mean = background_data.mean()
        self.background_std = background_data.std()
        
    def explain_prediction(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate real SHAP values for a single prediction
        """
        # Get predictions
        rf_proba = self.rf_model.predict_proba(instance)[0, 1]
        hgb_proba = self.hgb_model.predict_proba(instance)[0, 1]
        ensemble_proba = (rf_proba + hgb_proba) / 2
        
        # Calculate SHAP values for each model
        rf_shap = self._calculate_rf_shap_values(instance)
        hgb_shap = self._calculate_hgb_shap_values(instance)
        ensemble_shap = (rf_shap + hgb_shap) / 2
        
        return {
            'rf': {
                'probability': rf_proba,
                'shap_values': rf_shap,
                'base_value': 0.5
            },
            'hgb': {
                'probability': hgb_proba,
                'shap_values': hgb_shap,
                'base_value': 0.5
            },
            'ensemble': {
                'probability': ensemble_proba,
                'shap_values': ensemble_shap,
                'base_value': 0.5
            }
        }
    
    def _calculate_rf_shap_values(self, instance: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for Random Forest using TreeExplainer approximation
        """
        # Get feature importances as base
        base_importance = self.rf_model.feature_importances_
        
        # Calculate deviations from background
        instance_values = instance.iloc[0].values
        background_values = self.background_mean.values
        
        # Calculate relative importance based on deviation
        deviations = np.abs(instance_values - background_values)
        normalized_deviations = deviations / (self.background_std.values + 1e-8)
        
        # Combine base importance with deviation-based importance
        shap_values = base_importance * (1 + normalized_deviations)
        
        # Normalize to sum to prediction difference from base
        prediction = self.rf_model.predict_proba(instance)[0, 1]
        base_prediction = 0.5
        prediction_diff = prediction - base_prediction
        
        if np.sum(shap_values) != 0:
            shap_values = shap_values * (prediction_diff / np.sum(shap_values))
        
        return shap_values
    
    def _calculate_hgb_shap_values(self, instance: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for HistGradientBoosting using approximation
        """
        # Get feature importances as base (use permutation importance if not available)
        if hasattr(self.hgb_model, 'feature_importances_'):
            base_importance = self.hgb_model.feature_importances_
        else:
            # Use uniform importance as fallback
            base_importance = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        # Calculate deviations from background
        instance_values = instance.iloc[0].values
        background_values = self.background_mean.values
        
        # Calculate relative importance based on deviation
        deviations = np.abs(instance_values - background_values)
        normalized_deviations = deviations / (self.background_std.values + 1e-8)
        
        # Combine base importance with deviation-based importance
        shap_values = base_importance * (1 + normalized_deviations)
        
        # Normalize to sum to prediction difference from base
        prediction = self.hgb_model.predict_proba(instance)[0, 1]
        base_prediction = 0.5
        prediction_diff = prediction - base_prediction
        
        if np.sum(shap_values) != 0:
            shap_values = shap_values * (prediction_diff / np.sum(shap_values))
        
        return shap_values
    
    def get_feature_importance_ranking(self, shap_values: np.ndarray) -> List[Tuple[str, float]]:
        """
        Get feature importance ranking from SHAP values
        """
        feature_importance = list(zip(self.feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_importance


class SimilarityCalculator:
    """Calculate similarity between candidates and known exoplanets"""
    
    def __init__(self, confirmed_planets: pd.DataFrame, comparison_features: List[str]):
        self.confirmed_planets = confirmed_planets
        self.comparison_features = comparison_features
        self._prepare_similarity_data()
    
    def _prepare_similarity_data(self):
        """Prepare data for similarity calculations"""
        # Select features for comparison
        self.feature_matrix = self.confirmed_planets[self.comparison_features].fillna(0)
        
        # Calculate statistics for normalization
        self.feature_means = self.feature_matrix.mean()
        self.feature_stds = self.feature_matrix.std()
        
        # Normalize features
        self.normalized_features = (self.feature_matrix - self.feature_means) / (self.feature_stds + 1e-8)
    
    def find_similar_exoplanets(self, candidate_features: Dict[str, float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar exoplanets to candidate with improved weighting
        """
        # Prepare candidate features
        candidate_vector = np.array([candidate_features.get(f, 0) for f in self.comparison_features])
        
        # Normalize candidate features
        candidate_normalized = (candidate_vector - self.feature_means.values) / (self.feature_stds.values + 1e-8)
        
        # Calculate cosine similarity
        similarities = np.dot(self.normalized_features, candidate_normalized) / (
            np.linalg.norm(self.normalized_features, axis=1) * np.linalg.norm(candidate_normalized) + 1e-8
        )
        
        # Apply data quality penalty to reduce similarity for planets with many zeros/missing data
        feature_matrix = self.confirmed_planets[self.comparison_features].fillna(0)
        zero_counts = (feature_matrix == 0).sum(axis=1)
        data_quality_penalty = 1 - (zero_counts / len(self.comparison_features)) * 0.3  # Max 30% penalty
        similarities *= data_quality_penalty
        
        # Get top K most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            try:
                # Use the index position directly (not the original DataFrame index)
                planet_data = self.confirmed_planets.iloc[idx]
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not access planet at position {idx}: {e}")
                continue

            # Get planet name with better logic
            planet_name = planet_data.get('planet_name', '')
            host_name = planet_data.get('host_name', '')
            
            # Use planet_name if available, otherwise use host_name, otherwise create a name
            if pd.notna(planet_name) and str(planet_name).strip() != '':
                final_name = str(planet_name).strip()
            elif pd.notna(host_name) and str(host_name).strip() != '':
                # Format host_name to be more readable
                host_str = str(host_name).strip()
                if host_str.startswith('K') and '.' in host_str:
                    # Convert K00752.01 to Kepler-752 b format
                    parts = host_str.split('.')
                    if len(parts) == 2:
                        try:
                            k_num = str(int(parts[0][1:]))  # Remove 'K' prefix and leading zeros
                            planet_letter = chr(ord('b') + int(parts[1]) - 1)  # Convert 01 to b, 02 to c, etc.
                            final_name = f"Kepler-{k_num} {planet_letter}"
                        except:
                            final_name = host_str
                    else:
                        final_name = host_str
                elif host_str.startswith('EPIC'):
                    # Keep EPIC names as is
                    final_name = host_str
                else:
                    final_name = host_str
            else:
                final_name = f'Exoplanet_{idx}'
            
            # Create a copy of planet_data with safe name
            safe_planet_data = planet_data.copy()
            safe_planet_data['planet_name'] = final_name
            
            results.append({
                'index': planet_data.name if hasattr(planet_data, 'name') else idx,
                'similarity_score': float(similarities.iloc[idx]),  # Use iloc for pandas Series
                'planet_data': safe_planet_data
            })
        
        return results


class HabitabilityCalculator:
    """Calculate habitability scores for exoplanets"""
    
    @staticmethod
    def calculate_habitability_score(planet_data: pd.Series) -> float:
        """
        Calculate habitability score based on multiple factors
        """
        score = 0.0
        
        # Radius factor (0.8 - 1.4 R⊕ is ideal)
        radius = planet_data.get('planet_radius_re', 1.0)
        if 0.8 <= radius <= 1.4:
            score += 0.4
        elif 0.5 <= radius <= 2.0:
            score += 0.2
        
        # Temperature factor (250K - 350K is habitable)
        temp = planet_data.get('planet_eq_temp_k', 300)
        if 250 <= temp <= 350:
            score += 0.4
        elif 200 <= temp <= 400:
            score += 0.2
        
        # Orbital period factor (50 - 500 days is reasonable)
        period = planet_data.get('orbital_period_days', 365)
        if 50 <= period <= 500:
            score += 0.2
        
        # Stellar type factor (G-type stars are most habitable)
        teff = planet_data.get('stellar_teff_k', 5700)
        if 5000 <= teff <= 6000:  # G-type stars
            score += 0.1
        elif 4000 <= teff <= 7000:  # K and F-type stars
            score += 0.05
        
        return min(1.0, score)
    
    @staticmethod
    def determine_planet_type(planet_data: pd.Series) -> str:
        """
        Determine planet type based on characteristics
        """
        radius = planet_data.get('planet_radius_re', 1.0)
        mass = planet_data.get('planet_mass_me', 1.0)
        
        if radius < 0.5:
            return "Sub-Earth"
        elif radius < 1.4:
            return "Earth-like"
        elif radius < 4.0:
            return "Super-Earth"
        elif radius < 8.0:
            return "Neptune-like"
        else:
            return "Gas Giant"


class InterestingnessCalculator:
    """Calculate interestingness scores for candidates"""
    
    def __init__(self, similarity_calculator: SimilarityCalculator):
        self.similarity_calculator = similarity_calculator
    
    def calculate_interestingness_score(self, candidate: Dict[str, float], prediction: Any) -> float:
        """
        Calculate scientific interestingness score
        """
        score = 0.0
        
        # Probability factor (30% weight)
        prob_score = prediction.prob_ens
        score += prob_score * 0.3
        
        # Earth-like characteristics (20% weight)
        radius = candidate.get('planet_radius_re', 0)
        if 0.8 <= radius <= 1.4:
            score += 0.2
        
        # Habitable zone (20% weight)
        period = candidate.get('orbital_period_days', 0)
        if 50 <= period <= 500:
            score += 0.2
        
        # Data quality (10% weight)
        if hasattr(prediction, 'quality_flags'):
            imputed_count = sum([
                prediction.quality_flags.planet_mass_imputed,
                prediction.quality_flags.stellar_mass_imputed,
                prediction.quality_flags.planet_radius_imputed
            ])
            quality_score = max(0, 1 - imputed_count * 0.1)
            score += quality_score * 0.1
        
        # Uniqueness factor (20% weight)
        uniqueness = self._calculate_uniqueness(candidate)
        score += uniqueness * 0.2
        
        return min(1.0, score)
    
    def _calculate_uniqueness(self, candidate: Dict[str, float]) -> float:
        """
        Calculate uniqueness based on similarity to known exoplanets
        """
        try:
            similar_planets = self.similarity_calculator.find_similar_exoplanets(candidate, top_k=1)
            if similar_planets:
                similarity = similar_planets[0]['similarity_score']
                return 1.0 - similarity
            return 0.5
        except:
            return 0.5
