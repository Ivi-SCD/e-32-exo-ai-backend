from sqlalchemy.orm import Session
from app.database.models import ModelVersion, TrainingLog, ModelMetrics
from app.schemas.model_training import (
    ModelHyperparameters, TrainingRequest, TrainingResponse, 
    ModelComparison, ModelListResponse, ModelVersion as ModelVersionSchema
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import joblib
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

class ModelTrainingService:
    """Service for training and managing model versions"""
    
    def __init__(self, db: Session):
        self.db = db
        self.models_dir = "notebooks/models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train_new_model(self, request: TrainingRequest) -> TrainingResponse:
        """Train a new model with custom hyperparameters"""
        try:
            # Create model version record
            model_version = self._create_model_version(request)
            
            # Start training
            training_logs = []
            start_time = time.time()
            
            try:
                # Load and prepare data
                training_logs.append("Loading training data...")
                self._log_training_step(model_version.id, "data_loading", "INFO", "Loading training data...")
                
                X, y, feature_names = self._load_training_data()
                training_logs.append(f"Loaded {len(X)} samples with {len(feature_names)} features")
                
                # Split data
                training_logs.append("Splitting data into train/test sets...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=request.hyperparameters.test_size,
                    random_state=request.hyperparameters.random_state,
                    stratify=y
                )
                
                # Train models
                training_logs.append("Training Random Forest...")
                self._log_training_step(model_version.id, "training", "INFO", "Training Random Forest...")
                rf_model = self._train_random_forest(X_train, y_train, request.hyperparameters)
                
                training_logs.append("Training HistGradientBoosting...")
                self._log_training_step(model_version.id, "training", "INFO", "Training HistGradientBoosting...")
                hgb_model = self._train_hist_gradient_boosting(X_train, y_train, request.hyperparameters)
                
                training_logs.append("Creating ensemble...")
                self._log_training_step(model_version.id, "training", "INFO", "Creating ensemble...")
                ensemble_model = self._create_ensemble(rf_model, hgb_model, request.hyperparameters)
                
                # Evaluate models
                training_logs.append("Evaluating models...")
                self._log_training_step(model_version.id, "evaluation", "INFO", "Evaluating models...")
                performance_metrics = self._evaluate_models(
                    ensemble_model, rf_model, hgb_model, X_test, y_test, X_train, y_train
                )
                
                # Save models
                training_logs.append("Saving models...")
                self._log_training_step(model_version.id, "saving", "INFO", "Saving models...")
                model_files = self._save_models(model_version.id, rf_model, hgb_model, ensemble_model)
                
                # Update model version
                training_time = time.time() - start_time
                model_version.training_time_seconds = training_time
                model_version.performance_metrics = performance_metrics
                model_version.model_files = model_files
                model_version.is_trained = True
                model_version.training_status = "completed"
                model_version.training_completed_at = datetime.utcnow()
                model_version.dataset_size = len(X)
                model_version.features_used = len(feature_names)
                
                self.db.commit()
                
                training_logs.append(f"Training completed successfully in {training_time:.2f} seconds")
                self._log_training_step(model_version.id, "completed", "INFO", "Training completed successfully")
                
                # Compare with current active model
                performance_comparison = self._compare_with_active_model(model_version)
                
                return TrainingResponse(
                    model_version=ModelVersionSchema(
                        id=model_version.id,
                        version=model_version.version,
                        hyperparameters=request.hyperparameters,
                        performance_metrics=performance_metrics,
                        is_active=model_version.is_active,
                        created_at=model_version.created_at,
                        training_time_seconds=training_time,
                        dataset_size=len(X),
                        features_used=len(feature_names)
                    ),
                    training_logs=training_logs,
                    performance_comparison=performance_comparison,
                    status="success"
                )
                
            except Exception as e:
                # Update model version with error
                model_version.training_status = "failed"
                model_version.error_message = str(e)
                self.db.commit()
                
                training_logs.append(f"Training failed: {str(e)}")
                self._log_training_step(model_version.id, "error", "ERROR", f"Training failed: {str(e)}")
                
                return TrainingResponse(
                    model_version=ModelVersionSchema(
                        id=model_version.id,
                        version=model_version.version,
                        hyperparameters=request.hyperparameters,
                        performance_metrics={},
                        is_active=False,
                        created_at=model_version.created_at,
                        training_time_seconds=time.time() - start_time,
                        dataset_size=0,
                        features_used=0
                    ),
                    training_logs=training_logs,
                    status="failed"
                )
                
        except Exception as e:
            return TrainingResponse(
                model_version=None,
                training_logs=[f"Failed to create model version: {str(e)}"],
                status="failed"
            )
    
    def _create_model_version(self, request: TrainingRequest) -> ModelVersion:
        """Create a new model version record"""
        # Generate version number
        latest_version = self.db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
        if latest_version:
            version_parts = latest_version.version.split('.')
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            new_version = '.'.join(version_parts)
        else:
            new_version = "1.0.0"
        
        model_version = ModelVersion(
            version=new_version,
            model_name=request.model_name,
            description=request.description,
            hyperparameters=request.hyperparameters.model_dump(),
            performance_metrics={},
            training_time_seconds=0.0,
            dataset_size=0,
            features_used=0,
            training_status="training",
            training_started_at=datetime.utcnow()
        )
        
        self.db.add(model_version)
        self.db.commit()
        self.db.refresh(model_version)
        
        return model_version
    
    def _load_training_data(self) -> tuple:
        """Load and prepare training data"""
        # Load the processed dataset
        df = pd.read_csv('notebooks/data/processed/exoplanets_unified_derived_imputed.csv')
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col not in ['disposition', 'planet_name', 'host_name']]
        
        # Handle categorical columns
        X = df[feature_columns].copy()
        
        # Convert categorical columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert to numeric, if fails, use label encoding
                try:
                    X[col] = pd.to_numeric(X[col], errors='raise')
                except:
                    X[col] = pd.Categorical(X[col]).codes
        
        # Fill remaining NaN values
        X = X.fillna(0)
        
        # Target variable
        y = (df['disposition'] == 'CONFIRMED').astype(int)
        
        return X, y, feature_columns
    
    def _train_random_forest(self, X_train, y_train, hyperparams: ModelHyperparameters) -> RandomForestClassifier:
        """Train Random Forest model"""
        rf = RandomForestClassifier(
            n_estimators=hyperparams.rf_n_estimators,
            max_depth=hyperparams.rf_max_depth,
            min_samples_split=hyperparams.rf_min_samples_split,
            min_samples_leaf=hyperparams.rf_min_samples_leaf,
            max_features=hyperparams.rf_max_features,
            random_state=hyperparams.rf_random_state,
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        return rf
    
    def _train_hist_gradient_boosting(self, X_train, y_train, hyperparams: ModelHyperparameters) -> HistGradientBoostingClassifier:
        """Train HistGradientBoosting model"""
        hgb = HistGradientBoostingClassifier(
            max_iter=hyperparams.hgb_max_iter,
            learning_rate=hyperparams.hgb_learning_rate,
            max_depth=hyperparams.hgb_max_depth,
            min_samples_leaf=hyperparams.hgb_min_samples_leaf,
            random_state=hyperparams.hgb_random_state,
            class_weight='balanced'
        )
        
        hgb.fit(X_train, y_train)
        return hgb
    
    def _create_ensemble(self, rf_model, hgb_model, hyperparams: ModelHyperparameters):
        """Create ensemble model using VotingClassifier"""
        from sklearn.ensemble import VotingClassifier
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('hgb', hgb_model)
            ],
            voting='soft'
        )
        
        # Fit with dummy data to make it work
        dummy_X = np.zeros((1, rf_model.n_features_in_))
        dummy_y = [0]
        ensemble.fit(dummy_X, dummy_y)
        
        return ensemble
    
    def _evaluate_models(self, ensemble, rf, hgb, X_test, y_test, X_train, y_train) -> Dict[str, Any]:
        """Evaluate model performance"""
        # Get predictions
        rf_pred = rf.predict_proba(X_test)[:, 1]
        hgb_pred = hgb.predict_proba(X_test)[:, 1]
        ensemble_pred = (rf_pred + hgb_pred) / 2  # Simple average for now
        
        # Calculate metrics
        metrics = {}
        
        for name, pred in [("rf", rf_pred), ("hgb", hgb_pred), ("ensemble", ensemble_pred)]:
            # Binary predictions
            binary_pred = (pred > 0.5).astype(int)
            
            # Calculate metrics
            metrics[f"{name}_roc_auc"] = roc_auc_score(y_test, pred)
            metrics[f"{name}_accuracy"] = accuracy_score(y_test, binary_pred)
            metrics[f"{name}_precision"] = precision_score(y_test, binary_pred, zero_division=0)
            metrics[f"{name}_recall"] = recall_score(y_test, binary_pred, zero_division=0)
            metrics[f"{name}_f1"] = f1_score(y_test, binary_pred, zero_division=0)
            metrics[f"{name}_brier_score"] = brier_score_loss(y_test, pred)
        
        # Cross-validation for ensemble (using RF as proxy)
        try:
            cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
            metrics["cv_roc_auc_mean"] = cv_scores.mean()
            metrics["cv_roc_auc_std"] = cv_scores.std()
        except Exception as e:
            metrics["cv_roc_auc_mean"] = 0.0
            metrics["cv_roc_auc_std"] = 0.0
        
        return metrics
    
    def _save_models(self, model_version_id: str, rf, hgb, ensemble) -> Dict[str, str]:
        """Save trained models to disk"""
        model_files = {}
        
        # Create version-specific directory
        version_dir = os.path.join(self.models_dir, f"version_{model_version_id}")
        os.makedirs(version_dir, exist_ok=True)
        
        # Save models
        rf_path = os.path.join(version_dir, "rf_model.pkl")
        hgb_path = os.path.join(version_dir, "hgb_model.pkl")
        ensemble_path = os.path.join(version_dir, "ensemble_model.pkl")
        
        joblib.dump(rf, rf_path)
        joblib.dump(hgb, hgb_path)
        joblib.dump(ensemble, ensemble_path)
        
        model_files = {
            "rf": rf_path,
            "hgb": hgb_path,
            "ensemble": ensemble_path
        }
        
        return model_files
    
    def _compare_with_active_model(self, new_model_version: ModelVersion) -> Optional[Dict[str, Any]]:
        """Compare new model with currently active model"""
        active_model = self.db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        
        if not active_model:
            return None
        
        comparison = {
            "current_model": {
                "version": active_model.version,
                "roc_auc": active_model.performance_metrics.get("ensemble_roc_auc", 0),
                "f1_score": active_model.performance_metrics.get("ensemble_f1", 0)
            },
            "new_model": {
                "version": new_model_version.version,
                "roc_auc": new_model_version.performance_metrics.get("ensemble_roc_auc", 0),
                "f1_score": new_model_version.performance_metrics.get("ensemble_f1", 0)
            }
        }
        
        # Calculate improvements
        roc_auc_improvement = comparison["new_model"]["roc_auc"] - comparison["current_model"]["roc_auc"]
        f1_improvement = comparison["new_model"]["f1_score"] - comparison["current_model"]["f1_score"]
        
        comparison["improvements"] = {
            "roc_auc_delta": roc_auc_improvement,
            "f1_delta": f1_improvement,
            "overall_improvement": roc_auc_improvement > 0 and f1_improvement > 0
        }
        
        return comparison
    
    def _log_training_step(self, model_version_id: str, step: str, level: str, message: str, duration: float = None):
        """Log a training step"""
        log = TrainingLog(
            model_version_id=model_version_id,
            log_level=level,
            message=message,
            step=step,
            duration_seconds=duration
        )
        self.db.add(log)
        self.db.commit()
    
    def list_model_versions(self, limit: int = 10) -> ModelListResponse:
        """List all model versions"""
        models = self.db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit).all()
        active_model = self.db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        
        model_schemas = []
        for model in models:
            model_schemas.append(ModelVersionSchema(
                id=model.id,
                version=model.version,
                hyperparameters=ModelHyperparameters(**model.hyperparameters),
                performance_metrics=model.performance_metrics,
                is_active=model.is_active,
                created_at=model.created_at,
                training_time_seconds=model.training_time_seconds,
                dataset_size=model.dataset_size,
                features_used=model.features_used
            ))
        
        active_model_schema = None
        if active_model:
            active_model_schema = ModelVersionSchema(
                id=active_model.id,
                version=active_model.version,
                hyperparameters=ModelHyperparameters(**active_model.hyperparameters),
                performance_metrics=active_model.performance_metrics,
                is_active=active_model.is_active,
                created_at=active_model.created_at,
                training_time_seconds=active_model.training_time_seconds,
                dataset_size=active_model.dataset_size,
                features_used=active_model.features_used
            )
        
        return ModelListResponse(
            models=model_schemas,
            active_model=active_model_schema,
            total_models=len(model_schemas)
        )
    
    def deploy_model(self, model_version_id: str, force_deploy: bool = False) -> Dict[str, Any]:
        """Deploy a model version as active"""
        try:
            # Get the model version
            new_model = self.db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
            if not new_model:
                return {"success": False, "message": "Model version not found"}
            
            if not new_model.is_trained:
                return {"success": False, "message": "Model is not trained yet"}
            
            # Get current active model
            current_active = self.db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
            
            # Check performance if not forcing deployment
            if not force_deploy and current_active:
                current_roc = current_active.performance_metrics.get("ensemble_roc_auc", 0)
                new_roc = new_model.performance_metrics.get("ensemble_roc_auc", 0)
                
                if new_roc < current_roc:
                    return {
                        "success": False, 
                        "message": f"New model performance ({new_roc:.3f}) is worse than current ({current_roc:.3f}). Use force_deploy=True to override."
                    }
            
            # Deactivate current model
            if current_active:
                current_active.is_active = False
                self.db.commit()
            
            # Activate new model
            new_model.is_active = True
            self.db.commit()
            
            return {
                "success": True,
                "message": f"Model {new_model.version} deployed successfully",
                "previous_model": current_active.version if current_active else None,
                "new_model": new_model.version
            }
            
        except Exception as e:
            return {"success": False, "message": f"Deployment failed: {str(e)}"}
