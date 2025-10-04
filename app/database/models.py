from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """Usuários do sistema"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sessions = relationship("Session", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")

class Session(Base):
    """Sessões de usuário"""
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for anonymous users
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    predictions = relationship("Prediction", back_populates="session")
    batch_jobs = relationship("BatchJob", back_populates="session")

class Prediction(Base):
    """Predições de exoplanetas"""
    __tablename__ = "predictions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Input data
    input_data = Column(JSON, nullable=False)
    
    # Prediction results
    prob_rf = Column(Float, nullable=False)
    prob_hgb = Column(Float, nullable=False)
    prob_ens = Column(Float, nullable=False)
    label_rf = Column(String(50), nullable=False)
    label_hgb = Column(String(50), nullable=False)
    label_ens = Column(String(50), nullable=False)
    
    # Physical properties
    planet_data = Column(JSON, nullable=True)  # Planet, Star, Transit data
    quality_flags = Column(JSON, nullable=True)
    
    # Metadata
    processing_time = Column(Float, nullable=False)
    data_quality_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="predictions")
    user = relationship("User", back_populates="predictions")
    explanations = relationship("Explanation", back_populates="prediction")
    comparisons = relationship("Comparison", back_populates="prediction")

class Explanation(Base):
    """Explicações das predições"""
    __tablename__ = "explanations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id = Column(String(36), ForeignKey("predictions.id"), nullable=False)
    
    # Explanation data
    model_name = Column(String(50), nullable=False)
    base_value = Column(Float, nullable=False)
    prediction_probability = Column(Float, nullable=False)
    feature_importances = Column(JSON, nullable=False)
    explanation_summary = Column(Text, nullable=False)
    
    # Overall metrics
    confidence_score = Column(Float, nullable=False)
    key_factors = Column(JSON, nullable=False)
    overall_summary = Column(Text, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="explanations")

class Comparison(Base):
    """Comparações com exoplanetas similares"""
    __tablename__ = "comparisons"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id = Column(String(36), ForeignKey("predictions.id"), nullable=False)
    
    # Comparison data
    similar_exoplanets = Column(JSON, nullable=False)
    comparison_summary = Column(Text, nullable=False)
    uniqueness_score = Column(Float, nullable=False)
    scientific_interest = Column(String(50), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="comparisons")

class BatchJob(Base):
    """Jobs de processamento em lote"""
    __tablename__ = "batch_jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Job data
    filename = Column(String(255), nullable=True)
    dataset_type = Column(String(50), nullable=True)  # kepler, k2, tess, unified
    total_candidates = Column(Integer, nullable=False)
    processed_candidates = Column(Integer, nullable=False)
    
    # Results
    results = Column(JSON, nullable=True)
    summary_statistics = Column(JSON, nullable=True)
    
    # Status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    processing_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    session = relationship("Session", back_populates="batch_jobs")
    candidates = relationship("BatchCandidate", back_populates="batch_job")

class BatchCandidate(Base):
    """Candidatos individuais em jobs de lote"""
    __tablename__ = "batch_candidates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_job_id = Column(String(36), ForeignKey("batch_jobs.id"), nullable=False)
    
    # Candidate data
    candidate_id = Column(String(100), nullable=False)
    input_data = Column(JSON, nullable=False)
    prediction_results = Column(JSON, nullable=False)
    interestingness_score = Column(Float, nullable=False)
    ranking_position = Column(Integer, nullable=False)
    key_highlights = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    batch_job = relationship("BatchJob", back_populates="candidates")

class ModelMetrics(Base):
    """Métricas de performance do modelo"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    
    # Performance metrics
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    balanced_accuracy = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    
    # Training data
    training_samples = Column(Integer, nullable=False)
    features_used = Column(Integer, nullable=False)
    threshold = Column(Float, nullable=False)
    
    # Metadata
    missions = Column(JSON, nullable=False)  # ["K2", "Kepler", "TESS"]
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ModelVersion(Base):
    """Versões de modelos treinados"""
    __tablename__ = "model_versions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version = Column(String(50), nullable=False, unique=True)
    model_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Hyperparameters
    hyperparameters = Column(JSON, nullable=False)
    
    # Performance metrics
    performance_metrics = Column(JSON, nullable=False)
    
    # Training metadata
    training_time_seconds = Column(Float, nullable=False)
    dataset_size = Column(Integer, nullable=False)
    features_used = Column(Integer, nullable=False)
    
    # Model files (paths to saved models)
    model_files = Column(JSON, nullable=True)  # {"rf": "path", "hgb": "path", "ensemble": "path"}
    
    # Status
    is_active = Column(Boolean, default=False)
    is_trained = Column(Boolean, default=False)
    training_status = Column(String(50), default="pending")  # pending, training, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    training_logs = relationship("TrainingLog", back_populates="model_version")

class TrainingLog(Base):
    """Logs de treinamento de modelos"""
    __tablename__ = "training_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(String(36), ForeignKey("model_versions.id"), nullable=False)
    
    # Log content
    log_level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Additional context
    step = Column(String(100), nullable=True)  # "data_loading", "training", "evaluation", etc.
    duration_seconds = Column(Float, nullable=True)
    
    # Relationships
    model_version = relationship("ModelVersion", back_populates="training_logs")
