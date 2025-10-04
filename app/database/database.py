from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
from datetime import datetime, timezone
from app.settings import settings


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True, 
    pool_recycle=300,   
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with default data"""
    create_tables()
    
    db = SessionLocal()
    try:
        from .models import ModelMetrics
        
        existing_metrics = db.query(ModelMetrics).filter(ModelMetrics.is_active == True).first()
        
        if not existing_metrics:
            default_metrics = ModelMetrics(
                model_name="Ensemble Model v2.0",
                version="2.0.0",
                roc_auc=0.983,
                pr_auc=0.973,
                balanced_accuracy=0.929,
                brier_score=0.051,
                f1_score=0.888,
                recall=0.969,
                precision=0.830,
                accuracy=0.960,
                training_samples=21225,
                features_used=67,
                threshold=0.363,
                missions=["K2", "Kepler", "TESS"],
                created_at=datetime.now(timezone.utc),
                is_active=True
            )
            
            db.add(default_metrics)
            db.commit()
            print("✅ Default model metrics added to database")
        else:
            print("✅ Model metrics already exist in database")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()
