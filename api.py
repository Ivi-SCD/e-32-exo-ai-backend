from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.api.router import api_router
from app.database import init_database
from app.settings import settings

app = FastAPI(
    title="NASA Exoplanet Detection & Analysis API",
    description="""
    🚀 **Advanced System for Detection and Analysis of Exoplanets**
    
    Complete machine learning system for detection, explanation and analysis of candidates for exoplanets
    based on real data from the TESS, Kepler and K2 missions of NASA.
    
    ## 🎯 **Main Features:**
    
    ### 📊 **Candidate Prediction**
    - Modelo ensemble (HistGradientBoosting + RandomForest)
    - Trained with 21k+ samples and 65+ features
    - AUC of 0.983 on the validation set
    - Intelligent imputation of missing values
    
    ### 🔍 **Explainability**
    - Detailed explanation of model decisions
    - Feature importance using SHAP values
    - Category analysis (orbital, stellar, transit, etc.)
    - Confidence score in the explanation
    
    ### 🌍 **Comparison with Known Exoplanets**
    - Search for similar confirmed exoplanets
    - Similarity and habitability score
    - Automatic classification of planet type
    - Scientific interest evaluation
    
    ### ⚡ **Batch Processing**
    - Batch processing of multiple candidates
    - Ordering by "interestingness"
    - Automatic ranking by scientific interest
    - Summary statistics of the batch
    
    
    ## 🛠 **Available Endpoints:**
    
    - **`/api/v1/predict/`** - Candidate Prediction
    - **`/api/v1/explain/`** - Explanation of decisions
    - **`/api/v1/compare/`** - Comparison with similar exoplanets
    - **`/api/v1/batch/`** - Batch Processing
    
    ## 📈 **Performance:**
    - **Recall**: 96.9% ± 0.8%
    - **F1-Score**: 88.8%
    - **Precision**: 83.0%
    - **Accuracy**: 96.0%
    
    ## 🎯 **Use Cases:**
    - Automatic screening of candidates for exoplanets
    - Prioritization of follow-up observations
    - Scientific interest evaluation
    - Educational and scientific dissemination
    
    ---
    
    **Developed for the NASA Hackathon (TEAM: E-32)** 🏆
    """,
    version="2.0.0",
    contact = {
        "name": "Ivisson Alves",
        "url": "https://www.linkedin.com/in/ivi-aiengineer/",
        "email": "ivipnascimento@hotmail.com",
    },
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_redoc else None
)

app.add_middleware(
    CORSMiddleware,
    **settings.get_cors_config()
)

app.include_router(api_router, prefix="/api")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing database...")
    init_database()
    print("✅ Database initialized successfully!")

@app.get("/")
async def root():
    return {
        "message": "NASA Exoplanet Detection & Analysis API - E-32 Running ... 🚀",
        "version": "2.0.0",
        "status": "operational",
        "features": {
            "prediction": "✅ Active",
            "explainability": "✅ Active", 
            "comparison": "✅ Active",
            "batch_processing": "✅ Active"
        },
        "endpoints": {
            "prediction": "/api/v1/predict/",
            "explainability": "/api/v1/explain/",
            "comparison": "/api/v1/compare/",
            "batch": "/api/v1/batch/"
        },
        "datasets": {
            "TESS": "https://archive.stsci.edu/missions-and-data/tess",
            "Kepler": "https://archive.stsci.edu/kepler",
            "K2": "https://archive.stsci.edu/k2"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level,
        reload=settings.api_reload
    )