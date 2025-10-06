# 🌌 Exo.AI Backend - NASA Exoplanet Detection & Analysis API

> Advanced system for exoplanet detection and analysis using Machine Learning based on real data from NASA's TESS, Kepler, and K2 missions.

![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688?logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-316192?logo=postgresql)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-f7931e?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker)

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Docker Deployment](#docker-deployment)
  - [Environment Variables](#environment-variables)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Machine Learning Models](#-machine-learning-models)
- [Performance](#-performance)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Overview

The **Exo.AI Backend** is a comprehensive machine learning system for detecting, explaining, and analyzing exoplanet candidates. The system processes raw astronomical data, performs advanced feature engineering, and trains ensemble models optimized for high recall in exoplanet detection.

### 🎯 Objectives

- **For Users**: Visualize, learn and test exoplanet detections
- **For Scientists**: View advanced metrics, train new models, and manipulate the system with their own creativity

### 📊 Data Foundation

- **TESS**: Transiting Exoplanet Survey Satellite
- **Kepler**: Kepler Space Telescope  
- **K2**: Extended Kepler Mission
- **21,000+** processed samples
- **65+** derived features

## ✨ Key Features

### 🤖 Candidate Prediction
- **Ensemble Model**: HistGradientBoosting + RandomForest
- **Training**: 21k+ samples with 65+ features
- **Performance**: AUC of 0.983 on validation set
- **Intelligent Imputation**: Automatic handling of missing values

### 🔍 Explainability
- **Detailed Explanation**: Model decisions explained
- **Feature Importance**: SHAP values
- **Category Analysis**: Orbital, stellar, transit, etc.
- **Confidence Score**: Reliability in explanation

### 🌍 Comparison with Known Exoplanets
- **Similarity Search**: Similar confirmed exoplanets
- **Similarity Score**: Habitability scoring
- **Automatic Classification**: Planet type classification
- **Interest Evaluation**: Scientific interest assessment

### ⚡ Batch Processing
- **Batch Processing**: Multiple candidates
- **Interest Ranking**: Automatic ranking
- **Summary Statistics**: Complete batch analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   ML Models     │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (Scikit-Learn)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Database      │    │   SHAP Explainer│
                       │ (PostgreSQL)    │◄──►│   (Explainability)│
                       └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

### Backend Framework
- **FastAPI 0.100.0** - Modern and fast web framework
- **Python 3.11** - Programming language
- **Pydantic 2.0** - Data validation
- **Uvicorn** - ASGI server

### Machine Learning
- **Scikit-Learn 1.4.0** - ML library
- **Pandas 2.0** - Data manipulation
- **NumPy 1.24** - Numerical computing
- **Joblib 1.3** - Model serialization

### Database
- **PostgreSQL 15** - Primary database
- **SQLAlchemy 2.0** - ORM
- **Psycopg2** - PostgreSQL driver

### Visualization & Analysis
- **Matplotlib 3.7** - Data visualization
- **Seaborn 0.12** - Statistical visualization

### Containerization
- **Docker** - Containerization
- **Docker Compose** - Service orchestration

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+**
- **Docker & Docker Compose** (optional)
- **Git**

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/e-32-exo-ai-backend.git
cd e-32-exo-ai-backend
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up the database**
```bash
# Start PostgreSQL and create database
createdb nasa_exoplanets
```

5. **Run migrations**
```bash
python -c "from app.database import init_database; init_database()"
```

6. **Start the application**
```bash
python api.py
```

### Docker Deployment

1. **Clone and navigate to directory**
```bash
git clone https://github.com/your-username/e-32-exo-ai-backend.git
cd e-32-exo-ai-backend
```

2. **Run with Docker Compose**
```bash
docker-compose up -d
```

3. **Verify services are running**
```bash
docker-compose ps
```

The API will be available at: `http://localhost:8000`

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
DEBUG=false

# Database Configuration
DATABASE_URL=postgresql://nasa_user:nasa_password@localhost:5433/nasa_exoplanets

# Logging Configuration
LOG_LEVEL=info

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "https://your-frontend.com"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["*"]
CORS_ALLOW_HEADERS=["*"]

# Model Configuration
MODEL_PATH=./notebooks/models
DATA_PATH=./notebooks/data

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Cache Configuration
CACHE_TTL=300

# Upload Configuration
MAX_UPLOAD_SIZE=10485760
ALLOWED_FILE_TYPES=["csv", "xlsx", "xls"]

# Processing Configuration
MAX_BATCH_SIZE=1000
PROCESSING_TIMEOUT=300

# Development Configuration
ENABLE_DOCS=true
ENABLE_REDOC=true
```

## 📁 Project Structure

```
e-32-exo-ai-backend/
├── app/
│   ├── api/
│   │   ├── router.py              # Main router
│   │   └── v1/
│   │       └── endpoints/         # API endpoints
│   │           ├── batch.py       # Batch processing
│   │           ├── compare.py     # Exoplanet comparison
│   │           ├── dashboard.py   # Scientific dashboard
│   │           ├── explain.py     # Explainability
│   │           ├── model_training.py # Model training
│   │           └── prediction.py  # Predictions
│   │
│   ├── database/
│   │   ├── database.py           # Database configuration
│   │   └── models.py             # Database models
│   │
│   ├── schemas/
│   │   ├── batch.py              # Batch schemas
│   │   ├── comparison.py         # Comparison schemas
│   │   ├── dashboard.py          # Dashboard schemas
│   │   ├── explainability.py     # Explainability schemas
│   │   ├── model_training.py     # Training schemas
│   │   ├── physics.py            # Physics schemas
│   │   └── prediction.py         # Prediction schemas
│   │
│   ├── services/
│   │   ├── database_service.py   # Database services
│   │   ├── model_service.py      # ML services
│   │   ├── model_training_service.py # Training services
│   │   └── shap_explainer.py     # SHAP explainability
│   │
│   └── settings/
│       └── config.py             # Application settings
│
├── notebooks/
│   ├── 01_analysis.ipynb         # Exploratory analysis
│   ├── 02_statistical_analysis.ipynb # Statistical analysis
│   ├── 03_modeling.ipynb         # Modeling
│   ├── 04_recall_optimization.ipynb # Recall optimization
│   ├── data/
│   │   ├── processed/            # Processed data
│   │   └── raw/                  # Raw data
│   └── models/                   # Trained models
│
├── api.py                        # Main application
├── docker-compose.yml            # Docker configuration
├── Dockerfile                    # Docker image
├── init.sql                      # Database initialization script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔌 API Endpoints

### 🎯 Prediction

#### `POST /api/v1/predict/user`
Prediction for users with basic features.

**Request Body:**
```json
{
  "orbital_period_days": 12.3,
  "transit_depth_ppm": 1500,
  "planet_radius_re": 1.05,
  "planet_mass_me": 1.0,
  "stellar_teff_k": 5700,
  "stellar_radius_rsun": 1.0,
  "stellar_mass_msun": 1.0
}
```

#### `POST /api/v1/predict/scientist`
Prediction for scientists with advanced features.

**Request Body:**
```json
{
  "orbital_period_days": 12.3,
  "transit_depth": 1500,
  "transit_duration": 3.2,
  "planet_radius_re": 1.05,
  "planet_mass_me": 1.0,
  "stellar_teff_k": 5700,
  "stellar_radius_rsun": 1.0,
  "stellar_mass_msun": 1.0,
  "radius_ratio": 0.01,
  "semi_major_axis_au": 0.1,
  "equilibrium_temp_recalc_k": 300,
  "log_orbital_period": 1.09,
  "period_mass_interaction": 12.3,
  "stellar_teff_bin": 5700
}
```

### 🔍 Explainability

#### `POST /api/v1/explain/`
Detailed explanation of model decisions using SHAP.

### 🌍 Comparison

#### `POST /api/v1/compare/`
Comparison with similar confirmed exoplanets.

### ⚡ Batch Processing

#### `POST /api/v1/batch/`
Simultaneous processing of multiple candidates.

### 📊 Scientific Dashboard

#### `GET /api/v1/dashboard/`
System metrics and statistics.

### 🤖 Model Training

#### `POST /api/v1/model/train/`
Training new models with updated data.

## 🧠 Machine Learning Models

### Ensemble Model
- **Random Forest**: Robust classifier for non-linear data
- **Histogram Gradient Boosting**: Optimized for high performance
- **Ensemble Voting**: Intelligent combination of models

### Processed Features
- **Orbital**: Period, transit duration, semi-major axis
- **Stellar**: Temperature, radius, mass, spectral type
- **Planetary**: Radius, mass, equilibrium temperature
- **Derived**: Log period, interactions, temperature bins

### Optimization
- **Recall**: 96.9% ± 0.8% (exoplanet detection)
- **F1-Score**: 88.8% (precision-recall balance)
- **Precision**: 83.0% (false positive reduction)
- **Accuracy**: 96.0% (overall accuracy)

## 📈 Performance

### System Metrics
- **AUC**: 0.983 (area under ROC curve)
- **Response Time**: < 2 seconds per prediction
- **Throughput**: 100+ predictions/minute
- **Availability**: 99.9% uptime

### Implemented Optimizations
- **Model Caching**: Single loading on initialization
- **Intelligent Imputation**: Missing value handling
- **Batch Processing**: Parallel processing
- **Rate Limiting**: Load control
- **Connection Pooling**: Database optimization

## 🔧 Development

### Run in Development Mode

```bash
# Install development dependencies
pip install -r requirements.txt

# Set environment variables
export DEBUG=true
export API_RELOAD=true

# Start in development mode
python api.py
```

### API Documentation

Access interactive documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🤝 Contributing

1. **Fork** the project
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Code Standards
- Use **type hints** in all functions
- Document functions with **docstrings**
- Follow **PEP 8** standard
- Use **conventional commits**


## 👥 Team

**Developed for NASA Hackathon - Team E-32** 🏆

- **Ivisson Alves** - AI Engineer & Backend Developer
  - LinkedIn: [@ivi-aiengineer](https://www.linkedin.com/in/ivi-aiengineer/)
  - Email: ivipnascimento@hotmail.com

## 🔗 Useful Links

- **API Documentation**: `/docs`
- **TESS Data**: [archive.stsci.edu/tess](https://archive.stsci.edu/missions-and-data/tess)
- **Kepler Data**: [archive.stsci.edu/kepler](https://archive.stsci.edu/kepler)
- **K2 Data**: [archive.stsci.edu/k2](https://archive.stsci.edu/k2)

---

**🚀 Detect Your Exoplanet Anywhere for Anyone** - Exo.AI Backend v2.0.0