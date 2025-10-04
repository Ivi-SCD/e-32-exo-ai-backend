from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from app.schemas import BatchCandidateRequest, BatchProcessingResponse
from app.services import ModelService
from app.database import get_db
from app.services.database_service import DatabaseService
from sqlalchemy.orm import Session
import pandas as pd
import io
import json

router = APIRouter()
model_service = ModelService()

class DatasetProcessor:
    """Helper class for processing different dataset formats"""
    
    @staticmethod
    def detect_dataset_type(df: pd.DataFrame) -> str:
        """Detect the type of dataset (kepler, k2, tess) based on column names"""
        columns = df.columns.tolist()
        
        # Check for Kepler-specific columns
        if any('koi_' in col.lower() for col in columns):
            return 'kepler'
        
        # Check for K2-specific columns (pl_name, epic_)
        if any('epic_' in col.lower() for col in columns) or 'pl_name' in columns:
            return 'k2'
        
        # Check for TESS-specific columns
        if any('toi' in col.lower() for col in columns) or 'tid' in columns:
            return 'tess'
        
        # Default to unified format
        return 'unified'
    
    @staticmethod
    def filter_candidates_by_dataset(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Filter candidates based on dataset type"""
        
        if dataset_type == 'kepler':
            # Kepler: keep CANDIDATE, remove CONFIRMED and FALSE POSITIVE
            if 'koi_disposition' in df.columns:
                df = df[df['koi_disposition'] == 'CANDIDATE']
        
        elif dataset_type == 'k2':
            # K2: keep CANDIDATE, remove CONFIRMED, FALSE POSITIVE, REFUTED
            if 'disposition' in df.columns:
                df = df[df['disposition'] == 'CANDIDATE']
        
        elif dataset_type == 'tess':
            # TESS: keep PC (Planet Candidate), remove FP (False Positive), CP (Confirmed Planet)
            if 'tfopwg_disp' in df.columns:
                # Keep PC (Planet Candidate) and KP (Known Planet) as candidates
                df = df[df['tfopwg_disp'].isin(['PC', 'KP'])]
        
        else:  # unified format
            # Original logic for unified dataset
            disposition_columns = [col for col in df.columns if 'disposition' in col.lower()]
            if disposition_columns:
                disposition_col = disposition_columns[0]
                disposition_values = ['CONFIRMED', 'FALSE POSITIVE', 'REFUTED']
                df = df[~df[disposition_col].isin(disposition_values)]
                df = df[(df[disposition_col].isna()) | (df[disposition_col] == '') | (df[disposition_col] == 'CANDIDATE')]
        
        return df
    
    @staticmethod
    def convert_to_candidates(df: pd.DataFrame, dataset_type: str) -> list:
        """Convert DataFrame to candidates list using dataset-specific mapping"""
        
        # Define column mappings for each dataset
        mappings = {
            'kepler': {
                'orbital_period_days': 'koi_period',
                'transit_depth_ppm': 'koi_depth',
                'planet_radius_re': 'koi_prad',
                'stellar_teff_k': 'koi_steff',
                'stellar_radius_rsun': 'koi_srad',
                'stellar_mass_msun': 'koi_smass'
            },
            'k2': {
                'orbital_period_days': 'pl_orbper',
                'transit_depth_ppm': 'pl_trandep',
                'planet_radius_re': 'pl_rade',
                'stellar_teff_k': 'st_teff',
                'stellar_radius_rsun': 'st_rad',
                'stellar_mass_msun': 'st_mass'
            },
            'tess': {
                'orbital_period_days': 'pl_orbper',
                'transit_depth_ppm': 'pl_trandep',
                'planet_radius_re': 'pl_rade',
                'stellar_teff_k': 'st_teff',
                'stellar_radius_rsun': 'st_rad'
                # Note: TESS doesn't have stellar_mass_msun
            },
            'unified': {}  # Use original column names
        }
        
        mapping = mappings.get(dataset_type, {})
        candidates = []
        
        for idx, row in df.iterrows():
            candidate = {}
            
            # Use mapping if available, otherwise use original column names
            if mapping:
                for standard_col, dataset_col in mapping.items():
                    if dataset_col in df.columns:
                        value = row[dataset_col]
                        if not pd.isna(value):
                            try:
                                candidate[standard_col] = float(value)
                            except (ValueError, TypeError):
                                continue
            else:
                # Original logic for unified dataset
                for col in df.columns:
                    value = row[col]
                    if col in ['host_name', 'planet_name'] or pd.isna(value):
                        continue
                    try:
                        candidate[col] = float(value)
                    except (ValueError, TypeError):
                        continue
            
            if candidate:  # Only add if we have some data
                candidates.append(candidate)
        
        return candidates

@router.post("/process", response_model=BatchProcessingResponse)
async def process_batch_candidates(
    request: BatchCandidateRequest,
    db: Session = Depends(get_db)
):
    """
    Processes multiple candidates and sorts them by "interestingness".

    This endpoint allows you to process a list of exoplanet candidates
    and returns results sorted by scientific interest:

    **Available sorting criteria:**
    - `interestingness`: Combined score of probability, habitability, and uniqueness
    - `probability`: Sorts by probability of being an exoplanet
    - `similarity`: Sorts by uniqueness (less similar = more interesting)

    **For each candidate, returns:**
    - Complete prediction with all models
    - Scientific interest score (0-1)
    - Ranking position
    - Main highlights of the candidate
    - Batch summary statistics

    **Example usage:**
    ```json
    {
        "candidates": [
            {
                "orbital_period_days": 365.25,
                "transit_depth_ppm": 1500,
                "planet_radius_re": 1.0,
                "stellar_teff_k": 5778
            },
            {
                "orbital_period_days": 12.3,
                "transit_depth_ppm": 800,
                "planet_radius_re": 0.8,
                "stellar_teff_k": 5200
            }
        ],
        "sort_by": "interestingness"
    }
    ```
    """
    try:
        # Validate number of candidates
        if len(request.candidates) > 100:
            raise HTTPException(
                status_code=400, 
                detail="Maximum of 100 candidates per batch. To process more, divide into smaller batches."
            )
        
        if len(request.candidates) == 0:
            raise HTTPException(
                status_code=400,
                detail="List of candidates cannot be empty."
            )
        
        # Validate sorting criterion
        valid_sort_criteria = ["interestingness", "probability", "similarity"]
        if request.sort_by not in valid_sort_criteria:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sorting criterion. Use one of the following: {valid_sort_criteria}"
            )
        
        # Get or create session
        db_service = DatabaseService(db)
        session = db_service.get_or_create_anonymous_session()
        
        # Save batch job to database
        batch_job = db_service.save_batch_job(
            session_id=session.id,
            filename=None,  # Manual batch request
            dataset_type="manual",
            total_candidates=len(request.candidates)
        )
        
        # Process batch with session tracking
        result = model_service.process_batch_candidates(request)
        
        # Update batch job with results
        db_service.update_batch_job(
            batch_job_id=batch_job.id,
            status="completed",
            processed_candidates=len(result.results),
            results=result.model_dump(),
            summary_statistics=result.summary_statistics,
            processing_time=result.processing_time_seconds
        )
        
        # Add batch job ID to response
        result.batch_job_id = batch_job.id
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-csv", response_model=BatchProcessingResponse)
async def process_csv_file(
    file: UploadFile = File(..., description="CSV file with exoplanet data"),
    max_rows: int = Query(default=100, ge=1, le=1000, description="Maximum number of rows to process"),
    sort_by: str = Query(default="interestingness", description="Sorting criterion (interestingness, probability, similarity)"),
    db: Session = Depends(get_db)
):
    """
    Process exoplanet data from a CSV file and analyze each planet.
    
    This endpoint accepts a CSV file with exoplanet data and returns analysis for each planet:
    
    **CSV Requirements:**
    - Must contain columns for exoplanet features (orbital_period_days, transit_depth_ppm, etc.)
    - Will automatically filter out rows with disposition values (CONFIRMED, CANDIDATE, FALSE POSITIVE, etc.)
    - Processes only the first N rows (default: 100, max: 1000)
    
    **Returns for each planet:**
    - Complete prediction with all models
    - Scientific interest score (0-1)
    - Ranking position
    - Main highlights of the candidate
    - Batch summary statistics
    
    **Example CSV columns:**
    - host_name, planet_name, orbital_period_days, transit_depth_ppm, planet_radius_re, stellar_teff_k, etc.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="File must be a CSV file"
            )
        
        # Read CSV file
        content = await file.read()
        try:
            # Try to read with error handling for malformed CSV and comments
            df = pd.read_csv(io.StringIO(content.decode('utf-8')), low_memory=False, on_bad_lines='skip', comment='#')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading CSV file: {str(e)}"
            )
        
        # Check if file is empty
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty"
            )
        
        # Detect dataset type and apply appropriate filtering
        dataset_type = DatasetProcessor.detect_dataset_type(df)
        df = DatasetProcessor.filter_candidates_by_dataset(df, dataset_type)
        
        # Limit number of rows
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        # Check if we have data after filtering
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid data found after filtering disposition values"
            )
        
        # Convert DataFrame to list of dictionaries using dataset-specific mapping
        candidates = DatasetProcessor.convert_to_candidates(df, dataset_type)
        
        if not candidates:
            raise HTTPException(
                status_code=400,
                detail="No valid numeric data found in CSV file"
            )
        
        # Validate sorting criterion
        valid_sort_criteria = ["interestingness", "probability", "similarity"]
        if sort_by not in valid_sort_criteria:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sorting criterion. Use one of the following: {valid_sort_criteria}"
            )
        
        # Create batch request
        batch_request = BatchCandidateRequest(
            candidates=candidates,
            sort_by=sort_by
        )
        
        # Get or create session
        db_service = DatabaseService(db)
        session = db_service.get_or_create_anonymous_session()
        
        # Save batch job to database
        batch_job = db_service.save_batch_job(
            session_id=session.id,
            filename=file.filename,
            dataset_type=dataset_type,
            total_candidates=len(candidates)
        )
        
        # Process the batch
        result = model_service.process_batch_candidates(batch_request)
        
        # Update batch job with results
        db_service.update_batch_job(
            batch_job_id=batch_job.id,
            status="completed",
            processed_candidates=len(result.results),
            results=result.model_dump(),
            summary_statistics=result.summary_statistics,
            processing_time=result.processing_time_seconds
        )
        
        # Add metadata about the CSV processing
        result.summary_statistics.update({
            "csv_filename": file.filename,
            "dataset_type": dataset_type,
            "total_rows_in_csv": len(pd.read_csv(io.StringIO(content.decode('utf-8')), comment='#', on_bad_lines='skip')),
            "rows_after_filtering": len(df),
            "candidates_processed": len(candidates),
            "batch_job_id": batch_job.id
        })
        
        # Add batch job ID to response
        result.batch_job_id = batch_job.id
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

