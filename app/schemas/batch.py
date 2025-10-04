from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pydantic import Field
from app.schemas import PredictionGeneralResponse

class BatchCandidateRequest(BaseModel):
    candidates: List[Dict[str, Any]] = Field(..., description="Candidates to process")
    sort_by: str = Field(default="interestingness", description="Sorting criterion (interestingness, probability, similarity)")

class CandidateResult(BaseModel):
    candidate_id: str = Field(..., description="Candidate ID")
    prediction: PredictionGeneralResponse = Field(..., description="Prediction result")
    interestingness_score: float = Field(..., description="Scientific interest score")
    ranking_position: int = Field(..., description="Ranking position")
    key_highlights: List[str] = Field(..., description="Key highlights of the candidate")

class BatchProcessingResponse(BaseModel):
    results: List[CandidateResult] = Field(..., description="Ordered results of candidates")
    total_processed: int = Field(..., description="Total candidates processed")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    summary_statistics: Dict[str, Any] = Field(..., description="Summary statistics of the results")
    batch_job_id: Optional[str] = Field(None, description="Batch job ID for tracking")