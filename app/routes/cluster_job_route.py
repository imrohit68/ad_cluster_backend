from typing import List

from fastapi import APIRouter, Depends

from app.models.cluster_models import ClusteringJobDTO
from app.routes.deps.dependencies import get_cluster_job_service
from app.services.db_services.cluster_job_db_operations import ClusteringJobService

router = APIRouter(tags=["Clustering Jobs"])



@router.get("/jobs", response_model=List[ClusteringJobDTO])
async def get_jobs(
    cluster_job_service: ClusteringJobService = Depends(get_cluster_job_service)
):
    return await cluster_job_service.get_all_jobs()