from typing import List

from fastapi import APIRouter, Depends

from app.models.cluster_models import  ClusterDTO
from app.routes.deps.dependencies import  get_cluster_service
from app.services.db_services.cluster_db_operations import ClusterService


router = APIRouter(tags=["Clusters"])

@router.get("/clusters/{job_id}", response_model=List[ClusterDTO])
async def get_clusters(
    job_id: str,
    cluster_service: ClusterService = Depends(get_cluster_service)
):
    return await cluster_service.get_clusters_by_job(job_id)