from typing import List

from app.core.mongo_client import mongodb
from app.models.cluster_models import ClusteringJobDTO
from app.repository.custer_job_repository import ClusteringJobRepository


class ClusteringJobService:
    def __init__(self):
        collection = mongodb.db["clustering_jobs"]
        self.repository = ClusteringJobRepository(collection)

    async def get_all_jobs(self) -> List[ClusteringJobDTO]:
        jobs = await self.repository.get_all()
        return [ClusteringJobDTO(**job.model_dump()) for job in jobs]
