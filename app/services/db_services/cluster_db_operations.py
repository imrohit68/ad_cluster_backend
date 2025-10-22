from typing import List

from app.core.mongo_client import mongodb
from app.models.cluster_models import ClusterDTO
from app.repository.cluster_repository import ClusterRepository


class ClusterService:
    def __init__(self):
        collection = mongodb.db["clusters"]
        self.repository = ClusterRepository(collection)

    async def get_clusters_by_job(self, job_id: str) -> List[ClusterDTO]:
        return await self.repository.get_by_job_id(job_id)
