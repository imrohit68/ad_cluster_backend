from typing import List
from motor.motor_asyncio import AsyncIOMotorCollection
from app.models.cluster_models import Cluster, ClusterDTO


class ClusterRepository:
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def save(self, cluster: Cluster) -> str:
        """Save a single cluster"""
        cluster_dict = cluster.model_dump()  # Pydantic V2 method
        result = await self.collection.insert_one(cluster_dict)
        return str(result.inserted_id)

    async def batch_save(self, clusters: List[Cluster]) -> List[str]:
        """
        Save multiple clusters at once.

        Args:
            clusters: List of Cluster models

        Returns:
            List of inserted IDs as strings
        """
        if not clusters:
            return []

        cluster_dicts = [c.model_dump() for c in clusters]
        result = await self.collection.insert_many(cluster_dicts)
        return [str(_id) for _id in result.inserted_ids]


    async def get_by_job_id(self, job_id: str) -> List[ClusterDTO]:
        """
        Fetch all clusters for a specific job_id and return as DTOs.
        """
        cursor = self.collection.find({"job_id": job_id})
        clusters = await cursor.to_list(length=None)

        # Map Mongo documents to DTOs, ignoring _id
        return [
            ClusterDTO(**{k: v for k, v in c.items() if k != "_id"})
            for c in clusters
        ]
