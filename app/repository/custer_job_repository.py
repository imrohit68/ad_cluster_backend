from typing import List

from motor.motor_asyncio import AsyncIOMotorCollection

from app.models.cluster_models import ClusteringJob


class ClusteringJobRepository:
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def save(self, job: ClusteringJob) -> str:
        job_dict = job.model_dump(by_alias=True, exclude_none=True)
        result = await self.collection.insert_one(job_dict)
        return str(result.inserted_id)

    async def get_by_id(self, job_id: str):
        job_data = await self.collection.find_one({"_id": job_id})
        return job_data


    async def get_all(self) -> List[ClusteringJob]:
        """Fetch all clustering jobs from the collection"""
        cursor = self.collection.find({})
        jobs_data = await cursor.to_list(length=None)
        return [ClusteringJob(**dict(data)) for data in jobs_data]


