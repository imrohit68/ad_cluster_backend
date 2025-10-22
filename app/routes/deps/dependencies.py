from app.services.db_services.cluster_db_operations import ClusterService
from app.services.db_services.cluster_job_db_operations import ClusteringJobService
from app.services.image_download_service import ImageDownloader
from app.services.image_process_pipeline_service import ProductionPipeline
from fastapi import Request


def get_cluster_service() -> ClusterService:
    return ClusterService()

def get_cluster_job_service() -> ClusteringJobService:
    return ClusteringJobService()

def get_pipeline(request: Request) -> ProductionPipeline:
    return request.app.state.pipeline

def get_downloader(request: Request) -> ImageDownloader:
    return request.app.state.downloader
