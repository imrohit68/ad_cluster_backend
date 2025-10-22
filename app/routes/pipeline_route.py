from fastapi import APIRouter, Depends

from app.models.cluster_models import ProcessRequest
from app.routes.deps.dependencies import get_pipeline
from app.services.image_process_pipeline_service import ProductionPipeline

router = APIRouter(tags=["Pipeline"])



@router.post("/process")
async def process_images(
    request_body: ProcessRequest,
    pipeline: ProductionPipeline = Depends(get_pipeline),
):
    job_id = await pipeline.run_pipeline_and_save(request_body.urls,request_body.cluster_job_name)
    return {"job_id": job_id}