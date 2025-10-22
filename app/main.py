from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.staticfiles import StaticFiles

from app.core.mongo_client import connect_to_mongo, close_mongo_connection
from app.routes import cluster_job_route, clusters_route, image_fetch_route, pipeline_route
from app.services.clustering_service import RecursiveAdClusterer
from app.services.emdedding_generator_serivce import DINOv2EmbeddingGenerator
from app.services.image_download_service import ImageDownloader
from app.services.image_process_pipeline_service import ProductionPipeline, PipelineConfig
from app.utilities.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""

    await connect_to_mongo()

    app.state.downloader = ImageDownloader(max_workers=16)
    app.state.embedder = DINOv2EmbeddingGenerator()
    app.state.clusterer = RecursiveAdClusterer()
    app.state.pipeline = ProductionPipeline(
        downloader=app.state.downloader,
        embedder=app.state.embedder,
        clusterer=app.state.clusterer,
        config=PipelineConfig()
    )


    logger.info("Application startup complete: Services initialized.")


    yield

    await close_mongo_connection()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Ad Clustering Backend",
    lifespan=lifespan
)
app.include_router(cluster_job_route.router)
app.include_router(clusters_route.router)
app.include_router(image_fetch_route.router)
app.include_router(pipeline_route.router)


app.mount("/static", StaticFiles(directory="static", html=True), name="static")
















