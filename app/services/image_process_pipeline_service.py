"""
Production Pipeline: Image Download → Embedding → Clustering (Stateless)
------------------------------------------------------------------------
Thread-safe, stateless integration of modular services for production.
"""
import asyncio
import functools
import threading
import queue
import time

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from starlette.concurrency import run_in_threadpool

from app.core.mongo_client import mongodb
from app.repository.cluster_repository import ClusterRepository
from app.repository.custer_job_repository import ClusteringJobRepository
from app.services.clustering_service import RecursiveAdClusterer
from app.services.emdedding_generator_serivce import DINOv2EmbeddingGenerator
from app.services.image_download_service import ImageDownloader

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration parameters for the production pipeline."""
    queue_size: int = 64
    embedding_batch_size: int = 16
    max_download_workers: int = 16
    download_timeout: int = 20
    download_max_retries: int = 3
    use_attention: bool = True
    emphasis_strength: float = 2.0


class ProductionPipeline:
    """Stateless production pipeline integrating download, embedding, and clustering."""

    def __init__(
        self,
        downloader: ImageDownloader,
        embedder: DINOv2EmbeddingGenerator,
        clusterer: RecursiveAdClusterer,
        config: Optional[PipelineConfig] = None
    ):
        self.config = config or PipelineConfig()
        self.downloader = downloader
        self.embedder = embedder
        self.clusterer = clusterer

    def run_pipeline(self, urls: List[str], cluster_params: Optional[Dict] = None) -> Optional[Dict]:
        """Run full production pipeline (stateless)."""

        start_time = time.time()
        logger.info(f"Starting production pipeline for {len(urls)} URLs")

        # Shared structures (per-run)
        image_queue = queue.Queue(maxsize=self.config.queue_size)
        done_signal = object()
        results_lock = threading.Lock()

        all_embeddings, all_metadata = [], []
        embed_times = []
        processed_count = 0
        download_stats = {}

        # Producer thread
        def producer():
            nonlocal download_stats
            try:
                logger.info("Producer: Starting image downloads")
                result = self.downloader.download_images(urls)
                download_stats = result["stats"]

                for item in result["success"]:
                    image_queue.put({
                        "index": item.index,
                        "url": item.url,
                        "image": item.image
                    })

                image_queue.put(done_signal)
                logger.info("Producer: Downloads completed")
            except Exception as e:
                logger.exception(f"Producer error: {e}")

        # Consumer thread
        def consumer():
            nonlocal processed_count
            batch_images, batch_urls, batch_indices = [], [], []
            batch_num = 0

            while True:
                try:
                    item = image_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is done_signal:
                    if batch_images:
                        self._process_batch(
                            batch_images, batch_indices, batch_urls,
                            all_embeddings, all_metadata, embed_times, results_lock, batch_num + 1
                        )
                    break

                batch_images.append(item["image"])
                batch_urls.append(item["url"])
                batch_indices.append(item["index"])

                if len(batch_images) >= self.config.embedding_batch_size:
                    batch_num += 1
                    self._process_batch(
                        batch_images, batch_indices, batch_urls,
                        all_embeddings, all_metadata, embed_times, results_lock, batch_num
                    )
                    batch_images, batch_urls, batch_indices = [], [], []

                image_queue.task_done()
                processed_count += 1

            logger.info(f"Consumer: Completed embedding generation for {processed_count} images")

        # Start threads
        producer_thread = threading.Thread(target=producer, name="ProducerThread")
        consumer_thread = threading.Thread(target=consumer, name="ConsumerThread")

        producer_thread.start()
        consumer_thread.start()
        producer_thread.join()
        consumer_thread.join()


        # Combine embeddings
        final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        logger.info(f"Embedding phase complete: {final_embeddings.shape}")

        clustering_result = None

        if processed_count > 0:
            params = {
                "initial_threshold": 0.75,
                "min_threshold": 0.96,
                "threshold_step": 0.05,
                "k_neighbors": 30,
                "min_split_improvement": 0.02
            }
            if cluster_params:
                params.update(cluster_params)

            logger.info("Starting clustering phase")
            embeddings_data = {
                "embeddings": final_embeddings,
                "image_paths": [m["url"] for m in all_metadata],
                "metadata": {}
            }
            clustering_result = self.clusterer.cluster_recursive(
                embeddings_data=embeddings_data, **params
            )
            logger.info(f"Clustering complete: {len(clustering_result['flat_clusters'])} clusters")

        total_time = time.time() - start_time

        logger.info(f"Pipeline complete in {total_time:.2f}s")

        return clustering_result

    async def save_clustering_result(
            self,
            clustering_result: Dict,
            clustering_job_name: str
    ) -> Optional[str]:
        """
        Save clustering result metadata to database.

        Args:
            clustering_result: Result dict from run_pipeline()

        Returns:
            Job ID string if successful, None if clustering_result is None
        """
        if clustering_result is None:
            logger.warning("No clustering result to save")
            return None

        # Generate and save metadata
        metadata = self.clusterer.get_metadata_model(clustering_result,clustering_job_name)
        clustering_jobs_collection = ClusteringJobRepository(mongodb.db["clustering_jobs"])

        job_id = await clustering_jobs_collection.save(metadata)

        logger.info(f"Saved clustering job with ID: {job_id}")
        return job_id

    async def save_clusters_with_job(
            self,
            clustering_result: Dict,
            job_id: str
    ) -> Optional[list[str]]:
        """
        Generate cluster models from clustering result with given job ID.

        Args:
            clustering_result: Result dict from run_pipeline()
            job_id: Parent job ID to associate clusters with

        Returns:
            List of Cluster models, or None if clustering_result is None
        """
        if clustering_result is None:
            logger.warning("No clustering result to process")
            return None

        # Generate cluster models
        cluster_models = self.clusterer.get_cluster_models(clustering_result,job_id)

        logger.info(f"Created {len(cluster_models)} clusters for job {job_id}")

        clustering_collection = ClusterRepository(mongodb.db["clusters"])

        result = await clustering_collection.batch_save(cluster_models)

        return result

    import functools
    from fastapi.concurrency import run_in_threadpool

    async def run_pipeline_and_save(
            self,
            urls: List[str],
            clusteing_job_name: str,
            cluster_params: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Run full pipeline in a thread (downloads + embeddings + clustering)
        and save results to database asynchronously.

        Args:
            urls: List of image URLs to process
            cluster_params: Optional clustering parameters

        Returns:
            Job ID string if successful, None otherwise
        """
        # Run the blocking pipeline in a thread to avoid blocking FastAPI event loop
        clustering_result = await run_in_threadpool(
            functools.partial(self.run_pipeline, urls, cluster_params)
        )

        # Save clustering result asynchronously
        job_id = await self.save_clustering_result(clustering_result, clusteing_job_name)
        if job_id:
            await self.save_clusters_with_job(clustering_result, job_id)

        return job_id

    def _process_batch(
        self,
        images, indices, urls,
        all_embeddings, all_metadata, embed_times, lock, batch_num
    ):
        """Internal: process embedding batch safely."""
        start = time.time()
        try:
            result = self.embedder.extract_embeddings_from_images(
                images=images,
                image_ids=urls,
                batch_size=self.config.embedding_batch_size,
                use_attention_weights=self.config.use_attention,
                emphasis_strength=self.config.emphasis_strength,
            )

            with lock:
                all_embeddings.append(result["embeddings"])
                all_metadata.extend(
                    {"index": i, "url": u} for i, u in zip(indices, urls)
                )

            elapsed = time.time() - start
            embed_times.append(elapsed)
            logger.info(f"Batch {batch_num} complete ({elapsed:.2f}s)")

        except Exception as e:
            logger.exception(f"Batch {batch_num} failed: {e}")

