"""
Stateless Image Downloader for Production Use
---------------------------------------------
Thread-safe and reusable instance for FastAPI apps.
"""

import concurrent.futures
import requests
import time
from io import BytesIO
from typing import List, Dict
from PIL import Image
import logging

from app.models.download_image_models import DownloadResult, DownloadError

logger = logging.getLogger(__name__)


class ImageDownloader:
    """Concurrent, stateless downloader with retry logic."""

    def __init__(
        self,
        max_workers: int = 16,
        timeout: int = 20,
        max_retries: int = 3,
        sleep_between_retries: float = 1.5,
    ):
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_between_retries = sleep_between_retries
        self.vibe_my_ad_base_url = "https://vibemyad.com/api/test-assignment"

    def _download_single(self, idx: int, url: str):
        """Download one image with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                resp = requests.get(url, timeout=self.timeout, stream=True)
                resp.raise_for_status()

                img = Image.open(BytesIO(resp.content)).convert("RGB")
                elapsed = time.time() - start
                return DownloadResult(index=idx, url=url, image=img, elapsed=elapsed)

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"[Retry {attempt}/{self.max_retries}] {url}: {e}")
                    time.sleep(self.sleep_between_retries)
                else:
                    logger.error(f"Download failed for {url}: {e}")
                    return DownloadError(index=idx, url=url, error=str(e))

    def download_images(self, urls: List[str]) -> Dict[str, List]:
        """Download multiple images concurrently."""
        start_time = time.time()
        downloaded, failed = [], []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_single, idx, url): url
                for idx, url in enumerate(urls)
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if isinstance(result, DownloadResult):
                    downloaded.append(result)
                elif isinstance(result, DownloadError):
                    failed.append(result)

        total_time = time.time() - start_time
        logger.info(
            f"Download Summary → Success: {len(downloaded)} | Fail: {len(failed)} | "
            f"Time: {total_time:.2f}s | Speed: {len(urls) / total_time:.2f} img/s"
        )

        return {
            "success": downloaded,
            "failed": failed,
            "stats": {
                "total": len(urls),
                "successful": len(downloaded),
                "failed": len(failed),
                "total_time": total_time,
                "avg_time_per_img": total_time / max(len(downloaded), 1)
            }
        }

    def fetch_image_urls(self,brand_key: str, timeout: int = 30) -> List[str]:
        """
        Fetches image URLs for a given brand key.

        Args:
            brand_key (str): Brand identifier (e.g., 'uber', 'nike', etc.)
            timeout (int): Request timeout in seconds.

        Returns:
            List[str]: List of image URLs.
        """
        try:
            params = {"brand_key": brand_key}
            response = requests.get(self.vibe_my_ad_base_url, params=params, timeout=timeout)
            response.raise_for_status()

            data = response.json().get("data", [])
            urls = [item.get("image_url") for item in data if item.get("image_url")]

            return urls

        except requests.exceptions.RequestException as e:
            print(f"Error fetching image URLs: {e}")
            return []
