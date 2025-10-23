"""
Image downloader service for downloading images with timeout and retry logic.
"""

import concurrent.futures
import requests
import time
from io import BytesIO
from typing import List, Dict, Optional
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

    def download_single_image(self, url: str, timeout: int = 20, max_retries: int = 3) -> Optional[Dict]:
        """Download a single image and return PIL Image."""
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes).convert('RGB')

            return {"success": True, "image": image}
        except Exception as e:
            logger.warning(f"Download failed: {url} - {e}")
            return None
