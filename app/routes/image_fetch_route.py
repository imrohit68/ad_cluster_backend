from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.routes.deps.dependencies import  get_downloader
from app.services.image_download_service import ImageDownloader

router = APIRouter(tags=["Pipeline"])

@router.get("/fetch-images/{brand_key}", response_model=List[str])
async def fetch_images(brand_key: str,
 downloader: ImageDownloader = Depends(get_downloader)
):
    """
    Fetch image URLs for a specific brand key.
    """
    urls = downloader.fetch_image_urls(brand_key)
    if not urls:
        raise HTTPException(status_code=404, detail="No images found for this brand key.")
    return urls