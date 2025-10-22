"""
Download Models
---------------
Pydantic models for image download results and errors.
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional
from PIL import Image


class DownloadResult(BaseModel):
    """
    Represents a successfully downloaded image.
    """
    index: int = Field(..., description="Index of the image in the original list")
    url: str = Field(..., description="Source URL of the downloaded image")
    elapsed: float = Field(..., description="Time taken to download the image (in seconds)")
    image: Optional[Image.Image] = Field(
        None,
        description="In-memory PIL Image (excluded from serialization)"
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Image.Image: lambda v: f"<PIL.Image size={v.size if v else None}>"
        }


class DownloadError(BaseModel):
    """
    Represents a failed download attempt.
    """
    index: int = Field(..., description="Index of the image in the original list")
    url: str = Field(..., description="URL that failed to download")
    error: str = Field(..., description="Error message or exception text")
