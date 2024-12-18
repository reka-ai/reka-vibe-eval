from typing import Tuple
import httpx
from pathlib import Path
import time

def validate_image_url(media_url: str) -> Tuple[str, str]:
    """Validate image URL and return media type.
    
    Args:
        media_url: URL of the image
        
    Returns:
        Tuple of (media_type, extension)
        
    Raises:
        ValueError: If image extension is not supported
    """
    ext = Path(media_url).suffix.lower()[1:]  # Remove the dot
    if ext == "jpg":
        ext = "jpeg"
    if ext not in ["jpeg", "png"]:
        raise ValueError(f"Unsupported image extension: {ext}")
    return f"image/{ext}", ext

def get_image_data(media_url: str) -> bytes:
    """Fetch image data from URL.
    
    Args:
        media_url: URL of the image
        
    Returns:
        bytes: Raw image data
        
    Raises:
        httpx.HTTPError: If image fetch fails
    """
    response = httpx.get(media_url)
    response.raise_for_status()
    return response.content 

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_requests: int, time_window: int):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = 0
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        if self.requests >= self.max_requests:
            time.sleep(self.time_window)
            self.requests = 0
        
        self.requests += 1