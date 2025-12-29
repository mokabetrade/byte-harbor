from __future__ import annotations

import time
from typing import Iterable
import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop
from config.settings import BackgroundRemovalConfig
from logger_config import logger

from ben2 import BEN_Base

class BackgroundRemovalService:
    def __init__(self, settings: BackgroundRemovalConfig):
        """
        Initialize the BackgroundRemovalService.
        """
        self.settings = settings

        # Set padding percentage and output size for centering and resizing
        self.padding_percentage = self.settings.padding_percentage
        self.limit_padding = self.settings.limit_padding
        self.output_size = self.settings.output_image_size

        # Set device
        self.device = f"cuda:{settings.gpu}" if torch.cuda.is_available() else "cpu"

        # Set BEN model
        self.model: BEN_Base | None = None

        # Set transform
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.settings.model_id} model...")

        # Load model
        try:
            self.model = BEN_Base.from_pretrained(self.settings.model_id).to(self.device).eval()
            logger.success(f"{self.settings.model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.settings.model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.settings.model_id} model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.settings.model_id} model not initialized.")

    def remove_background(self, image: Image.Image | Iterable[Image.Image]) -> Image.Image | Iterable[Image.Image]:
        """
        Remove the background from the image.
        """
        # try:
        t1 = time.time()
        
        # Image (H, W, C=3) -> Tensor (B>=1, C=4, H',W')
        foreground_tensor = self._remove_background(image)
        outputs = tuple(self._crop_and_center(f) for f in foreground_tensor)

        outputs = tuple(to_pil_image(o[:3]) for o in outputs)
        image_without_background = outputs

        removal_time = time.time() - t1
        logger.success(f"Background remove - Time: {removal_time:.2f}s - OutputSize: {outputs[0].size}")

        return image_without_background

    def _remove_background(self, image: Image.Image | Iterable[Image.Image]) -> torch.Tensor:
        """
        Remove the background from the image.
        """
        with torch.no_grad():
            foreground = self.model.inference(image.copy())
        foregrounds = foreground if isinstance(foreground, Iterable) else (foreground,)

        return torch.stack(tuple(self.transforms(f) for f in foregrounds), dim=0).to(self.device)  # Tensor shape: (B>=1, C=4, H', W')

    def _crop_and_center(self, foreground_tensor: torch.Tensor) -> torch.Tensor:
        """
        Remove the background from the image.
        """

        # Normalize tensor value for background removal model, reshape for model batch processing
        tensor_rgb = foreground_tensor[:3]
        mask = foreground_tensor[-1]

        # Get bounding box indices
        bbox_indices = torch.argwhere(mask > 0.8)
        logger.info(f"BBOX len: {len(bbox_indices)}")
        if len(bbox_indices) == 0:
            crop_args = dict(top = 0, left = 0, height = mask.shape[1], width = mask.shape[0])
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 1])
            w_min, w_max = torch.aminmax(bbox_indices[:, 0])
            width, height = w_max - w_min, h_max - h_min
            center =  (h_max + h_min) / 2, (w_max + w_min) / 2
            size = max(width, height)
            padded_size_factor = 1 + self.padding_percentage
            size = int(size * padded_size_factor)

            top = int(center[1] - size // 2)
            left = int(center[0] - size // 2)
            bottom = int(center[1] + size // 2)
            right = int(center[0] + size // 2)

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[1], bottom)
                right = min(mask.shape[0], right)
            
            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left
            )
        

        logger.info(f"CROP: {crop_args}")
        mask = mask.unsqueeze(0)
        # Concat mask with image and blacken the background: (C=3, H, W) | (1, H, W) -> (C=4, H, W)
        tensor_rgba = torch.cat([tensor_rgb*mask, mask], dim=-3)
        output = resized_crop(tensor_rgba, **crop_args, size = self.output_size, antialias=False)
        return output