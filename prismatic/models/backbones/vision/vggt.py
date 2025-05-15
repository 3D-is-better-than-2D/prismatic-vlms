from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from functools import partial
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

from prismatic.models.backbones.vision.base_vision import TimmViTBackbone
from prismatic.overwatch import initialize_overwatch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images



overwatch = initialize_overwatch(__name__)

class VGGTBackbone(TimmViTBackbone):
    """
    VGGT backbone for extracting VGGT features from images.
    """
    def __init__(
        self
    ) -> None:
        super().__init__(
            "vggt",
            "vit_large_patch14_reg4_dinov2.lvd142m",
            image_resize_strategy="resize-crop",
            default_image_size=224,
        )
        
        # Initialize VGGT model
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda")
        self.model.requires_grad_(False)
        self.model.eval()
        
        # Set feature dimension (VGGT-1B output dimension)
        self._embed_dim = 2048  # VGGT-1B feature dimension
        
    def forward(self, image_paths: Union[str, List[str]]) -> torch.Tensor:
        """Extract VGGT features from input images using the aggregator.
        
        Args:
            image_paths: Either a single image path, a list of image paths, or a dictionary containing image paths
                        under the "image_paths" key.
        
        Returns:
            torch.Tensor: Extracted VGGT features in shape [batch_size, num_features, feature_dim]
        """
        # Handle different input types
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        print(image_paths)
        # Load and preprocess images using VGGT's utility
        images = load_and_preprocess_images(image_paths).to("cuda")
        if images.ndim == 4:
            images = images[None]
        # Get aggregated tokens from VGGT
        # change dtype of images
        images = images.to(torch.bfloat16)
        aggregated_tokens, _ = self.model.aggregator(images)
        
        # Get the final layer features
        vggt_features = aggregated_tokens[-1]
        
        print(f"VGGT features shape: {vggt_features.shape}")
        # Reshape to match expected format [batch_size, num_features, feature_dim]
        # VGGT features are already in the right shape, but we ensure it
        features = vggt_features.view(vggt_features.size(0), -1, self.embed_dim)
        
        return features
    
    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    # def get_fsdp_wrapping_policy(self) -> callable:
    #     """Return a simple FSDP policy that wraps the VGGT model."""
    #     vggt_wrap_policy = partial(_module_wrap_policy, module_classes={VGGT})
    #     return partial(_or_policy, policies=[vggt_wrap_policy])

    # @property
    # def default_image_resolution(self) -> Tuple[int, int, int]:
    #     return self._default_image_resolution

    # @property
    # def embed_dim(self) -> int:
    #     return self._embed_dim

    # @property
    # def num_patches(self) -> int:
    #     return self._num_patches

    # @property
    # def half_precision_dtype(self) -> torch.dtype:
    #     return self._half_precision_dtype 