import torch
import logging
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DINOv2EmbeddingGenerator:
    """Production-ready DINOv2 embedding generator for ad image deduplication."""

    def __init__(self, model_name: str = "facebook/dinov2-large"):
        """
        Initialize DINOv2 model and processor.

        Args:
            model_name: DINOv2 variant ("small", "base", "large").
        """
        self.device = self._get_device()
        logger.info(f"Initializing DINOv2 model: {model_name} on {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        try:
            self.model.set_attn_implementation("eager")
            logger.info("Set attention implementation to 'eager' for attention map extraction.")
        except Exception as e:
            logger.warning(f"Could not set attn_implementation to 'eager': {e}")

        self.model.eval()

        self.model_name = model_name
        self.embedding_dim = self.model.config.hidden_size

        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

    def _get_device(self) -> str:
        """Detect best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def extract_embeddings_from_images(
        self,
        images: List[Image.Image],
        image_ids: List[str] = None,
        batch_size: int = 16,
        use_attention_weights: bool = True,
        emphasis_strength: float = 2.0
    ) -> Dict:
        """
        Extract embeddings for in-memory PIL images.

        Args:
            images: List of PIL.Image objects.
            image_ids: Optional identifiers for images.
            batch_size: Number of images per batch.
            use_attention_weights: Enable attention-based weighting.
            emphasis_strength: Foreground emphasis (1.0–3.0).

        Returns:
            Dictionary with embeddings, successful IDs, failed IDs, and metadata.
        """
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]

        embeddings_list, successful_ids, failed_ids = [], [], []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_ids = image_ids[i:i + batch_size]
            valid_images, valid_ids = [], []

            for img, img_id in zip(batch_images, batch_ids):
                try:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    valid_images.append(img)
                    valid_ids.append(img_id)
                except Exception as e:
                    failed_ids.append(img_id)
                    logger.warning(f"Image conversion failed for {img_id}: {e}")

            if not valid_images:
                continue

            try:
                inputs = self.processor(images=valid_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = (
                        self.model(**inputs, output_attentions=True)
                        if use_attention_weights else
                        self.model(**inputs)
                    )

                last_hidden = outputs.last_hidden_state
                cls_token = last_hidden[:, 0]
                patch_tokens = last_hidden[:, 1:]

                if use_attention_weights and hasattr(outputs, "attentions") and outputs.attentions:
                    try:
                        attention_weights = outputs.attentions[-1]
                        avg_attention = attention_weights.mean(dim=1)
                        cls_to_patches = avg_attention[:, 0, 1:]
                        patch_self_attention = avg_attention[:, 1:, 1:].mean(dim=-1)
                        combined_attention = cls_to_patches + patch_self_attention

                        attention_min = combined_attention.min(dim=1, keepdim=True)[0]
                        attention_max = combined_attention.max(dim=1, keepdim=True)[0]
                        spatial_weights = (combined_attention - attention_min) / (
                            attention_max - attention_min + 1e-8
                        )
                        spatial_weights = torch.exp(spatial_weights * emphasis_strength)

                        seq_len = patch_tokens.shape[1]
                        spatial_weights = spatial_weights * (
                            seq_len / spatial_weights.sum(dim=1, keepdim=True)
                        )
                        spatial_weights = spatial_weights.unsqueeze(-1)

                        weighted_patches = patch_tokens * spatial_weights
                        weighted_mean = weighted_patches.sum(dim=1) / seq_len
                    except Exception as e:
                        logger.warning(f"Attention processing failed: {e}")
                        weighted_mean = patch_tokens.mean(dim=1)
                else:
                    weighted_mean = patch_tokens.mean(dim=1)

                max_pool = patch_tokens.max(dim=1)[0]

                embedding = torch.cat(
                    [cls_token, weighted_mean, weighted_mean, max_pool],
                    dim=-1
                )
                batch_embeddings = embedding.cpu().numpy()
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / (norms + 1e-8)

                embeddings_list.append(batch_embeddings)
                successful_ids.extend(valid_ids)

            except Exception as e:
                failed_ids.extend(valid_ids)
                logger.error(f"Batch processing failed: {e}", exc_info=True)

        all_embeddings = np.vstack(embeddings_list) if embeddings_list else np.array([])

        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim * 4,
            "device": self.device,
            "attention_weighting": use_attention_weights,
            "emphasis_strength": emphasis_strength if use_attention_weights else None,
            "total_images": len(images),
            "successful": len(successful_ids),
            "failed": len(failed_ids),
            "timestamp": datetime.now().isoformat()
        }

        return {
            "embeddings": all_embeddings,
            "image_ids": successful_ids,
            "failed_ids": failed_ids,
            "metadata": metadata
        }
