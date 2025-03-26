import torch
from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
from PIL import Image

from .utils import load_model_and_processor


class VLM:

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model,
        device="cpu",
        attn_mode=None,
        dtype=torch.bfloat16,
        pool_factor=0,
    ):
        instance = cls()

        instance.model, instance.processor = load_model_and_processor(
            pretrained_model, device=device, attn_mode=attn_mode, dtype=dtype
        )

        instance.pretrained_model = pretrained_model
        instance.device = device
        instance.attn_mode = attn_mode
        instance.dtype = dtype
        instance.pool_factor = pool_factor
        instance.pool_tokens = bool(pool_factor)

        if instance.pool_tokens:
            instance.token_pooler = HierarchicalTokenPooler(pool_factor=pool_factor)
        else:
            instance.token_pooler = None

        return instance

    @property
    def dim(self):
        return self.model.dim

    def embed_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries]

        with torch.inference_mode():
            processed_queries = self.processor.process_queries(queries).to(self.device)
            embedded_queries = self.model(**processed_queries)

        return embedded_queries

    def embed_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]

        with torch.inference_mode():
            processed_images = self.processor.process_images(images).to(self.device)
            embedded_images = self.model(**processed_images)

        if self.pool_tokens:
            embedded_images = self.token_pooler.pool_embeddings(
                embedded_images,
                padding=True,
                padding_side=self.processor.tokenizer.padding_side,
            )

        return embedded_images
