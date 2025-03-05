from pathlib import Path

import torch
from colpali_engine.models import (
    ColIdefics3,
    ColIdefics3Processor,
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2Processor,
)
from pdf2image import convert_from_path
from PIL import Image


class MultiModalRAGModel:

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model,
        device="cpu",
        attn_mode=None,
        dtype=torch.bfloat16,
        index_root="./index",
    ):
        instance = cls()

        instance.model, instance.processor = load_model_and_processor(
            pretrained_model, device=device, attn_mode=attn_mode, dtype=dtype
        )

        instance.device = device
        instance.attn_mode = attn_mode
        instance.dtype = dtype
        instance.index_root = index_root
        instance.index_name = None
        instance._doc_idx = {}
        instance._doc_id_to_name = {}
        instance._last_index_id = -1
        instance._index_embeddings = []
        instance._embed_id_to_doc_id = {}

        return instance

    @property
    def doc_idx_to_name(self):
        return self._doc_idx_to_name

    @property
    def doc_idx(self):
        return self._doc_idx

    @property
    def embed_id_to_doc_id(self):
        return self._embed_id_to_doc_id

    @property
    def index_embeddings(self):
        return self._index_embeddings

    def embed_queries(self, query):
        if isinstance(query, str):
            query = [query]

        with torch.inference_mode():
            processed_query = self.processor.process_query(query).to(self.device)
            embedded_query = self.model(**processed_query)

        return embedded_query

    def embed_images(self, image):
        if isinstance(image, Image.Image):
            images = [image]

        with torch.inference_mode():
            processed_images = self.processor.process_images(images).to(self.device)
            embedded_images = self.model(**processed_images)

        return embedded_images

    def create_index(self, source, index_name, doc_idx=None, metadata=None):

        index_path = Path(self.create_index) / Path(index_name)
        if index_path.exists():
            print(f"Index path {index_path} already exists")

        self.index_name = index_name

        source_path = Path(source)
        if source.isdir():
            source_files = source_path.iterdir()
        else:
            source_files = [source_path]

        using_default_index = doc_idx is None
        for i, source_file in enumerate(source_files):
            if using_default_index:
                doc_id = self._last_index_id + 1
            else:
                doc_id = doc_idx[i]
            doc_meta = metadata[doc_id] if metadata else None

            self.add_to_index(source_file, doc_id, metadata=doc_meta)

    def add_to_index(
        self,
        item,
        doc_id=None,
        page_id=1,
        metadata=None,
    ):

        if self.index_name is None:
            raise RuntimeError("There is no index loaded.")

        pages = process_item(item)
        embedded_pages = self.embed_images(pages)

        for i, embedding in enumerate(torch.unbind(embedded_pages.cpu())):
            embed_id = len(self._index_embeddings)
            self._index_embeddings.append(embedding)
            self._emebd_id_to_doc_id[embed_id] = {
                "doc_id": doc_id,
                "page_id": i + 1,
            }


def load_model_and_processor(
    pretrained_model, device="cpu", attn_mode=None, dtype=torch.bfloat16
):
    if "colpali" in pretrained_model:
        model_class = ColPali
        processor_class = ColPaliProcessor
    elif "colSmol" in pretrained_model:
        model_class = ColIdefics3
        processor_class = ColIdefics3Processor
    elif "colQwen" in pretrained_model:
        model_class = ColQwen2
        processor_class = ColQwen2Processor
    else:
        raise ValueError(f"Unknown model {pretrained_model}")

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=pretrained_model,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_mode,
    ).eval()
    processor = processor_class.from_pretrained(pretrained_model)

    return model, processor


def process_item(item):
    from tempfile import TemporaryDirectory

    if isinstance(item, Image):
        return [item]

    path = Path(item)
    if path.suffix.lower() == ".pdf":
        images = convert_from_path(path)
        with TemporaryDirectory() as temp_dir:
            image_paths = convert_from_path(
                path, output_folder=temp_dir, paths_only=True
            )

            return [Image.open(image_path) for image_path in image_paths]

    elif item.suffix.lower() in [".jpg", ".jpeg", ".png", "bmp"]:
        return [Image.open(item)]
