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
        low_memory=False,
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
        instance._doc_id_to_metadata = {}
        instance._index_embeddings = []
        instance._embed_id_to_doc_id = {}
        instance.low_memory = low_memory

        return instance

    @property
    def doc_id_to_name(self):
        return self._doc_id_to_name

    @property
    def doc_idx(self):
        return self._doc_idx

    @property
    def embed_id_to_doc_id(self):
        return self._embed_id_to_doc_id

    @property
    def index_embeddings(self):
        return self._index_embeddings

    @property
    def doc_id_to_metadata(self):
        return self._doc_id_to_metadata

    def embed_queries(self, query):
        if isinstance(query, str):
            query = [query]

        with torch.inference_mode():
            processed_query = self.processor.process_query(query).to(self.device)
            embedded_query = self.model(**processed_query)

        return embedded_query

    def embed_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]

        with torch.inference_mode():
            processed_images = self.processor.process_images(images).to(self.device)
            embedded_images = self.model(**processed_images)

        return embedded_images

    def create_index(self, source, index_name, doc_idx=None, metadata=None):

        index_path = Path(self.index_root) / Path(index_name)
        if index_path.exists():
            print(f"Index path {index_path} already exists")

        source_path = Path(source)
        doc_files = list(source_path.iterdir())

        doc_idx = range(len(doc_files)) if doc_idx is None else doc_idx
        if len(doc_idx) != len(doc_files):
            raise ValueError("Different number of doc IDs and docs")
        if metadata is not None:
            if len(metadata) != len(doc_files):
                raise ValueError("Different number of metadata and docs")

        self.index_name = index_name

        for i, (doc_id, doc_file) in enumerate(zip(doc_idx, doc_files)):
            doc_meta = metadata[i] if metadata else None
            print(f"Adding doc {doc_file} with doc_id {doc_id} and metadata {doc_meta}")

            self.add_to_index(doc_file, doc_id, metadata=doc_meta)

    def add_to_index(
        self,
        doc_path,
        doc_id=None,
        metadata=None,
    ):

        if self.index_name is None:
            raise RuntimeError("There is no index loaded.")

        pages = process_doc(doc_path)
        print("Embedding pages", print(len(pages)), print(pages))
        if self.low_memory:
            print("Low memory mode")
            for page_num, page in enumerate(pages):
                print(f"Embedding page {page_num}")
                embedded_page = self.embed_images(page)[0].cpu()
                embed_id = len(self._index_embeddings)
                self._index_embeddings.append(embedded_page)
                self._embed_id_to_doc_id[embed_id] = {
                    "doc_id": doc_id,
                    "page_id": page_num + 1,
                }
        else:
            embedded_pages = self.embed_images(pages)

            for page_num, embedding in enumerate(torch.unbind(embedded_pages.cpu())):
                embed_id = len(self._index_embeddings)
                self._index_embeddings.append(embedding)
                self._embed_id_to_doc_id[embed_id] = {
                    "doc_id": doc_id,
                    "page_id": page_num + 1,
                }

        self._doc_id_to_name[doc_id] = doc_path.as_posix()
        self._doc_id_to_metadata[doc_id] = metadata
        self._doc_idx.append(doc_id)


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


def process_doc(doc_path):
    print(f"Processing item {doc_path}")

    doc_path = Path(doc_path)
    if doc_path.suffix.lower() == ".pdf":
        print("It's a PDF")
        images = convert_from_path(doc_path)
        return images

    elif doc_path.suffix.lower() in [".jpg", ".jpeg", ".png", "bmp"]:
        print("It's an image")
        return [Image.open(doc_path)]

    else:
        raise ValueError(f"Unknown file type {doc_path.suffix}")
