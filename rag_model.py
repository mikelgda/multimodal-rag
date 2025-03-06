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
        max_pages_per_batch=1,
    ):
        instance = cls()

        instance.model, instance.processor = load_model_and_processor(
            pretrained_model, device=device, attn_mode=attn_mode, dtype=dtype
        )

        instance.pretrained_model = pretrained_model
        instance.device = device
        instance.attn_mode = attn_mode
        instance.dtype = dtype
        instance.index_name = None
        instance._saved_docs = {}
        instance._index_embeddings = []
        instance._embed_id_to_doc_id = {}
        instance.max_pages_per_batch = max_pages_per_batch

        return instance

    @classmethod
    def from_index(
        cls, index_path, device="cpu", attn_mode=None, dtype=None, max_pages_per_batch=1
    ):
        import json

        index_path = Path(index_path)
        config = json.load((index_path / "config.json").open())
        docs = json.load((index_path / "docs.json").open())
        embed_id_to_doc_id = json.load((index_path / "embed_id_to_doc_id.json").open())

        instance = cls()

        dtype = dtype if dtype else eval(config["dtype"])
        instance.model, instance.processor = load_model_and_processor(
            config["pretrained_model"],
            device=device,
            attn_mode=attn_mode,
            dtype=dtype,
        )

        embeddings = []
        for i in range(config["embedding_chunks"]):
            embeddings.extend(
                torch.load((index_path / "embeddings" / f"embeddings_{i}.pt"))
            )

        instance.pretrained_model = config["pretrained_model"]
        instance.device = device
        instance.attn_mode = attn_mode
        instance.dtype = dtype
        instance.index_name = config["index_name"]
        instance._saved_docs = docs
        instance._index_embeddings = embeddings
        instance._embed_id_to_doc_id = embed_id_to_doc_id
        instance.max_pages_per_batch = max_pages_per_batch

        return instance

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

        return embedded_images

    def create_index(
        self,
        source,
        index_name,
        doc_idx=None,
        metadata=None,
        overwrite=True,
        max_pages_per_batch=None,
    ):

        if self.index_name:
            print(f"There already exists an index {self.index_name}")

        source_path = Path(source)
        doc_files = list(source_path.iterdir())

        doc_idx = range(len(doc_files)) if doc_idx is None else doc_idx
        if len(doc_idx) != len(doc_files):
            raise ValueError("Different number of doc IDs and docs")
        if metadata is not None:
            if len(metadata) != len(doc_files):
                raise ValueError("Different number of metadata and docs")

        self.index_name = index_name
        max_pages_per_batch = (
            max_pages_per_batch if max_pages_per_batch else self.max_pages_per_batch
        )

        for i, (doc_id, doc_file) in enumerate(zip(doc_idx, doc_files)):
            doc_meta = metadata[i] if metadata else None
            print(f"Adding doc {doc_file} with doc_id {doc_id} and metadata {doc_meta}")

            self.add_to_index(
                doc_file,
                doc_id,
                metadata=doc_meta,
                overwrite=overwrite,
                max_pages_per_batch=max_pages_per_batch,
            )

    def add_to_index(
        self,
        doc_path,
        doc_id=None,
        metadata=None,
        overwrite=False,
        max_pages_per_batch=None,
    ):

        if self.index_name is None:
            raise RuntimeError("There is no index loaded.")

        if doc_id in self._saved_docs and not overwrite:
            raise ValueError(f"Doc {doc_id} already exists in the index")

        doc_path = Path(doc_path)

        pages = process_doc(doc_path)
        print(f"Embedding doc: {doc_path} with {len(pages)} pages")
        for i in range(0, len(pages), max_pages_per_batch):
            pages_chunk = pages[i : i + max_pages_per_batch]
            print(f"Page chunk {i} of length {len(pages_chunk)}")
            embedded_pages = self.embed_images(pages_chunk)
            for page_num, embedding in enumerate(torch.unbind(embedded_pages.cpu())):
                embed_id = len(self._index_embeddings)
                self._index_embeddings.append(embedding)
                self._embed_id_to_doc_id[embed_id] = {
                    "doc_id": doc_id,
                    "page_id": i + page_num + 1,
                }

        self._saved_docs[doc_id] = {
            "doc_path": doc_path.as_posix(),
            "pages": len(pages),
            "metadata": metadata,
        }
        print("Saved!")
        print()

    def clear_index(self):
        self._index_embeddings = []
        self._embed_id_to_doc_id = {}
        self._saved_docs = {}
        self.index_name = None

    def search(self, queries, top_k=3, with_scores=True):
        embedded_queries = torch.unbind(self.embed_queries(queries).cpu())
        scores = self.processor.score(embedded_queries, self._index_embeddings)
        top_idx = scores.cpu().numpy().argsort()[0, -3:][::-1]

        doc_page_ids = [self._embed_id_to_doc_id[idx] for idx in top_idx]

        results = []
        for doc_page_idx in doc_page_ids:
            doc_result = {}
            doc_result.update(doc_page_idx)
            doc_result.update(self._saved_docs[doc_page_idx["doc_id"]])
            results.append(doc_result)

        if with_scores:
            return results, scores.cpu().numpy()[0, top_idx]
        else:
            return results

    def save_index(self, save_path=None, em_chunk_size=500):
        import json

        save_path = Path(save_path) / self.index_name
        save_path.mkdir(parents=True, exist_ok=True)

        embeddings_path = save_path / "embeddings"
        embeddings_path.mkdir(exist_ok=True)
        for i in range(0, len(self._index_embeddings), em_chunk_size):
            torch.save(
                self._index_embeddings[i : i + em_chunk_size],
                embeddings_path / f"embeddings_{i}.pt",
            )

        config = {
            "index_name": self.index_name,
            "pretrained_model": self.pretrained_model,
            "dtype": str(self.dtype),
            "embedding_chunks": i + 1,
        }

        json.dump(config, (save_path / "config.json").open("w"))

        json.dump(self._saved_docs, (save_path / "docs.json").open("w"), indent=4)
        json.dump(
            self._embed_id_to_doc_id,
            (save_path / "embed_id_to_doc_id.json").open("w"),
            indent=4,
        )


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

    doc_path = Path(doc_path)
    if doc_path.suffix.lower() == ".pdf":
        images = convert_from_path(doc_path)
        return images

    elif doc_path.suffix.lower() in [".jpg", ".jpeg", ".png", "bmp"]:
        return [Image.open(doc_path)]

    else:
        raise ValueError(f"Unknown file type {doc_path.suffix}")
