from pathlib import Path
from uuid import uuid4

import torch
from qdrant_client import models

from .utils import process_doc
from .vlm import VLM


class SimpleMultiModalRetriever:

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

        instance.vlm = VLM.from_pretrained(
            pretrained_model, device=device, attn_mode=attn_mode, dtype=dtype
        )

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
        instance.vlm = VLM.from_pretrained(
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
            embedded_pages = self.vlm.embed_images(pages_chunk)
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
        embedded_queries = torch.unbind(self.vlm.embed_queries(queries).cpu())
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


class QdrantMultiModalRetriever:

    def __init__(self, vlm, client, save_docs_path=None):
        self.vlm = vlm
        self.client = client
        self.collection = None
        if save_docs_path is not None:
            save_docs_path = Path(save_docs_path)
            if save_docs_path.exists():
                print(
                    f"WARNING: the path {save_docs_path} already exits. Data may be overwritten."
                )
            else:
                save_docs_path.mkdir(parents=True)
        self.save_docs_path = save_docs_path

    def create_collection(
        self,
        collection_name,
        on_disk_payload=True,
        quantization_quantile=0.99,
        quantization_always_ram=True,
    ):
        success = self.client.create_collection(
            collection_name=collection_name,  # the name of the collection
            on_disk_payload=on_disk_payload,  # store the payload on disk
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=100
            ),  # it can be useful to swith this off when doing a bulk upload and then manually trigger the indexing once the upload is done
            vectors_config=models.VectorParams(
                size=self.vlm.dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=quantization_quantile,
                        always_ram=quantization_always_ram,
                    ),
                ),
            ),
        )
        if success:
            self.collection = collection_name
        else:
            print(f"There was an error creating the collection {collection_name}")

    def choose_collection(self, collection_name):
        if collection_name not in [
            x.name for x in self.client.get_collections().collections
        ]:
            raise ValueError(f"Collection {collection_name} does not exist.")
        else:
            self.collection = collection_name

    def upsert_to_qdrant(self, batch):
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=False,
            )
        except Exception as e:
            print(f"Error during upsert: {e}")
            return False
        return True

    def save_item(self, item, id=None, payload={}):

        if id is None:
            id = str(uuid4())
        page_embedding = self.vlm.embed_images(item)[0].cpu().float().numpy().tolist()
        point = models.PointStruct(
            id=id,
            vector=page_embedding,
            payload=payload,
        )

        self.upsert_to_qdrant([point])

    def save_document(self, doc, payload={}, batch_size=2):
        doc = Path(doc)
        pages = process_doc(doc)
        num_pages = len(pages)
        doc_points = []
        for i in range(0, num_pages, batch_size):
            batch = pages[i : i + batch_size]
            batch_embedding = self.vlm.embed_images(batch)
            for j, page_embedding in enumerate(
                torch.unbind(batch_embedding.cpu().float())
            ):
                payload = {
                    "doc_name": doc.name,
                    "page_num": i + j + 1,
                }
                id = str(uuid4())
                if self.save_docs_path is not None:
                    pages[i + j].save(self.save_docs_path / f"{id}.jpg")
                multivector = page_embedding.numpy().tolist()
                doc_points.append(
                    models.PointStruct(id=id, vector=multivector, payload=payload)
                )

        self.upsert_to_qdrant(doc_points)

    def save_from_folder(self, folder_path, batch_size=2):
        folder_path = Path(folder_path)
        doc_files = list(folder_path.iterdir())
        for doc_file in doc_files:
            print(f"Saving doc {doc_file}")
            self.save_document(doc_file, batch_size=batch_size)

    def list_items(self, limit=10):
        return self.client.scroll(collection_name=self.collection, limit=limit)[0]

    def search(self, query, limit=5, timeout=60):
        query_embedding = (
            self.vlm.embed_queries([query])[0].cpu().float().numpy().tolist()
        )
        search_result = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=limit,
            timeout=timeout,
        )

        return search_result

    def item_id_exists(self, id):

        return bool(self.client.retrieve(self.collection, ids=[id]))
