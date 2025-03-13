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
from PIL.Image import Image


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
