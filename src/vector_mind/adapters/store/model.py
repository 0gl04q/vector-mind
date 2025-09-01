from typing import Literal

from pydantic import BaseModel
from .const import BASE_EMBEDDING_MODEL, CPU_DEVICE


class ChromaConfig(BaseModel):
    persist_directory: str
    collection_name: str
    embedding_model_name: str = BASE_EMBEDDING_MODEL
    device: Literal["cpu", "cuda"] = CPU_DEVICE
    normalize_embeddings: bool = True


class SimilaritySearch(BaseModel):
    query: str
    with_score: bool = True
    k: int = 4
    filters: dict | None = None
