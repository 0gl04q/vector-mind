from typing import Literal

from pydantic import BaseModel
from .const import BASE_EMBEDDING_MODEL, CPU_DEVICE


class ChromaConfig(BaseModel):
        """
        Класс ChromaConfig представляет конфигурацию для работы с хранилищем Chroma.

        Атрибуты:
            persist_directory (str): Путь к директории для сохранения данных.
            collection_name (str): Название коллекции в хранилище.
            embedding_model_name (str): Название модели эмбеддингов (по умолчанию BASE_EMBEDDING_MODEL).
            device (Literal["cpu", "cuda"]): Устройство для вычислений (по умолчанию CPU_DEVICE).
            normalize_embeddings (bool): Флаг нормализации эмбеддингов (по умолчанию True).
        """
        persist_directory: str
        collection_name: str
        embedding_model_name: str = BASE_EMBEDDING_MODEL
        device: Literal["cpu", "cuda"] = CPU_DEVICE
        normalize_embeddings: bool = True


class SimilaritySearch(BaseModel):
    """
    Класс SimilaritySearch представляет параметры для выполнения поиска по схожести.

    Атрибуты:
        query (str): Запрос для поиска.
        with_score (bool): Флаг включения оценки схожести в результат (по умолчанию True).
        k (int): Количество ближайших соседей для поиска (по умолчанию 4).
        filters (dict | None): Дополнительные фильтры для поиска (по умолчанию None).
    """
    query: str
    with_score: bool = True
    k: int = 4
    filters: dict | None = None