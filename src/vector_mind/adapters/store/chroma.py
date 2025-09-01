from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from loguru import logger

from .model import ChromaConfig, SimilaritySearch


class ChromaStore:
    def __init__(self, config: ChromaConfig):
        self.config = config

        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model_name,
            model_kwargs={"device": config.device},
            encode_kwargs={"normalize_embeddings": config.normalize_embeddings},
        )

        self._store = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=embeddings,
            collection_name=config.collection_name
        )

    async def similarity_search(self, params: SimilaritySearch) -> list:
        try:
            if params.with_score:
                results = self._store.similarity_search_with_score(params.query, k=params.k, filter=params.filters)
            else:
                results = self._store.similarity_search(params.query, k=params.k, filter=params.filters)
            logger.debug(f"Найдено {len(results)} результатов")
            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise e
