from typing import AsyncGenerator

from langchain_openai.chat_models.base import BaseChatModel

from loguru import logger

from .model import StreamRequest


class Chat:
    """
    Класс Chat предоставляет адаптер для взаимодействия с языковой моделью (LLM).
    """

    def __init__(self, llm: BaseChatModel):
        """
        Инициализирует объект Chat с заданной языковой моделью.

        :param llm: Экземпляр языковой модели, реализующий интерфейс BaseChatModel.
        """
        self.llm = llm

    async def stream_response(self, request: StreamRequest) -> AsyncGenerator[str, None]:
        """
        Асинхронно генерирует потоковый ответ на основе запроса.

        :param request: Экземпляр StreamRequest, содержащий системный и пользовательский запросы.
        :yield: Часть ответа (строка), сгенерированная языковой моделью.
        :raises Exception: В случае возникновения ошибки при генерации ответа.
        """
        logger.info(f"Начинаем генерацию ответа для запроса: {request.human_prompt[:50]}...")

        try:
            async for chunk in self.llm.astream([request.system_prompt, request.human_prompt]):
                if chunk.content:
                    logger.debug(f"Получен чанк: {chunk.content[:50]}...")
                    yield chunk.content

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            raise e
