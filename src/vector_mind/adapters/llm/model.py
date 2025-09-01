from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel


class StreamRequest(BaseModel):
    """
    Модель StreamRequest представляет запрос для потоковой генерации ответа.

    Атрибуты:
        system_prompt (SystemMessage): Системное сообщение, задающее контекст для генерации.
        human_prompt (HumanMessage): Сообщение от пользователя, на которое требуется ответ.
    """
    system_prompt: SystemMessage
    human_prompt: HumanMessage
