# Vector Mind

Система для работы с векторными базами данных и языковыми моделями, построенная на основе Chroma и LangChain.

## Описание

Vector Mind - это Python-библиотека, которая предоставляет удобные адаптеры для:
- Работы с векторными хранилищами данных (Chroma)
- Взаимодействия с языковыми моделями (LLM)
- Семантического поиска по документам
- Потоковой генерации ответов

## Возможности

- 🔍 **Семантический поиск**: Поиск релевантных документов по векторному сходству
- 🤖 **Интеграция с LLM**: Поддержка различных языковых моделей через LangChain
- 📊 **Векторное хранилище**: Работа с Chroma для хранения и поиска эмбеддингов
- ⚡ **Потоковая генерация**: Асинхронная генерация ответов с поддержкой стриминга
- 🎯 **Гибкая конфигурация**: Настраиваемые параметры для моделей и хранилища

## Архитектура

```
src/vector_mind/
├── adapters/
│   ├── llm/           # Адаптеры для языковых моделей
│   │   ├── chat.py    # Класс Chat для работы с LLM
│   │   └── model.py   # Модели данных для LLM
│   └── store/         # Адаптеры для векторного хранилища
│       ├── chroma.py  # Класс ChromaStore
│       ├── model.py   # Модели конфигурации
│       └── const.py   # Константы
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd vector-mind
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Базовый пример

```python
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.vector_mind.adapters.llm import Chat, StreamRequest
from src.vector_mind.adapters.store import ChromaConfig, SimilaritySearch
from src.vector_mind.adapters.store.chroma import ChromaStore

async def main():
    # Настройка языковой модели
    llm = ChatOpenAI(
        model="meta-llama-3.1-8b-instruct",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("apikey"),
        temperature=0.2
    )

    # Конфигурация векторного хранилища
    config = ChromaConfig(
        persist_directory="../chroma_db",
        collection_name="obsidian",
        embedding_model_name="BAAI/bge-m3",
        device="cpu",
        normalize_embeddings=True
    )

    # Поиск релевантных документов
    query = "Объясни теорию относительности простыми словами."
    db = ChromaStore(config)
    
    params = SimilaritySearch(
        query=query,
        with_score=True,
        k=4,
        filters=None
    )
    
    results = await db.similarity_search(params)
    context = "\n".join([doc.page_content for doc, _ in results])

    # Генерация ответа
    chat = Chat(llm)
    request = StreamRequest(
        system_prompt=SystemMessage("Ты опытный преподаватель, объясняющий сложные концепции простыми словами."),
        human_prompt=HumanMessage(f"Вопрос: {query}\nКонтекст: {context}"),
    )

    # Потоковый вывод ответа
    response = []
    async for chunk in chat.stream_response(request):
        response.append(chunk)
        print(chunk, end="", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
```

## Требования

- Python 3.8+
- langchain-core
- langchain-openai
- langchain-chroma
- langchain-huggingface
- pydantic
- loguru

## Поддержка

Если у вас есть вопросы или предложения, создайте issue в репозитории.
