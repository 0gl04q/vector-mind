import asyncio

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.vector_mind.adapters.llm import Chat, StreamRequest
from src.vector_mind.adapters.store import ChromaConfig, SimilaritySearch
from src.vector_mind.adapters.store.chroma import ChromaStore


async def main():
    llm = ChatOpenAI(
        model="meta-llama-3.1-8b-instruct",
        base_url="http://localhost:1234/v1",
        api_key=SecretStr("apikey"),
        temperature=0.2
    )

    config = ChromaConfig(
        persist_directory="../chroma_db",
        collection_name="obsidian",
        embedding_model_name="BAAI/bge-m3",
        device="cpu",
        normalize_embeddings=True
    )

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

    chat = Chat(llm)

    request = StreamRequest(
        system_prompt=SystemMessage("Ты опытный преподаватель, объясняющий сложные концепции простыми словами."),
        human_prompt=HumanMessage(f"Вопрос: {query}\nКонтекст:{context}"),
    )

    response = []
    async for chunk in chat.stream_response(request):
        response.append(chunk)

    print("".join(response))


if __name__ == '__main__':
    asyncio.run(main())