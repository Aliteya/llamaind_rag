from ..core import settings

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import os
import json
import asyncio

qa_prompt = PromptTemplate(
    "Отвечай ТОЛЬКО на английском языке и ТОЛЬКО используя предоставленные данные.\n"
    "Формат ответа:\n"
    "1. [пункт]\n"
    "   - [Детализация пунктов]\n"
    "2. [Следующий пункт]\n"
    "Используй только данные из этих разделов:\n{context_str}\n\n"
    "Вопрос: {query_str}\n"
    "Ответ:"
)

def setup_qdrant(cache_dir: str, collection_name: str):
    os.makedirs(cache_dir, exist_ok=True)
    client = qdrant_client.QdrantClient(path=cache_dir)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return vector_store

def load_json_db(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = []
    for item in data:
        file_path = item["file_path"]
        for chunk in item["chunks"]:
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
            document = Document(text=text, metadata={"file_path": file_path, **metadata})
            documents.append(document)
    return documents

def create_ingestion_pipeline():
    Settings.llm = OpenAI(
        model="gpt-4o-mini", 
        api_key=settings.get_llm_key(), 
        temperature=0.6
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
          api_key=settings.get_llm_key()
    )
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            ]
        )
    return pipeline

def setup_rag_pipeline(nodes, vector_store):
    index = VectorStoreIndex(nodes, vector_store=vector_store)
    query_engine = index.as_query_engine(text_qa_template=qa_prompt)
    return query_engine

async def data_process(documents, pipeline: IngestionPipeline, num_workers=5):
    nodes = await pipeline.arun(documents, num_workers=num_workers)
    return nodes

async def proccess_questions(query_engine, questions: list) -> list:
    async def process_question(question):
        try:
            response = await query_engine.aquery(question)
            retrieved_chunks = [node.node.text for node in response.source_nodes]

            if not retrieved_chunks:
                answer = "Не могу найти информацию по этому вопросу."
            else:
                answer = response.response

            return {
                "question": question,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks
            }
        except Exception as e:
            return {
                "question": question,
                "answer": "Вопрос выходит за пределы базы знаний.",
                "retrieved_chunks": []
            }

    tasks = [process_question(question) for question in questions]
    responses = await asyncio.gather(*tasks)
    return responses