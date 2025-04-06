from ..core import settings
from ..logging import logger

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, PromptTemplate
from llama_index.core.schema import TextNode
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
    "Answer ONLY in English and ONLY using the data provided.\n"
    "Answer format:\n"
    "1. [item]\n"
    " - [Item details]\n"
    "2. [Next item]\n"
    "Use only data from these sections:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

# def setup_qdrant(cache_dir: str, collection_name: str):
#     os.makedirs(cache_dir, exist_ok=True)
#     client = qdrant_client.QdrantClient(path=cache_dir)
#     vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
#     return vector_store

def load_json_db(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = []
    for item in data:
        file_path = item["file_path"]
        for chunk in item["chunks"]:
            if not chunk["text"].strip():
                logger.warning(f"Пустой чанк в файле {item['file_path']}")
                continue
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
            document = Document(text=text, metadata={"file_path": file_path, **metadata})
            documents.append(document)
    logger.info(f"Загружено документов: {len(documents)}")
    if not documents:
        raise ValueError("Документы не загружены! Проверьте путь к db.json и его структуру.")
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
        transformations=[]
        )
    return pipeline

def setup_rag_pipeline(nodes):
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine(text_qa_template=qa_prompt)
    return query_engine

async def data_process(documents, pipeline: IngestionPipeline, num_workers=5):
    nodes = [
        TextNode(
            text=doc.text,
            metadata=doc.metadata
        ) 
        for doc in documents
    ]
    logger.info(f"Сгенерировано узлов (nodes): {len(nodes)}")
    if not nodes:
        raise ValueError("Не удалось создать узлы. Проверьте пайплайн обработки.")
    return nodes

async def proccess_questions(query_engine, questions: list) -> list:
    async def process_question(question):
        try:
            logger.info(f"Запрос: {question}")
            response = await query_engine.aquery(question)
            logger.info(f"{response}")
            logger.info(f"Найдено чанков: {len(response.source_nodes)}")
            logger.debug(f"Топ-1 чанк: {response.source_nodes[0].node.text[:200]}...")
            retrieved_chunks = [node.node.text for node in response.source_nodes]
            logger.debug(f"Запрос: {question}")
            logger.debug(f"Найдено чанков: {len(response.source_nodes)}")
            logger.debug(f"Топ-1 чанк: {response.source_nodes[0].node.text[:200]}...")

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
                "answer": "The question is outside the knowledge base.",
                "retrieved_chunks": []
            }

    tasks = [process_question(question) for question in questions]
    responses = await asyncio.gather(*tasks)
    return responses