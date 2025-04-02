from ..core import settings

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

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
def initialize_rag_pipeline(data_dir: str):
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=settings.get_llm_key(),
        temperature=0.4
    )
    
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=settings.get_llm_key()
    )

    index = VectorStoreIndex.from_documents(documents)
    
    return index.as_query_engine(
        text_qa_template=qa_prompt
        )

# def initialize_rag_pipeline(data_dir: str):
#     documents = SimpleDirectoryReader(data_dir).load_data()
#     Settings.llm = OpenAI(model="gpt-4o-mini", api_key=settings.get_llm_key(), temperature=0.2)
#     Settings.embed_model = OpenAIEmbedding(api_key=settings.get_llm_key())
#     index = VectorStoreIndex.from_documents(documents)
#     return  index.as_query_engine(text_qa_template=qa_prompt, similarity_top_k=3, verbose=True)

def proccess_questions(query_engine, questions: list) -> list:
    responses = []
    for question in questions:
        try:
            response = query_engine.query(question)
            retrieved_chunks = [node.node.text for node in response.source_nodes]

            if not retrieved_chunks:
                answer = "Не могу найти информацию по этому вопросу."
            else:
                answer = response.response

            responses.append({
                "question": question,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks
            }) 
        except:
            responses.append({
                "question": question,
                "answer": "Вопрос выходит за пределы базы знаний.",
                "retrieved_chunks": []
            }) 
    return responses


# def proccess_questions(query_engine, questions: list) -> list:
    # responses = []
    # for question in questions:
    #     try:
    #         response = query_engine.query(question)

    #         retrieved_data = []
    #         for node in response.source_nodes:
    #             file_path = node.node.metadata.get("source", "unknown")
    #             chunk_content = node.node.text
    #             images = node.node.metadata.get("images", [])

    #             retrieved_data.append({
    #                 "file_path": file_path,
    #                 "chunks": [chunk_content],
    #                 "images": images
    #             })

    #         # retrieved_chunks = [node.node.text for node in response.source_nodes]

    #         if not retrieved_data:
    #             answer = "Не могу найти информацию по этому вопросу."
    #             retrieved_data = {
    #                 "project": "assistant_rag",
    #                 "chunks": [],
    #                 "images": []
    #             }
    #         else:
    #             answer = response.response
    #             retrieved_chunks = {
    #                 "project": "assistant_rag",
    #                 "chunks": [entry["chunks"][0] for entry in retrieved_data],
    #                 "images": [img for entry in retrieved_data for img in entry["images"]]
    #             }

    #         responses.append({
    #             "question": question,
    #             "answer": answer,
    #             "retrieved_chunks": retrieved_chunks
    #         }) 

    #     except:
    #         responses.append({
    #             "question": question,
    #             "answer": "Вопрос выходит за пределы базы знаний.",
    #             "retrieved_chunks": {
    #                 "project": "assistant_rag",
    #                 "chunks": [],
    #                 "images": []
    #             }
    #         }) 
    # return responses