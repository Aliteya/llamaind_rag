from ..core import settings
from ragas import evaluate
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._noise_sensitivity import NoiseSensitivity
from datasets import Dataset

# Используем актуальные импорты из langchain_community
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

def evaluate_responses(responses: list, ground_truth: list) -> list:
    # Создаем датасет в формате, совместимом с ragas
    dataset = [
        {
            "question": responses[i]["question"],
            "ground_truth": ground_truth[i],
            "answer": responses[i]["answer"],
            "contexts": responses[i]["retrieved_chunks"]
        }
        for i in range(len(responses))
    ]
    dataset = Dataset.from_list(dataset)

    # Определяем метрики
    metrics = [FactualCorrectness(), NoiseSensitivity()]

    # Инициализируем LLM и Embeddings
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.get_llm_key())
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.get_llm_key())

    # Оцениваем датасет
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )

    return scores


def prepare_data(responses: list) -> list:
    # Формируем заголовки таблицы
    rows = [["Question", "Answer", "Retrieved Chunks", "Factual Correctness", "Noise Sensitivity"]]

    # Добавляем данные в таблицу
    for response in responses:
        chunks_str = "\n".join(response["retrieved_chunks"]) # Ограничиваем длину вывода
        rows.append([
            response["question"],
            response["answer"],
            chunks_str,
            # score.get("factual_correctness", ""),
            # score.get("noise_sensitivity", "")
        ])
    return rows