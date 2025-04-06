from ..core import settings
from ..logging import logger

from ragas import evaluate
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._noise_sensitivity import NoiseSensitivity
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI

def evaluate_responses(responses: list, ground_truth: list) -> list:
    dataset = [
        {
            "user_input": responses[i]["question"],
            "response": responses[i]["answer"],
            "reference": ground_truth[i],
            "retrieved_contexts": responses[i]["retrieved_chunks"]
        }
        for i in range(len(responses))
    ]
    dataset = Dataset.from_list(dataset)

    metrics = [FactualCorrectness(), NoiseSensitivity()]

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.get_llm_key())
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.get_llm_key())

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    logger.info(f"scores: {scores}")
    return scores


def prepare_data(scores: list, responses: list) -> list:
    rows = [["Question", "Answer", "Retrieved Chunks", "Factual Correctness", "Noise Sensitivity"]]

    for i in range(len(responses)):
        chunks_str = "\n".join(responses[i]["retrieved_chunks"])
        rows.append([
            responses[i]["question"],
            responses[i]["answer"],
            chunks_str,     
        ])
    rows.append([str(scores['factual_correctness']), 
            str(scores['noise_sensitivity'])])
    return rows