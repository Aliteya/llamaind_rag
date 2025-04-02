from ragas.metrics._factual_correctness import FactualCorrectness 
from ragas.metrics._noise_sensitivity import NoiseSensitivity
from ragas import evaluate

def evaluate_responses(responses: list) -> list:
    dataset = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["retrieved_chunks"]
        }
        for r in responses
    ]
    
    metrics = [FactualCorrectness, NoiseSensitivity]
    scores = evaluate(dataset, metrics=metrics)

    for i, score in enumerate(scores):
        responses[i]["faithfulness"] = score["faithfulness"]
        responses[i]["noise_sensitivity"] = score["noise_sensitivity"]

    return responses

def prepare_data(responses: list) -> list:
    rows = [["Question", "Answer", "Retrieved Chunks", "faithfulness", "noise_sensitivity"]]
    for response in responses:
        chunks_str = "\n".join(response["retrieved_chunks"])
        rows.append([
            response["question"],
            response["answer"],
            chunks_str,
            response.get("faithfulness", ""),
            response.get("noise_sensitivity", "")
        ])
    return rows