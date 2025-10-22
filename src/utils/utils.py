import random
from src.data.base_qa_dataset import BaseQADataset
from typing import List, Dict

def get_n_random_samples(dataset: BaseQADataset, n: int, seed: int) -> List[Dict]:
    random.seed(seed)
    dataset_len = len(dataset)
    qa = []
    indices = random.sample(range(dataset_len), n)
    for i in indices:
        qa.append(dataset[i])
    return qa

def get_success_rate(samples: List[Dict], llm_results: List[Dict]) -> float:
    correct = sum(1 for sample, pred in zip(samples, llm_results) if sample["label"] == pred["predicted_label"])
    return correct / len(samples)
