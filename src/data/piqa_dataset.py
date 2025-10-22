import urllib.request
import json
from src.config.settings import DATASET_URL, LABELS_URL
from src.data.base_qa_dataset import BaseQADataset

class PIQADataset(BaseQADataset):

    def __init__(self, data=None):
        self.dataset_url = DATASET_URL
        self.labels_url = LABELS_URL
        if data is None:
            self.data = self.load_data()
        else:
            self.data = data


    def load_data(self) -> list:
        with urllib.request.urlopen(self.dataset_url) as f:
            data = [json.loads(line.decode("utf-8")) for line in f]

        with urllib.request.urlopen(self.labels_url) as f:
            labels = [line.decode("utf-8").strip() for line in f]
        
        for d, l in zip(data, labels):
            d['label'] = int(l)
        
        return data
