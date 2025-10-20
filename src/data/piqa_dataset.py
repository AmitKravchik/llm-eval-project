import urllib.request
import json
from config.config import DATASET_URL, LABELS_URL

class PIQADataset:

    def __init__(self):
        self.data = self.load_data()
        self.dataset_url = DATASET_URL
        self.labels_url = LABELS_URL


    def load_data(self):
        with urllib.request.urlopen(self.dataset_url) as f:
            data = [json.loads(line.decode("utf-8")) for line in f]

        with urllib.request.urlopen(self.labels_url) as f:
            labels = [line.decode("utf-8").strip() for line in f]
        
        for d, l in zip(data, labels):
            d['label'] = int(l)
        
        return data


    def get_example(self, index):
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[index]

