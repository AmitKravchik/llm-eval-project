from abc import ABC, abstractmethod

class BaseQADataset(ABC):
    def __init__(self, data):
        self.data = data
 
    def __getitem__(self, index):
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    @abstractmethod
    def load_data(self):
        pass


    