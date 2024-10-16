import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, train_path, test_path) -> None:
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
    
    def load_data(self) -> pd.DataFrame:
        train = pd.read_json(self.train_path)
        test = pd.read_json(self.test_path)
        return train, test
