from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    report: str

class TextClassifier(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        pass

    @abstractmethod
    def evaluate(self, texts: List[str], true_labels: List[str]) -> EvaluationMetrics:
        pass

    @staticmethod
    @abstractmethod
    def load_dataset(csv_file: str, text_column: str, label_column: str) -> Dict[str, List[Any]]:
        pass