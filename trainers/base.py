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

class TextClassifierTrainer(ABC):
    @abstractmethod
    def load_dataset(csv_file: str, text_column: str, label_column: str) -> Dict[str, List[Any]]:
        pass

    @abstractmethod
    def train(self, texts: List[str]) -> None:
        pass
