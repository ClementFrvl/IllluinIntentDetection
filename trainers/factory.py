from trainers.setfit import SetfitClassifierTrainer
from trainers.tfidf import TfidfClassifierTrainer
from trainers.base import TextClassifierTrainer
import sys
from typing import List, Dict, Any
from trainers.base import EvaluationMetrics

class ClassifierTrainerFactory:
    @staticmethod
    def create_trainer(classifier_type: str) -> TextClassifierTrainer:
        if classifier_type.lower() == 'setfit':
            return SetfitClassifierTrainer()
        elif classifier_type.lower() == 'tfidf':
            return TfidfClassifierTrainer()
        else:
            print(f"Unsupported classifier type: {classifier_type}")
            sys.exit(1)

    @staticmethod
    def load_dataset(csv_file: str, text_column: str, label_column: str) -> Dict[str, List[Any]]:
        pass

    def train(self, texts: List[str]) -> None:
        pass

    def evaluate(self, texts: List[str], true_labels: List[str]) -> EvaluationMetrics:
        pass
