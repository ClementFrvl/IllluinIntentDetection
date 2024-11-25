from models.setfit import SetfitClassifier
from models.tfidf import TfidfClassifier
from models.base import TextClassifier
import sys
from typing import List

class ClassifierFactory:
    @staticmethod
    def create_classifier(classifier_type: str, model_path: str) -> TextClassifier:
        if classifier_type.lower() == 'setfit':
            return SetfitClassifier(model_path)
        elif classifier_type.lower() == 'tfidf':
            return TfidfClassifier(model_path)
        else:
            print(f"Unsupported classifier type: {classifier_type}")
            sys.exit(1)

    @staticmethod
    def get_available_models(type: str) -> List[str]:
        if type.lower() == 'setfit':
            return SetfitClassifier.get_available_models()
        elif type.lower() == 'tfidf':
            return TfidfClassifier.get_available_models()
        else:
            print(f"Unsupported classifier type: {type}")
            return []