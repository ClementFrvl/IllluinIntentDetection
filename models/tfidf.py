import os
from utils.constants import TFIDF_MODELS_PATH, DATA_PATH
from models.base import TextClassifier, EvaluationMetrics
import sys
from typing import List, Dict, Any
import time
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import pandas as pd


class TfidfClassifier(TextClassifier):
    def __init__(self, model_path: str):
        self.model_path: str = model_path
        self.model: Pipeline = self.load_model()
        self.label_encoder = self.load_label_encoder()

    def load_model(self) -> Pipeline:
        try:
            model = joblib.load(f"{TFIDF_MODELS_PATH}/{self.model_path}/tfidf_model.joblib")
            return model
        except Exception as e:
            print(f"Error loading TF-IDF model: {e}")
            sys.exit(1)

    def load_label_encoder(self):
        try:
            label_encoder = joblib.load(f"{TFIDF_MODELS_PATH}/{self.model_path}/label_encoder.joblib")
            return label_encoder
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            sys.exit(1)

    def predict(self, texts: List[str]) -> List[str]:
        start_time = time.time()
        X = texts
        predictions = self.model.predict(X)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if len(texts) == 1:
            print(f"Prediction time: {elapsed_time:.4f} seconds")
        # Decode the numerical predictions to original labels
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        return decoded_predictions.tolist()

    def evaluate(self, texts: List[str], true_labels: List[str]) -> EvaluationMetrics:
        predictions = self.predict(texts)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        report = classification_report(true_labels, predictions, zero_division=0)

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            report=report
        )

    @staticmethod
    def load_dataset(csv_file: str, text_column: str, label_column: str) -> Dict[str, List[Any]]:
        texts: List[str] = []
        labels: List[int] = []

        try:
            df = pd.read_csv(DATA_PATH + csv_file)
            for _, row in df.iterrows():
                texts.append(row[text_column])
                labels.append(row[label_column])
            return {'texts': texts, 'labels': labels}
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    @staticmethod
    def get_available_models() -> list:
        return os.listdir(TFIDF_MODELS_PATH)