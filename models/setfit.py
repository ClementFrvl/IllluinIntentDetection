import pandas as pd
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel
from utils.constants import DATA_PATH, SETFIT_MODELS, MODELS_PATH, UTILS_PATH
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from setfit import SetFitModel
import joblib
import os

@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    report: str

class TextClassifier:
    def __init__(self, model_path: str):
        self.model_path: str = model_path
        self.model: SetFitModel = self.load_model()
        self.label_encoder = self.load_label_encoder()

    @staticmethod
    def get_available_models() -> list:
        return SETFIT_MODELS

    def load_model(self) -> SetFitModel:
        try:
            # Check if the folder exists
            if not os.path.exists(MODELS_PATH + self.model_path):
                print(f"Model not saved locally, checking Hugging Face Hub for model: {self.model_path}")
                model = SetFitModel.from_pretrained(self.model_path)
            else:
                print(f"Loading local model from: {MODELS_PATH + self.model_path}")
                model = SetFitModel.from_pretrained(MODELS_PATH + self.model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def load_label_encoder(self) -> LabelEncoder:
        try:
            label_encoder = joblib.load(UTILS_PATH + 'label_encoder.joblib')
            return label_encoder
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            sys.exit(1)

    def predict(self, texts: List[str]) -> List[int]:
        start_time = time.time()
        predictions = self.model.predict(texts)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Prediction time: {elapsed_time:.4f} seconds")

        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        return decoded_predictions.tolist()
    
    def evaluate(self, texts: List[str], true_labels: List[int]) -> EvaluationMetrics:
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
