from trainers.base import TextClassifierTrainer, EvaluationMetrics
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from utils.constants import DATA_PATH, TFIDF_MODELS_PATH
import sys
import joblib
import os

class TfidfClassifierTrainer(TextClassifierTrainer):
    def __init__(self) -> None:
        super().__init__()

    def set(self, save_to: str, classifier: str, **classifier_params) -> None:
        self.save_to = save_to
        self.classifier = self.set_classifier(classifier)

    def set_classifier(self, classifier: str, **classifier_params) -> Any:
        classifiers = {
            'LogisticRegression': LogisticRegression,
            'RidgeClassifier': RidgeClassifier,
            'SGDClassifier': SGDClassifier,
            'LinearSVC': LinearSVC,
            'SVC': SVC,
            'MultinomialNB': MultinomialNB,
            'ComplementNB': ComplementNB,
            'BernoulliNB': BernoulliNB,
            'DecisionTree': DecisionTreeClassifier,
            'RandomForest': RandomForestClassifier,
            'GradientBoosting': GradientBoostingClassifier,
            'AdaBoost': AdaBoostClassifier,
            'KNeighbors': KNeighborsClassifier,
            'MLPClassifier': MLPClassifier,
        }
        
        if classifier in classifiers:
            return classifiers[classifier](**classifier_params)
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")
        
    def load_dataset(self, csv_file: str, text_column: str, label_column: str) -> Dict[str, List[Any]]:
        texts: List[str] = []
        labels: List[int] = []

        try:
            df = pd.read_csv(DATA_PATH + csv_file)
            for _, row in df.iterrows():
                texts.append(row[text_column])
                labels.append(row[label_column])
            
            self.X = texts
            self.y = labels

        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    def train(self, **tfidf_params) -> None:
        
        label_encoder = LabelEncoder()
        train_labels_encoded = label_encoder.fit_transform(self.y)

        # Create a pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('clf', self.classifier),
        ])

        # Train the model
        pipeline.fit(self.X, train_labels_encoded)

        # Create a directory to save the model
        model_dir = f'{TFIDF_MODELS_PATH}{self.save_to}'
        os.makedirs(model_dir, exist_ok=True)

        # Save the model and label encoder
        joblib.dump(pipeline, f'{model_dir}/tfidf_model.joblib')
        joblib.dump(label_encoder, f'{model_dir}/label_encoder.joblib')

        print(f"Model saved to {model_dir}")