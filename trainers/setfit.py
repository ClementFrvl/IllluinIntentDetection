from trainers.base import TextClassifierTrainer
from datasets import Dataset
from typing import List, Dict, Any
import pandas as pd
from utils.constants import DATA_PATH, SETFIT_MODELS_PATH
from setfit import SetFitModel
from setfit import Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import sys
import joblib

class SetfitClassifierTrainer(TextClassifierTrainer):
    def __init__(self) -> None:
        super().__init__()

    def set(self, save_to: str, model: str) -> None:
        self.save_to = save_to
        print("Loading model: ", model)
        self.model = SetFitModel.from_pretrained(model)

    def load_dataset(self, csv_file: str, text_column: str, label_column: str) -> Dict[str, List[Any]]:
        texts: List[str] = []
        labels: List[int] = []

        try:
            df = pd.read_csv(DATA_PATH + csv_file)
            for _, row in df.iterrows():
                texts.append(row[text_column])
                labels.append(row[label_column])
            
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.dataset = Dataset.from_pandas(pd.DataFrame({ 'text': texts, 'label': encoded_labels }))

        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    def train(self, **train_params) -> None:
        args = TrainingArguments(**train_params)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset,
        )

        trainer.train()

        print("Training completed. Saving model...")
        self.model.save_pretrained(f"{SETFIT_MODELS_PATH}/{self.save_to}")
        joblib.dump(self.label_encoder, f"{SETFIT_MODELS_PATH}/{self.save_to}/label_encoder.joblib")

        print("Saved model to: ", f"{SETFIT_MODELS_PATH}/{self.save_to}")