# Text Classifier CLI

This project is a command-line interface (CLI) application for text classification using two different approaches:

- **SetFit Classifier**: Utilizes the SetFit library for efficient few-shot text classification with Sentence Transformers.
- **TF-IDF Classifier**: Employs traditional TF-IDF vectorization combined with machine learning algorithms like Logistic Regression, SVM, etc.

The CLI allows you to train, evaluate, and make predictions using either classifier, providing flexibility and ease of use.

## Table of Contents

- [Objectives](#objectives)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Displaying Help](#displaying-help)
  - [Predicting Text Labels](#predicting-text-labels)
  - [Evaluating Models](#evaluating-models)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Objectives

- Provide a CLI tool for text classification using both modern and traditional methods.
- Enable predictions and evaluations on test datasets in CSV format.
- Ensure efficient prediction times (approximately one second per input).
- Maintain high code quality by adhering to Python conventions (typing, classes).

## Features

- **Multiple Classifiers**: Choose between SetFit and TF-IDF classifiers.
- **Flexible Model Selection**: Use pre-trained models or your own custom-trained models.
- **Evaluation Metrics**: Obtain detailed evaluation metrics, including accuracy, precision, recall, F1 score, and classification reports.
- **Easy Integration**: Modular codebase allows for easy extension and integration with other projects.

## Prerequisites

- Python 3.7 or higher
- `pip` package manager
- High memory usage for setfit models (2GB minimum for camemBERT)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ClementFrvl/IllluinIntentDetection
   cd text-classifier-cli
   ```

2. **Create a Virtual Environment (Optional but Recommended):**:
   ```bash
   python -m venv venv
    source venv/bin/activate
   ```

3. **Install Required Dependencies:**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The CLI provides two main commands:

- predict: Predict labels for input texts.
- evaluate: Evaluate a model on a test CSV dataset.

### Displaying Help

To display the help message and see all available options:
```bash
python main.py -h
```
   
   Sample output
   
```bash
usage: main.py [-h] {predict,evaluate} ...

Text Classifier CLI

positional arguments:
  {predict,evaluate}  Commands
    predict           Predict labels for input texts
    evaluate          Evaluate the model on a test CSV dataset

optional arguments:
  -h, --help          show this help message and exit

Available SetFit Models: ['camembert', 'camembert_preprocessed']
Available TF-IDF Models: ['tfidf_sgd', 'tfidf_logreg_default', 'tfidf_logreg_bigrams', 'tfidf_svm', 'tfidf_sgd_char', 'tfidf_sgd_bigrams']

Use 'python main.py <command> -h' for more information on a command.
```

### Training Models

**SetFit Classifier**:
```bash
python3 main.py train-setfit 
   --dataset_path your_dataset_path
   --save_to your_model_name
   --text_column text_column_name
   --label_column label_column_name
   --model your_model_name
   --setfit_params all_params (dict)
```

**TF-IDF Classifier**:
```bash
python3 main.py train-setfit 
   --dataset_path your_dataset_path
   --save_to your_model_name
   --text_column text_column_name
   --label_column label_column_name
   --classifier your_sklearn_classifier
   --tfidf_params all_params (dict)
   --classifier_params all_classifier_params (dict)
```

### Predicting Text Labels

```bash
python main.py predict "Your text here" \
    --classifier_type {setfit,tfidf} \
    --model_path path_to_model
```

**Parameters**:
- "Your text here": Input text(s) to classify.
- --classifier_type: Type of classifier to use (setfit or tfidf).
- --model_path: Path or identifier of the model to use.

**Example**

```bash
python3 main.py predict "Reserver un vol Londres Paris" 
    --classifier_type setfit 
    --model_path camembert
```

Sample Output:

```bash
Loading local model from: models/setfit_local_models/camembert
Prediction time: 0.2274 seconds
Input Text: Reserver un vol Londres Paris
Predicted Label: book_flight
```

### Evaluating Models

```bash
python main.py evaluate path_to_csv
    --classifier_type {setfit,tfidf}
    --model_path path_to_model
    --text_column text_column_name
    --label_column label_column_name
```

**Parameters:**

- path_to_csv: Path to the CSV file containing test data.
- --text_column: Name of the column containing texts (default: text).
- --label_column: Name of the column containing true labels (default: label).

**Example:**

```bash
python3 main.py evaluate intent-detection-train.csv 
    --classifier_type tfidf 
    --model_path tfidf_svm 
    --text_column text 
    --label_column label
```

Sample Output:

```bash
Evaluation Metrics:
Accuracy: 0.7600
Precision: 0.8539
Recall: 0.7600
F1 Score: 0.7529

Classification Report:
                   precision    recall  f1-score   support

      book_flight       0.83      0.83      0.83         6
       book_hotel       1.00      0.57      0.73         7
         carry_on       1.00      0.38      0.55         8
    flight_status       1.00      0.67      0.80         6
     lost_luggage       0.88      1.00      0.93         7
     out_of_scope       0.57      1.00      0.72        21
        translate       1.00      0.86      0.92         7
     travel_alert       1.00      0.40      0.57         5
travel_suggestion       1.00      0.62      0.77         8

         accuracy                           0.76        75
        macro avg       0.92      0.70      0.76        75
     weighted avg       0.85      0.76      0.75        75
```

## Contributing

TODO