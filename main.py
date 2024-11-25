import argparse
from models.setfit import TextClassifier
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
import sys

def main():
    available_models = TextClassifier.get_available_models()

    parser = argparse.ArgumentParser(
        description="SetFit Text Classifier CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"Local Models:\n - " + "\n - ".join(available_models)
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Predict Command
    predict_parser = subparsers.add_parser('predict', help='Predict labels for input texts')
    predict_parser.add_argument('texts', nargs='+', help='Input text(s) to classify.')
    predict_parser.add_argument('--model_path', type=str, default='setfit_models/', help='Path to the saved model.')

    # Evaluate Command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the model on a test CSV dataset')
    evaluate_parser.add_argument('csv_file', type=str, help='Path to the test dataset CSV file.')
    evaluate_parser.add_argument('--text_column', type=str, default='text', help='Name of the text column in the CSV.')
    evaluate_parser.add_argument('--label_column', type=str, default='label', help='Name of the label column in the CSV.')
    evaluate_parser.add_argument('--model_path', type=str, default='setfit_models/', help='Path to the saved model.')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    classifier = TextClassifier(model_path=args.model_path)

    if args.command == 'predict':
        predictions = classifier.predict(args.texts)
        for text, pred in zip(args.texts, predictions):
            print(f"Input Text: {text}")
            print(f"Predicted Label: {pred}")

    elif args.command == 'evaluate':
        dataset = classifier.load_dataset(args.csv_file, args.text_column, args.label_column)
        metrics = classifier.evaluate(dataset['texts'], dataset['labels'])
        print("Evaluation Metrics:")
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall: {metrics.recall:.4f}")
        print(f"F1 Score: {metrics.f1_score:.4f}")
        print("\nClassification Report:")
        print(metrics.report)

if __name__ == '__main__':
    main()