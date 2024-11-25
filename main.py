import argparse
from models.factory import ClassifierFactory
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
import sys

def main():
    setfit_models = ClassifierFactory.get_available_models('setfit')
    tfidf_models = ClassifierFactory.get_available_models('tfidf')

    parser = argparse.ArgumentParser(
        description="Text Classifier CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"Available SetFit Models: {setfit_models}\nAvailable TF-IDF Models: {tfidf_models}"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--classifier_type', type=str, default='setfit', choices=['setfit', 'tfidf'], help='Type of classifier to use.')
    common_parser.add_argument('--model_path', type=str, default='saved_model/', help='Path or identifier of the model to use.')

    # Predict Command
    predict_parser = subparsers.add_parser('predict', help='Predict labels for input texts', parents=[common_parser])
    predict_parser.add_argument('texts', nargs='+', help='Input text(s) to classify.')

    # Evaluate Command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the model on a test CSV dataset', parents=[common_parser])
    evaluate_parser.add_argument('csv_file', type=str, help='Path to the test dataset CSV file.')
    evaluate_parser.add_argument('--text_column', type=str, default='text', help='Name of the text column in the CSV.')
    evaluate_parser.add_argument('--label_column', type=str, default='label', help='Name of the label column in the CSV.')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    classifier = ClassifierFactory.create_classifier(args.classifier_type, args.model_path)

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