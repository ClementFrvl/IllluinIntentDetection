import argparse
from models.factory import ClassifierFactory
from trainers.factory import ClassifierTrainerFactory
import ast
import sys

def parse_dict(arg):
    try:
        return ast.literal_eval(arg)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {arg}. Error: {e}")


def main():
    setfit_models = ClassifierFactory.get_available_models('setfit')
    tfidf_models = ClassifierFactory.get_available_models('tfidf')

    parser = argparse.ArgumentParser(
        description="Text Classifier CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"Available SetFit Models: {setfit_models}\nAvailable TF-IDF Models: {tfidf_models}\n\nUse 'python main.py <command> -h' for more information on a command."
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--classifier_type', type=str, default='setfit', choices=['setfit', 'tfidf'], help='Type of classifier to use.')
    common_parser.add_argument('--model_path', type=str, default='', help='Path or identifier of the model to use. (Leave empty when training a new model)')

    # Predict Command
    predict_parser = subparsers.add_parser('predict', help='Predict labels for input texts', parents=[common_parser])
    predict_parser.add_argument('texts', nargs='+', help='Input text(s) to classify.')

    # Evaluate Command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the model on a test CSV dataset', parents=[common_parser])
    evaluate_parser.add_argument('csv_file', type=str, help='Path to the test dataset CSV file.')
    evaluate_parser.add_argument('--text_column', type=str, default='text', help='Name of the text column in the CSV.')
    evaluate_parser.add_argument('--label_column', type=str, default='label', help='Name of the label column in the CSV.')

    # Train Setfit Command
    train_setfit_parser = subparsers.add_parser('train-setfit', help='Train a new Setfit model')
    train_setfit_parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training dataset CSV file.')
    train_setfit_parser.add_argument('--text_column', type=str, default='text', help='Name of the text column in the CSV.')
    train_setfit_parser.add_argument('--label_column', type=str, default='label', help='Name of the label column in the CSV.')
    train_setfit_parser.add_argument('--save_to', type=str, required=True, help='Directory to save the trained model.')
    train_setfit_parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Sentence similarity model to base itself on for training.')
    train_setfit_parser.add_argument('--setfit_params', type=parse_dict, default={ 'num_epochs': 3, 'batch_size': 16 }, help='SetFit training parameters.')   

    # Train TF-IDF Command
    train_tfidf_parser = subparsers.add_parser('train-tfidf', help='Train a new TF-IDF model')
    train_tfidf_parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training dataset CSV file.')
    train_tfidf_parser.add_argument('--text_column', type=str, default='text', help='Name of the text column in the CSV.')
    train_tfidf_parser.add_argument('--label_column', type=str, default='label', help='Name of the label column in the CSV.')
    train_tfidf_parser.add_argument('--save_to', type=str, required=True, help='Directory to save the trained model.')
    train_tfidf_parser.add_argument('--classifier', type=str, default='LogisticRegression', help='Classifier to use for training. (Must be a valid scikit-learn classifier)')
    train_tfidf_parser.add_argument('--tfidf_params', type=parse_dict, default={ 'max_features': 1000, 'ngram_range': (1, 2) }, help='TF-IDF parameters for the vectorizer.')
    train_tfidf_parser.add_argument('--classifier_params', type=parse_dict, default={}, help='Parameters for the classifier.')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'predict':
        classifier = ClassifierFactory.create_classifier(args.classifier_type, args.model_path)
        predictions = classifier.predict(args.texts)
        for text, pred in zip(args.texts, predictions):
            print(f"Input Text: {text}")
            print(f"Predicted Label: {pred}")

    elif args.command == 'evaluate':
        classifier = ClassifierFactory.create_classifier(args.classifier_type, args.model_path)
        dataset = classifier.load_dataset(args.csv_file, args.text_column, args.label_column)
        metrics = classifier.evaluate(dataset['texts'], dataset['labels'])
        print("Evaluation Metrics:")
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall: {metrics.recall:.4f}")
        print(f"F1 Score: {metrics.f1_score:.4f}")
        print("\nClassification Report:")
        print(metrics.report)

    elif args.command == 'train-setfit':
        trainer = ClassifierTrainerFactory.create_trainer('setfit')
        trainer.set(args.save_to, args.model)
        trainer.load_dataset(args.dataset_path, args.text_column, args.label_column)
        trainer.train(**args.setfit_params)

    elif args.command == 'train-tfidf':
        trainer = ClassifierTrainerFactory.create_trainer('tfidf')
        trainer.set(args.save_to, args.classifier, **args.classifier_params)
        trainer.load_dataset(args.dataset_path, args.text_column, args.label_column)
        trainer.train(**args.tfidf_params)        

if __name__ == '__main__':
    main()