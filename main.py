import argparse
import os
from train import train
from evaluate import evaluate
import config
from preprocessing import ImagePreprocessor
from config import PreprocessingConfig

def main():
    # --- 1. Run Preprocessing ---
    print("--- Starting Preprocessing ---")
    # Initialize preprocessor with the config class
    preprocessor = ImagePreprocessor(config=PreprocessingConfig)
    preprocessor.run_pipeline()
    print("--- Preprocessing Finished ---\n")

    # --- 2. Run Training & Evaluation ---
    parser = argparse.ArgumentParser(description='Training and evaluation pipeline.')
    # The data_dir is now determined by the preprocessing output
    parser.add_argument('--model_name', type=str, default=config.TrainingConfig.MODEL_NAME, choices=config.TrainingConfig.AVAILABLE_MODELS, help='Model to train and evaluate')
    parser.add_argument('--epochs', type=int, default=config.TrainingConfig.EPOCHS, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=config.TrainingConfig.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=config.TrainingConfig.BATCH_SIZE, help='Batch size')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save the trained model')
    args = parser.parse_args()

    # Create model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model_path = os.path.join(args.model_dir, f'{args.model_name}.pth')
    
    # Use the output directory from preprocessing as the data source for training
    training_data_dir = PreprocessingConfig.OUTPUT_DIR

    # Train the model
    print("--- Starting Training ---")
    train(
        data_dir=training_data_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        model_path=model_path,
        model_dir=args.model_dir
    )
    print("--- Training Finished ---")

    # Evaluate the model on the validation set
    print("\n--- Starting Validation Evaluation ---")
    evaluate(
        data_dir=os.path.join(training_data_dir, 'val'), # Explicitly use val subdirectory
        model_name=args.model_name,
        model_path=model_path
    )
    print("--- Validation Evaluation Finished ---")

    # Evaluate the model on the test set
    print("\n--- Starting Test Evaluation ---")
    evaluate(
        data_dir=os.path.join(training_data_dir, 'test'), # Explicitly use test subdirectory
        model_name=args.model_name,
        model_path=model_path
    )
    print("--- Test Evaluation Finished ---")

if __name__ == '__main__':
    main()