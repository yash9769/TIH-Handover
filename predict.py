import torch
from torchvision import transforms
import cv2
import json
import os
import argparse
from train import get_model  # We reuse the model-building function
import config

def predict(image_path, model_path):
    """
    Makes a prediction on a single image using a trained model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model (.pth) file.
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Class Mappings ---
    # The class mapping is saved alongside the model, e.g., 'resnet50_class_to_idx.json'
    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    if not os.path.exists(class_to_idx_path):
        print(f"Error: Class mapping file not found at {class_to_idx_path}")
        return

    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    
    # Invert the dictionary to map index to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # --- 2. Load the Model ---
    # Infer model name from the filename (e.g., 'models/resnet50.pth' -> 'resnet50')
    model_name = os.path.basename(os.path.splitext(model_path)[0])
    if model_name not in config.TrainingConfig.AVAILABLE_MODELS:
        print(f"Error: Model name '{model_name}' inferred from path is not in AVAILABLE_MODELS.")
        print(f"Please ensure the model file is named after one of: {config.TrainingConfig.AVAILABLE_MODELS}")
        return

    # Initialize the model architecture
    model = get_model(model_name, num_classes)
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set the model to evaluation mode

    # --- 3. Prepare the Image ---
    # Define the same transformations used during validation/testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    # Load the image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    # The preprocessing script already resizes, so we assume the input image for prediction
    # should also be resized to the model's expected input size.
    image = cv2.resize(image, config.IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB

    # Apply transformations
    image_tensor = transform(image)
    
    # Add a batch dimension (models expect a batch of images)
    # The shape changes from [C, H, W] to [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    # Move the tensor to the appropriate device
    image_tensor = image_tensor.to(device)

    # --- 4. Make a Prediction ---
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(image_tensor)
        # Get the index of the highest score
        _, predicted_idx = torch.max(outputs, 1)

    # --- 5. Display the Result ---
    predicted_class = idx_to_class[predicted_idx.item()]
    
    print("\n--- Prediction Result ---")
    print(f"Predicted Class: {predicted_class}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the class of a single image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth model file.')
    
    args = parser.parse_args()
    
    predict(args.image_path, args.model_path)
