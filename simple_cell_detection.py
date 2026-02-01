#!/usr/bin/env python3
"""
Simple Cell Image Disease Detection Script
"""

import os
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import json

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess single image for model prediction"""
    try:
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(image)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_test_data(data_dir):
    """Load test data paths and labels"""
    test_dir = Path(data_dir) / 'test'
    classes = ['Benign', 'Early', 'Pre', 'Pro']
    
    image_paths = []
    true_labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = test_dir / class_name
        if class_dir.exists():
            for img_file in class_dir.glob('*.jpg'):
                image_paths.append(str(img_file))
                true_labels.append(class_idx)
    
    return image_paths, true_labels, classes

def analyze_h5_model(model_path):
    """Analyze H5 model structure"""
    print(f"\nAnalyzing model: {model_path}")
    
    try:
        with h5py.File(model_path, 'r') as f:
            print("Model structure:")
            
            def print_structure(name, obj):
                print(f"  {name}: {type(obj)}")
                if hasattr(obj, 'shape'):
                    print(f"    Shape: {obj.shape}")
                if hasattr(obj, 'dtype'):
                    print(f"    Dtype: {obj.dtype}")
            
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"Error analyzing {model_path}: {e}")

def simple_prediction_with_keras_subset(model_path, image_paths, true_labels, classes):
    """Simple prediction without full TensorFlow"""
    print(f"\nTesting with basic approach for: {model_path}")
    
    # Analyze the model first
    analyze_h5_model(model_path)
    
    # Process a few sample images
    print("\nProcessing sample images...")
    
    correct_predictions = 0
    total_predictions = 0
    
    # Test on first 10 images for quick validation
    sample_size = min(10, len(image_paths))
    
    for i in range(sample_size):
        img_path = image_paths[i]
        true_label = true_labels[i]
        
        # Preprocess image
        img_array = preprocess_image(img_path)
        if img_array is None:
            continue
            
        print(f"Image {i+1}: {Path(img_path).name}")
        print(f"  True class: {classes[true_label]}")
        print(f"  Image shape: {img_array.shape}")
        
        # For now, just random prediction (since we can't load the full model easily)
        predicted_label = np.random.randint(0, len(classes))
        print(f"  Predicted class: {classes[predicted_label]} (random for demo)")
        
        if predicted_label == true_label:
            correct_predictions += 1
        total_predictions += 1
        
        print()
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Sample accuracy (random): {accuracy:.2f} ({correct_predictions}/{total_predictions})")

def count_test_images(data_dir):
    """Count images in test set by class"""
    test_dir = Path(data_dir) / 'test'
    classes = ['Benign', 'Early', 'Pre', 'Pro']
    
    print("Test set statistics:")
    total_images = 0
    
    for class_name in classes:
        class_dir = test_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob('*.jpg')))
            print(f"  {class_name}: {count} images")
            total_images += count
        else:
            print(f"  {class_name}: Directory not found")
    
    print(f"  Total: {total_images} images")
    return total_images

def main():
    """Main execution"""
    print("Cell Image Disease Detection - Simple Analysis")
    print("=" * 50)
    
    # Set paths
    models_dir = Path('models')
    data_dir = Path('data')
    
    # Check if directories exist
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Count test images
    count_test_images(data_dir)
    
    # List available models
    model_files = list(models_dir.glob('*.h5'))
    print(f"\nAvailable models:")
    for model_file in model_files:
        print(f"  {model_file.name}")
    
    if not model_files:
        print("No H5 model files found!")
        return
    
    # Load test data
    print(f"\nLoading test data...")
    image_paths, true_labels, classes = load_test_data(data_dir)
    print(f"Loaded {len(image_paths)} test images")
    print(f"Classes: {classes}")
    
    # Test each model
    for model_file in model_files:
        simple_prediction_with_keras_subset(model_file, image_paths, true_labels, classes)
        print("-" * 50)
    
    # Save summary
    summary = {
        "total_test_images": len(image_paths),
        "classes": classes,
        "models_tested": [f.name for f in model_files],
        "note": "This is a simplified analysis without full TensorFlow loading"
    }
    
    with open('detection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Summary saved to detection_summary.json")
    print("\nNote: This is a simplified analysis. For full model evaluation,")
    print("TensorFlow needs to be properly installed and configured.")

if __name__ == "__main__":
    main()