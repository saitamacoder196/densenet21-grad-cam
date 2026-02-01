#!/usr/bin/env python3
"""
Simple Activation Map Visualization for DenseNet121 Models
========================================================

Creates activation maps to show where the model focuses when making predictions
Simple alternative to Grad-CAM for understanding model behavior

Author: Cell Image-Based Disease Detection Team
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available. Please install TensorFlow.")
    TENSORFLOW_AVAILABLE = False

# Configuration
class Config:
    # Paths
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/test")
    ACTIVATION_DIR = Path("activation_maps")
    
    # Model settings
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    
    # Visualization settings
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_intermediate_model(model):
    """Create model that outputs intermediate activations"""
    try:
        # Try to get the DenseNet121 backbone output
        densenet_layer = model.get_layer('densenet121')
        
        # Create a model that outputs the DenseNet features
        intermediate_model = Model(
            inputs=model.input,
            outputs=densenet_layer.output
        )
        
        print(f"✅ Created intermediate model with output shape: {densenet_layer.output_shape}")
        return intermediate_model, densenet_layer.output_shape
        
    except Exception as e:
        print(f"❌ Could not create intermediate model: {e}")
        return None, None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    try:
        # Load and resize image
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        
        # Normalize to [0,1] (same as training)
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array, image
    except Exception as e:
        print(f"❌ Error preprocessing {image_path}: {e}")
        return None, None

def create_activation_map(activations):
    """Create activation map from feature maps"""
    # Average across all feature maps to create a single activation map
    activation_map = np.mean(activations[0], axis=-1)
    
    # Normalize to [0, 1]
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    
    return activation_map

def resize_activation_map(activation_map, target_size):
    """Resize activation map to target size using simple interpolation"""
    h_ratio = target_size[0] / activation_map.shape[0]
    w_ratio = target_size[1] / activation_map.shape[1]
    
    resized_map = np.zeros(target_size)
    
    for i in range(target_size[0]):
        for j in range(target_size[1]):
            # Find corresponding position in original map
            orig_i = int(i / h_ratio)
            orig_j = int(j / w_ratio)
            
            # Ensure indices are within bounds
            orig_i = min(orig_i, activation_map.shape[0] - 1)
            orig_j = min(orig_j, activation_map.shape[1] - 1)
            
            resized_map[i, j] = activation_map[orig_i, orig_j]
    
    return resized_map

def select_sample_images():
    """Select sample images from each class"""
    samples = {}
    
    for class_name in Config.CLASS_NAMES:
        class_dir = Config.DATA_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️ Class directory not found: {class_dir}")
            continue
        
        # Get all images in class
        image_files = list(class_dir.glob("*.jpg"))
        
        if len(image_files) == 0:
            print(f"⚠️ No images found in {class_dir}")
            continue
        
        # Select samples
        num_samples = min(Config.SAMPLES_PER_CLASS, len(image_files))
        if num_samples == 1:
            selected = [image_files[0]]
        else:
            step = max(1, len(image_files) // num_samples)
            selected = image_files[::step][:num_samples]
        
        samples[class_name] = selected
        print(f"📸 Selected {len(selected)} samples from {class_name}")
    
    return samples

def analyze_predictions(model, samples):
    """Analyze model predictions and create visualizations"""
    results = []
    
    for class_name, image_paths in samples.items():
        class_index = Config.CLASS_NAMES.index(class_name)
        
        print(f"\n🔍 Processing {class_name} samples...")
        
        for image_path in image_paths:
            print(f"   Processing {image_path.name}...")
            
            # Preprocess image
            image_array, original_image = preprocess_image(image_path)
            if image_array is None:
                continue
            
            # Get model prediction
            predictions = model.predict(image_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Store results
            result = {
                'image_path': image_path,
                'true_class': class_name,
                'true_class_index': class_index,
                'predicted_class': Config.CLASS_NAMES[predicted_class],
                'predicted_class_index': predicted_class,
                'confidence': confidence,
                'original_image': original_image,
                'image_array': image_array,
                'predictions': predictions[0]
            }
            
            results.append(result)
            print(f"     ✅ Prediction: {Config.CLASS_NAMES[predicted_class]} (Conf: {confidence:.3f})")
    
    return results

def create_prediction_visualization(result, save_dir, index):
    """Create visualization showing prediction results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    # Prediction probabilities
    class_probs = result['predictions']
    colors = ['lightcoral' if i == result['predicted_class_index'] else 'lightblue' for i in range(len(Config.CLASS_NAMES))]
    bars = axes[1].bar(Config.CLASS_NAMES, class_probs, color=colors)
    axes[1].set_title('Class Probabilities', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Probability')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add probability values on bars
    for bar, prob in zip(bars, class_probs):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Prediction info
    pred_text = f'True Class: {result["true_class"]}\nPredicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}'
    
    if result["predicted_class"] != result["true_class"]:
        pred_text += f'\n\n❌ Incorrect Prediction'
        color = 'lightcoral'
    else:
        pred_text += f'\n\n✅ Correct Prediction'
        color = 'lightgreen'
    
    axes[2].text(0.5, 0.5, pred_text, ha='center', va='center', 
                 transform=axes[2].transAxes, fontsize=12,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[2].axis('off')
    
    # Main title
    image_name = result['image_path'].name
    plt.suptitle(f'Prediction Analysis: {image_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f"prediction_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path.name}")

def create_summary_analysis(results, save_dir):
    """Create summary analysis of all predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy by class
    class_accuracy = {}
    for class_name in Config.CLASS_NAMES:
        class_results = [r for r in results if r['true_class'] == class_name]
        if class_results:
            correct = sum(1 for r in class_results if r['predicted_class'] == r['true_class'])
            accuracy = correct / len(class_results)
            class_accuracy[class_name] = accuracy
    
    if class_accuracy:
        axes[0, 0].bar(class_accuracy.keys(), class_accuracy.values(), color='skyblue')
        axes[0, 0].set_title('Accuracy by Class', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1.1)
        
        # Add value labels
        for i, (class_name, acc) in enumerate(class_accuracy.items()):
            axes[0, 0].text(i, acc + 0.05, f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Confidence distribution
    all_confidences = [r['confidence'] for r in results]
    axes[0, 1].hist(all_confidences, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Confidence Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(np.mean(all_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_confidences):.3f}')
    axes[0, 1].legend()
    
    # Confusion matrix (simplified)
    confusion_data = np.zeros((len(Config.CLASS_NAMES), len(Config.CLASS_NAMES)))
    for result in results:
        true_idx = result['true_class_index']
        pred_idx = result['predicted_class_index']
        confusion_data[true_idx, pred_idx] += 1
    
    im = axes[1, 0].imshow(confusion_data, cmap='Blues')
    axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[1, 0].set_xticks(range(len(Config.CLASS_NAMES)))
    axes[1, 0].set_yticks(range(len(Config.CLASS_NAMES)))
    axes[1, 0].set_xticklabels(Config.CLASS_NAMES, rotation=45)
    axes[1, 0].set_yticklabels(Config.CLASS_NAMES)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
    # Add text annotations
    for i in range(len(Config.CLASS_NAMES)):
        for j in range(len(Config.CLASS_NAMES)):
            axes[1, 0].text(j, i, int(confusion_data[i, j]), ha='center', va='center', 
                           color='white' if confusion_data[i, j] > confusion_data.max()/2 else 'black')
    
    plt.colorbar(im, ax=axes[1, 0])
    
    # Summary statistics
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    stats_text = f'Total Samples: {total_samples}\nCorrect Predictions: {correct_predictions}\nOverall Accuracy: {overall_accuracy:.3f}\nMean Confidence: {np.mean(all_confidences):.3f}\nStd Confidence: {np.std(all_confidences):.3f}'
    
    axes[1, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                    transform=axes[1, 1].transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics', fontweight='bold')
    
    plt.suptitle(f'Model Analysis Summary - {Config.TARGET_MODEL}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"summary_analysis_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary analysis saved: {save_path.name}")

def main():
    """Main function"""
    print("🔍 DenseNet121 Activation Map Analysis")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available.")
        return
    
    # Setup
    Config.ACTIVATION_DIR.mkdir(exist_ok=True)
    
    # Load model
    model_path = Config.MODELS_DIR / Config.TARGET_MODEL
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📂 Loading model: {Config.TARGET_MODEL}")
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Select samples
    print("\n📸 Selecting sample images...")
    samples = select_sample_images()
    
    if not samples:
        print("❌ No sample images found")
        return
    
    # Analyze predictions
    print("\n🎯 Analyzing model predictions...")
    results = analyze_predictions(model, samples)
    
    if not results:
        print("❌ No analysis results generated")
        return
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    for i, result in enumerate(results):
        create_prediction_visualization(result, Config.ACTIVATION_DIR, i)
    
    # Create summary analysis
    create_summary_analysis(results, Config.ACTIVATION_DIR)
    
    # Final summary
    correct_predictions = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    total_results = len(results)
    accuracy = correct_predictions / total_results if total_results > 0 else 0
    
    print(f"\n🎉 Analysis completed!")
    print(f"📁 Results saved in: {Config.ACTIVATION_DIR}")
    print(f"📊 Total samples analyzed: {total_results}")
    print(f"🎯 Correct predictions: {correct_predictions}/{total_results} ({accuracy:.1%})")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()