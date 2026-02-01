#!/usr/bin/env python3
"""
Grad-CAM Visualization for DenseNet121 Cell Classification Models
===============================================================

Visualizes how the models "see" cell images by highlighting important regions
using Gradient-weighted Class Activation Mapping (Grad-CAM)

Author: Cell Image-Based Disease Detection Team
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.densenet import preprocess_input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available. Please install TensorFlow to run Grad-CAM.")
    TENSORFLOW_AVAILABLE = False

# Configuration
class Config:
    # Paths
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/test")
    GRADCAM_DIR = Path("gradcam_results")
    
    # Model settings
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    
    # Grad-CAM settings
    SAMPLES_PER_CLASS = 3  # Number of samples to visualize per class
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"  # Best performing model
    
    # Visualization settings
    HEATMAP_ALPHA = 0.4
    COLORMAP = cv2.COLORMAP_JET
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class GradCAM:
    """Grad-CAM implementation for DenseNet121"""
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        self.grad_model = self._build_grad_model()
    
    def _find_target_layer(self):
        """Find the best convolutional layer for Grad-CAM in DenseNet121"""
        # Look for the last convolutional layer in DenseNet121 backbone
        target_layers = [
            'densenet121',  # DenseNet121 model name
            'bn',           # Final batch norm in DenseNet121
            'conv5_block32_2_conv',  # Last conv block
            'conv5_block16_2_conv',  # Alternative layer
        ]
        
        for layer_name in target_layers:
            try:
                layer = self.model.get_layer(layer_name)
                if hasattr(layer, 'output'):
                    print(f"🎯 Using target layer: {layer_name}")
                    return layer_name
            except:
                continue
        
        # Fallback: find last convolutional layer
        conv_layers = []
        for layer in self.model.layers:
            if 'conv' in layer.name.lower() and hasattr(layer, 'filters'):
                conv_layers.append(layer.name)
        
        if conv_layers:
            target = conv_layers[-1]
            print(f"🎯 Using fallback target layer: {target}")
            return target
        
        # Final fallback: use a layer before global pooling
        for i, layer in enumerate(self.model.layers):
            if 'global' in layer.name.lower() and 'pool' in layer.name.lower():
                if i > 0:
                    target = self.model.layers[i-1].name
                    print(f"🎯 Using pre-pooling layer: {target}")
                    return target
        
        raise ValueError("❌ Could not find suitable target layer for Grad-CAM")
    
    def _build_grad_model(self):
        """Build gradient model for Grad-CAM computation"""
        try:
            target_layer = self.model.get_layer(self.layer_name)
            grad_model = keras.Model(
                inputs=self.model.input,
                outputs=[target_layer.output, self.model.output]
            )
            return grad_model
        except Exception as e:
            print(f"❌ Error building gradient model: {e}")
            return None
    
    def generate_heatmap(self, image, class_index, eps=1e-8):
        """Generate Grad-CAM heatmap for given image and class"""
        if self.grad_model is None:
            return None
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, class_index]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()

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

def create_gradcam_overlay(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Create Grad-CAM overlay on original image"""
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]))
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = heatmap_colored * alpha + image_array * (1 - alpha)
    overlay = np.uint8(overlay)
    
    return overlay

def select_sample_images():
    """Select representative sample images from each class"""
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
        
        # Select samples (evenly spaced)
        num_samples = min(Config.SAMPLES_PER_CLASS, len(image_files))
        step = max(1, len(image_files) // num_samples)
        selected = image_files[::step][:num_samples]
        
        samples[class_name] = selected
        print(f"📸 Selected {len(selected)} samples from {class_name}")
    
    return samples

def visualize_gradcam_for_samples(model, gradcam, samples):
    """Generate Grad-CAM visualizations for sample images"""
    results = []
    
    for class_name, image_paths in samples.items():
        class_index = Config.CLASS_NAMES.index(class_name)
        
        print(f"\n🔍 Processing {class_name} samples...")
        
        for i, image_path in enumerate(image_paths):
            print(f"   Processing {image_path.name}...")
            
            # Preprocess image
            image_array, original_image = preprocess_image(image_path)
            if image_array is None:
                continue
            
            # Get model prediction
            predictions = model.predict(image_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Generate Grad-CAM for true class
            heatmap_true = gradcam.generate_heatmap(image_array, class_index)
            
            # Generate Grad-CAM for predicted class (if different)
            heatmap_pred = None
            if predicted_class != class_index:
                heatmap_pred = gradcam.generate_heatmap(image_array, predicted_class)
            
            # Create overlays
            if heatmap_true is not None:
                overlay_true = create_gradcam_overlay(original_image, heatmap_true, Config.HEATMAP_ALPHA)
                
                overlay_pred = None
                if heatmap_pred is not None:
                    overlay_pred = create_gradcam_overlay(original_image, heatmap_pred, Config.HEATMAP_ALPHA)
                
                result = {
                    'image_path': image_path,
                    'true_class': class_name,
                    'true_class_index': class_index,
                    'predicted_class': Config.CLASS_NAMES[predicted_class],
                    'predicted_class_index': predicted_class,
                    'confidence': confidence,
                    'original_image': original_image,
                    'overlay_true': overlay_true,
                    'overlay_pred': overlay_pred,
                    'heatmap_true': heatmap_true,
                    'heatmap_pred': heatmap_pred
                }
                
                results.append(result)
    
    return results

def plot_gradcam_results(results, save_dir):
    """Create comprehensive Grad-CAM visualization plots"""
    save_dir.mkdir(exist_ok=True)
    
    # Group results by class
    results_by_class = {}
    for result in results:
        class_name = result['true_class']
        if class_name not in results_by_class:
            results_by_class[class_name] = []
        results_by_class[class_name].append(result)
    
    # Create individual visualizations
    individual_count = 0
    for class_name, class_results in results_by_class.items():
        for i, result in enumerate(class_results):
            plot_single_gradcam(result, save_dir, individual_count)
            individual_count += 1
    
    # Create class comparison plot
    plot_class_comparison(results_by_class, save_dir)
    
    # Create summary grid
    plot_summary_grid(results, save_dir)

def plot_single_gradcam(result, save_dir, index):
    """Plot single Grad-CAM result with multiple views"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(result['original_image'])
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # True class heatmap
    if result['heatmap_true'] is not None:
        im1 = axes[0, 1].imshow(result['heatmap_true'], cmap='jet', alpha=0.8)
        axes[0, 1].set_title(f'Heatmap - True Class\n({result["true_class"]})', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # True class overlay
    axes[0, 2].imshow(result['overlay_true'])
    axes[0, 2].set_title(f'Grad-CAM Overlay\n({result["true_class"]})', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Prediction info
    pred_text = f'Predicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}'
    if result["predicted_class"] != result["true_class"]:
        pred_text += f'\n❌ Incorrect Prediction'
        axes[1, 0].text(0.5, 0.5, pred_text, ha='center', va='center', 
                       transform=axes[1, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    else:
        pred_text += f'\n✅ Correct Prediction'
        axes[1, 0].text(0.5, 0.5, pred_text, ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 0].axis('off')
    
    # Predicted class heatmap (if different)
    if result['heatmap_pred'] is not None:
        im2 = axes[1, 1].imshow(result['heatmap_pred'], cmap='jet', alpha=0.8)
        axes[1, 1].set_title(f'Heatmap - Predicted Class\n({result["predicted_class"]})', fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Predicted class overlay
        axes[1, 2].imshow(result['overlay_pred'])
        axes[1, 2].set_title(f'Grad-CAM Overlay\n({result["predicted_class"]})', fontweight='bold')
        axes[1, 2].axis('off')
    else:
        # Same prediction, show difference analysis
        axes[1, 1].text(0.5, 0.5, 'Correct Prediction\nSame as True Class', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].axis('off')
        
        axes[1, 2].text(0.5, 0.5, 'Model Focus Analysis\nConsistent with Ground Truth', 
                       ha='center', va='center', transform=axes[1, 2].transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].axis('off')
    
    # Main title
    image_name = result['image_path'].name
    plt.suptitle(f'Grad-CAM Analysis: {image_name}\nTrue: {result["true_class"]} | Predicted: {result["predicted_class"]} (Conf: {result["confidence"]:.3f})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f"gradcam_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path}")

def plot_class_comparison(results_by_class, save_dir):
    """Create class comparison visualization"""
    fig, axes = plt.subplots(len(Config.CLASS_NAMES), Config.SAMPLES_PER_CLASS, 
                            figsize=(Config.SAMPLES_PER_CLASS * 4, len(Config.CLASS_NAMES) * 3))
    
    if len(Config.CLASS_NAMES) == 1:
        axes = [axes]
    
    for class_idx, (class_name, results) in enumerate(results_by_class.items()):
        for sample_idx, result in enumerate(results[:Config.SAMPLES_PER_CLASS]):
            row = class_idx
            col = sample_idx
            
            if len(Config.CLASS_NAMES) == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            ax.imshow(result['overlay_true'])
            
            # Title with prediction info
            title = f"{result['image_path'].name}\n"
            if result['predicted_class'] == result['true_class']:
                title += f"✅ {result['confidence']:.3f}"
            else:
                title += f"❌ Pred: {result['predicted_class']} ({result['confidence']:.3f})"
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
    
    # Add class labels
    for class_idx, class_name in enumerate(Config.CLASS_NAMES):
        if len(Config.CLASS_NAMES) > 1:
            axes[class_idx, 0].set_ylabel(class_name, fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
    
    plt.suptitle('Grad-CAM Analysis by Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"gradcam_class_comparison_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Class comparison saved: {save_path}")

def plot_summary_grid(results, save_dir):
    """Create summary grid of all results"""
    n_results = len(results)
    cols = min(6, n_results)
    rows = (n_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if rows == 1:
        if cols == 1:
            axes = [axes]
        else:
            axes = [axes]
    
    for i, result in enumerate(results):
        row = i // cols
        col = i % cols
        
        if rows == 1:
            ax = axes[col] if cols > 1 else axes[0]
        else:
            ax = axes[row, col]
        
        ax.imshow(result['overlay_true'])
        
        # Short title
        title = f"{result['true_class']}\n"
        if result['predicted_class'] == result['true_class']:
            title += f"✅ {result['confidence']:.2f}"
        else:
            title += f"❌ {result['predicted_class']} ({result['confidence']:.2f})"
        
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_results, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            ax = axes[col] if cols > 1 else axes[0]
        else:
            ax = axes[row, col]
        ax.axis('off')
    
    plt.suptitle(f'Grad-CAM Summary Grid - {Config.TARGET_MODEL}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"gradcam_summary_grid_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Summary grid saved: {save_path}")

def main():
    """Main Grad-CAM visualization function"""
    print("🔍 DenseNet121 Grad-CAM Visualization")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Please install TensorFlow to run Grad-CAM.")
        return
    
    # Setup
    Config.GRADCAM_DIR.mkdir(exist_ok=True)
    
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
    
    # Initialize Grad-CAM
    print("🔧 Initializing Grad-CAM...")
    try:
        gradcam = GradCAM(model)
        print("✅ Grad-CAM initialized")
    except Exception as e:
        print(f"❌ Error initializing Grad-CAM: {e}")
        return
    
    # Select sample images
    print("📸 Selecting sample images...")
    samples = select_sample_images()
    
    if not samples:
        print("❌ No sample images found")
        return
    
    total_samples = sum(len(paths) for paths in samples.values())
    print(f"✅ Total samples selected: {total_samples}")
    
    # Generate Grad-CAM visualizations
    print("🎨 Generating Grad-CAM visualizations...")
    results = visualize_gradcam_for_samples(model, gradcam, samples)
    
    if not results:
        print("❌ No visualizations generated")
        return
    
    print(f"✅ Generated {len(results)} visualizations")
    
    # Create plots
    print("📊 Creating visualization plots...")
    plot_gradcam_results(results, Config.GRADCAM_DIR)
    
    # Summary
    print(f"\n🎉 Grad-CAM visualization completed!")
    print(f"📁 Results saved in: {Config.GRADCAM_DIR}")
    print(f"📊 Total visualizations: {len(results)}")
    
    # Analysis summary
    correct_predictions = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    print(f"🎯 Correct predictions: {correct_predictions}/{len(results)} ({correct_predictions/len(results)*100:.1f}%)")
    
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()