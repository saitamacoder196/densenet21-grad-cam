#!/usr/bin/env python3
"""
Simplified Grad-CAM Visualization for DenseNet121 Cell Classification
===================================================================

Visualizes how the models "see" cell images without OpenCV dependency
using Gradient-weighted Class Activation Mapping (Grad-CAM)

Author: Cell Image-Based Disease Detection Team
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
    SAMPLES_PER_CLASS = 2  # Number of samples to visualize per class
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"  # Best performing model
    
    # Visualization settings
    HEATMAP_ALPHA = 0.4
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class SimpleGradCAM:
    """Simplified Grad-CAM implementation for DenseNet121"""
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        self.grad_model = self._build_grad_model()
    
    def _find_target_layer(self):
        """Find the best layer for Grad-CAM in DenseNet121"""
        # Strategy: find the last layer before global pooling
        layer_names = [layer.name for layer in self.model.layers]
        
        # Look for batch normalization layer before global pooling (typical in DenseNet)
        target_candidates = [
            'batch_normalization_2',  # From our model architecture
            'batch_normalization_1',
            'batch_normalization',
            'bn',
            'densenet121'
        ]
        
        for candidate in target_candidates:
            if candidate in layer_names:
                print(f"🎯 Using target layer: {candidate}")
                return candidate
        
        # Fallback: find layer before global pooling
        for i, layer in enumerate(self.model.layers):
            if 'global' in layer.name.lower() and 'pool' in layer.name.lower() and i > 0:
                target = self.model.layers[i-1].name
                print(f"🎯 Using pre-pooling layer: {target}")
                return target
        
        # Final fallback: use middle layer
        mid_layer = self.model.layers[len(self.model.layers)//2].name
        print(f"🎯 Using middle layer as fallback: {mid_layer}")
        return mid_layer
    
    def _build_grad_model(self):
        """Build gradient model for Grad-CAM computation"""
        try:
            target_layer = self.model.get_layer(self.layer_name)
            
            # Check if layer has output_shape attribute
            if hasattr(target_layer, 'output_shape'):
                print(f"   Target layer output shape: {target_layer.output_shape}")
            else:
                print(f"   Target layer: {target_layer.name} (shape not available)")
            
            grad_model = keras.Model(
                inputs=self.model.input,
                outputs=[target_layer.output, self.model.output]
            )
            return grad_model
        except Exception as e:
            print(f"❌ Error building gradient model: {e}")
            print("🔧 Trying alternative layer selection...")
            return self._try_alternative_layers()
    
    def _try_alternative_layers(self):
        """Try alternative layers for Grad-CAM"""
        # Print all layer names for debugging
        print("   Available layers:")
        for i, layer in enumerate(self.model.layers):
            print(f"     {i}: {layer.name} - {type(layer).__name__}")
        
        # Try each layer individually, starting from the most suitable ones
        target_layers_to_try = [
            'densenet121',  # The backbone CNN
            'global_average_pooling2d_1',  # After DenseNet features
            'dense_3',  # First dense layer
            'dense_4',  # Second dense layer
        ]
        
        for layer_name in target_layers_to_try:
            try:
                target_layer = self.model.get_layer(layer_name)
                grad_model = keras.Model(
                    inputs=self.model.input,
                    outputs=[target_layer.output, self.model.output]
                )
                self.layer_name = target_layer.name
                print(f"   ✅ Successfully using layer: {target_layer.name}")
                return grad_model
            except Exception as e:
                print(f"   ❌ Failed to use layer {layer_name}: {e}")
                continue
        
        return None
    
    def generate_heatmap(self, image, class_index, eps=1e-8):
        """Generate Grad-CAM heatmap for given image and class"""
        if self.grad_model is None:
            return None
        
        try:
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(image)
                loss = predictions[:, class_index]
            
            # Compute gradients of loss w.r.t. feature maps
            grads = tape.gradient(loss, conv_outputs)
            
            # Check if we got valid gradients
            if grads is None:
                print(f"⚠️ No gradients computed for class {class_index}")
                return None
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by importance
            conv_outputs = conv_outputs[0]
            
            # Handle different output dimensions
            if len(conv_outputs.shape) == 3:  # Spatial feature maps
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
            elif len(conv_outputs.shape) == 1:  # Global features
                # Create a uniform heatmap for global features
                heatmap = tf.ones((7, 7)) * tf.reduce_mean(pooled_grads)
            else:
                print(f"⚠️ Unexpected feature map shape: {conv_outputs.shape}")
                return None
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val > eps:
                heatmap = heatmap / max_val
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"❌ Error generating heatmap: {e}")
            return None

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

def create_simple_overlay(image, heatmap, alpha=0.4):
    """Create simplified Grad-CAM overlay without OpenCV"""
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Resize heatmap to match image size using numpy
    from skimage.transform import resize
    try:
        # Try using skimage if available
        heatmap_resized = resize(heatmap, (image_array.shape[0], image_array.shape[1]), 
                               preserve_range=True, anti_aliasing=True)
    except ImportError:
        # Fallback to simple nearest neighbor interpolation
        h_ratio = image_array.shape[0] / heatmap.shape[0]
        w_ratio = image_array.shape[1] / heatmap.shape[1]
        
        heatmap_resized = np.zeros((image_array.shape[0], image_array.shape[1]))
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                hi = int(i / h_ratio)
                wi = int(j / w_ratio)
                hi = min(hi, heatmap.shape[0] - 1)
                wi = min(wi, heatmap.shape[1] - 1)
                heatmap_resized[i, j] = heatmap[hi, wi]
    
    # Create colored heatmap manually (red colormap)
    heatmap_colored = np.zeros((heatmap_resized.shape[0], heatmap_resized.shape[1], 3))
    heatmap_colored[:, :, 0] = heatmap_resized  # Red channel
    heatmap_colored[:, :, 1] = 0  # Green channel
    heatmap_colored[:, :, 2] = 1 - heatmap_resized  # Blue channel (inverse for better contrast)
    
    # Normalize to [0, 255]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Create overlay
    overlay = heatmap_colored * alpha + image_array * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
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
        if num_samples == 1:
            selected = [image_files[0]]
        else:
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
            
            if heatmap_true is not None:
                # Create overlay
                overlay_true = create_simple_overlay(original_image, heatmap_true, Config.HEATMAP_ALPHA)
                
                result = {
                    'image_path': image_path,
                    'true_class': class_name,
                    'true_class_index': class_index,
                    'predicted_class': Config.CLASS_NAMES[predicted_class],
                    'predicted_class_index': predicted_class,
                    'confidence': confidence,
                    'original_image': original_image,
                    'overlay_true': overlay_true,
                    'heatmap_true': heatmap_true
                }
                
                results.append(result)
                print(f"     ✅ Generated visualization (Pred: {Config.CLASS_NAMES[predicted_class]}, Conf: {confidence:.3f})")
            else:
                print(f"     ❌ Failed to generate heatmap")
    
    return results

def plot_gradcam_results(results, save_dir):
    """Create Grad-CAM visualization plots"""
    save_dir.mkdir(exist_ok=True)
    
    # Create individual visualizations
    for i, result in enumerate(results):
        plot_single_gradcam(result, save_dir, i)
    
    # Create summary grid
    plot_summary_grid(results, save_dir)

def plot_single_gradcam(result, save_dir, index):
    """Plot single Grad-CAM result"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(result['heatmap_true'], cmap='jet')
    axes[1].set_title(f'Grad-CAM Heatmap\n({result["true_class"]})', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(result['overlay_true'])
    axes[2].set_title(f'Overlay\n({result["true_class"]})', fontweight='bold')
    axes[2].axis('off')
    
    # Prediction info
    pred_text = f'Predicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}\nTrue Class: {result["true_class"]}'
    
    if result["predicted_class"] != result["true_class"]:
        pred_text += f'\n\n❌ Incorrect Prediction'
        color = 'lightcoral'
    else:
        pred_text += f'\n\n✅ Correct Prediction'
        color = 'lightgreen'
    
    axes[3].text(0.5, 0.5, pred_text, ha='center', va='center', 
                 transform=axes[3].transAxes, fontsize=12,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[3].axis('off')
    
    # Main title
    image_name = result['image_path'].name
    plt.suptitle(f'Grad-CAM Analysis: {image_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f"gradcam_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path.name}")

def plot_summary_grid(results, save_dir):
    """Create summary grid of all results"""
    n_results = len(results)
    cols = min(4, n_results)
    rows = (n_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Handle single row/column cases
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, result in enumerate(results):
        row = i // cols
        col = i % cols
        
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row][col]
        
        ax.imshow(result['overlay_true'])
        
        # Title with prediction info
        title = f"{result['true_class']}\n{result['image_path'].name}\n"
        if result['predicted_class'] == result['true_class']:
            title += f"✅ Confidence: {result['confidence']:.3f}"
        else:
            title += f"❌ Predicted: {result['predicted_class']}\nConfidence: {result['confidence']:.3f}"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_results, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            ax = axes[col] if cols > 1 else axes[0]
        else:
            ax = axes[row][col] if cols > 1 else axes[row]
        ax.axis('off')
    
    plt.suptitle(f'Grad-CAM Summary - {Config.TARGET_MODEL}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"gradcam_summary_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Summary grid saved: {save_path.name}")

def main():
    """Main Grad-CAM visualization function"""
    print("🔍 Simple DenseNet121 Grad-CAM Visualization")
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
        print(f"   Total layers: {len(model.layers)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize Grad-CAM
    print("\n🔧 Initializing Simple Grad-CAM...")
    try:
        gradcam = SimpleGradCAM(model)
        if gradcam.grad_model is not None:
            print("✅ Grad-CAM initialized successfully")
        else:
            print("❌ Failed to initialize Grad-CAM")
            return
    except Exception as e:
        print(f"❌ Error initializing Grad-CAM: {e}")
        return
    
    # Select sample images
    print("\n📸 Selecting sample images...")
    samples = select_sample_images()
    
    if not samples:
        print("❌ No sample images found")
        return
    
    total_samples = sum(len(paths) for paths in samples.values())
    print(f"✅ Total samples selected: {total_samples}")
    
    # Generate Grad-CAM visualizations
    print("\n🎨 Generating Grad-CAM visualizations...")
    results = visualize_gradcam_for_samples(model, gradcam, samples)
    
    if not results:
        print("❌ No visualizations generated")
        return
    
    print(f"\n✅ Generated {len(results)} visualizations")
    
    # Create plots
    print("\n📊 Creating visualization plots...")
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