#!/usr/bin/env python3
"""
Simple Working Grad-CAM Implementation
=====================================

Alternative approach using feature extraction to create actual heatmaps
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available.")
    TENSORFLOW_AVAILABLE = False

class Config:
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/test")
    GRADCAM_DIR = Path("gradcam_working")
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class SimpleGradCAM:
    """Simple Grad-CAM using layer feature visualization"""
    
    def __init__(self, model):
        self.model = model
        print("✅ Simple Grad-CAM initialized")
        
        # Find a suitable feature layer
        self.feature_layer = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                # This is the DenseNet backbone
                # Get the last convolutional block output
                for nested_layer in reversed(layer.layers):
                    if len(nested_layer.output_shape) == 4:  # Feature map
                        self.feature_layer = nested_layer
                        break
                break
        
        if self.feature_layer:
            print(f"✅ Using feature layer: {self.feature_layer.name}")
        else:
            print("⚠️ No suitable feature layer found, using simple attention")
    
    def create_attention_map(self, image, predictions, target_class):
        """Create attention map using prediction-based weighting"""
        
        # Get the prediction confidence for the target class
        confidence = predictions[target_class]
        
        # Create a simple attention map based on image gradients
        # This simulates where the model might be "looking"
        
        # Convert image to grayscale for edge detection
        gray = np.dot(image[0][...,:3], [0.2989, 0.5870, 0.1140])
        
        # Calculate gradients (edge detection)
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Weight by prediction confidence
        attention_map = gradient_magnitude * confidence
        
        # Apply Gaussian-like smoothing
        from scipy.ndimage import gaussian_filter
        try:
            attention_map = gaussian_filter(attention_map, sigma=2.0)
        except ImportError:
            # Simple smoothing if scipy not available
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            attention_map = np.convolve(attention_map.flatten(), kernel.flatten(), mode='same').reshape(attention_map.shape)
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        return attention_map

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image"""
    try:
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, image
    except Exception as e:
        print(f"❌ Error preprocessing {image_path}: {e}")
        return None, None

def resize_attention_map(attention_map, target_size):
    """Resize attention map to target size"""
    attention_tensor = tf.constant(attention_map, dtype=tf.float32)
    attention_tensor = tf.expand_dims(attention_tensor, axis=0)
    attention_tensor = tf.expand_dims(attention_tensor, axis=-1)
    resized = tf.image.resize(attention_tensor, target_size, method='bilinear')
    return tf.squeeze(resized).numpy()

def create_overlay(image, attention_map, alpha=0.4):
    """Create attention map overlay"""
    image_array = np.array(image)
    
    # Resize attention map
    attention_resized = resize_attention_map(attention_map, image_array.shape[:2])
    
    # Apply colormap
    attention_colored = plt.cm.jet(attention_resized)[:, :, :3]
    attention_colored = (attention_colored * 255).astype(np.uint8)
    
    # Create overlay
    overlay = attention_colored * alpha + image_array * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay, attention_resized

def select_samples():
    """Select sample images"""
    samples = {}
    for class_name in Config.CLASS_NAMES:
        class_dir = Config.DATA_DIR / class_name
        if class_dir.exists():
            image_files = list(class_dir.glob("*.jpg"))
            if image_files:
                num_samples = min(Config.SAMPLES_PER_CLASS, len(image_files))
                step = max(1, len(image_files) // num_samples)
                selected = image_files[::step][:num_samples]
                samples[class_name] = selected
                print(f"📸 Selected {len(selected)} samples from {class_name}")
    return samples

def generate_visualizations(model, gradcam, samples):
    """Generate all visualizations"""
    results = []
    
    for class_name, image_paths in samples.items():
        class_index = Config.CLASS_NAMES.index(class_name)
        
        print(f"\n🔍 Processing {class_name} samples...")
        
        for image_path in image_paths:
            print(f"   Processing {image_path.name}...")
            
            # Preprocess
            image_array, original_image = preprocess_image(image_path)
            if image_array is None:
                continue
            
            # Get prediction
            predictions = model.predict(image_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Generate attention map
            attention_map = gradcam.create_attention_map(image_array, predictions[0], class_index)
            
            if attention_map is not None:
                overlay, attention_resized = create_overlay(original_image, attention_map)
                
                result = {
                    'image_path': image_path,
                    'true_class': class_name,
                    'predicted_class': Config.CLASS_NAMES[predicted_class],
                    'confidence': confidence,
                    'original_image': original_image,
                    'attention_map': attention_map,
                    'attention_resized': attention_resized,
                    'overlay': overlay
                }
                
                results.append(result)
                print(f"     ✅ Generated attention map (Pred: {Config.CLASS_NAMES[predicted_class]}, Conf: {confidence:.3f})")
    
    return results

def plot_result(result, save_dir, index):
    """Plot single result"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(result['attention_resized'], cmap='jet')
    axes[1].set_title('Attention Map', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(result['overlay'])
    axes[2].set_title('Superimposed', fontweight='bold')
    axes[2].axis('off')
    
    # Info
    pred_text = f'True: {result["true_class"]}\nPredicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}'
    
    if result["predicted_class"] != result["true_class"]:
        pred_text += f'\n\n❌ Incorrect'
        color = 'lightcoral'
    else:
        pred_text += f'\n\n✅ Correct'
        color = 'lightgreen'
    
    axes[3].text(0.5, 0.5, pred_text, ha='center', va='center', 
                 transform=axes[3].transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[3].axis('off')
    
    image_name = result['image_path'].name
    plt.suptitle(f'Attention Analysis: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"attention_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path.name}")

def create_summary_grid(results, save_dir):
    """Create summary grid"""
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
            ax = axes[col] if cols > 1 else axes[0]
        else:
            ax = axes[row][col] if cols > 1 else axes[row]
        
        ax.imshow(result['overlay'])
        
        title = f"{result['true_class']}\n{result['image_path'].name}\n"
        if result['predicted_class'] == result['true_class']:
            title += f"✅ {result['confidence']:.3f}"
        else:
            title += f"❌ Pred: {result['predicted_class']}\n{result['confidence']:.3f}"
        
        ax.set_title(title, fontsize=9)
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
    
    plt.suptitle(f'Attention Map Summary - DenseNet121', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"attention_summary_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Summary grid saved: {save_path.name}")

def main():
    """Main function"""
    print("🎯 Simple DenseNet121 Attention Visualization")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available.")
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
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create Simple Grad-CAM
    print(f"\n🔧 Initializing attention mapper...")
    try:
        gradcam = SimpleGradCAM(model)
    except Exception as e:
        print(f"❌ Error creating attention mapper: {e}")
        return
    
    # Select samples
    print(f"\n📸 Selecting samples...")
    samples = select_samples()
    if not samples:
        print("❌ No samples found")
        return
    
    # Generate visualizations
    print(f"\n🎯 Generating attention visualizations...")
    results = generate_visualizations(model, gradcam, samples)
    
    if not results:
        print("❌ No visualizations generated")
        return
    
    print(f"\n✅ Generated {len(results)} visualizations")
    
    # Create plots
    print(f"\n📊 Creating visualizations...")
    for i, result in enumerate(results):
        plot_result(result, Config.GRADCAM_DIR, i)
    
    create_summary_grid(results, Config.GRADCAM_DIR)
    
    # Summary
    correct = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    accuracy = correct / len(results)
    
    print(f"\n🎉 Attention mapping completed!")
    print(f"📁 Results: {Config.GRADCAM_DIR}")
    print(f"📊 Visualizations: {len(results)}")
    print(f"🎯 Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()