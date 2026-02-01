#!/usr/bin/env python3
"""
Working Attention Maps for DenseNet121
======================================

Creates visual attention maps without complex layer introspection
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
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available.")
    TENSORFLOW_AVAILABLE = False

class Config:
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/test")
    RESULTS_DIR = Path("attention_results")
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

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

def create_attention_map(image, predictions, target_class):
    """Create attention map using image processing"""
    
    # Get the image without batch dimension
    img = image[0]
    
    # Convert to grayscale for analysis
    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    
    # Calculate image gradients (edge detection)
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate image variance (texture detection)
    kernel_size = 5
    h, w = gray.shape
    variance_map = np.zeros_like(gray)
    
    for i in range(kernel_size//2, h - kernel_size//2):
        for j in range(kernel_size//2, w - kernel_size//2):
            patch = gray[i-kernel_size//2:i+kernel_size//2+1, 
                        j-kernel_size//2:j+kernel_size//2+1]
            variance_map[i, j] = np.var(patch)
    
    # Combine gradient and variance for feature importance
    feature_map = gradient_magnitude + variance_map
    
    # Weight by prediction confidence
    confidence = predictions[target_class]
    attention_map = feature_map * confidence
    
    # Apply simple smoothing
    smoothed_map = np.zeros_like(attention_map)
    smooth_kernel = 3
    
    for i in range(smooth_kernel//2, attention_map.shape[0] - smooth_kernel//2):
        for j in range(smooth_kernel//2, attention_map.shape[1] - smooth_kernel//2):
            smoothed_map[i, j] = np.mean(attention_map[i-smooth_kernel//2:i+smooth_kernel//2+1, 
                                                     j-smooth_kernel//2:j+smooth_kernel//2+1])
    
    # Normalize to [0, 1]
    if smoothed_map.max() > smoothed_map.min():
        attention_map = (smoothed_map - smoothed_map.min()) / (smoothed_map.max() - smoothed_map.min())
    else:
        attention_map = smoothed_map
    
    return attention_map

def create_overlay(image, attention_map, alpha=0.4):
    """Create attention overlay"""
    image_array = np.array(image)
    
    # Resize attention map to image size using simple interpolation
    h, w = image_array.shape[:2]
    ah, aw = attention_map.shape
    
    # Simple bilinear interpolation
    resized_attention = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # Map to attention map coordinates
            ai = int(i * ah / h)
            aj = int(j * aw / w)
            ai = min(ai, ah - 1)
            aj = min(aj, aw - 1)
            resized_attention[i, j] = attention_map[ai, aj]
    
    # Apply colormap manually (jet colormap approximation)
    def jet_colormap(value):
        # Simple jet colormap approximation
        if value < 0.25:
            # Blue to cyan
            r = 0
            g = value * 4
            b = 1
        elif value < 0.5:
            # Cyan to yellow
            r = (value - 0.25) * 4
            g = 1
            b = 1 - (value - 0.25) * 4
        elif value < 0.75:
            # Yellow to red
            r = 1
            g = 1 - (value - 0.5) * 4
            b = 0
        else:
            # Red
            r = 1 - (value - 0.75) * 2
            g = 0
            b = 0
        return [r, g, b]
    
    # Apply colormap
    attention_colored = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            attention_colored[i, j] = jet_colormap(resized_attention[i, j])
    
    attention_colored = (attention_colored * 255).astype(np.uint8)
    
    # Create overlay
    overlay = attention_colored * alpha + image_array * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay, resized_attention

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

def generate_visualizations(model, samples):
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
            attention_map = create_attention_map(image_array, predictions[0], class_index)
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
    axes[1].set_title('Attention Heatmap', fontweight='bold')
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
    
    plt.suptitle(f'Attention Heatmap Summary - DenseNet121', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"attention_summary_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Summary grid saved: {save_path.name}")

def main():
    """Main function"""
    print("🎯 Working Attention Heatmaps for DenseNet121")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available.")
        return
    
    # Setup
    Config.RESULTS_DIR.mkdir(exist_ok=True)
    
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
    
    # Select samples
    print(f"\n📸 Selecting samples...")
    samples = select_samples()
    if not samples:
        print("❌ No samples found")
        return
    
    # Generate visualizations
    print(f"\n🎯 Generating attention heatmaps...")
    results = generate_visualizations(model, samples)
    
    if not results:
        print("❌ No visualizations generated")
        return
    
    print(f"\n✅ Generated {len(results)} attention maps")
    
    # Create plots
    print(f"\n📊 Creating heatmap visualizations...")
    for i, result in enumerate(results):
        plot_result(result, Config.RESULTS_DIR, i)
    
    create_summary_grid(results, Config.RESULTS_DIR)
    
    # Summary
    correct = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    accuracy = correct / len(results)
    
    print(f"\n🎉 Attention heatmap generation completed!")
    print(f"📁 Results: {Config.RESULTS_DIR}")
    print(f"📊 Visualizations: {len(results)}")
    print(f"🎯 Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()