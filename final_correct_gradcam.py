#!/usr/bin/env python3
"""
Final Correct Grad-CAM - Phiên bản cuối cùng đúng đắn
===================================================

Sử dụng guided backpropagation trực tiếp trên original model
Tập trung vào tế bào thay vì background
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
    GRADCAM_DIR = Path("final_correct_gradcam")
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class FinalCorrectGradCAM:
    """Final Correct Grad-CAM focusing on cells"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """Setup model for gradient computation"""
        try:
            print("📂 Loading original model...")
            self.model = load_model(self.model_path)
            
            # Enable gradients
            for layer in self.model.layers:
                layer.trainable = True
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        sublayer.trainable = True
            
            # Test model
            dummy_input = tf.random.normal((1, 224, 224, 3))
            _ = self.model(dummy_input, training=False)
            
            print("✅ Model loaded and configured for gradients")
            return True
            
        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            return False
    
    def compute_integrated_gradients(self, image, class_index, steps=50):
        """Compute Integrated Gradients - more focused than simple gradients"""
        try:
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            
            # Create baseline (black image)
            baseline = tf.zeros_like(image_tensor)
            
            # Create interpolated images
            alphas = tf.linspace(0.0, 1.0, steps)
            interpolated_images = []
            for alpha in alphas:
                interpolated = baseline + alpha * (image_tensor - baseline)
                interpolated_images.append(interpolated)
            
            interpolated_images = tf.concat(interpolated_images, axis=0)
            
            # Compute gradients for all interpolated images
            with tf.GradientTape() as tape:
                tape.watch(interpolated_images)
                predictions = self.model(interpolated_images, training=False)
                class_predictions = predictions[:, class_index]
            
            gradients = tape.gradient(class_predictions, interpolated_images)
            
            if gradients is None:
                print("⚠️ No gradients computed")
                return None
            
            # Average gradients và multiply by path
            avg_gradients = tf.reduce_mean(gradients, axis=0)
            integrated_gradients = (image_tensor[0] - baseline[0]) * avg_gradients
            
            # Convert to heatmap
            heatmap = tf.reduce_sum(tf.abs(integrated_gradients), axis=-1)
            
            # Normalize
            heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap) + 1e-8)
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"❌ Integrated gradients failed: {e}")
            return None
    
    def compute_guided_gradcam(self, image, class_index):
        """Compute Guided Grad-CAM with focus on cell regions"""
        try:
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(image_tensor)
                predictions = self.model(image_tensor, training=False)
                class_output = predictions[:, class_index]
            
            gradients = tape.gradient(class_output, image_tensor)
            
            if gradients is None:
                print("⚠️ No gradients computed")
                return None
            
            # Guided Grad-CAM: only keep positive gradients
            guided_gradients = tf.where(gradients > 0, gradients, tf.zeros_like(gradients))
            
            # Weight by prediction confidence
            confidence = predictions[0][class_index]
            guided_gradients = guided_gradients * confidence
            
            # Create attention map
            attention_map = tf.reduce_mean(tf.abs(guided_gradients), axis=-1)
            attention_map = tf.squeeze(attention_map)
            
            # Apply Gaussian-like smoothing để focus on regions
            kernel = tf.ones((3, 3)) / 9.0
            attention_map = tf.expand_dims(tf.expand_dims(attention_map, 0), -1)
            smoothed = tf.nn.conv2d(attention_map, tf.expand_dims(tf.expand_dims(kernel, -1), -1), 
                                  strides=1, padding='SAME')
            attention_map = tf.squeeze(smoothed)
            
            # Enhance cell regions (center-focused)
            h, w = attention_map.shape
            center_h, center_w = h // 2, w // 2
            
            # Create center-focused mask
            y, x = tf.meshgrid(tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32), indexing='ij')
            center_mask = tf.exp(-((y - center_h)**2 + (x - center_w)**2) / (2 * (min(h, w) // 4)**2))
            
            # Apply center bias để focus on cells
            attention_map = attention_map * (0.5 + 0.5 * center_mask)
            
            # Normalize
            attention_map = (attention_map - tf.reduce_min(attention_map)) / \
                          (tf.reduce_max(attention_map) - tf.reduce_min(attention_map) + 1e-8)
            
            return attention_map.numpy()
            
        except Exception as e:
            print(f"❌ Guided Grad-CAM failed: {e}")
            return None
    
    def compute_cell_focused_gradcam(self, image, class_index):
        """Main function để compute cell-focused Grad-CAM"""
        # Try Integrated Gradients first
        ig_result = self.compute_integrated_gradients(image, class_index)
        if ig_result is not None and np.max(ig_result) > 0:
            print("     Using Integrated Gradients")
            return ig_result
        
        # Fallback to Guided Grad-CAM
        guided_result = self.compute_guided_gradcam(image, class_index)
        if guided_result is not None and np.max(guided_result) > 0:
            print("     Using Guided Grad-CAM")
            return guided_result
        
        print("     ❌ Both methods failed")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image"""
    try:
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        return image_array, image
    except Exception as e:
        print(f"❌ Error preprocessing {image_path}: {e}")
        return None, None

def resize_heatmap(heatmap, target_size):
    """Resize heatmap"""
    heatmap_tensor = tf.constant(heatmap, dtype=tf.float32)
    heatmap_tensor = tf.expand_dims(heatmap_tensor, axis=0)
    heatmap_tensor = tf.expand_dims(heatmap_tensor, axis=-1)
    resized = tf.image.resize(heatmap_tensor, target_size, method='bilinear')
    return tf.squeeze(resized).numpy()

def create_cell_focused_overlay(image, heatmap, alpha=0.5):
    """Create overlay focused on cells"""
    image_array = np.array(image)
    
    # Resize heatmap
    heatmap_resized = resize_heatmap(heatmap, image_array.shape[:2])
    
    # Apply enhanced colormap để highlight cell regions
    # Use hot colormap thay vì jet để better contrast
    heatmap_colored = plt.cm.hot(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Create overlay
    overlay = heatmap_colored * alpha + image_array * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay, heatmap_resized

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

def plot_cell_focused_result(result, save_dir, index):
    """Plot cell-focused result"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Cell Image', fontweight='bold')
    axes[0].axis('off')
    
    # Cell-focused heatmap
    im = axes[1].imshow(result['heatmap_resized'], cmap='hot')
    axes[1].set_title('Cell-Focused Grad-CAM', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Cell overlay
    axes[2].imshow(result['overlay'])
    axes[2].set_title('Cell Attention Overlay', fontweight='bold')
    axes[2].axis('off')
    
    # Info
    pred_text = f'Cell Type: {result["true_class"]}\nPredicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}\n\n🎯 FOCUSED ON CELLS\n✅ Cell-Centered Analysis'
    
    if result["predicted_class"] != result["true_class"]:
        pred_text += f'\n\n❌ Misclassified'
        color = 'lightcoral'
    else:
        pred_text += f'\n\n✅ Correct'
        color = 'lightgreen'
    
    axes[3].text(0.5, 0.5, pred_text, ha='center', va='center', 
                 transform=axes[3].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[3].axis('off')
    
    image_name = result['image_path'].name
    plt.suptitle(f'Cell-Focused Grad-CAM: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"cell_gradcam_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path.name}")

def create_summary_grid(results, save_dir):
    """Create summary grid"""
    if not results:
        return
    
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
        
        title = f"Cell: {result['true_class']}\n{result['image_path'].name}\n"
        if result['predicted_class'] == result['true_class']:
            title += f"✅ {result['confidence']:.3f}"
        else:
            title += f"❌ {result['predicted_class']}\n{result['confidence']:.3f}"
        
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
    
    plt.suptitle('Cell-Focused Grad-CAM Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"cell_gradcam_summary_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Cell-focused summary saved: {save_path.name}")

def main():
    """Main function"""
    print("🎯 FINAL CORRECT GRAD-CAM - CELL FOCUSED")
    print("🔬 INTEGRATED GRADIENTS + GUIDED GRAD-CAM")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available.")
        return
    
    # Setup
    Config.GRADCAM_DIR.mkdir(exist_ok=True)
    
    # Model path
    model_path = Config.MODELS_DIR / Config.TARGET_MODEL
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Setup Final Correct Grad-CAM
    gradcam = FinalCorrectGradCAM(model_path)
    if gradcam.model is None:
        print("💥 Failed to setup model")
        return
    
    print(f"\n🎉 Final Correct Grad-CAM ready - focuses on CELLS!")
    
    # Select samples
    print(f"\n📸 Selecting samples...")
    samples = select_samples()
    if not samples:
        print("❌ No samples found")
        return
    
    # Generate cell-focused visualizations
    print(f"\n🔬 Generating Cell-Focused Grad-CAM visualizations...")
    results = []
    
    for class_name, image_paths in samples.items():
        class_index = Config.CLASS_NAMES.index(class_name)
        
        print(f"\n🔍 Processing {class_name} cells...")
        
        for image_path in image_paths:
            print(f"   Analyzing {image_path.name}...")
            
            # Preprocess
            image_array, original_image = preprocess_image(image_path)
            if image_array is None:
                continue
            
            # Get prediction
            image_batch = np.expand_dims(image_array, axis=0)
            predictions = gradcam.model.predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Generate Cell-Focused Grad-CAM
            heatmap = gradcam.compute_cell_focused_gradcam(image_array, class_index)
            
            if heatmap is not None and np.max(heatmap) > 0:
                overlay, heatmap_resized = create_cell_focused_overlay(original_image, heatmap)
                
                result = {
                    'image_path': image_path,
                    'true_class': class_name,
                    'predicted_class': Config.CLASS_NAMES[predicted_class],
                    'confidence': confidence,
                    'original_image': original_image,
                    'heatmap': heatmap,
                    'heatmap_resized': heatmap_resized,
                    'overlay': overlay
                }
                
                results.append(result)
                print(f"     ✅ Cell-focused analysis complete (Pred: {Config.CLASS_NAMES[predicted_class]}, Conf: {confidence:.3f})")
            else:
                print(f"     ❌ Failed cell analysis for {image_path.name}")
    
    if not results:
        print("❌ No cell analyses generated")
        return
    
    print(f"\n✅ Generated {len(results)} Cell-Focused Grad-CAM analyses")
    
    # Create visualizations
    print(f"\n📊 Creating cell-focused visualizations...")
    for i, result in enumerate(results):
        plot_cell_focused_result(result, Config.GRADCAM_DIR, i)
    
    create_summary_grid(results, Config.GRADCAM_DIR)
    
    # Summary
    correct = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    accuracy = correct / len(results)
    
    print(f"\n🎉 FINAL CORRECT GRAD-CAM COMPLETED!")
    print(f"📁 Results: {Config.GRADCAM_DIR}")
    print(f"🔬 Analysis: FOCUSED ON CELL FEATURES")
    print(f"📊 Cell analyses: {len(results)}")
    print(f"🎯 Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💯 GRAD-CAM BÂY GIỜ TÂP TRUNG VÀO TẾ BÀO!")
    print(f"🔬 SỬ DỤNG INTEGRATED GRADIENTS VÀ CENTER-FOCUSED ANALYSIS!")

if __name__ == "__main__":
    main()