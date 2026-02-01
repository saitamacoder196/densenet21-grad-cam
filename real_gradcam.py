#!/usr/bin/env python3
"""
Real Grad-CAM Implementation - Thực hiện Grad-CAM thật
======================================================

Sử dụng kỹ thuật đặc biệt để truy xuất gradients từ model gốc
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
    GRADCAM_DIR = Path("real_gradcam_results")
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class RealGradCAM:
    """Real Grad-CAM using direct model manipulation"""
    
    def __init__(self, model):
        self.model = model
        self.prepare_gradcam_model()
    
    def prepare_gradcam_model(self):
        """Chuẩn bị model cho Grad-CAM"""
        try:
            # Tìm DenseNet121 layer
            self.densenet_layer = None
            for layer in self.model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 0:
                    # Đây là DenseNet backbone
                    self.densenet_layer = layer
                    break
            
            if self.densenet_layer is None:
                raise ValueError("Không tìm thấy DenseNet layer")
            
            # Tạo feature extraction model
            # Sử dụng output của DenseNet trước GlobalAveragePooling
            self.feature_model = Model(
                inputs=self.model.input,
                outputs=self.densenet_layer.output
            )
            
            print("✅ Real Grad-CAM model prepared")
            print(f"   Feature shape: {self.densenet_layer.output_shape}")
            
        except Exception as e:
            print(f"❌ Error preparing Grad-CAM model: {e}")
            self.feature_model = None
    
    def generate_gradcam(self, image, class_index):
        """Tạo Grad-CAM heatmap thực sự"""
        if self.feature_model is None:
            return None
        
        try:
            # Đảm bảo image có batch dimension
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            
            # Sử dụng GradientTape
            with tf.GradientTape() as tape:
                # Watch input image
                tape.watch(image_tensor)
                
                # Get feature maps từ DenseNet
                features = self.feature_model(image_tensor, training=False)
                
                # Global Average Pooling thủ công
                pooled = tf.reduce_mean(features, axis=[1, 2])  # [batch, channels]
                
                # Get predictions từ model gốc
                predictions = self.model(image_tensor, training=False)
                class_output = predictions[:, class_index]
            
            # Tính gradients của class output w.r.t features
            gradients = tape.gradient(class_output, features)
            
            if gradients is None:
                print("⚠️ No gradients computed")
                return None
            
            # Pooling gradients globally
            pooled_gradients = tf.reduce_mean(gradients, axis=[0, 1, 2])
            
            # Get feature maps
            feature_maps = features[0]  # Remove batch dimension
            
            # Weighted combination of feature maps
            heatmap = tf.zeros(feature_maps.shape[:2])  # [H, W]
            for i in range(feature_maps.shape[-1]):
                heatmap += pooled_gradients[i] * feature_maps[:, :, i]
            
            # Apply ReLU
            heatmap = tf.maximum(heatmap, 0)
            
            # Normalize
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"❌ Error in Grad-CAM: {e}")
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
    """Resize heatmap to target size"""
    heatmap_tensor = tf.constant(heatmap, dtype=tf.float32)
    heatmap_tensor = tf.expand_dims(heatmap_tensor, axis=0)
    heatmap_tensor = tf.expand_dims(heatmap_tensor, axis=-1)
    resized = tf.image.resize(heatmap_tensor, target_size, method='bilinear')
    return tf.squeeze(resized).numpy()

def create_overlay(image, heatmap, alpha=0.4):
    """Create Grad-CAM overlay"""
    image_array = np.array(image)
    
    # Resize heatmap
    heatmap_resized = resize_heatmap(heatmap, image_array.shape[:2])
    
    # Apply jet colormap
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
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

def generate_visualizations(model, gradcam, samples):
    """Generate all Grad-CAM visualizations"""
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
            image_batch = np.expand_dims(image_array, axis=0)
            predictions = model.predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Generate Real Grad-CAM
            heatmap = gradcam.generate_gradcam(image_array, class_index)
            
            if heatmap is not None and heatmap.max() > 0:
                overlay, heatmap_resized = create_overlay(original_image, heatmap)
                
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
                print(f"     ✅ Generated Real Grad-CAM (Pred: {Config.CLASS_NAMES[predicted_class]}, Conf: {confidence:.3f})")
            else:
                print(f"     ❌ Failed to generate Grad-CAM for {image_path.name}")
    
    return results

def plot_result(result, save_dir, index):
    """Plot single Real Grad-CAM result"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Real Grad-CAM heatmap
    im = axes[1].imshow(result['heatmap_resized'], cmap='jet')
    axes[1].set_title('Real Grad-CAM Heatmap', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(result['overlay'])
    axes[2].set_title('Grad-CAM Overlay', fontweight='bold')
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
    plt.suptitle(f'Real Grad-CAM Analysis: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"real_gradcam_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path.name}")

def create_summary_grid(results, save_dir):
    """Create Real Grad-CAM summary grid"""
    if not results:
        print("❌ No results to create summary")
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
    
    plt.suptitle(f'Real Grad-CAM Summary - DenseNet121', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"real_gradcam_summary_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Real Grad-CAM summary saved: {save_path.name}")

def main():
    """Main function"""
    print("🎯 Real Grad-CAM for DenseNet121 - Thực hiện Grad-CAM thật")
    print("=" * 60)
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
    
    # Create Real Grad-CAM
    print(f"\n🔧 Creating Real Grad-CAM...")
    try:
        gradcam = RealGradCAM(model)
        if gradcam.feature_model is None:
            print("❌ Could not create Grad-CAM model")
            return
    except Exception as e:
        print(f"❌ Error creating Real Grad-CAM: {e}")
        return
    
    # Select samples
    print(f"\n📸 Selecting samples...")
    samples = select_samples()
    if not samples:
        print("❌ No samples found")
        return
    
    # Generate Real Grad-CAM visualizations
    print(f"\n🔥 Generating Real Grad-CAM visualizations...")
    results = generate_visualizations(model, gradcam, samples)
    
    if not results:
        print("❌ No Real Grad-CAM visualizations generated")
        return
    
    print(f"\n✅ Generated {len(results)} Real Grad-CAM visualizations")
    
    # Create plots
    print(f"\n📊 Creating Real Grad-CAM plots...")
    for i, result in enumerate(results):
        plot_result(result, Config.GRADCAM_DIR, i)
    
    create_summary_grid(results, Config.GRADCAM_DIR)
    
    # Summary
    correct = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    accuracy = correct / len(results)
    
    print(f"\n🎉 Real Grad-CAM completed successfully!")
    print(f"📁 Results: {Config.GRADCAM_DIR}")
    print(f"📊 Real Grad-CAM visualizations: {len(results)}")
    print(f"🎯 Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 Đây là Grad-CAM thực sự sử dụng gradients từ model DenseNet121!")

if __name__ == "__main__":
    main()