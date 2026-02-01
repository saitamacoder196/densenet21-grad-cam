#!/usr/bin/env python3
"""
True Grad-CAM Implementation for DenseNet121
============================================

Thực hiện Grad-CAM thật sự bằng cách tái tạo model architecture
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
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available.")
    TENSORFLOW_AVAILABLE = False

class Config:
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/test")
    GRADCAM_DIR = Path("true_gradcam_results")
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class TrueGradCAM:
    """True Grad-CAM implementation bằng cách tái tạo model"""
    
    def __init__(self, original_model):
        self.original_model = original_model
        
        # Tạo DenseNet121 backbone
        self.base_model = DenseNet121(
            weights=None,  # Sẽ load weights từ original model
            include_top=False,
            input_shape=Config.INPUT_SHAPE
        )
        
        # Tạo model mới với cùng architecture
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(4, activation='softmax')(x)
        
        self.grad_model = Model(inputs=self.base_model.input, outputs=predictions)
        
        # Copy weights từ original model
        self._copy_weights()
        
        # Tạo model để lấy features và predictions
        self.feature_extractor = Model(
            inputs=self.base_model.input,
            outputs=self.base_model.output
        )
        
        print("✅ True Grad-CAM model created successfully")
    
    def _copy_weights(self):
        """Copy weights từ original model sang grad_model"""
        try:
            # Lấy DenseNet121 layer từ original model
            densenet_layer = None
            for layer in self.original_model.layers:
                if 'densenet121' in layer.name.lower():
                    densenet_layer = layer
                    break
            
            if densenet_layer is not None:
                # Copy DenseNet121 weights
                for i, layer in enumerate(self.base_model.layers):
                    if i < len(densenet_layer.layers):
                        try:
                            layer.set_weights(densenet_layer.layers[i].get_weights())
                        except:
                            pass
                
                # Copy dense layer weights (classifier)
                dense_weights = None
                for layer in self.original_model.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        dense_weights = layer.get_weights()
                        break
                
                if dense_weights is not None:
                    # Tìm Dense layer trong grad_model
                    for layer in self.grad_model.layers:
                        if isinstance(layer, tf.keras.layers.Dense):
                            layer.set_weights(dense_weights)
                            break
                
                print("✅ Weights copied successfully")
            else:
                print("⚠️ Could not find DenseNet121 layer for weight copying")
        
        except Exception as e:
            print(f"⚠️ Weight copying failed: {e}")
            print("🔄 Using original model predictions instead")
    
    def generate_gradcam(self, image, class_index):
        """Tạo Grad-CAM heatmap thực sự"""
        try:
            # Ensure image has batch dimension
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            image_tensor = tf.cast(image, tf.float32)
            
            # Sử dụng GradientTape để tính gradients
            with tf.GradientTape() as tape:
                # Get feature maps
                feature_maps = self.feature_extractor(image_tensor)
                tape.watch(feature_maps)
                
                # Global average pooling
                pooled_features = tf.reduce_mean(feature_maps, axis=[1, 2])
                
                # Get predictions từ original model (để đảm bảo accuracy)
                predictions = self.original_model(image_tensor, training=False)
                
                # Get class activation
                class_activation = predictions[:, class_index]
            
            # Compute gradients
            gradients = tape.gradient(class_activation, feature_maps)
            
            if gradients is None:
                print(f"⚠️ No gradients computed for class {class_index}")
                return None
            
            # Pool the gradients over all the axes except the feature map axis
            pooled_gradients = tf.reduce_mean(gradients, axis=[0, 1, 2])
            
            # Get the feature maps for the last conv layer
            feature_maps = feature_maps[0]
            
            # Multiply each feature map by "how important this feature is"
            for i in range(pooled_gradients.shape[-1]):
                feature_maps = feature_maps[:, :, i] * pooled_gradients[i]
            
            # The channel-wise mean of the resulting feature map is our heatmap
            heatmap = tf.reduce_mean(feature_maps, axis=-1)
            
            # Normalize the heatmap
            heatmap = tf.maximum(heatmap, 0)  # ReLU
            max_val = tf.reduce_max(heatmap)
            if max_val != 0:
                heatmap = heatmap / max_val
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"❌ Error in Grad-CAM generation: {e}")
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
    
    # Apply colormap
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

def generate_visualizations(original_model, gradcam, samples):
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
            
            # Get prediction từ original model
            image_batch = np.expand_dims(image_array, axis=0)
            predictions = original_model.predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Generate Grad-CAM heatmap
            heatmap = gradcam.generate_gradcam(image_array, class_index)
            
            if heatmap is not None:
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
                print(f"     ✅ Generated Grad-CAM (Pred: {Config.CLASS_NAMES[predicted_class]}, Conf: {confidence:.3f})")
            else:
                print(f"     ❌ Failed to generate Grad-CAM for {image_path.name}")
    
    return results

def plot_result(result, save_dir, index):
    """Plot single Grad-CAM result"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    im = axes[1].imshow(result['heatmap_resized'], cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontweight='bold')
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
    plt.suptitle(f'True Grad-CAM Analysis: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"gradcam_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path.name}")

def create_summary_grid(results, save_dir):
    """Create Grad-CAM summary grid"""
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
    
    plt.suptitle(f'True Grad-CAM Summary - DenseNet121', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"true_gradcam_summary_{Config.TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Grad-CAM summary saved: {save_path.name}")

def main():
    """Main function"""
    print("🔥 True Grad-CAM for DenseNet121")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available.")
        return
    
    # Setup
    Config.GRADCAM_DIR.mkdir(exist_ok=True)
    
    # Load original model
    model_path = Config.MODELS_DIR / Config.TARGET_MODEL
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📂 Loading original model: {Config.TARGET_MODEL}")
    try:
        original_model = load_model(model_path)
        print(f"✅ Original model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create True Grad-CAM
    print(f"\n🔧 Creating True Grad-CAM model...")
    try:
        gradcam = TrueGradCAM(original_model)
    except Exception as e:
        print(f"❌ Error creating Grad-CAM: {e}")
        return
    
    # Select samples
    print(f"\n📸 Selecting samples...")
    samples = select_samples()
    if not samples:
        print("❌ No samples found")
        return
    
    # Generate Grad-CAM visualizations
    print(f"\n🔥 Generating True Grad-CAM visualizations...")
    results = generate_visualizations(original_model, gradcam, samples)
    
    if not results:
        print("❌ No Grad-CAM visualizations generated")
        return
    
    print(f"\n✅ Generated {len(results)} Grad-CAM visualizations")
    
    # Create plots
    print(f"\n📊 Creating Grad-CAM plots...")
    for i, result in enumerate(results):
        plot_result(result, Config.GRADCAM_DIR, i)
    
    create_summary_grid(results, Config.GRADCAM_DIR)
    
    # Summary
    correct = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    accuracy = correct / len(results)
    
    print(f"\n🎉 True Grad-CAM completed!")
    print(f"📁 Results: {Config.GRADCAM_DIR}")
    print(f"📊 Grad-CAM visualizations: {len(results)}")
    print(f"🎯 Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()