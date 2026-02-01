#!/usr/bin/env python3
"""
Working Grad-CAM Implementation for DenseNet121
==============================================

Fixed implementation that properly handles the nested DenseNet121 model structure
to create actual heatmaps showing where the model focuses.

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
    print("❌ TensorFlow not available.")
    TENSORFLOW_AVAILABLE = False

# Configuration
class Config:
    # Paths
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/test")
    GRADCAM_DIR = Path("gradcam_heatmaps")
    
    # Model settings
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    
    # Grad-CAM settings
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    
    # Target layers (from inspection results)
    TARGET_LAYERS = [
        'conv5_block16_2_conv',  # Last conv layer
        'conv5_block15_2_conv',  # Second to last
        'conv4_block24_2_conv'   # Earlier high-level features
    ]
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class WorkingGradCAM:
    """Working Grad-CAM implementation for DenseNet121"""
    
    def __init__(self, model, target_layer_name='conv5_block16_2_conv'):
        self.model = model
        self.densenet_model = model.get_layer('densenet121')
        self.target_layer_name = target_layer_name
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self):
        """Build gradient model targeting specific DenseNet layer"""
        try:
            # Get the target layer from within DenseNet121
            target_layer = self.densenet_model.get_layer(self.target_layer_name)
            print(f"✅ Found target layer: {self.target_layer_name}")
            print(f"   Output shape: {target_layer.output_shape}")
            
            # Create gradient model that outputs both conv features and final predictions
            grad_model = Model(
                inputs=self.model.input,
                outputs=[target_layer.output, self.model.output]
            )
            
            return grad_model
            
        except Exception as e:
            print(f"❌ Error building gradient model for {self.target_layer_name}: {e}")
            return None
    
    def generate_heatmap(self, image, class_index):
        """Generate Grad-CAM heatmap"""
        if self.grad_model is None:
            return None
        
        try:
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(image)
                loss = predictions[:, class_index]
            
            # Get gradients of loss w.r.t. conv_outputs
            grads = tape.gradient(loss, conv_outputs)
            
            if grads is None:
                print(f"⚠️ No gradients computed for class {class_index}")
                return None
            
            # Global average pooling on gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Multiply each channel by its corresponding gradient
            conv_outputs = conv_outputs[0]  # Remove batch dimension
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Apply ReLU to keep only positive influence
            heatmap = tf.maximum(heatmap, 0)
            
            # Normalize heatmap
            max_val = tf.reduce_max(heatmap)
            if max_val != 0:
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

def resize_heatmap_bilinear(heatmap, target_size):
    """Resize heatmap using bilinear interpolation"""
    # Use TensorFlow's resize function
    heatmap_tensor = tf.constant(heatmap, dtype=tf.float32)
    heatmap_tensor = tf.expand_dims(heatmap_tensor, axis=0)  # Add batch dim
    heatmap_tensor = tf.expand_dims(heatmap_tensor, axis=-1)  # Add channel dim
    
    # Resize using bilinear interpolation
    resized = tf.image.resize(heatmap_tensor, target_size, method='bilinear')
    
    # Remove extra dimensions
    resized = tf.squeeze(resized)
    
    return resized.numpy()

def create_gradcam_overlay(image, heatmap, alpha=0.4):
    """Create Grad-CAM overlay on original image"""
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Resize heatmap to match image size
    heatmap_resized = resize_heatmap_bilinear(heatmap, image_array.shape[:2])
    
    # Create colored heatmap (red colormap)
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Normalize image to [0, 255]
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    
    # Create overlay
    overlay = heatmap_colored * alpha + image_array * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay, heatmap_resized

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

def generate_gradcam_visualizations(model, samples):
    """Generate Grad-CAM visualizations for all target layers"""
    all_results = []
    
    for layer_name in Config.TARGET_LAYERS:
        print(f"\n🎯 Processing layer: {layer_name}")
        
        # Create Grad-CAM for this layer
        gradcam = WorkingGradCAM(model, layer_name)
        if gradcam.grad_model is None:
            print(f"❌ Skipping {layer_name} - failed to create gradient model")
            continue
        
        layer_results = []
        
        for class_name, image_paths in samples.items():
            class_index = Config.CLASS_NAMES.index(class_name)
            
            print(f"\n   🔍 Processing {class_name} samples...")
            
            for image_path in image_paths:
                print(f"      Processing {image_path.name}...")
                
                # Preprocess image
                image_array, original_image = preprocess_image(image_path)
                if image_array is None:
                    continue
                
                # Get model prediction
                predictions = model.predict(image_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                
                # Generate heatmap for true class
                heatmap = gradcam.generate_heatmap(image_array, class_index)
                
                if heatmap is not None:
                    # Create overlay
                    overlay, heatmap_resized = create_gradcam_overlay(original_image, heatmap)
                    
                    result = {
                        'layer_name': layer_name,
                        'image_path': image_path,
                        'true_class': class_name,
                        'true_class_index': class_index,
                        'predicted_class': Config.CLASS_NAMES[predicted_class],
                        'predicted_class_index': predicted_class,
                        'confidence': confidence,
                        'original_image': original_image,
                        'heatmap': heatmap,
                        'heatmap_resized': heatmap_resized,
                        'overlay': overlay
                    }
                    
                    layer_results.append(result)
                    print(f"         ✅ Generated heatmap (Pred: {Config.CLASS_NAMES[predicted_class]}, Conf: {confidence:.3f})")
                else:
                    print(f"         ❌ Failed to generate heatmap")
        
        all_results.extend(layer_results)
    
    return all_results

def plot_gradcam_comparison(result, save_dir, index):
    """Plot Grad-CAM result with multiple views"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(result['heatmap_resized'], cmap='jet')
    axes[1].set_title(f'Grad-CAM Heatmap\n{result["layer_name"]}', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(result['overlay'])
    axes[2].set_title(f'Superimposed\n({result["true_class"]})', fontweight='bold')
    axes[2].axis('off')
    
    # Prediction info
    pred_text = f'True: {result["true_class"]}\nPredicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}\nLayer: {result["layer_name"]}'
    
    if result["predicted_class"] != result["true_class"]:
        pred_text += f'\n\n❌ Incorrect'
        color = 'lightcoral'
    else:
        pred_text += f'\n\n✅ Correct'
        color = 'lightgreen'
    
    axes[3].text(0.5, 0.5, pred_text, ha='center', va='center', 
                 transform=axes[3].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[3].axis('off')
    
    # Main title
    image_name = result['image_path'].name
    plt.suptitle(f'Grad-CAM Analysis: {image_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    safe_layer_name = result['layer_name'].replace('_', '-')
    save_path = save_dir / f"gradcam_{index:03d}_{result['true_class']}_{safe_layer_name}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      💾 Saved: {save_path.name}")

def create_layer_comparison(results, save_dir):
    """Create comparison of different layers for same image"""
    # Group by image
    results_by_image = {}
    for result in results:
        image_key = f"{result['true_class']}_{result['image_path'].name}"
        if image_key not in results_by_image:
            results_by_image[image_key] = []
        results_by_image[image_key].append(result)
    
    for image_key, image_results in results_by_image.items():
        if len(image_results) < 2:  # Need at least 2 layers to compare
            continue
        
        # Sort by layer name for consistent ordering
        image_results.sort(key=lambda x: x['layer_name'])
        
        fig, axes = plt.subplots(2, len(image_results), figsize=(4 * len(image_results), 8))
        
        if len(image_results) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, result in enumerate(image_results):
            # Heatmap
            im1 = axes[0, i].imshow(result['heatmap_resized'], cmap='jet')
            axes[0, i].set_title(f'Heatmap\n{result["layer_name"]}', fontweight='bold', fontsize=10)
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Overlay
            axes[1, i].imshow(result['overlay'])
            axes[1, i].set_title(f'Overlay\n{result["layer_name"]}', fontweight='bold', fontsize=10)
            axes[1, i].axis('off')
        
        # Main title
        sample_result = image_results[0]
        main_title = f'Layer Comparison: {sample_result["image_path"].name}\nTrue: {sample_result["true_class"]} | Predicted: {sample_result["predicted_class"]} (Conf: {sample_result["confidence"]:.3f})'
        plt.suptitle(main_title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        save_path = save_dir / f"layer_comparison_{image_key.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Layer comparison saved: {save_path.name}")

def main():
    """Main Grad-CAM function"""
    print("🔥 Working DenseNet121 Grad-CAM Visualization")
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
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Select samples
    print(f"\n📸 Selecting sample images...")
    samples = select_sample_images()
    
    if not samples:
        print("❌ No sample images found")
        return
    
    total_samples = sum(len(paths) for paths in samples.values())
    print(f"✅ Total samples selected: {total_samples}")
    
    # Generate Grad-CAM visualizations
    print(f"\n🔥 Generating Grad-CAM heatmaps...")
    print(f"   Target layers: {Config.TARGET_LAYERS}")
    
    results = generate_gradcam_visualizations(model, samples)
    
    if not results:
        print("❌ No Grad-CAM visualizations generated")
        return
    
    print(f"\n✅ Generated {len(results)} Grad-CAM visualizations")
    
    # Create individual plots
    print(f"\n📊 Creating individual visualizations...")
    for i, result in enumerate(results):
        plot_gradcam_comparison(result, Config.GRADCAM_DIR, i)
    
    # Create layer comparisons
    print(f"\n🔄 Creating layer comparisons...")
    create_layer_comparison(results, Config.GRADCAM_DIR)
    
    # Summary
    correct_predictions = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    total_results = len(results)
    accuracy = correct_predictions / total_results if total_results > 0 else 0
    
    print(f"\n🎉 Grad-CAM visualization completed!")
    print(f"📁 Results saved in: {Config.GRADCAM_DIR}")
    print(f"📊 Total visualizations: {total_results}")
    print(f"🎯 Correct predictions: {correct_predictions}/{total_results} ({accuracy:.1%})")
    print(f"🔥 Heatmaps generated for {len(Config.TARGET_LAYERS)} layers")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()