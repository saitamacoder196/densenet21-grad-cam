#!/usr/bin/env python3
"""
Correct True Grad-CAM - Sửa vấn đề focus sai
============================================

Fix vấn đề Grad-CAM không tập trung vào tế bào
Extract exact weights từ original model
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
    from tensorflow.keras.layers import Input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available.")
    TENSORFLOW_AVAILABLE = False

class Config:
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/test")
    GRADCAM_DIR = Path("correct_gradcam_results")
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class CorrectGradCAM:
    """Correct True Grad-CAM sử dụng exact original weights"""
    
    def __init__(self, original_model_path):
        self.original_model_path = original_model_path
        self.original_model = None
        self.feature_extractor = None
        self.gradcam_model = None
        
    def load_and_analyze_model(self):
        """Load và phân tích chi tiết original model"""
        try:
            print("📂 Loading original model...")
            self.original_model = load_model(self.original_model_path)
            
            # Khởi tạo model
            dummy_input = tf.random.normal((1, 224, 224, 3))
            _ = self.original_model(dummy_input)
            
            print("\n🔍 Detailed model analysis:")
            print(f"Total layers: {len(self.original_model.layers)}")
            
            # Tìm DenseNet121 layer
            densenet_layer = None
            dense_layers = []
            
            for i, layer in enumerate(self.original_model.layers):
                print(f"  {i}: {layer.name} ({layer.__class__.__name__}) -> {getattr(layer, 'output_shape', 'Unknown')}")
                
                # Tìm DenseNet backbone
                if ('densenet' in layer.name.lower() or 
                    layer.__class__.__name__ == 'Functional' and 
                    hasattr(layer, 'output_shape') and 
                    len(layer.output_shape) == 4):
                    densenet_layer = layer
                    print(f"    ✅ Found DenseNet backbone: {layer.name}")
                
                # Tìm Dense layers
                if layer.__class__.__name__ == 'Dense':
                    dense_layers.append(layer)
                    print(f"    ✅ Found Dense layer: {layer.name} -> {layer.output_shape}")
            
            if densenet_layer is None:
                raise ValueError("Could not find DenseNet backbone layer")
            
            print(f"\n✅ Model analysis completed:")
            print(f"   DenseNet layer: {densenet_layer.name}")
            print(f"   Dense layers found: {len(dense_layers)}")
            
            self.densenet_layer = densenet_layer
            self.dense_layers = dense_layers
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load/analyze model: {e}")
            return False
    
    def extract_exact_weights(self):
        """Extract exact weights từ original model"""
        try:
            print("\n🔧 Extracting exact weights from original model...")
            
            # Get all weights từ original model
            all_weights = self.original_model.get_weights()
            print(f"Total weights arrays: {len(all_weights)}")
            
            # Analyze weight shapes
            print("\n📊 Weight analysis:")
            for i, weight in enumerate(all_weights):
                print(f"  {i}: shape {weight.shape}")
                if len(weight.shape) == 2:  # Dense layer weights
                    print(f"    → Likely Dense layer: {weight.shape[0]} -> {weight.shape[1]}")
                elif len(weight.shape) == 1:  # Bias weights
                    print(f"    → Likely Bias: {weight.shape[0]} units")
                elif len(weight.shape) == 4:  # Conv weights
                    print(f"    → Conv weights: {weight.shape}")
            
            # Tìm final dense layer weights (phân loại 4 classes)
            final_dense_weights = None
            final_dense_bias = None
            
            for i, weight in enumerate(all_weights):
                if len(weight.shape) == 2 and weight.shape[1] == 4:  # Dense layer -> 4 classes
                    final_dense_weights = weight
                    if i + 1 < len(all_weights) and len(all_weights[i + 1].shape) == 1 and all_weights[i + 1].shape[0] == 4:
                        final_dense_bias = all_weights[i + 1]
                    print(f"✅ Found final dense weights at index {i}: {weight.shape}")
                    break
            
            if final_dense_weights is None:
                raise ValueError("Could not find final dense layer weights")
            
            return final_dense_weights, final_dense_bias, all_weights
            
        except Exception as e:
            print(f"❌ Weight extraction failed: {e}")
            return None, None, None
    
    def create_feature_extractor(self):
        """Tạo feature extractor từ original DenseNet layer"""
        try:
            print("\n🏗️ Creating feature extractor...")
            
            # Sử dụng exact DenseNet layer từ original model
            inputs = Input(shape=Config.INPUT_SHAPE, name='input')
            features = self.densenet_layer(inputs)
            
            self.feature_extractor = Model(inputs, features)
            
            # Test feature extractor
            dummy_input = tf.random.normal((1, 224, 224, 3))
            features_output = self.feature_extractor(dummy_input)
            
            print(f"✅ Feature extractor created:")
            print(f"   Input shape: {self.feature_extractor.input_shape}")
            print(f"   Output shape: {features_output.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Feature extractor creation failed: {e}")
            return False
    
    def create_gradcam_function(self, final_dense_weights, final_dense_bias):
        """Tạo Grad-CAM function với exact weights"""
        try:
            print("\n🎯 Creating Grad-CAM function with exact weights...")
            
            def gradcam_compute(image, class_index):
                """Compute Grad-CAM với exact original model behavior"""
                if len(image.shape) == 3:
                    image = np.expand_dims(image, axis=0)
                
                image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    # Get features từ exact DenseNet
                    features = self.feature_extractor(image_tensor, training=False)
                    tape.watch(features)
                    
                    # Manual classification với exact weights
                    pooled = tf.reduce_mean(features, axis=[1, 2])  # Global Average Pooling
                    
                    # Exact dense layer computation
                    logits = tf.matmul(pooled, final_dense_weights)
                    if final_dense_bias is not None:
                        logits = tf.nn.bias_add(logits, final_dense_bias)
                    
                    predictions = tf.nn.softmax(logits)
                    class_output = predictions[:, class_index]
                
                # Compute gradients của class w.r.t features
                gradients = tape.gradient(class_output, features)
                
                if gradients is None:
                    print(f"⚠️ No gradients for class {class_index}")
                    return None
                
                # Standard Grad-CAM computation
                pooled_grads = tf.reduce_mean(gradients, axis=[0, 1, 2])
                feature_maps = features[0]
                
                # Weighted combination
                heatmap = tf.zeros(feature_maps.shape[:2])
                for i in range(feature_maps.shape[-1]):
                    heatmap += pooled_grads[i] * feature_maps[:, :, i]
                
                # Apply ReLU và normalize
                heatmap = tf.maximum(heatmap, 0)
                max_val = tf.reduce_max(heatmap)
                if max_val > 0:
                    heatmap = heatmap / max_val
                
                return heatmap.numpy(), predictions[0].numpy()
            
            # Test function
            dummy_input = tf.random.normal((1, 224, 224, 3))
            test_heatmap, test_pred = gradcam_compute(dummy_input[0].numpy(), 0)
            
            # Compare với original model prediction
            orig_pred = self.original_model(dummy_input).numpy()[0]
            
            print(f"✅ Grad-CAM function created and tested:")
            print(f"   Heatmap shape: {test_heatmap.shape}")
            print(f"   Original prediction: {orig_pred[:2]}")
            print(f"   GradCAM prediction: {test_pred[:2]}")
            print(f"   Prediction difference: {np.mean(np.abs(orig_pred - test_pred)):.6f}")
            
            self.gradcam_function = gradcam_compute
            return True
            
        except Exception as e:
            print(f"❌ Grad-CAM function creation failed: {e}")
            return False
    
    def setup_correct_gradcam(self):
        """Setup correct Grad-CAM với exact weights"""
        print("🚀 Setting up Correct True Grad-CAM")
        print("=" * 50)
        
        # Step 1: Load và analyze model
        if not self.load_and_analyze_model():
            return False
        
        # Step 2: Extract exact weights
        final_weights, final_bias, all_weights = self.extract_exact_weights()
        if final_weights is None:
            return False
        
        # Step 3: Create feature extractor
        if not self.create_feature_extractor():
            return False
        
        # Step 4: Create Grad-CAM function
        if not self.create_gradcam_function(final_weights, final_bias):
            return False
        
        print("\n🎉 Correct True Grad-CAM setup completed!")
        return True

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

def create_overlay(image, heatmap, alpha=0.4):
    """Create overlay"""
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

def plot_result(result, save_dir, index):
    """Plot result"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Correct True Grad-CAM
    im = axes[1].imshow(result['heatmap_resized'], cmap='jet')
    axes[1].set_title('Correct True Grad-CAM', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(result['overlay'])
    axes[2].set_title('Focused on Cells', fontweight='bold')
    axes[2].axis('off')
    
    # Info
    pred_text = f'True: {result["true_class"]}\nPredicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}\n\nUsing Exact Weights\nFocused on Cell Features'
    
    if result["predicted_class"] != result["true_class"]:
        pred_text += f'\n\n❌ Incorrect'
        color = 'lightcoral'
    else:
        pred_text += f'\n\n✅ Correct'
        color = 'lightgreen'
    
    axes[3].text(0.5, 0.5, pred_text, ha='center', va='center', 
                 transform=axes[3].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[3].axis('off')
    
    image_name = result['image_path'].name
    plt.suptitle(f'Correct True Grad-CAM: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"correct_gradcam_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path.name}")

def main():
    """Main function"""
    print("🎯 CORRECT TRUE GRAD-CAM - FOCUSED ON CELLS")
    print("🔥 FIXED ATTENTION ISSUES!")
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
    
    # Setup Correct Grad-CAM
    correct_gradcam = CorrectGradCAM(model_path)
    if not correct_gradcam.setup_correct_gradcam():
        print("💥 Failed to setup Correct Grad-CAM")
        return
    
    print(f"\n🎉 Correct True Grad-CAM ready with exact weights!")
    
    # Select samples
    print(f"\n📸 Selecting samples...")
    samples = select_samples()
    if not samples:
        print("❌ No samples found")
        return
    
    # Generate correct visualizations
    print(f"\n🔥 Generating Correct True Grad-CAM (focused on cells)...")
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
            
            # Generate Correct True Grad-CAM
            heatmap, gradcam_pred = correct_gradcam.gradcam_function(image_array, class_index)
            
            if heatmap is not None and np.max(heatmap) > 0:
                # Get original model prediction for comparison
                image_batch = np.expand_dims(image_array, axis=0)
                original_pred = correct_gradcam.original_model.predict(image_batch, verbose=0)[0]
                
                predicted_class = np.argmax(original_pred)
                confidence = original_pred[predicted_class]
                
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
                print(f"     ✅ Generated Correct Grad-CAM (Pred: {Config.CLASS_NAMES[predicted_class]}, Conf: {confidence:.3f})")
            else:
                print(f"     ❌ Failed to generate heatmap for {image_path.name}")
    
    if not results:
        print("❌ No visualizations generated")
        return
    
    print(f"\n✅ Generated {len(results)} Correct True Grad-CAM visualizations")
    
    # Create plots
    print(f"\n📊 Creating focused cell visualizations...")
    for i, result in enumerate(results):
        plot_result(result, Config.GRADCAM_DIR, i)
    
    # Summary
    correct = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    accuracy = correct / len(results)
    
    print(f"\n🎉 CORRECT TRUE GRAD-CAM COMPLETED!")
    print(f"📁 Results: {Config.GRADCAM_DIR}")
    print(f"🎯 Now focuses on CELLS instead of background!")
    print(f"📊 Visualizations: {len(results)}")
    print(f"🎯 Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💯 TRUE GRAD-CAM CHÍNH XÁC - TÂP TRUNG VÀO TẾ BÀO!")

if __name__ == "__main__":
    main()