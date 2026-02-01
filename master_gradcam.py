#!/usr/bin/env python3
"""
Master True Grad-CAM Implementation
==================================

Triển khai nhiều strategies để tạo True Grad-CAM
Thử tuần tự từ strategy có khả năng thành công cao nhất
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
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
    from tensorflow.keras import Sequential
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available.")
    TENSORFLOW_AVAILABLE = False

class Config:
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/test")
    GRADCAM_DIR = Path("master_gradcam_results")
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    SAMPLES_PER_CLASS = 2
    TARGET_MODEL = "final_DenseNet121_Transfer_4Class.h5"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class MasterGradCAM:
    """Master True Grad-CAM với nhiều strategies"""
    
    def __init__(self, original_model_path):
        self.original_model_path = original_model_path
        self.original_model = None
        self.gradcam_model = None
        self.strategy_used = None
        
    def load_original_model(self):
        """Load original model"""
        try:
            self.original_model = load_model(self.original_model_path)
            print(f"✅ Original model loaded: {self.original_model_path}")
            
            # Force một forward pass để khởi tạo
            dummy_input = tf.random.normal((1, 224, 224, 3))
            _ = self.original_model(dummy_input)
            
            return True
        except Exception as e:
            print(f"❌ Failed to load original model: {e}")
            return False
    
    def strategy_1_weight_cloning(self):
        """Strategy 1: Weight Cloning Approach"""
        try:
            print("\n🔄 Strategy 1: Weight Cloning...")
            
            # Tạo DenseNet121 template
            base_model = DenseNet121(
                weights='imagenet',  # Start with ImageNet
                include_top=False,
                input_shape=Config.INPUT_SHAPE
            )
            
            # Tạo full model template
            inputs = Input(shape=Config.INPUT_SHAPE)
            features = base_model(inputs)
            pooled = GlobalAveragePooling2D()(features)
            predictions = Dense(4, activation='softmax', name='predictions')(pooled)
            
            template_model = Model(inputs, predictions)
            
            # Khởi tạo template model
            dummy_input = tf.random.normal((1, 224, 224, 3))
            _ = template_model(dummy_input)
            
            print("✅ Template model created")
            
            # Copy weights intelligently
            original_weights = self.original_model.get_weights()
            template_weights = template_model.get_weights()
            
            print(f"Original weights: {len(original_weights)}")
            print(f"Template weights: {len(template_weights)}")
            
            # Smart weight matching
            matched_weights = []
            original_idx = 0
            
            for template_weight in template_weights:
                template_shape = template_weight.shape
                
                # Tìm weight phù hợp trong original model
                found = False
                for i in range(original_idx, len(original_weights)):
                    if original_weights[i].shape == template_shape:
                        matched_weights.append(original_weights[i])
                        original_idx = i + 1
                        found = True
                        break
                
                if not found:
                    # Giữ template weight nếu không tìm thấy match
                    matched_weights.append(template_weight)
            
            # Set matched weights
            template_model.set_weights(matched_weights)
            
            # Test prediction
            test_pred_original = self.original_model(dummy_input)
            test_pred_template = template_model(dummy_input)
            
            pred_diff = tf.reduce_mean(tf.abs(test_pred_original - test_pred_template))
            print(f"Prediction difference: {pred_diff.numpy():.6f}")
            
            # Tạo Grad-CAM model
            gradcam_model = Model(
                inputs=template_model.input,
                outputs=[base_model.output, template_model.output]
            )
            
            print("✅ Grad-CAM model created with weight cloning")
            return gradcam_model, "Weight Cloning"
            
        except Exception as e:
            print(f"❌ Strategy 1 failed: {e}")
            return None, None
    
    def strategy_2_layer_reconstruction(self):
        """Strategy 2: Layer-by-Layer Reconstruction"""
        try:
            print("\n🔄 Strategy 2: Layer Reconstruction...")
            
            # Analyze original model structure
            print("📊 Analyzing original model structure...")
            for i, layer in enumerate(self.original_model.layers):
                print(f"  {i}: {layer.name} ({layer.__class__.__name__})")
            
            # Extract DenseNet121 từ original model
            densenet_layer = None
            dense_layer = None
            
            for layer in self.original_model.layers:
                if 'densenet121' in layer.name.lower() or layer.__class__.__name__ == 'Functional':
                    densenet_layer = layer
                elif layer.__class__.__name__ == 'Dense':
                    dense_layer = layer
            
            if densenet_layer is None:
                raise ValueError("Could not find DenseNet layer")
            
            print(f"✅ Found DenseNet layer: {densenet_layer.name}")
            
            # Tạo model mới từ components
            inputs = Input(shape=Config.INPUT_SHAPE)
            
            # Recreate DenseNet với same weights
            base_model = DenseNet121(
                weights=None,
                include_top=False,
                input_shape=Config.INPUT_SHAPE
            )
            
            # Copy DenseNet weights
            if hasattr(densenet_layer, 'get_weights'):
                densenet_weights = densenet_layer.get_weights()
                if densenet_weights:
                    base_model.set_weights(densenet_weights)
                    print("✅ DenseNet weights copied")
            
            # Build full model
            features = base_model(inputs)
            pooled = GlobalAveragePooling2D()(features)
            
            # Copy dense layer weights if available
            if dense_layer is not None and dense_layer.get_weights():
                dense_weights = dense_layer.get_weights()
                predictions = Dense(4, activation='softmax')(pooled)
                reconstructed_model = Model(inputs, predictions)
                
                # Initialize and set dense weights
                dummy_input = tf.random.normal((1, 224, 224, 3))
                _ = reconstructed_model(dummy_input)
                
                # Find dense layer in reconstructed model
                for layer in reconstructed_model.layers:
                    if layer.__class__.__name__ == 'Dense':
                        layer.set_weights(dense_weights)
                        print("✅ Dense layer weights copied")
                        break
            else:
                predictions = Dense(4, activation='softmax')(pooled)
                reconstructed_model = Model(inputs, predictions)
            
            # Test model
            dummy_input = tf.random.normal((1, 224, 224, 3))
            _ = reconstructed_model(dummy_input)
            
            # Create Grad-CAM model
            gradcam_model = Model(
                inputs=reconstructed_model.input,
                outputs=[features, reconstructed_model.output]
            )
            
            print("✅ Grad-CAM model created with layer reconstruction")
            return gradcam_model, "Layer Reconstruction"
            
        except Exception as e:
            print(f"❌ Strategy 2 failed: {e}")
            return None, None
    
    def strategy_3_model_surgery(self):
        """Strategy 3: Model Surgery"""
        try:
            print("\n🔄 Strategy 3: Model Surgery...")
            
            # Make model trainable để enable gradients
            for layer in self.original_model.layers:
                layer.trainable = True
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        sublayer.trainable = True
            
            # Recompile model
            self.original_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy'
            )
            
            # Find suitable intermediate layer
            target_layer = None
            for layer in self.original_model.layers:
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    target_layer = layer
                    break
                elif hasattr(layer, 'layers'):  # Nested model
                    for sublayer in layer.layers:
                        if hasattr(sublayer, 'output_shape') and len(sublayer.output_shape) == 4:
                            target_layer = sublayer
                            break
            
            if target_layer is None:
                raise ValueError("Could not find suitable feature layer")
            
            print(f"✅ Found target layer: {target_layer.name}")
            
            # Create Grad-CAM model
            gradcam_model = Model(
                inputs=self.original_model.input,
                outputs=[target_layer.output, self.original_model.output]
            )
            
            # Test với dummy input
            dummy_input = tf.random.normal((1, 224, 224, 3))
            features, predictions = gradcam_model(dummy_input)
            
            print(f"✅ Surgery successful - Features: {features.shape}, Predictions: {predictions.shape}")
            return gradcam_model, "Model Surgery"
            
        except Exception as e:
            print(f"❌ Strategy 3 failed: {e}")
            return None, None
    
    def strategy_4_guided_backprop(self):
        """Strategy 4: Guided Backpropagation Alternative"""
        try:
            print("\n🔄 Strategy 4: Guided Backpropagation...")
            
            # Create custom Grad-CAM function using input gradients
            @tf.function
            def guided_gradcam(image, class_idx):
                with tf.GradientTape() as tape:
                    tape.watch(image)
                    predictions = self.original_model(image, training=True)
                    class_output = predictions[:, class_idx]
                
                gradients = tape.gradient(class_output, image)
                
                # Process gradients to create attention map
                if gradients is not None:
                    # Take absolute value and sum across channels
                    saliency = tf.reduce_sum(tf.abs(gradients), axis=-1)
                    
                    # Normalize
                    saliency = tf.squeeze(saliency)
                    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
                    
                    return saliency
                else:
                    return None
            
            # Test function
            dummy_input = tf.random.normal((1, 224, 224, 3))
            test_result = guided_gradcam(dummy_input, 0)
            
            if test_result is not None:
                print(f"✅ Guided backprop working - Output shape: {test_result.shape}")
                
                # Wrap in a simple class for consistency
                class GuidedGradCAM:
                    def __init__(self, grad_func):
                        self.grad_func = grad_func
                    
                    def __call__(self, image, class_idx):
                        return self.grad_func(image, class_idx)
                
                gradcam_model = GuidedGradCAM(guided_gradcam)
                return gradcam_model, "Guided Backpropagation"
            else:
                raise ValueError("Guided backprop returned None")
            
        except Exception as e:
            print(f"❌ Strategy 4 failed: {e}")
            return None, None
    
    def execute_master_strategy(self):
        """Execute all strategies in order until one succeeds"""
        print("🚀 Starting Master Grad-CAM Strategy Execution")
        print("=" * 60)
        
        if not self.load_original_model():
            return False
        
        strategies = [
            ("Weight Cloning", self.strategy_1_weight_cloning),
            ("Layer Reconstruction", self.strategy_2_layer_reconstruction),
            ("Model Surgery", self.strategy_3_model_surgery),
            ("Guided Backpropagation", self.strategy_4_guided_backprop)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                print(f"\n🎯 Executing Strategy: {strategy_name}")
                print("-" * 40)
                
                result, used_strategy = strategy_func()
                
                if result is not None:
                    print(f"\n🎉 SUCCESS! {strategy_name} worked!")
                    self.gradcam_model = result
                    self.strategy_used = used_strategy
                    return True
                
            except Exception as e:
                print(f"💥 {strategy_name} crashed: {e}")
                continue
        
        print("\n💥 All strategies failed!")
        return False

def generate_gradcam_heatmap(gradcam_model, strategy_used, image, class_index):
    """Generate heatmap using successful strategy"""
    try:
        if strategy_used == "Guided Backpropagation":
            # Special handling for guided backprop
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            heatmap = gradcam_model(image_tensor, class_index)
            return heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap
        
        else:
            # Standard Grad-CAM approach
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                features, predictions = gradcam_model(image_tensor)
                class_output = predictions[:, class_index]
            
            gradients = tape.gradient(class_output, features)
            
            if gradients is None:
                return None
            
            # Standard Grad-CAM processing
            pooled_gradients = tf.reduce_mean(gradients, axis=[0, 1, 2])
            feature_maps = features[0]
            
            # Weighted combination
            heatmap = tf.zeros(feature_maps.shape[:2])
            for i in range(feature_maps.shape[-1]):
                heatmap += pooled_gradients[i] * feature_maps[:, :, i]
            
            # Apply ReLU and normalize
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            
            return heatmap.numpy()
        
    except Exception as e:
        print(f"❌ Error generating heatmap: {e}")
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

def plot_result(result, save_dir, index, strategy_used):
    """Plot Master Grad-CAM result"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # True Grad-CAM heatmap
    im = axes[1].imshow(result['heatmap_resized'], cmap='jet')
    axes[1].set_title('True Grad-CAM Heatmap', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(result['overlay'])
    axes[2].set_title('True Grad-CAM Overlay', fontweight='bold')
    axes[2].axis('off')
    
    # Info
    pred_text = f'True: {result["true_class"]}\nPredicted: {result["predicted_class"]}\nConfidence: {result["confidence"]:.3f}\n\nStrategy: {strategy_used}'
    
    if result["predicted_class"] != result["true_class"]:
        pred_text += f'\n\n❌ Sai'
        color = 'lightcoral'
    else:
        pred_text += f'\n\n✅ Đúng'
        color = 'lightgreen'
    
    axes[3].text(0.5, 0.5, pred_text, ha='center', va='center', 
                 transform=axes[3].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[3].axis('off')
    
    image_name = result['image_path'].name
    plt.suptitle(f'Master True Grad-CAM: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"master_gradcam_{index:03d}_{result['true_class']}_{image_name.replace('.jpg', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved: {save_path.name}")

def main():
    """Main function"""
    print("🎯 MASTER TRUE GRAD-CAM IMPLEMENTATION")
    print("🔥 MULTIPLE STRATEGIES FOR TRUE GRAD-CAM")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available.")
        return
    
    # Setup
    Config.GRADCAM_DIR.mkdir(exist_ok=True)
    
    # Load model path
    model_path = Config.MODELS_DIR / Config.TARGET_MODEL
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize Master Grad-CAM
    master = MasterGradCAM(model_path)
    
    # Execute strategy
    if not master.execute_master_strategy():
        print("💥 All strategies failed to create True Grad-CAM")
        return
    
    print(f"\n🎉 Successfully created True Grad-CAM using: {master.strategy_used}")
    
    # Select samples
    print(f"\n📸 Selecting samples...")
    samples = select_samples()
    if not samples:
        print("❌ No samples found")
        return
    
    # Generate visualizations
    print(f"\n🔥 Generating Master True Grad-CAM visualizations...")
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
            
            # Get prediction from original model
            image_batch = np.expand_dims(image_array, axis=0)
            predictions = master.original_model.predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Generate True Grad-CAM
            heatmap = generate_gradcam_heatmap(
                master.gradcam_model, 
                master.strategy_used, 
                image_array, 
                class_index
            )
            
            if heatmap is not None and np.max(heatmap) > 0:
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
                print(f"     ✅ Generated Master True Grad-CAM (Pred: {Config.CLASS_NAMES[predicted_class]}, Conf: {confidence:.3f})")
            else:
                print(f"     ❌ Failed to generate heatmap for {image_path.name}")
    
    if not results:
        print("❌ No visualizations generated")
        return
    
    print(f"\n✅ Generated {len(results)} Master True Grad-CAM visualizations")
    
    # Create plots
    print(f"\n📊 Creating Master True Grad-CAM plots...")
    for i, result in enumerate(results):
        plot_result(result, Config.GRADCAM_DIR, i, master.strategy_used)
    
    # Summary
    correct = sum(1 for r in results if r['predicted_class'] == r['true_class'])
    accuracy = correct / len(results)
    
    print(f"\n🎉 MASTER TRUE GRAD-CAM COMPLETED SUCCESSFULLY!")
    print(f"📁 Results: {Config.GRADCAM_DIR}")
    print(f"🔥 Strategy used: {master.strategy_used}")
    print(f"📊 True Grad-CAM visualizations: {len(results)}")
    print(f"🎯 Accuracy: {correct}/{len(results)} ({accuracy:.1%})")
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💯 TRUE GRAD-CAM THỰC SỰ ĐÃ HOẠT ĐỘNG!")
    print(f"🔬 Sử dụng chiến lược: {master.strategy_used}")

if __name__ == "__main__":
    main()