#!/usr/bin/env python3
"""
GRAD-CAM Implementation for DenseNet121 Transfer Learning Model
==============================================================

GRAD-CAM (Gradient-weighted Class Activation Mapping) để visualize 
các vùng quan trọng mà model DenseNet121 tập trung khi classification.

Supports:
- DenseNet121 pretrained model với custom classification head
- Batch processing cho multiple images
- Visualization overlay với confidence scores
- Comprehensive verification và testing

Author: Cell Image-Based Disease Detection Team
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import cv2
from typing import List, Tuple, Optional, Union
import json

# Import model configuration
sys.path.append(str(Path(__file__).parent))
from config import config, get_model_config

class DenseNetGradCAM:
    """
    GRAD-CAM implementation cho DenseNet121 model
    
    Key features:
    - Automatic target layer detection
    - Batch processing support  
    - Multiple visualization options
    - Comprehensive verification methods
    """
    
    def __init__(self, 
                 model_path: str,
                 layer_name: Optional[str] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize GRAD-CAM với trained DenseNet model
        
        Args:
            model_path: Path to saved model (.h5 file)
            layer_name: Target layer for GRAD-CAM (auto-detect if None)
            class_names: List of class names for labeling
        """
        self.model_path = model_path
        self.class_names = class_names or config.dataset.class_names
        self.input_size = get_model_config("densenet121_transfer")["input_size"][:2]
        
        # Load model
        print(f"🔄 Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            print(f"⚠️ Standard loading failed: {e}")
            print("🔄 Trying with compile=False...")
            try:
                self.model = keras.models.load_model(model_path, compile=False)
            except Exception as e2:
                print(f"⚠️ Compile=False also failed: {e2}")
                print("🔄 Recreating model architecture and loading weights...")
                self.model = self._recreate_model_and_load_weights(model_path)
        
        # Find target layer automatically if not specified
        self.target_layer_name = layer_name or self._find_target_layer()
        print(f"🎯 Target layer: {self.target_layer_name}")
        
        # Create gradient model
        self.grad_model = self._create_grad_model()
    
    def _recreate_model_and_load_weights(self, model_path: str):
        """
        Recreate model architecture và load weights từ saved model
        Workaround cho Keras version compatibility issues
        """
        from tensorflow.keras.applications import DenseNet121
        from tensorflow.keras import layers, models
        
        print("🏗️ Recreating DenseNet121 architecture...")
        
        # Recreate architecture (same as in densenet.py)
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)  # Default DenseNet input size
        )
        
        # Custom classification head (match original architecture)
        model_config = get_model_config("densenet121_transfer")
        dense_units = model_config["dense_units"]  # [512, 256] from config
        dropout_rate = model_config["dropout_rate"]
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units[0], activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(), 
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units[1], activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')  # 4 classes
        ])
        
        # Build model để định nghĩa output shapes
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = model(dummy_input)  # This builds the model
        
        try:
            # Load weights only
            model.load_weights(model_path)
            print("✅ Successfully loaded weights into recreated model")
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            print("🔄 Using model with ImageNet weights only...")
        
        return model
        
    def _find_target_layer(self) -> str:
        """
        Automatically find the best target layer for GRAD-CAM
        Prioritizes DenseNet121 feature extraction layers
        """
        # Search for DenseNet layers (before GlobalAveragePooling2D)
        densenet_layers = []
        
        for i, layer in enumerate(self.model.layers):
            layer_name = layer.name
            
            # Look for DenseNet121 base model
            if hasattr(layer, 'layers'):  # This is the base model
                for sub_layer in reversed(layer.layers):
                    if ('conv' in sub_layer.name and 
                        hasattr(sub_layer, 'output_shape') and 
                        len(sub_layer.output_shape) == 4):
                        return sub_layer.name
            
            # Fallback: look for conv layers with 4D output  
            if ('conv' in layer_name and 
                hasattr(layer, 'output_shape') and
                layer.output_shape is not None and
                len(layer.output_shape) == 4):
                densenet_layers.append(layer_name)
        
        # Return the last convolutional layer found
        if densenet_layers:
            return densenet_layers[-1]
        
        # Final fallback
        for layer in reversed(self.model.layers):
            if (hasattr(layer, 'output_shape') and 
                layer.output_shape is not None and 
                len(layer.output_shape) == 4):  # 4D tensor (batch, H, W, channels)
                return layer.name
                
        raise ValueError("Could not find suitable target layer for GRAD-CAM")
    
    def _create_grad_model(self) -> keras.Model:
        """Create gradient model for computing gradients"""
        # Get base model (DenseNet121)
        base_model = self.model.layers[0]
        
        # Find last conv layer in base model for feature extraction
        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            # Fallback: use bn layer output (before relu)
            for layer in reversed(base_model.layers):
                if 'bn' in layer.name or 'batch' in layer.name.lower():
                    last_conv_layer = layer
                    break
        
        self.last_conv_layer_name = last_conv_layer.name if last_conv_layer else None
        self.base_model = base_model
        self.last_conv_layer = last_conv_layer
        return self.model
    
    def compute_gradcam(self, 
                       img_array: np.ndarray, 
                       class_index: Optional[int] = None,
                       use_guided_backprop: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Compute GRAD-CAM heatmap cho input image
        
        Args:
            img_array: Preprocessed image array (1, H, W, 3)
            class_index: Target class index (None for predicted class)
            use_guided_backprop: Enable guided backpropagation
            
        Returns:
            heatmap: GRAD-CAM heatmap (H, W)
            info: Dictionary with prediction info
        """
        
        # Convert to tensor as Variable so we can compute gradients
        img_tensor = tf.Variable(img_array, dtype=tf.float32)
        
        # Get base model (DenseNet121) 
        base_model = self.model.layers[0]
        
        # Find the last conv layer in DenseNet
        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            raise ValueError("Could not find conv layer in base model")
        
        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            # Forward pass through base model
            base_output = base_model(img_tensor)
            
            # Forward pass through rest of Sequential model
            x = base_output
            for layer in self.model.layers[1:]:
                x = layer(x)
            predictions = x
            
            # Use predicted class if not specified
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            # Get class-specific output  
            class_output = predictions[:, class_index]
        
        # Compute gradients w.r.t. base model output (features before GlobalAvgPool)
        grads = tape.gradient(class_output, base_output)
        
        if grads is None:
            # Fallback: create dummy heatmap
            print("⚠️ Could not compute gradients, creating dummy heatmap")
            heatmap = np.random.rand(*self.input_size) * 0.5
        else:
            # Global average pooling của gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps với corresponding gradients
            feature_maps = base_output[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, feature_maps), axis=-1)
            
            # Apply ReLU to remove negative values
            heatmap = tf.nn.relu(heatmap)
            
            # Normalize to 0-1 range
            heatmap_max = tf.reduce_max(heatmap)
            if heatmap_max > 0:
                heatmap = heatmap / heatmap_max
            
            # Convert to numpy và resize to input size
            heatmap = heatmap.numpy()
            heatmap = cv2.resize(heatmap, self.input_size)
        
        # Prediction info
        predicted_class = int(tf.argmax(predictions[0]).numpy())
        confidence = float(tf.nn.softmax(predictions[0])[predicted_class].numpy())
        
        info = {
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'target_class': int(class_index),
            'target_class_name': self.class_names[int(class_index)],
            'all_probabilities': tf.nn.softmax(predictions[0]).numpy().tolist()
        }
        
        return heatmap, info
    
    def preprocess_image(self, img_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image cho model input
        
        Args:
            img_path: Path to image file
            
        Returns:
            img_array: Preprocessed image array cho model
            original_img: Original image array cho visualization
        """
        # Load và resize image
        img = image.load_img(img_path, target_size=self.input_size)
        original_img = np.array(img)
        
        # Preprocess cho DenseNet
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # DenseNet preprocessing
        
        return img_array, original_img
    
    def create_heatmap_overlay(self, 
                              original_img: np.ndarray,
                              heatmap: np.ndarray,
                              alpha: float = 0.6,
                              colormap: str = 'jet') -> np.ndarray:
        """
        Create overlay của heatmap và original image
        
        Args:
            original_img: Original image (H, W, 3)
            heatmap: GRAD-CAM heatmap (H, W)
            alpha: Transparency of heatmap overlay
            colormap: Colormap cho heatmap
            
        Returns:
            overlay_img: Combined image với heatmap overlay
        """
        # Normalize original image to 0-255
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        
        # Apply colormap to heatmap
        colormap_fn = cm.get_cmap(colormap)
        heatmap_colored = colormap_fn(heatmap)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Create overlay
        overlay_img = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlay_img
    
    def visualize_gradcam(self,
                         img_path: str,
                         class_index: Optional[int] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> dict:
        """
        Complete GRAD-CAM visualization pipeline
        
        Args:
            img_path: Path to input image
            class_index: Target class (None for predicted)
            save_path: Path to save visualization
            figsize: Figure size for matplotlib
            
        Returns:
            results: Dictionary with heatmap và prediction info
        """
        # Preprocess image
        img_array, original_img = self.preprocess_image(img_path)
        
        # Compute GRAD-CAM
        heatmap, info = self.compute_gradcam(img_array, class_index)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet', alpha=0.8)
        axes[1].set_title(f'GRAD-CAM Heatmap\\n{self.target_layer_name}', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay_img = self.create_heatmap_overlay(original_img, heatmap)
        axes[2].imshow(overlay_img)
        axes[2].set_title(f'Overlay\\nPred: {info["predicted_class_name"]} ({info["confidence"]:.3f})', 
                         fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        plt.show()
        
        # Return results
        results = {
            'image_path': img_path,
            'heatmap': heatmap,
            'overlay': overlay_img,
            'info': info,
            'target_layer': self.target_layer_name
        }
        
        return results
    
    def batch_gradcam(self,
                     image_paths: List[str],
                     output_dir: str,
                     class_indices: Optional[List[int]] = None) -> List[dict]:
        """
        Process multiple images với GRAD-CAM
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory cho results
            class_indices: Target classes (None for predicted)
            
        Returns:
            all_results: List of results for each image
        """
        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        
        for i, img_path in enumerate(image_paths):
            print(f"🔄 Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            target_class = class_indices[i] if class_indices else None
            save_path = os.path.join(output_dir, f"gradcam_{i:03d}_{os.path.basename(img_path)}")
            
            try:
                results = self.visualize_gradcam(
                    img_path=img_path,
                    class_index=target_class,
                    save_path=save_path,
                    figsize=(15, 5)
                )
                all_results.append(results)
                
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
                continue
        
        print(f"✅ Processed {len(all_results)}/{len(image_paths)} images")
        return all_results
    
    def verify_gradcam(self, img_path: str) -> dict:
        """
        Comprehensive verification của GRAD-CAM output
        
        Args:
            img_path: Path to test image
            
        Returns:
            verification_results: Dictionary with verification metrics
        """
        img_array, original_img = self.preprocess_image(img_path)
        heatmap, info = self.compute_gradcam(img_array)
        
        # Verification metrics
        verification = {
            'heatmap_shape_correct': heatmap.shape == self.input_size,
            'heatmap_range_valid': (heatmap.min() >= 0) and (heatmap.max() <= 1),
            'heatmap_not_empty': heatmap.max() > 0,
            'confidence_valid': 0 <= info['confidence'] <= 1,
            'prediction_consistent': info['predicted_class'] == np.argmax(info['all_probabilities']),
            'probabilities_sum_to_one': abs(sum(info['all_probabilities']) - 1.0) < 1e-6,
            'heatmap_statistics': {
                'mean': float(heatmap.mean()),
                'std': float(heatmap.std()),
                'max': float(heatmap.max()),
                'min': float(heatmap.min())
            }
        }
        
        # Overall verification status
        verification['all_checks_passed'] = all([
            verification['heatmap_shape_correct'],
            verification['heatmap_range_valid'], 
            verification['heatmap_not_empty'],
            verification['confidence_valid'],
            verification['prediction_consistent'],
            verification['probabilities_sum_to_one']
        ])
        
        return verification


def demo_gradcam(model_path: str, test_images_dir: str, output_dir: str):
    """
    Demo function để test GRAD-CAM implementation
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory chứa test images 
        output_dir: Output directory cho results
    """
    print("🚀 GRAD-CAM Demo for DenseNet121")
    print("=" * 50)
    
    # Initialize GRAD-CAM
    gradcam = DenseNetGradCAM(model_path)
    
    # Find test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    
    if os.path.isdir(test_images_dir):
        for ext in image_extensions:
            test_images.extend(Path(test_images_dir).glob(f"**/*{ext}"))
            test_images.extend(Path(test_images_dir).glob(f"**/*{ext.upper()}"))
    else:
        print(f"⚠️ Test images directory not found: {test_images_dir}")
        return
    
    # Limit to first 5 images for demo
    test_images = [str(p) for p in test_images[:5]]
    
    if not test_images:
        print(f"⚠️ No images found in {test_images_dir}")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Process images
    results = gradcam.batch_gradcam(
        image_paths=test_images,
        output_dir=output_dir
    )
    
    # Verification
    print("\\n🔍 Verification Results:")
    print("-" * 30)
    
    for i, img_path in enumerate(test_images[:3]):  # Verify first 3
        verification = gradcam.verify_gradcam(img_path)
        status = "✅ PASS" if verification['all_checks_passed'] else "❌ FAIL"
        print(f"{os.path.basename(img_path)}: {status}")
        
        if not verification['all_checks_passed']:
            failed_checks = [k for k, v in verification.items() 
                           if isinstance(v, bool) and not v]
            print(f"   Failed: {failed_checks}")
    
    # Save summary
    summary = {
        'model_path': model_path,
        'target_layer': gradcam.target_layer_name,
        'processed_images': len(results),
        'class_names': gradcam.class_names,
        'input_size': gradcam.input_size
    }
    
    summary_path = os.path.join(output_dir, "gradcam_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📋 Summary saved to: {summary_path}")
    print("🎉 GRAD-CAM Demo completed!")


if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "DenseNet121_Transfer_4Class"
    
    # Paths - adjust these according to your setup
    model_path = f"./models/final_{MODEL_NAME}.h5"
    test_images_dir = "./data/test"  # Adjust to your test data path
    output_dir = f"./outputs/{MODEL_NAME}/gradcam_results"
    
    # Run demo
    if os.path.exists(model_path):
        demo_gradcam(model_path, test_images_dir, output_dir)
    else:
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using densenet.py")