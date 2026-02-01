#!/usr/bin/env python3
"""
GradCAM Visualization cho DenseNet121 Transfer Learning Model
=============================================================

Script này tạo GradCAM heatmaps cho model DenseNet121 đã train.
Được xây dựng dựa trên densenet.py để đảm bảo tính nhất quán.

Cấu trúc model (từ densenet.py):
- Sequential([DenseNet121, GlobalAveragePooling2D, Dense(256), BN, Dropout, 
              Dense(128), BN, Dropout, Dense(4)])
- Preprocessing: rescale=1./255 (KHÔNG dùng preprocess_input của DenseNet)
- Input shape: (224, 224, 3)
- Classes: ['Benign', 'Early', 'Pre', 'Pro']

Author: GradCAM Implementation
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# ============================================================================
# STEP 0: CONFIGURATION (Giống densenet.py)
# ============================================================================

class GradCAMConfig:
    """Configuration matching densenet.py exactly"""
    # Model settings
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    NUM_CLASSES = 4
    
    # Paths
    PROJECT_DIR = Path(__file__).parent
    MODEL_PATH = PROJECT_DIR / "models" / "best_DenseNet121_Transfer_4Class_phase2.h5"
    DATA_DIR = PROJECT_DIR / "data"
    OUTPUT_DIR = PROJECT_DIR / "gradcam_densenet121_output"
    
    # GradCAM settings
    TARGET_LAYER_NAME = "conv5_block16_2_conv"  # Last conv layer in DenseNet121
    
    # Visualization settings
    ALPHA = 0.4  # Heatmap overlay transparency


def print_step(step_num, title):
    """Print step header for verification"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")


def print_output(label, value):
    """Print output for verification"""
    print(f"  ✓ {label}: {value}")


# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================

def load_model_safe(model_path):
    """
    Load model với fallback mechanisms.
    
    Output verification:
    - Model summary
    - Layer names
    - Input/output shapes
    """
    print_step(1, "LOAD MODEL")
    
    model_path = str(model_path)
    print_output("Model path", model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Try loading with different methods
    model = None
    load_method = None
    
    # Method 1: Standard load
    try:
        model = keras.models.load_model(model_path)
        load_method = "standard load_model()"
    except Exception as e:
        print(f"  ⚠ Standard load failed: {e}")
        
        # Method 2: Load without compile
        try:
            model = keras.models.load_model(model_path, compile=False)
            load_method = "load_model(compile=False)"
        except Exception as e2:
            print(f"  ⚠ Compile=False load failed: {e2}")
            raise RuntimeError(f"Cannot load model: {e2}")
    
    print_output("Load method", load_method)
    print_output("Model name", model.name)
    print_output("Input shape", model.input_shape)
    print_output("Output shape", model.output_shape)
    
    # Print layer structure
    print("\n  Layer structure:")
    for i, layer in enumerate(model.layers):
        trainable_status = "trainable" if layer.trainable else "frozen"
        if hasattr(layer, 'layers'):
            print(f"    [{i}] {layer.name} ({layer.__class__.__name__}) - {len(layer.layers)} sub-layers, {trainable_status}")
        else:
            print(f"    [{i}] {layer.name} ({layer.__class__.__name__}), {trainable_status}")
    
    return model


# ============================================================================
# STEP 2: ACCESS TARGET LAYER
# ============================================================================

def get_target_layer(model, target_layer_name):
    """
    Truy cập target convolutional layer cho GradCAM.
    
    Model structure: Sequential([DenseNet121, ...])
    -> model.layers[0] là DenseNet121 base model
    -> Trong đó có conv5_block16_2_conv
    
    Output verification:
    - Layer name
    - Layer output shape
    """
    print_step(2, "ACCESS TARGET LAYER")
    
    # Get base model (DenseNet121 is the first layer in Sequential)
    base_model = model.layers[0]
    print_output("Base model name", base_model.name)
    print_output("Base model type", base_model.__class__.__name__)
    
    # List available conv layers in last block for reference
    print("\n  Available conv layers in conv5_block16:")
    for layer in base_model.layers:
        if 'conv5_block16' in layer.name:
            print(f"    - {layer.name}: {layer.output_shape}")
    
    # Get target layer
    try:
        target_layer = base_model.get_layer(target_layer_name)
    except ValueError:
        # Try alternative layer names
        alternatives = ['conv5_block16_concat', 'conv5_block15_2_conv', 'bn']
        for alt in alternatives:
            try:
                target_layer = base_model.get_layer(alt)
                print(f"  ⚠ Using alternative layer: {alt}")
                break
            except:
                continue
        else:
            raise ValueError(f"Cannot find target layer: {target_layer_name}")
    
    print_output("Target layer name", target_layer.name)
    print_output("Target layer type", target_layer.__class__.__name__)
    print_output("Target layer output shape", target_layer.output_shape)
    
    return base_model, target_layer


# ============================================================================
# STEP 3: CREATE GRADIENT MODEL
# ============================================================================

def create_grad_model(model, base_model, target_layer):
    """
    Tạo gradient model với dual outputs:
    - Output 1: Feature maps từ target conv layer
    - Output 2: Predictions từ model
    
    Output verification:
    - Gradient model structure
    - Output shapes
    """
    print_step(3, "CREATE GRADIENT MODEL")
    
    # Enable gradient computation for all layers
    print("  Enabling gradients for all layers...")
    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                sub_layer.trainable = True
    
    # Create gradient model
    # Input: model input
    # Outputs: [target_layer_output, model_predictions]
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )
    
    print_output("Grad model inputs", grad_model.input_shape)
    print_output("Grad model output 1 (conv features)", grad_model.output_shape[0])
    print_output("Grad model output 2 (predictions)", grad_model.output_shape[1])
    
    # Verify outputs
    print("\n  Gradient model summary:")
    print(f"    Input: {grad_model.input_shape}")
    print(f"    Output[0] (features): {grad_model.output_shape[0]} - for heatmap")
    print(f"    Output[1] (predictions): {grad_model.output_shape[1]} - for class score")
    
    return grad_model


# ============================================================================
# STEP 4: PREPROCESSING (Giống densenet.py)
# ============================================================================

def preprocess_image(image_path):
    """
    Preprocess image GIỐNG HỆT với densenet.py training:
    - Resize to (224, 224)
    - Rescale by 1./255 (NOT using preprocess_input!)
    
    Output verification:
    - Image shape
    - Value range (should be 0-1)
    """
    print_step(4, "PREPROCESSING")
    
    print_output("Image path", image_path)
    
    # Load image
    img = Image.open(image_path)
    original_size = img.size
    print_output("Original size", original_size)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
        print_output("Converted to", "RGB")
    
    # Resize to model input size
    img = img.resize((GradCAMConfig.INPUT_SHAPE[0], GradCAMConfig.INPUT_SHAPE[1]))
    print_output("Resized to", img.size)
    
    # Convert to array
    img_array = np.array(img, dtype=np.float32)
    
    # CRITICAL: Use same preprocessing as training (rescale=1./255)
    # DO NOT use keras.applications.densenet.preprocess_input()
    img_array = img_array / 255.0
    
    print_output("Array shape", img_array.shape)
    print_output("Value range", f"[{img_array.min():.4f}, {img_array.max():.4f}]")
    print_output("Mean value", f"{img_array.mean():.4f}")
    
    # Verify range is correct (should be 0-1)
    if img_array.min() < -0.1 or img_array.max() > 1.1:
        print("  ⚠ WARNING: Value range seems incorrect!")
    else:
        print("  ✓ Value range is correct (0-1)")
    
    # Add batch dimension
    img_tensor = np.expand_dims(img_array, axis=0)
    print_output("Tensor shape (with batch)", img_tensor.shape)
    
    return img_tensor, img_array


# ============================================================================
# STEP 5: COMPUTE GRADCAM HEATMAP
# ============================================================================

def compute_gradcam(grad_model, img_tensor, class_index=None):
    """
    Compute GradCAM heatmap.
    
    Algorithm:
    1. Forward pass to get conv features and predictions
    2. Get gradients of class score w.r.t. conv features
    3. Global average pool gradients to get weights
    4. Weighted sum of feature maps
    5. ReLU and normalize
    
    Output verification:
    - Predicted class
    - Confidence score
    - Heatmap statistics
    """
    print_step(5, "COMPUTE GRADCAM HEATMAP")
    
    # Forward pass with gradient tape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        
        # Get predicted class if not specified
        if class_index is None:
            class_index = tf.argmax(predictions[0]).numpy()
        
        class_score = predictions[:, class_index]
    
    # Get prediction info
    predicted_class = GradCAMConfig.CLASS_NAMES[class_index]
    confidence = predictions[0, class_index].numpy()
    
    print_output("Predicted class index", class_index)
    print_output("Predicted class name", predicted_class)
    print_output("Confidence", f"{confidence:.4f} ({confidence*100:.2f}%)")
    
    # Print all class probabilities
    print("\n  All class probabilities:")
    for i, class_name in enumerate(GradCAMConfig.CLASS_NAMES):
        prob = predictions[0, i].numpy()
        marker = " <-- predicted" if i == class_index else ""
        print(f"    {class_name}: {prob:.4f} ({prob*100:.2f}%){marker}")
    
    # Compute gradients
    grads = tape.gradient(class_score, conv_outputs)
    
    if grads is None:
        print("  ⚠ ERROR: Gradients are None!")
        print("  Possible causes:")
        print("    - Layers not trainable")
        print("    - Disconnected graph")
        return None, class_index, confidence
    
    print_output("Gradients shape", grads.shape)
    print_output("Gradients range", f"[{grads.numpy().min():.6f}, {grads.numpy().max():.6f}]")
    
    # Global average pooling of gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    print_output("Pooled gradients shape", pooled_grads.shape)
    
    # Get conv output (remove batch dimension)
    conv_outputs = conv_outputs[0]
    print_output("Conv outputs shape", conv_outputs.shape)
    
    # Weighted combination of feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU - keep only positive contributions
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize to [0, 1]
    heatmap_max = tf.reduce_max(heatmap)
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max
    
    heatmap = heatmap.numpy()
    
    print_output("Heatmap shape", heatmap.shape)
    print_output("Heatmap range", f"[{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print_output("Heatmap mean", f"{heatmap.mean():.4f}")
    print_output("Heatmap non-zero ratio", f"{(heatmap > 0.1).sum() / heatmap.size:.4f}")
    
    return heatmap, class_index, confidence


# ============================================================================
# STEP 6: VISUALIZATION & SAVE
# ============================================================================

def create_visualization(original_image, heatmap, class_index, confidence, output_path, image_name):
    """
    Create and save GradCAM visualization.
    
    Outputs:
    1. Original image
    2. Heatmap (jet colormap)
    3. Overlay (original + heatmap)
    4. Combined figure
    
    Output verification:
    - Saved file paths
    """
    print_step(6, "VISUALIZATION & SAVE")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    print_output("Output directory", output_path)
    
    # Get class name
    class_name = GradCAMConfig.CLASS_NAMES[class_index]
    
    # Resize heatmap to match original image
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (original_image.shape[1], original_image.shape[0]),
        Image.BILINEAR
    )) / 255.0
    
    print_output("Resized heatmap shape", heatmap_resized.shape)
    
    # Apply colormap to heatmap
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Remove alpha channel
    
    # Create overlay
    overlay = original_image * (1 - GradCAMConfig.ALPHA) + heatmap_colored * GradCAMConfig.ALPHA
    overlay = np.clip(overlay, 0, 1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPredicted: {class_name} ({confidence*100:.1f}%)', fontsize=12)
    axes[2].axis('off')
    
    # Main title
    fig.suptitle(f'GradCAM Visualization - {image_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save combined figure
    combined_path = os.path.join(output_path, f"{image_name}_gradcam_combined.png")
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_output("Saved combined", combined_path)
    
    # Save individual images
    # Original
    original_path = os.path.join(output_path, f"{image_name}_original.png")
    plt.imsave(original_path, original_image)
    print_output("Saved original", original_path)
    
    # Heatmap
    heatmap_path = os.path.join(output_path, f"{image_name}_heatmap.png")
    plt.imsave(heatmap_path, heatmap_resized, cmap='jet')
    print_output("Saved heatmap", heatmap_path)
    
    # Overlay
    overlay_path = os.path.join(output_path, f"{image_name}_overlay.png")
    plt.imsave(overlay_path, overlay)
    print_output("Saved overlay", overlay_path)
    
    print(f"\n  ✓ All visualizations saved successfully!")
    
    return {
        'combined': combined_path,
        'original': original_path,
        'heatmap': heatmap_path,
        'overlay': overlay_path
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_gradcam(image_path, model_path=None, output_dir=None, class_index=None):
    """
    Run complete GradCAM pipeline.
    
    Args:
        image_path: Path to input image
        model_path: Path to model (default: from config)
        output_dir: Output directory (default: from config)
        class_index: Target class index (default: predicted class)
    
    Returns:
        Dictionary with results and file paths
    """
    print("\n" + "="*60)
    print("🔬 GradCAM for DenseNet121 Transfer Learning Model")
    print("="*60)
    
    # Use defaults from config
    if model_path is None:
        model_path = GradCAMConfig.MODEL_PATH
    if output_dir is None:
        output_dir = GradCAMConfig.OUTPUT_DIR
    
    # Get image name for output files
    image_name = Path(image_path).stem
    
    # STEP 1: Load model
    model = load_model_safe(model_path)
    
    # STEP 2: Get target layer
    base_model, target_layer = get_target_layer(model, GradCAMConfig.TARGET_LAYER_NAME)
    
    # STEP 3: Create gradient model
    grad_model = create_grad_model(model, base_model, target_layer)
    
    # STEP 4: Preprocess image
    img_tensor, img_array = preprocess_image(image_path)
    
    # STEP 5: Compute GradCAM
    heatmap, pred_class_index, confidence = compute_gradcam(grad_model, img_tensor, class_index)
    
    if heatmap is None:
        print("\n❌ GradCAM computation failed!")
        return None
    
    # STEP 6: Visualize and save
    saved_files = create_visualization(
        img_array, heatmap, pred_class_index, confidence, 
        output_dir, image_name
    )
    
    # Summary
    print("\n" + "="*60)
    print("✅ GRADCAM COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"  Image: {image_path}")
    print(f"  Predicted: {GradCAMConfig.CLASS_NAMES[pred_class_index]} ({confidence*100:.2f}%)")
    print(f"  Output: {output_dir}")
    print("="*60 + "\n")
    
    return {
        'image_path': image_path,
        'predicted_class': GradCAMConfig.CLASS_NAMES[pred_class_index],
        'predicted_index': pred_class_index,
        'confidence': confidence,
        'heatmap': heatmap,
        'saved_files': saved_files
    }


def process_multiple_images(image_paths, model_path=None, output_dir=None):
    """Process multiple images with the same model."""
    results = []
    
    # Load model once
    if model_path is None:
        model_path = GradCAMConfig.MODEL_PATH
    
    print("\n" + "="*60)
    print(f"🔬 Processing {len(image_paths)} images...")
    print("="*60)
    
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path}")
        result = run_gradcam(image_path, model_path, output_dir)
        if result:
            results.append(result)
    
    print(f"\n✅ Processed {len(results)}/{len(image_paths)} images successfully")
    return results


def find_test_images(num_per_class=1):
    """Find test images from each class for demo."""
    test_images = []
    
    for class_name in GradCAMConfig.CLASS_NAMES:
        class_dir = GradCAMConfig.DATA_DIR / "test" / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            test_images.extend(images[:num_per_class])
    
    return test_images


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GradCAM for DenseNet121')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run demo with test images')
    parser.add_argument('--class-index', type=int, help='Target class index (0-3)')
    
    args = parser.parse_args()
    
    if args.demo:
        # Demo mode: process one image from each class
        print("🎯 Running demo mode...")
        test_images = find_test_images(num_per_class=1)
        
        if not test_images:
            print("❌ No test images found in data/test/")
            sys.exit(1)
        
        print(f"Found {len(test_images)} test images")
        for img in test_images:
            print(f"  - {img}")
        
        results = process_multiple_images(
            [str(img) for img in test_images],
            args.model,
            args.output
        )
        
    elif args.image:
        # Single image mode
        result = run_gradcam(
            args.image,
            args.model,
            args.output,
            args.class_index
        )
        
    else:
        # No arguments: run with a sample image
        print("Usage:")
        print("  python gradcam_densenet121.py --demo          # Run with test images")
        print("  python gradcam_densenet121.py --image PATH    # Run with specific image")
        print("\nRunning demo mode by default...")
        
        test_images = find_test_images(num_per_class=1)
        if test_images:
            result = run_gradcam(str(test_images[0]))
        else:
            print("❌ No test images found. Please specify --image PATH")
