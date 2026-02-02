#!/usr/bin/env python3
"""
Grad-CAM for DenseNet121 Transfer Learning Model
=================================================

Simple and working Grad-CAM implementation for DenseNet121.

Usage:
    python gradcam_densenet121.py --model path/to/model.h5 --image path/to/image.jpg
    python gradcam_densenet121.py --model path/to/model.h5 --test-dir path/to/test/
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import warnings
warnings.filterwarnings('ignore')


class GradCAMDenseNet121:
    """Simple Grad-CAM for DenseNet121"""

    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']
    INPUT_SIZE = (224, 224)

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.base_model = None

        self._load_model()

    def _load_model(self):
        """Load model"""
        print(f"Loading model: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = keras.models.load_model(self.model_path, compile=False)

        # Tim DenseNet121 base
        for layer in self.model.layers:
            if 'densenet' in layer.name.lower():
                self.base_model = layer
                break

        if self.base_model is None:
            raise ValueError("Could not find DenseNet121 in model")

        # Build model bang cach goi voi dummy input
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.model(dummy_input, training=False)

        print(f"Model loaded: {self.model.name}")
        print(f"Base model: {self.base_model.name}")

    def preprocess_image(self, image_path):
        """Preprocess image"""
        image = load_img(image_path, target_size=self.INPUT_SIZE)
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, image

    def generate_gradcam(self, image_array, class_index=None):
        """Generate Grad-CAM heatmap using direct computation"""

        image_tensor = tf.cast(image_array, tf.float32)

        # Forward pass voi GradientTape
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)

            # Manual forward pass de lay intermediate output
            x = image_tensor

            # Qua base model (DenseNet121) va luu conv output
            conv_output = self.base_model(x, training=False)

            # Qua cac layer con lai (GAP, Dense, etc.)
            for layer in self.model.layers[1:]:
                conv_output = layer(conv_output, training=False)

            predictions = conv_output

            if class_index is None:
                class_index = tf.argmax(predictions[0])

            class_score = predictions[:, class_index]

        # Lay conv output tu base model (can chay lai)
        base_output = self.base_model(image_tensor, training=False)

        # Tinh gradient cua class score doi voi base output
        with tf.GradientTape() as tape:
            tape.watch(base_output)

            x = base_output
            for layer in self.model.layers[1:]:
                x = layer(x, training=False)

            predictions = x

            if class_index is None:
                class_index = tf.argmax(predictions[0])

            class_score = predictions[:, class_index]

        grads = tape.gradient(class_score, base_output)

        if grads is None:
            print("Warning: No gradients")
            predictions_np = predictions.numpy()[0]
            return None, predictions_np, int(np.argmax(predictions_np))

        # Pooled gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted combination
        conv_outputs = base_output[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU va normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        predictions_np = predictions.numpy()[0]
        predicted_class = int(np.argmax(predictions_np))

        return heatmap.numpy(), predictions_np, predicted_class

    def create_overlay(self, original_image, heatmap, alpha=0.4):
        """Create overlay"""
        image_array = np.array(original_image)

        # Resize heatmap
        heatmap_resized = tf.image.resize(
            heatmap[np.newaxis, :, :, np.newaxis],
            image_array.shape[:2]
        ).numpy().squeeze()

        # Apply colormap
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Blend
        overlay = (heatmap_colored * alpha + image_array * (1 - alpha))
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay, heatmap_resized

    def visualize(self, image_path, save_path=None):
        """Visualize Grad-CAM"""
        image_path = Path(image_path)

        image_array, original_image = self.preprocess_image(image_path)
        heatmap, predictions, predicted_class = self.generate_gradcam(image_array)

        if heatmap is None:
            print(f"Failed: {image_path.name}")
            return None

        overlay, heatmap_resized = self.create_overlay(original_image, heatmap)
        confidence = predictions[predicted_class]

        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(original_image)
        axes[0].set_title('Original', fontweight='bold')
        axes[0].axis('off')

        im = axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title(f'Heatmap\n({self.CLASS_NAMES[predicted_class]})', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontweight='bold')
        axes[2].axis('off')

        pred_text = f"Pred: {self.CLASS_NAMES[predicted_class]}\n"
        pred_text += f"Conf: {confidence:.3f}\n\n"
        for i, name in enumerate(self.CLASS_NAMES):
            pred_text += f"{name}: {predictions[i]:.3f}\n"

        axes[3].text(0.5, 0.5, pred_text, ha='center', va='center',
                    transform=axes[3].transAxes, fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[3].set_title('Predictions', fontweight='bold')
        axes[3].axis('off')

        plt.suptitle(f'Grad-CAM: {image_path.name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()

        return {
            'image_path': str(image_path),
            'predicted_class': self.CLASS_NAMES[predicted_class],
            'confidence': float(confidence),
            'predictions': predictions.tolist(),
            'overlay': overlay
        }

    def visualize_test_set(self, test_dir, output_dir, samples_per_class=3):
        """Visualize test set"""
        test_dir = Path(test_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for class_name in self.CLASS_NAMES:
            class_dir = test_dir / class_name

            if not class_dir.exists():
                print(f"Not found: {class_dir}")
                continue

            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if not images:
                continue

            step = max(1, len(images) // samples_per_class)
            selected = images[::step][:samples_per_class]

            print(f"\n{class_name}: {len(selected)} samples")

            for img_path in selected:
                save_path = output_dir / f"gradcam_{class_name}_{img_path.stem}.png"
                result = self.visualize(img_path, save_path=save_path)

                if result:
                    result['true_class'] = class_name
                    results.append(result)

                    status = "OK" if result['predicted_class'] == class_name else "WRONG"
                    print(f"   [{status}] {img_path.name}: {result['predicted_class']} ({result['confidence']:.3f})")

        # Summary
        self._create_summary(results, output_dir)
        return results

    def _create_summary(self, results, output_dir):
        """Create summary"""
        if not results:
            return

        n = len(results)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, result in enumerate(results):
            ax = axes[i]
            ax.imshow(result['overlay'])

            title = f"{result['predicted_class']}\n{result['confidence']:.3f}"
            if 'true_class' in result:
                if result['predicted_class'] == result['true_class']:
                    title = f"OK\n{title}"
                else:
                    title = f"WRONG (True: {result['true_class']})\nPred: {title}"

            ax.set_title(title, fontsize=9)
            ax.axis('off')

        for i in range(n, len(axes)):
            axes[i].axis('off')

        correct = sum(1 for r in results if r.get('true_class') == r['predicted_class'])
        plt.suptitle(f"Grad-CAM Summary\nAccuracy: {correct}/{len(results)} ({correct/len(results):.1%})",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSummary: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for DenseNet121')
    parser.add_argument('--model', required=True, help='Model path (.h5)')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--test-dir', help='Test directory with class subfolders')
    parser.add_argument('--output', default='gradcam_output', help='Output directory')
    parser.add_argument('--samples', type=int, default=3, help='Samples per class')

    args = parser.parse_args()

    if not args.image and not args.test_dir:
        parser.error("Specify --image or --test-dir")

    print("=" * 50)
    print("Grad-CAM for DenseNet121")
    print("=" * 50)

    gradcam = GradCAMDenseNet121(args.model)
    output_dir = Path(args.output)

    if args.image:
        save_path = output_dir / f"gradcam_{Path(args.image).stem}.png"
        result = gradcam.visualize(args.image, save_path=save_path)
        if result:
            print(f"\nPredicted: {result['predicted_class']} ({result['confidence']:.3f})")

    elif args.test_dir:
        results = gradcam.visualize_test_set(args.test_dir, output_dir, args.samples)
        if results:
            correct = sum(1 for r in results if r.get('true_class') == r['predicted_class'])
            print(f"\nAccuracy: {correct}/{len(results)} ({correct/len(results):.1%})")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
