#!/usr/bin/env python3
"""
DenseNet121 Model Evaluation Script
==================================

Đánh giá các model đã train từ folder models/ trên test dataset
Compatible với architecture từ densenet.py

Author: Cell Image-Based Disease Detection Team
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import load_model
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Using basic analysis mode.")
    TENSORFLOW_AVAILABLE = False

# Configuration
class Config:
    # Dataset paths
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data")
    TEST_DIR = DATA_DIR / "test"
    RESULTS_DIR = Path("evaluation_results")
    
    # Model configuration (matching densenet.py)
    INPUT_SHAPE = (224, 224, 3)
    CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']  # Based on test folder structure
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    
    # Results
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_environment():
    """Setup directories and environment"""
    print("🔧 Setting up environment...")
    
    # Create results directory
    Config.RESULTS_DIR.mkdir(exist_ok=True)
    
    # Check paths
    if not Config.MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {Config.MODELS_DIR}")
    
    if not Config.TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {Config.TEST_DIR}")
    
    # Check for model files
    model_files = list(Config.MODELS_DIR.glob("*.h5"))
    if not model_files:
        raise FileNotFoundError(f"No H5 model files found in {Config.MODELS_DIR}")
    
    print(f"✅ Found {len(model_files)} model files")
    print(f"📁 Results will be saved to: {Config.RESULTS_DIR}")
    
    return model_files

def create_test_generator():
    """Create test data generator matching densenet.py preprocessing"""
    print("🔄 Creating test data generator...")
    
    # Use same preprocessing as densenet.py
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        str(Config.TEST_DIR),
        target_size=Config.INPUT_SHAPE[:2],
        batch_size=Config.BATCH_SIZE,
        class_mode='sparse',
        shuffle=False,  # Important for evaluation
        seed=42
    )
    
    print(f"✅ Test generator created: {test_generator.samples} samples")
    print(f"📋 Class indices: {test_generator.class_indices}")
    
    # Verify class mapping matches our expected order
    expected_classes = {name: i for i, name in enumerate(Config.CLASS_NAMES)}
    actual_classes = test_generator.class_indices
    
    if expected_classes != actual_classes:
        print("⚠️ Class mapping mismatch detected!")
        print(f"Expected: {expected_classes}")
        print(f"Actual: {actual_classes}")
    
    return test_generator

def load_and_analyze_model(model_path):
    """Load model and analyze its structure"""
    print(f"\n📂 Loading model: {model_path.name}")
    
    try:
        model = load_model(model_path)
        
        # Model info
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        print(f"📊 Model loaded successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        return model, {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape)
        }
        
    except Exception as e:
        print(f"❌ Error loading model {model_path.name}: {e}")
        return None, None

def evaluate_single_model(model, model_name, test_generator):
    """Evaluate a single model on test data"""
    print(f"\n🎯 Evaluating {model_name}...")
    
    # Reset generator
    test_generator.reset()
    
    # Make predictions
    print("   Making predictions...")
    predictions = model.predict(test_generator, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    prediction_probabilities = np.max(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    
    # Classification report
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=Config.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Per-class analysis
    per_class_analysis = {}
    for i, class_name in enumerate(Config.CLASS_NAMES):
        class_indices = np.where(true_classes == i)[0]
        class_predictions = predicted_classes[class_indices] if len(class_indices) > 0 else []
        class_probabilities = prediction_probabilities[class_indices] if len(class_indices) > 0 else []
        
        per_class_analysis[class_name] = {
            'total_samples': len(class_indices),
            'correct_predictions': np.sum(class_predictions == i) if len(class_predictions) > 0 else 0,
            'mean_confidence': np.mean(class_probabilities) if len(class_probabilities) > 0 else 0,
            'std_confidence': np.std(class_probabilities) if len(class_probabilities) > 0 else 0,
            'precision': report[class_name]['precision'] if class_name in report else 0,
            'recall': report[class_name]['recall'] if class_name in report else 0,
            'f1_score': report[class_name]['f1-score'] if class_name in report else 0
        }
    
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   📊 Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"   📈 Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'per_class_analysis': per_class_analysis,
        'prediction_stats': {
            'total_predictions': len(predicted_classes),
            'mean_confidence': np.mean(prediction_probabilities),
            'std_confidence': np.std(prediction_probabilities),
            'min_confidence': np.min(prediction_probabilities),
            'max_confidence': np.max(prediction_probabilities)
        }
    }

def plot_confusion_matrix(cm, model_name, accuracy):
    """Plot enhanced confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
                cbar_kws={'label': 'Count'},
                square=True, linewidths=0.5)
    
    plt.title(f'{model_name}\nConfusion Matrix (Accuracy: {accuracy:.4f})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Save plot
    plot_path = Config.RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Confusion matrix saved: {plot_path}")
    return plot_path

def plot_class_performance(results_data):
    """Plot per-class performance comparison across models"""
    if len(results_data) <= 1:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['precision', 'recall', 'f1_score']
    
    # Prepare data
    model_names = [r['model_name'] for r in results_data]
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Data for each class
        class_data = {class_name: [] for class_name in Config.CLASS_NAMES}
        
        for result in results_data:
            for class_name in Config.CLASS_NAMES:
                value = result['per_class_analysis'][class_name][metric]
                class_data[class_name].append(value)
        
        # Plot
        x = np.arange(len(model_names))
        width = 0.2
        
        for j, class_name in enumerate(Config.CLASS_NAMES):
            ax.bar(x + j*width, class_data[class_name], width, 
                   label=class_name, alpha=0.8)
        
        ax.set_title(f'{metric.title()} by Class', fontweight='bold')
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.title())
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    # Overall accuracy comparison
    axes[1, 1].clear()
    accuracies = [r['accuracy'] for r in results_data]
    bars = axes[1, 1].bar(model_names, accuracies, color='skyblue', alpha=0.8)
    axes[1, 1].set_title('Overall Accuracy Comparison', fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    plot_path = Config.RESULTS_DIR / f"models_comparison_{Config.TIMESTAMP}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Model comparison saved: {plot_path}")
    return plot_path

def save_comprehensive_results(results_data, model_info_data):
    """Save comprehensive evaluation results"""
    
    # Prepare summary
    summary = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'test_dataset_path': str(Config.TEST_DIR),
            'total_test_samples': sum([r['prediction_stats']['total_predictions'] for r in results_data]) // len(results_data),
            'class_names': Config.CLASS_NAMES,
            'num_models_evaluated': len(results_data)
        },
        'model_performance': {},
        'model_info': model_info_data,
        'comparison_summary': {}
    }
    
    # Individual model results
    for result in results_data:
        model_name = result['model_name']
        summary['model_performance'][model_name] = {
            'accuracy': result['accuracy'],
            'macro_f1': result['classification_report']['macro avg']['f1-score'],
            'weighted_f1': result['classification_report']['weighted avg']['f1-score'],
            'per_class_metrics': result['per_class_analysis'],
            'prediction_confidence': result['prediction_stats']
        }
    
    # Comparison summary
    if len(results_data) > 1:
        best_accuracy = max(results_data, key=lambda x: x['accuracy'])
        best_f1 = max(results_data, key=lambda x: x['classification_report']['macro avg']['f1-score'])
        
        summary['comparison_summary'] = {
            'best_accuracy_model': {
                'name': best_accuracy['model_name'],
                'accuracy': best_accuracy['accuracy']
            },
            'best_f1_model': {
                'name': best_f1['model_name'],
                'f1_score': best_f1['classification_report']['macro avg']['f1-score']
            },
            'accuracy_range': {
                'min': min([r['accuracy'] for r in results_data]),
                'max': max([r['accuracy'] for r in results_data]),
                'std': np.std([r['accuracy'] for r in results_data])
            }
        }
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    summary = convert_numpy_types(summary)
    
    # Save JSON
    json_path = Config.RESULTS_DIR / f"evaluation_summary_{Config.TIMESTAMP}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save CSV summary
    csv_data = []
    for result in results_data:
        model_name = result['model_name']
        csv_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Macro_F1': result['classification_report']['macro avg']['f1-score'],
            'Weighted_F1': result['classification_report']['weighted avg']['f1-score'],
            'Mean_Confidence': result['prediction_stats']['mean_confidence'],
            'Total_Params': model_info_data.get(model_name, {}).get('total_params', 'N/A')
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = Config.RESULTS_DIR / f"models_summary_{Config.TIMESTAMP}.csv"
    csv_df.to_csv(csv_path, index=False)
    
    print(f"📄 Summary saved: {json_path}")
    print(f"📊 CSV saved: {csv_path}")
    
    return summary

def print_evaluation_summary(results_data):
    """Print comprehensive evaluation summary"""
    print(f"\n{'='*60}")
    print("🏆 EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    if not results_data:
        print("❌ No models were successfully evaluated")
        return
    
    print(f"📊 Total models evaluated: {len(results_data)}")
    print(f"📁 Test samples: {results_data[0]['prediction_stats']['total_predictions']}")
    print(f"🏷️ Classes: {', '.join(Config.CLASS_NAMES)}")
    
    print(f"\n{'Model Performance:':<30}")
    print("-" * 60)
    
    # Sort by accuracy
    sorted_results = sorted(results_data, key=lambda x: x['accuracy'], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['model_name']:<30} "
              f"Acc: {result['accuracy']:.4f} | "
              f"F1: {result['classification_report']['macro avg']['f1-score']:.4f}")
    
    if len(results_data) > 1:
        best_model = sorted_results[0]
        print(f"\n🥇 Best Model: {best_model['model_name']}")
        print(f"   📈 Accuracy: {best_model['accuracy']:.4f}")
        print(f"   🎯 Macro F1: {best_model['classification_report']['macro avg']['f1-score']:.4f}")
        
        # Per-class performance for best model
        print(f"\n📋 Per-class performance (Best Model):")
        for class_name in Config.CLASS_NAMES:
            metrics = best_model['per_class_analysis'][class_name]
            print(f"   {class_name:<10}: P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f} F1={metrics['f1_score']:.3f}")
    
    print(f"\n📁 Results saved in: {Config.RESULTS_DIR}")

def main():
    """Main evaluation function"""
    print("🩺 DenseNet121 Model Evaluation")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Please install TensorFlow to run evaluation.")
        return
    
    try:
        # Setup
        model_files = setup_environment()
        
        # Create test generator
        test_generator = create_test_generator()
        
        # GPU setup
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🚀 Using GPU: {len(gpus)} device(s)")
            except RuntimeError as e:
                print(f"⚠️ GPU setup error: {e}")
        else:
            print("💻 Using CPU")
        
        results_data = []
        model_info_data = {}
        
        # Evaluate each model
        for model_file in model_files:
            model_name = model_file.stem
            
            # Load model
            model, model_info = load_and_analyze_model(model_file)
            if model is None:
                continue
            
            model_info_data[model_name] = model_info
            
            # Evaluate
            result = evaluate_single_model(model, model_name, test_generator)
            results_data.append(result)
            
            # Plot confusion matrix
            plot_confusion_matrix(
                np.array(result['confusion_matrix']), 
                model_name, 
                result['accuracy']
            )
            
            # Clear memory
            del model
            tf.keras.backend.clear_session()
        
        # Generate comparison plots
        if len(results_data) > 1:
            plot_class_performance(results_data)
        
        # Save comprehensive results
        save_comprehensive_results(results_data, model_info_data)
        
        # Print summary
        print_evaluation_summary(results_data)
        
        print(f"\n🎉 Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()