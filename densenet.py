#!/usr/bin/env python3
"""
DenseNet121 Transfer Learning cho phân loại 4 lớp ALL-IDB4
=========================================================

Kiến trúc: DenseNet121 pretrained + custom classification head
Classes: Benign, Early_Pre_B_ALL, Pre_B_ALL, Pro_B_ALL

Author: Cell Image-Based Disease Detection Team
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import unified configuration - robust path resolution
import sys
from pathlib import Path

def find_project_src():
    """Find the src directory containing config.py"""
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        config_path = current_path / "src" / "config.py"
        if config_path.exists():
            return current_path / "src"
        current_path = current_path.parent
    return None

project_src = find_project_src()
if project_src:
    sys.path.insert(0, str(project_src))

from config import config, get_model_config

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Get model-specific configuration
model_config = get_model_config("densenet121_transfer")
MODEL_NAME = "DenseNet121_Transfer_4Class"

class Config:
    # Dataset
    BASE_PATH = str(Path(__file__).parent.parent)
    DATASET_PATH = config.dataset.full_dataset_path
    CLASS_NAMES = config.dataset.class_names
    NUM_CLASSES = config.dataset.num_classes
    CLASS_WEIGHTS = config.dataset.class_weights
    
    # Model
    MODEL_NAME = MODEL_NAME
    INPUT_SHAPE = model_config["input_size"]
    DROPOUT_RATE = model_config["dropout_rate"]
    L2_REG = 1e-4
    DENSE_UNITS = model_config["dense_units"]
    
    # Training
    BATCH_SIZE = model_config["batch_size"]
    EPOCHS = model_config["epochs"]
    INITIAL_LR = model_config["learning_rate"]
    FINE_TUNE_LR = config.training.learning_rate_finetune
    
    # Transfer learning params
    FREEZE_LAYERS = 200  # Freeze first 200 layers
    FREEZE_EPOCHS = model_config["freeze_epochs"]
    FINE_TUNE_EPOCHS = model_config["finetune_epochs"]

def create_densenet121_transfer_model(fine_tune=False, unfreeze_layers=None):
    """Tạo DenseNet121 transfer learning model

    Args:
        fine_tune: Nếu True, sẽ unfreeze layers cho fine-tuning
        unfreeze_layers: Số layers cuối được unfreeze (None = tất cả layers)
    """
    print(f"🏗️ Xây dựng DenseNet121 Transfer Learning model (fine_tune={fine_tune})...")

    # Load pre-trained DenseNet121
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=Config.INPUT_SHAPE
    )

    total_layers = len(base_model.layers)
    print(f"📊 Total layers in DenseNet121: {total_layers}")

    # Freeze base model layers
    if not fine_tune:
        for layer in base_model.layers[:Config.FREEZE_LAYERS]:
            layer.trainable = False
        print(f"🔒 Frozen first {Config.FREEZE_LAYERS} layers")
    else:
        # Fine-tuning: unfreeze layers
        if unfreeze_layers is not None and unfreeze_layers < total_layers:
            # Chỉ unfreeze N layers cuối cùng
            freeze_until = total_layers - unfreeze_layers
            for layer in base_model.layers[:freeze_until]:
                layer.trainable = False
            for layer in base_model.layers[freeze_until:]:
                layer.trainable = True
            print(f"🔓 Unfrozen last {unfreeze_layers} layers (frozen first {freeze_until} layers)")
        else:
            # Unfreeze tất cả
            for layer in base_model.layers:
                layer.trainable = True
            print("🔓 All layers unfrozen for fine-tuning")
    
    # Add custom classification head optimized for DenseNet
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        
        # Dense layers optimized for DenseNet
        layers.Dense(Config.DENSE_UNITS[0], activation='relu',
                    kernel_regularizer=keras.regularizers.l2(Config.L2_REG)),
        layers.BatchNormalization(),
        layers.Dropout(Config.DROPOUT_RATE),
        
        layers.Dense(Config.DENSE_UNITS[1], activation='relu',
                    kernel_regularizer=keras.regularizers.l2(Config.L2_REG)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    
    model._name = Config.MODEL_NAME
    return model

def create_data_generators():
    """Tạo data generators với DenseNet preprocessing"""
    print("🔄 Tạo data generators...")
    
    # Training augmentation - optimized for DenseNet
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation/test data
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(Config.DATASET_PATH, "train"),
        target_size=Config.INPUT_SHAPE[:2],
        batch_size=Config.BATCH_SIZE,
        class_mode='sparse',
        shuffle=True,
        seed=SEED
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(Config.DATASET_PATH, "val"),
        target_size=Config.INPUT_SHAPE[:2],
        batch_size=Config.BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(Config.DATASET_PATH, "test"),
        target_size=Config.INPUT_SHAPE[:2],
        batch_size=Config.BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )
    
    print(f"✅ Train: {train_generator.samples}, Val: {validation_generator.samples}, Test: {test_generator.samples}")
    
    return train_generator, validation_generator, test_generator

def setup_callbacks(model_save_dir, phase="initial"):
    """Thiết lập callbacks"""
    os.makedirs(model_save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_save_dir, f"best_{Config.MODEL_NAME}_{phase}.h5")
    
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15 if phase == "initial" else 10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive LR reduction for DenseNet
            patience=7 if phase == "initial" else 5,
            min_lr=1e-8,
            verbose=1
        ),
        # Cosine annealing for better convergence
        callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.95 if epoch > 10 else lr,
            verbose=0
        )
    ]
    
    return callbacks_list

def train_model_two_phase(model, train_gen, val_gen, model_save_dir):
    """Two-phase training: freeze then fine-tune"""
    
    # Phase 1: Frozen base model
    print("\n🚀 PHASE 1: Training với frozen DenseNet121 base")
    print("=" * 50)
    
    # Compile with custom optimizer settings for DenseNet
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=Config.INITIAL_LR,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_list = setup_callbacks(model_save_dir, "phase1")
    
    steps_per_epoch = train_gen.samples // Config.BATCH_SIZE
    validation_steps = val_gen.samples // Config.BATCH_SIZE
    
    history1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.FREEZE_EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=Config.CLASS_WEIGHTS,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n🔥 PHASE 2: Fine-tuning toàn bộ DenseNet121")
    print("=" * 50)
    
    # Recreate model for fine-tuning
    model = create_densenet121_transfer_model(fine_tune=True)
    
    # Load weights from phase 1
    phase1_model_path = os.path.join(model_save_dir, f"best_{Config.MODEL_NAME}_phase1.h5")
    if os.path.exists(phase1_model_path):
        print("📂 Loading phase 1 weights...")
        try:
            # compile=False để tránh warning về metrics chưa được build
            phase1_model = keras.models.load_model(phase1_model_path, compile=False)
            model.set_weights(phase1_model.get_weights())
        except Exception as e:
            print(f"⚠️ Could not load phase 1 weights: {e}")
    
    # Compile with lower learning rate for fine-tuning
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=Config.FINE_TUNE_LR,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_list = setup_callbacks(model_save_dir, "phase2")
    
    history2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=Config.CLASS_WEIGHTS,
        verbose=1
    )
    
    # Combine histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    return model, combined_history

def evaluate_model(model, test_generator, results_dir):
    """Đánh giá model với detailed analysis"""
    print("📊 Đánh giá model...")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Predict
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=Config.CLASS_NAMES,
        output_dict=True
    )
    
    print("\n📈 CLASSIFICATION REPORT:")
    print("=" * 60)
    print(classification_report(true_classes, predicted_classes, target_names=Config.CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Enhanced confusion matrix plot
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(f'{Config.MODEL_NAME} - Confusion Matrix\nOverall Accuracy: {report["accuracy"]:.4f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    cm_path = os.path.join(results_dir, f"{Config.MODEL_NAME}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional analysis: prediction confidence
    prediction_confidence = np.max(predictions, axis=1)
    confidence_by_class = {}
    for i, class_name in enumerate(Config.CLASS_NAMES):
        class_indices = np.where(true_classes == i)[0]
        if len(class_indices) > 0:
            confidence_by_class[class_name] = {
                'mean_confidence': np.mean(prediction_confidence[class_indices]),
                'std_confidence': np.std(prediction_confidence[class_indices])
            }
    
    # Save results
    results = {
        'model_name': Config.MODEL_NAME,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'test_accuracy': report['accuracy'],
        'prediction_confidence': confidence_by_class,
        'per_class_metrics': {
            Config.CLASS_NAMES[i]: {
                'precision': report[Config.CLASS_NAMES[i]]['precision'],
                'recall': report[Config.CLASS_NAMES[i]]['recall'],
                'f1-score': report[Config.CLASS_NAMES[i]]['f1-score']
            } for i in range(Config.NUM_CLASSES)
        }
    }
    
    results_path = os.path.join(results_dir, f"{Config.MODEL_NAME}_results.json")
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    results = convert_numpy_types(results)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {results_path}")
    return results

def plot_training_history(history, results_dir):
    """Vẽ enhanced training history"""
    if history is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Add vertical line to separate phases
    phase_separator = Config.FREEZE_EPOCHS
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Training', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].axvline(x=phase_separator, color='red', linestyle='--', alpha=0.7, label='Fine-tune start')
    axes[0, 0].set_title('Model Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Training', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].axvline(x=phase_separator, color='red', linestyle='--', alpha=0.7, label='Fine-tune start')
    axes[0, 1].set_title('Model Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision (disabled)
    axes[0, 2].text(0.5, 0.5, 'Precision metric removed\nto avoid shape mismatch', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[0, 2].transAxes, fontsize=12)
    axes[0, 2].set_title('Precision (Disabled)', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Recall (disabled)
    axes[1, 0].text(0.5, 0.5, 'Recall metric removed\nto avoid shape mismatch', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1, 0].transAxes, fontsize=12)
    axes[1, 0].set_title('Recall (Disabled)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate', linewidth=2, color='orange')
        axes[1, 1].axvline(x=phase_separator, color='red', linestyle='--', alpha=0.7, label='Fine-tune start')
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    # Loss vs Accuracy correlation
    axes[1, 2].scatter(history['loss'], history['accuracy'], alpha=0.6, label='Training', s=20)
    axes[1, 2].scatter(history['val_loss'], history['val_accuracy'], alpha=0.6, label='Validation', s=20)
    axes[1, 2].set_title('Loss vs Accuracy Correlation', fontweight='bold')
    axes[1, 2].set_xlabel('Loss')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{Config.MODEL_NAME} - Comprehensive Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, f"{Config.MODEL_NAME}_training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved to: {plot_path}")

def train_phase2_only(model_save_dir, train_gen, val_gen, unfreeze_layers=None):
    """Chỉ chạy Phase 2 với model đã train từ Phase 1

    Args:
        unfreeze_layers: Số layers cuối được unfreeze (None = tất cả)
    """
    print("\n🔥 PHASE 2 ONLY: Fine-tuning DenseNet121")
    print("=" * 50)

    # Tạo model mới với fine_tune=True
    model = create_densenet121_transfer_model(fine_tune=True, unfreeze_layers=unfreeze_layers)

    # Load weights từ Phase 1
    phase1_model_path = os.path.join(model_save_dir, f"best_{Config.MODEL_NAME}_phase1.h5")
    if not os.path.exists(phase1_model_path):
        print(f"❌ Phase 1 model not found: {phase1_model_path}")
        print("⚠️ Vui lòng chạy Phase 1 trước hoặc kiểm tra đường dẫn model.")
        return None, None

    print(f"📂 Loading phase 1 weights from: {phase1_model_path}")
    try:
        phase1_model = keras.models.load_model(phase1_model_path, compile=False)
        model.set_weights(phase1_model.get_weights())
        print("✅ Phase 1 weights loaded successfully!")
    except Exception as e:
        print(f"❌ Could not load phase 1 weights: {e}")
        return None, None

    # Compile với learning rate thấp cho fine-tuning
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=Config.FINE_TUNE_LR,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_list = setup_callbacks(model_save_dir, "phase2")

    steps_per_epoch = train_gen.samples // Config.BATCH_SIZE
    validation_steps = val_gen.samples // Config.BATCH_SIZE

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=Config.CLASS_WEIGHTS,
        verbose=1
    )

    return model, history.history


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='DenseNet121 Transfer Learning Training')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=0,
                       help='Chọn phase để chạy: 1=Phase1 only, 2=Phase2 only, 0=Both (default)')
    parser.add_argument('--phase1-model', type=str, default=None,
                       help='Đường dẫn tới model Phase 1 (chỉ dùng khi --phase=2)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (giảm xuống 4-8 nếu hết RAM)')
    parser.add_argument('--unfreeze-layers', type=int, default=None,
                       help='Số layers cuối được unfreeze trong Phase 2 (mặc định: all). Ví dụ: 50 = chỉ unfreeze 50 layers cuối')
    args = parser.parse_args()

    # Override batch size nếu được chỉ định
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
        print(f"⚙️ Override batch size: {args.batch_size}")

    print("🩺 DenseNet121 Transfer Learning 4-Class Training")
    print("=" * 50)
    
    # Setup directories
    output_dir = config.output.get_output_path(Config.MODEL_NAME)
    model_save_dir = config.output.get_output_path(Config.MODEL_NAME, config.output.models_subdir)
    results_dir = config.output.get_output_path(Config.MODEL_NAME, config.output.results_subdir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check dataset
    if not os.path.exists(Config.DATASET_PATH):
        print(f"❌ Dataset not found: {Config.DATASET_PATH}")
        return
    
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
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()

    # Xử lý theo phase được chọn
    if args.phase == 2:
        # Chỉ chạy Phase 2
        print(f"\n🎯 Mode: Phase 2 Only (Fine-tuning)")

        # Nếu có custom path cho phase1 model
        if args.phase1_model:
            # Copy model vào model_save_dir với tên đúng format
            import shutil
            target_path = os.path.join(model_save_dir, f"best_{Config.MODEL_NAME}_phase1.h5")
            if args.phase1_model != target_path:
                print(f"📋 Copying phase1 model to: {target_path}")
                shutil.copy(args.phase1_model, target_path)

        model, history = train_phase2_only(model_save_dir, train_gen, val_gen, unfreeze_layers=args.unfreeze_layers)

        if model is None:
            print("❌ Phase 2 training failed!")
            return

        combined_history = history

    elif args.phase == 1:
        # Chỉ chạy Phase 1
        print(f"\n🎯 Mode: Phase 1 Only (Frozen base)")

        model = create_densenet121_transfer_model(fine_tune=False)

        print("📊 Model Summary:")
        print(f"📈 Total parameters: {model.count_params():,}")
        print(f"🔒 Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

        # Compile
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=Config.INITIAL_LR,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks_list = setup_callbacks(model_save_dir, "phase1")

        steps_per_epoch = train_gen.samples // Config.BATCH_SIZE
        validation_steps = val_gen.samples // Config.BATCH_SIZE

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=Config.FREEZE_EPOCHS,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            class_weight=Config.CLASS_WEIGHTS,
            verbose=1
        )

        combined_history = history.history
        print(f"\n✅ Phase 1 completed! Model saved at: {model_save_dir}")
        print("💡 Để chạy Phase 2, sử dụng: python densenet.py --phase 2")

    else:
        # Chạy cả 2 phase (default)
        print(f"\n🎯 Mode: Full Training (Phase 1 + Phase 2)")

        model = create_densenet121_transfer_model(fine_tune=False)

        print("📊 Model Summary:")
        print(f"📈 Total parameters: {model.count_params():,}")
        print(f"🔒 Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

        model, combined_history = train_model_two_phase(model, train_gen, val_gen, model_save_dir)

    # Plot training history
    if combined_history:
        plot_training_history(combined_history, results_dir)

    # Load best model for evaluation
    if args.phase == 1:
        best_model_path = os.path.join(model_save_dir, f"best_{Config.MODEL_NAME}_phase1.h5")
    else:
        best_model_path = os.path.join(model_save_dir, f"best_{Config.MODEL_NAME}_phase2.h5")

    if os.path.exists(best_model_path):
        print("📂 Loading best model for evaluation...")
        model = keras.models.load_model(best_model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # Evaluate model
    results = evaluate_model(model, test_gen, results_dir)

    # Save final model
    phase_suffix = f"_phase{args.phase}" if args.phase in [1, 2] else ""
    final_model_path = os.path.join(model_save_dir, f"final_{Config.MODEL_NAME}{phase_suffix}.h5")
    model.save(final_model_path)

    print(f"\n🎉 Training completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"🎯 Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"🏆 Best class F1-score: {max([results['per_class_metrics'][cls]['f1-score'] for cls in Config.CLASS_NAMES]):.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()