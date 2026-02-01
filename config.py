#!/usr/bin/env python3
"""
Unified Configuration File for WBC Classification Training Scripts
================================================================

This module provides centralized configuration management for all training scripts
in the Cell Image Based Disease Detection project.

Author: Cell Image Based Disease Detection Project
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class DatasetConfig:
    """Dataset configuration settings."""
    # Base paths
    base_path: str = str(Path(__file__).parent.parent)
    dataset_path: str = os.getenv("DATASET_PATH", "split_dataset")  # Updated to use our split dataset
    
    # Class configuration
    class_names: List[str] = None
    num_classes: int = int(os.getenv("NUM_CLASSES", "4"))
    class_weights: Dict[int, float] = None
    
    # Image settings
    image_size: Tuple[int, int] = (int(os.getenv("IMAGE_WIDTH", "224")), int(os.getenv("IMAGE_HEIGHT", "224")))
    channels: int = int(os.getenv("IMAGE_CHANNELS", "3"))
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.class_names is None:
            class_names_str = os.getenv("CLASS_NAMES", "Benign,Early,Pre,Pro")
            self.class_names = class_names_str.split(",")
        
        if self.class_weights is None:
            # Updated weights based on our dataset distribution
            # Benign: 504, Early: 985, Pre: 963, Pro: 804
            self.class_weights = {
                0: float(os.getenv("CLASS_WEIGHT_0", "1.95")),  # Benign (504) - highest weight for smallest class
                1: float(os.getenv("CLASS_WEIGHT_1", "1.0")),   # Early (985) - baseline (largest class)
                2: float(os.getenv("CLASS_WEIGHT_2", "1.02")),  # Pre (963) - slightly higher than Early
                3: float(os.getenv("CLASS_WEIGHT_3", "1.22"))   # Pro (804) - higher weight for smaller class
            }
    
    @property
    def full_dataset_path(self) -> str:
        """Get full path to dataset."""
        return os.path.join(self.base_path, self.dataset_path)
    
    @property
    def train_path(self) -> str:
        """Get path to training data."""
        return os.path.join(self.full_dataset_path, "train")
    
    @property
    def val_path(self) -> str:
        """Get path to validation data."""
        return os.path.join(self.full_dataset_path, "val")
    
    @property
    def test_path(self) -> str:
        """Get path to test data."""
        return os.path.join(self.full_dataset_path, "test")

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    # Training cycles and epochs
    training_cycles: int = int(os.getenv("TRAINING_CYCLES", "3"))  # Default 3 training cycles as requested
    epochs_per_cycle: int = int(os.getenv("EPOCHS_PER_CYCLE", "100"))
    
    # Batch sizes (framework-specific defaults)
    batch_size_tensorflow: int = int(os.getenv("BATCH_SIZE_TENSORFLOW", "32"))
    batch_size_pytorch: int = int(os.getenv("BATCH_SIZE_PYTORCH", "32"))
    batch_size_yolo: int = int(os.getenv("BATCH_SIZE_YOLO", "32"))
    
    # Learning rates
    learning_rate_base: float = float(os.getenv("LEARNING_RATE_BASE", "0.001"))
    learning_rate_transfer: float = float(os.getenv("LEARNING_RATE_TRANSFER", "0.0001"))
    learning_rate_finetune: float = float(os.getenv("LEARNING_RATE_FINETUNE", "0.00001"))
    learning_rate_yolo: float = float(os.getenv("LEARNING_RATE_YOLO", "0.01"))
    
    # Optimizer settings
    optimizer: str = os.getenv("OPTIMIZER", "adam")
    beta_1: float = float(os.getenv("OPTIMIZER_BETA_1", "0.9"))
    beta_2: float = float(os.getenv("OPTIMIZER_BETA_2", "0.999"))
    epsilon: float = float(os.getenv("OPTIMIZER_EPSILON", "1e-7"))
    
    # Scheduler settings
    reduce_lr_factor: float = float(os.getenv("REDUCE_LR_FACTOR", "0.5"))
    reduce_lr_patience: int = int(os.getenv("REDUCE_LR_PATIENCE", "10"))
    reduce_lr_min_lr: float = float(os.getenv("REDUCE_LR_MIN_LR", "1e-7"))
    
    # Early stopping
    early_stopping_patience: int = int(os.getenv("EARLY_STOPPING_PATIENCE", "20"))
    early_stopping_min_delta: float = float(os.getenv("EARLY_STOPPING_MIN_DELTA", "0.001"))
    
    # Validation settings
    validation_split: float = float(os.getenv("VALIDATION_SPLIT", "0.15"))
    shuffle: bool = os.getenv("SHUFFLE", "True").lower() == "true"
    
    # GPU settings
    mixed_precision: bool = os.getenv("MIXED_PRECISION", "True").lower() == "true"
    memory_growth: bool = os.getenv("MEMORY_GROWTH", "True").lower() == "true"

@dataclass
class ModelConfig:
    """Model-specific configuration settings."""
    # Model architectures and their specific settings
    models: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize model configurations."""
        if self.models is None:
            self.models = {
                "enhanced_allcnn": {
                    "input_size": (224, 224, 3),
                    "batch_size": int(os.getenv("ENHANCED_ALLCNN_BATCH_SIZE", "32")),
                    "epochs": int(os.getenv("ENHANCED_ALLCNN_EPOCHS", "100")),
                    "learning_rate": float(os.getenv("ENHANCED_ALLCNN_LEARNING_RATE", "0.001")),
                    "dropout_rate": float(os.getenv("ENHANCED_ALLCNN_DROPOUT_RATE", "0.5")),
                    "use_batch_norm": os.getenv("ENHANCED_ALLCNN_USE_BATCH_NORM", "True").lower() == "true"
                },
                "resnet50_transfer": {
                    "input_size": (224, 224, 3),
                    "batch_size": int(os.getenv("RESNET50_BATCH_SIZE", "16")),
                    "epochs": int(os.getenv("RESNET50_EPOCHS", "100")),
                    "learning_rate": float(os.getenv("RESNET50_LEARNING_RATE", "0.0001")),
                    "freeze_epochs": int(os.getenv("RESNET50_FREEZE_EPOCHS", "50")),
                    "finetune_epochs": int(os.getenv("RESNET50_FINETUNE_EPOCHS", "50")),
                    "dense_units": [256, 128],
                    "dropout_rate": float(os.getenv("RESNET50_DROPOUT_RATE", "0.5"))
                },
                "simple_cnn": {
                    "input_size": (128, 128, 3),
                    "batch_size": int(os.getenv("SIMPLE_CNN_BATCH_SIZE", "64")),
                    "epochs": int(os.getenv("SIMPLE_CNN_EPOCHS", "120")),
                    "learning_rate": float(os.getenv("SIMPLE_CNN_LEARNING_RATE", "0.001")),
                    "conv_filters": [32, 64, 128],
                    "dense_units": [512, 256],
                    "dropout_rate": float(os.getenv("SIMPLE_CNN_DROPOUT_RATE", "0.5"))
                },
                "pytorch_lightning": {
                    "input_size": (128, 128),
                    "batch_size": int(os.getenv("PYTORCH_LIGHTNING_BATCH_SIZE", "32")),
                    "epochs": int(os.getenv("PYTORCH_LIGHTNING_EPOCHS", "100")),
                    "learning_rate": float(os.getenv("PYTORCH_LIGHTNING_LEARNING_RATE", "0.001")),
                    "weight_decay": float(os.getenv("PYTORCH_LIGHTNING_WEIGHT_DECAY", "1e-4")),
                    "step_size": int(os.getenv("PYTORCH_LIGHTNING_STEP_SIZE", "30")),
                    "gamma": float(os.getenv("PYTORCH_LIGHTNING_GAMMA", "0.1"))
                },
                "vgg16_transfer": {
                    "input_size": (224, 224, 3),
                    "batch_size": int(os.getenv("VGG16_BATCH_SIZE", "16")),
                    "epochs": int(os.getenv("VGG16_EPOCHS", "70")),
                    "learning_rate": float(os.getenv("VGG16_LEARNING_RATE", "0.0001")),
                    "freeze_epochs": int(os.getenv("VGG16_FREEZE_EPOCHS", "35")),
                    "finetune_epochs": int(os.getenv("VGG16_FINETUNE_EPOCHS", "35")),
                    "dense_units": [512, 256],
                    "dropout_rate": float(os.getenv("VGG16_DROPOUT_RATE", "0.5"))
                },
                "densenet121_transfer": {
                    "input_size": (224, 224, 3),
                    "batch_size": int(os.getenv("DENSENET121_BATCH_SIZE", "16")),
                    "epochs": int(os.getenv("DENSENET121_EPOCHS", "100")),
                    "learning_rate": float(os.getenv("DENSENET121_LEARNING_RATE", "0.0001")),
                    "freeze_epochs": int(os.getenv("DENSENET121_FREEZE_EPOCHS", "50")),
                    "finetune_epochs": int(os.getenv("DENSENET121_FINETUNE_EPOCHS", "50")),
                    "dense_units": [256, 128],
                    "dropout_rate": float(os.getenv("DENSENET121_DROPOUT_RATE", "0.5"))
                },
                "efficientnet_transfer": {
                    "input_size": (224, 224, 3),
                    "batch_size": int(os.getenv("EFFICIENTNET_BATCH_SIZE", "16")),
                    "epochs": int(os.getenv("EFFICIENTNET_EPOCHS", "100")),
                    "learning_rate": float(os.getenv("EFFICIENTNET_LEARNING_RATE", "0.0001")),
                    "freeze_epochs": int(os.getenv("EFFICIENTNET_FREEZE_EPOCHS", "50")),
                    "finetune_epochs": int(os.getenv("EFFICIENTNET_FINETUNE_EPOCHS", "50")),
                    "dense_units": [256, 128],
                    "dropout_rate": float(os.getenv("EFFICIENTNET_DROPOUT_RATE", "0.3"))
                },
                "vision_transformer": {
                    "input_size": (224, 224, 3),
                    "batch_size": int(os.getenv("VISION_TRANSFORMER_BATCH_SIZE", "8")),
                    "epochs": int(os.getenv("VISION_TRANSFORMER_EPOCHS", "100")),
                    "learning_rate": float(os.getenv("VISION_TRANSFORMER_LEARNING_RATE", "0.0001")),
                    "patch_size": int(os.getenv("VISION_TRANSFORMER_PATCH_SIZE", "16")),
                    "num_heads": int(os.getenv("VISION_TRANSFORMER_NUM_HEADS", "8")),
                    "num_layers": int(os.getenv("VISION_TRANSFORMER_NUM_LAYERS", "6")),
                    "hidden_dim": int(os.getenv("VISION_TRANSFORMER_HIDDEN_DIM", "256")),
                    "mlp_dim": int(os.getenv("VISION_TRANSFORMER_MLP_DIM", "512")),
                    "dropout_rate": float(os.getenv("VISION_TRANSFORMER_DROPOUT_RATE", "0.1"))
                },
                "custom_deep_cnn": {
                    "input_size": (224, 224, 3),
                    "batch_size": int(os.getenv("CUSTOM_DEEP_CNN_BATCH_SIZE", "32")),
                    "epochs": int(os.getenv("CUSTOM_DEEP_CNN_EPOCHS", "100")),
                    "learning_rate": float(os.getenv("CUSTOM_DEEP_CNN_LEARNING_RATE", "0.001")),
                    "conv_blocks": int(os.getenv("CUSTOM_DEEP_CNN_CONV_BLOCKS", "5")),
                    "base_filters": int(os.getenv("CUSTOM_DEEP_CNN_BASE_FILTERS", "32")),
                    "dense_units": [512, 256],
                    "dropout_rate": float(os.getenv("CUSTOM_DEEP_CNN_DROPOUT_RATE", "0.5"))
                },
                "yolov8_classification": {
                    "input_size": int(os.getenv("YOLOV8_INPUT_SIZE", "224")),
                    "batch_size": int(os.getenv("YOLOV8_BATCH_SIZE", "32")),
                    "epochs": int(os.getenv("YOLOV8_EPOCHS", "100")),
                    "learning_rate": float(os.getenv("YOLOV8_LEARNING_RATE", "0.01")),
                    "momentum": float(os.getenv("YOLOV8_MOMENTUM", "0.937")),
                    "weight_decay": float(os.getenv("YOLOV8_WEIGHT_DECAY", "0.0005")),
                    "warmup_epochs": int(os.getenv("YOLOV8_WARMUP_EPOCHS", "3")),
                    "warmup_momentum": float(os.getenv("YOLOV8_WARMUP_MOMENTUM", "0.8"))
                }
            }

@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # TensorFlow/Keras augmentation
    rotation_range: float = float(os.getenv("AUGMENTATION_ROTATION_RANGE", "20"))
    width_shift_range: float = float(os.getenv("AUGMENTATION_WIDTH_SHIFT_RANGE", "0.1"))
    height_shift_range: float = float(os.getenv("AUGMENTATION_HEIGHT_SHIFT_RANGE", "0.1"))
    shear_range: float = float(os.getenv("AUGMENTATION_SHEAR_RANGE", "0.1"))
    zoom_range: float = float(os.getenv("AUGMENTATION_ZOOM_RANGE", "0.1"))
    horizontal_flip: bool = os.getenv("AUGMENTATION_HORIZONTAL_FLIP", "True").lower() == "true"
    vertical_flip: bool = os.getenv("AUGMENTATION_VERTICAL_FLIP", "False").lower() == "true"
    brightness_range: Tuple[float, float] = (
        float(os.getenv("AUGMENTATION_BRIGHTNESS_MIN", "0.8")),
        float(os.getenv("AUGMENTATION_BRIGHTNESS_MAX", "1.2"))
    )
    fill_mode: str = os.getenv("AUGMENTATION_FILL_MODE", "nearest")
    
    # PyTorch augmentation
    pytorch_transforms: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize PyTorch transforms."""
        if self.pytorch_transforms is None:
            self.pytorch_transforms = {
                "rotation_degrees": float(os.getenv("PYTORCH_ROTATION_DEGREES", "20")),
                "translate": (0.1, 0.1),
                "scale": (0.9, 1.1),
                "shear": float(os.getenv("PYTORCH_SHEAR", "10")),
                "brightness": float(os.getenv("PYTORCH_BRIGHTNESS", "0.2")),
                "contrast": float(os.getenv("PYTORCH_CONTRAST", "0.2")),
                "saturation": float(os.getenv("PYTORCH_SATURATION", "0.2")),
                "hue": float(os.getenv("PYTORCH_HUE", "0.1"))
            }

@dataclass
class MetricsConfig:
    """Metrics and evaluation configuration."""
    # Standard metrics to calculate
    metrics: List[str] = None
    
    # Visualization settings
    save_confusion_matrix: bool = os.getenv("SAVE_CONFUSION_MATRIX", "True").lower() == "true"
    save_training_history: bool = os.getenv("SAVE_TRAINING_HISTORY", "True").lower() == "true"
    save_classification_report: bool = os.getenv("SAVE_CLASSIFICATION_REPORT", "True").lower() == "true"
    
    # Interpretability settings (for applicable models)
    save_model_interpretability: bool = os.getenv("SAVE_MODEL_INTERPRETABILITY", "True").lower() == "true"
    interpretability_methods: List[str] = None
    
    def __post_init__(self):
        """Initialize default metrics."""
        if self.metrics is None:
            metrics_str = os.getenv("METRICS", "accuracy,precision,recall,f1_score,confusion_matrix,classification_report,per_class_accuracy,macro_avg_precision,macro_avg_recall,macro_avg_f1,weighted_avg_precision,weighted_avg_recall,weighted_avg_f1")
            self.metrics = metrics_str.split(",")
        
        if self.interpretability_methods is None:
            interpretability_str = os.getenv("INTERPRETABILITY_METHODS", "integrated_gradients,gradient_shap,saliency")
            self.interpretability_methods = interpretability_str.split(",")

@dataclass
class OutputConfig:
    """Output and logging configuration."""
    # Base output directory
    results_dir: str = os.getenv("RESULTS_DIR", "training_results")
    
    # Subdirectories
    models_subdir: str = os.getenv("MODELS_SUBDIR", "models")
    results_subdir: str = os.getenv("RESULTS_SUBDIR", "results")
    logs_subdir: str = os.getenv("LOGS_SUBDIR", "logs")
    visualizations_subdir: str = os.getenv("VISUALIZATIONS_SUBDIR", "visualizations")
    
    # File formats
    model_save_format: str = os.getenv("MODEL_SAVE_FORMAT", "best")  # "best", "final", or "both"
    results_format: str = os.getenv("RESULTS_FORMAT", "json")
    plot_format: str = os.getenv("PLOT_FORMAT", "png")
    plot_dpi: int = int(os.getenv("PLOT_DPI", "300"))
    
    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    save_logs: bool = os.getenv("SAVE_LOGS", "True").lower() == "true"
    
    def get_output_path(self, model_name: str, subdir: str = None) -> str:
        """Get output path for a specific model and subdirectory."""
        base_path = Path(__file__).parent.parent
        output_path = base_path / self.results_dir / model_name
        
        if subdir:
            output_path = output_path / subdir
        
        output_path.mkdir(parents=True, exist_ok=True)
        return str(output_path)

class UnifiedConfig:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self):
        self.dataset = DatasetConfig()
        self.training = TrainingConfig()
        self.models = ModelConfig()
        self.augmentation = AugmentationConfig()
        self.metrics = MetricsConfig()
        self.output = OutputConfig()
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_name not in self.models.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return self.models.models[model_name]
    
    def update_config(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dataset": self.dataset.__dict__,
            "training": self.training.__dict__,
            "models": self.models.__dict__,
            "augmentation": self.augmentation.__dict__,
            "metrics": self.metrics.__dict__,
            "output": self.output.__dict__
        }
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        import json
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def load_config(self, filepath: str):
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Update configurations
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

# Global configuration instance
config = UnifiedConfig()

# Convenience functions for backward compatibility
def get_dataset_config():
    """Get dataset configuration."""
    return config.dataset

def get_training_config():
    """Get training configuration."""
    return config.training

def get_model_config(model_name: str):
    """Get model-specific configuration."""
    return config.get_model_config(model_name)

def get_augmentation_config():
    """Get augmentation configuration."""
    return config.augmentation

def get_metrics_config():
    """Get metrics configuration."""
    return config.metrics

def get_output_config():
    """Get output configuration."""
    return config.output

# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    print("=== Unified Configuration Test ===")
    print(f"Dataset path: {config.dataset.full_dataset_path}")
    print(f"Train path: {config.dataset.train_path}")
    print(f"Class names: {config.dataset.class_names}")
    print(f"Class weights: {config.dataset.class_weights}")
    print(f"Training cycles: {config.training.training_cycles}")
    print(f"Epochs per cycle: {config.training.epochs_per_cycle}")
    
    # Test model-specific config
    resnet_config = config.get_model_config("resnet50_transfer")
    print(f"ResNet50 config: {resnet_config}")
    
    # Test output path generation
    output_path = config.output.get_output_path("test_model", "models")
    print(f"Output path: {output_path}")
    
    print("✅ Configuration test completed successfully!")