#!/usr/bin/env python3
"""
Model Structure Inspector
========================
Examine the model structure to understand layers for Grad-CAM
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

def inspect_model_structure(model_path):
    """Inspect model structure in detail"""
    print(f"🔍 Inspecting model: {model_path}")
    
    # Load model
    model = load_model(model_path)
    
    print(f"\n📊 Model Summary:")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    print(f"\n🏗️ Layer Structure:")
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        try:
            output_shape = layer.output_shape
        except:
            output_shape = "Unknown"
        
        print(f"  {i:2d}: {layer.name:<30} | {layer_type:<20} | {output_shape}")
        
        # If it's a nested model, examine its layers too
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            print(f"      └─ Nested model with {len(layer.layers)} layers:")
            for j, nested_layer in enumerate(layer.layers[-5:]):  # Show last 5 layers
                nested_type = type(nested_layer).__name__
                try:
                    nested_shape = nested_layer.output_shape
                except:
                    nested_shape = "Unknown"
                print(f"         {j:2d}: {nested_layer.name:<25} | {nested_type:<15} | {nested_shape}")
            if len(layer.layers) > 5:
                print(f"         ... and {len(layer.layers) - 5} more layers")
    
    return model

def find_conv_layers(model):
    """Find convolutional layers in the model"""
    print(f"\n🎯 Finding convolutional layers:")
    
    conv_layers = []
    
    def find_conv_recursive(layers, prefix=""):
        nonlocal conv_layers
        for layer in layers:
            layer_type = type(layer).__name__
            
            # Check if it's a convolutional layer
            if 'conv' in layer_type.lower() or 'Conv' in layer_type:
                conv_layers.append((f"{prefix}{layer.name}", layer, layer_type))
                print(f"  ✅ {prefix}{layer.name} ({layer_type})")
            
            # Check nested models
            if hasattr(layer, 'layers'):
                find_conv_recursive(layer.layers, f"{prefix}{layer.name}/")
    
    find_conv_recursive(model.layers)
    
    if not conv_layers:
        print("  ❌ No convolutional layers found directly")
        
        # Look for other suitable layers
        print(f"\n🔍 Looking for other suitable layers:")
        for layer in model.layers:
            layer_type = type(layer).__name__
            if any(keyword in layer_type.lower() for keyword in ['batch', 'norm', 'activation', 'add']):
                try:
                    shape = layer.output_shape
                    if len(shape) == 4:  # (batch, height, width, channels)
                        print(f"  🎯 Potential target: {layer.name} ({layer_type}) - {shape}")
                except:
                    pass
    
    return conv_layers

def main():
    model_path = Path("models/final_DenseNet121_Transfer_4Class.h5")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = inspect_model_structure(model_path)
    conv_layers = find_conv_layers(model)
    
    print(f"\n📋 Summary:")
    print(f"Total layers: {len(model.layers)}")
    print(f"Convolutional layers found: {len(conv_layers)}")
    
    if conv_layers:
        print(f"\nBest candidates for Grad-CAM:")
        for name, layer, layer_type in conv_layers[-3:]:  # Last 3 conv layers
            print(f"  🎯 {name} ({layer_type})")

if __name__ == "__main__":
    main()