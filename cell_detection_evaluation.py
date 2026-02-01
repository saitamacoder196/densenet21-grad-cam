#!/usr/bin/env python3
"""
Cell Image Based Disease Detection and Evaluation Script
Uses DenseNet121 models to classify cell images into 4 classes: Benign, Early, Pre, Pro
"""

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CellDetectionEvaluator:
    def __init__(self, models_dir='models', data_dir='data'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.test_dir = self.data_dir / 'test'
        self.classes = ['Benign', 'Early', 'Pre', 'Pro']
        self.img_size = (224, 224)
        self.models = {}
        
    def load_models(self):
        """Load all available models"""
        print("Loading models...")
        model_files = list(self.models_dir.glob('*.h5'))
        
        for model_file in model_files:
            try:
                model_name = model_file.stem
                print(f"Loading {model_name}...")
                self.models[model_name] = load_model(model_file)
                print(f"✓ {model_name} loaded successfully")
            except Exception as e:
                print(f"✗ Error loading {model_file}: {e}")
        
        if not self.models:
            raise ValueError("No models could be loaded!")
            
        print(f"Successfully loaded {len(self.models)} models")
        
    def prepare_test_data(self):
        """Prepare test data generator"""
        print("Preparing test data...")
        
        # Create data generator for test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Found {self.test_generator.samples} test images")
        print(f"Classes: {list(self.test_generator.class_indices.keys())}")
        
    def predict_with_model(self, model_name):
        """Make predictions using specified model"""
        print(f"\nMaking predictions with {model_name}...")
        
        model = self.models[model_name]
        
        # Reset generator
        self.test_generator.reset()
        
        # Make predictions
        predictions = model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = self.test_generator.classes
        
        return predictions, predicted_classes, true_classes
        
    def evaluate_model(self, model_name):
        """Evaluate a single model"""
        predictions, predicted_classes, true_classes = self.predict_with_model(model_name)
        
        # Calculate accuracy
        accuracy = accuracy_score(true_classes, predicted_classes)
        
        # Generate classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.classes,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_class_accuracy(self, report, model_name):
        """Plot per-class accuracy"""
        class_scores = [report[cls]['f1-score'] for cls in self.classes]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.classes, class_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        plt.title(f'Per-Class F1-Score - {model_name}')
        plt.xlabel('Classes')
        plt.ylabel('F1-Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, class_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'class_accuracy_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_detailed_results(self, results, model_name):
        """Save detailed prediction results to CSV"""
        # Get file paths
        file_paths = self.test_generator.filepaths
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'image_path': file_paths,
            'true_class': [self.classes[i] for i in results['true_classes']],
            'predicted_class': [self.classes[i] for i in results['predicted_classes']],
            'correct': results['true_classes'] == results['predicted_classes']
        })
        
        # Add prediction probabilities for each class
        for i, class_name in enumerate(self.classes):
            results_df[f'prob_{class_name}'] = results['predictions'][:, i]
        
        # Save to CSV
        csv_filename = f'detailed_results_{model_name}.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"Detailed results saved to {csv_filename}")
        
        return results_df
    
    def print_summary_report(self, results):
        """Print comprehensive summary report"""
        model_name = results['model_name']
        accuracy = results['accuracy']
        report = results['classification_report']
        
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS - {model_name}")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nPer-Class Results:")
        print("-" * 50)
        
        for class_name in self.classes:
            if class_name in report:
                class_report = report[class_name]
                print(f"{class_name:10s} | Precision: {class_report['precision']:.3f} | "
                      f"Recall: {class_report['recall']:.3f} | "
                      f"F1-Score: {class_report['f1-score']:.3f} | "
                      f"Support: {int(class_report['support'])}")
        
        print("-" * 50)
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']
        print(f"{'Macro Avg':10s} | Precision: {macro_avg['precision']:.3f} | "
              f"Recall: {macro_avg['recall']:.3f} | "
              f"F1-Score: {macro_avg['f1-score']:.3f}")
        print(f"{'Weighted Avg':10s} | Precision: {weighted_avg['precision']:.3f} | "
              f"Recall: {weighted_avg['recall']:.3f} | "
              f"F1-Score: {weighted_avg['f1-score']:.3f}")
        
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting Cell Detection Evaluation...")
        print("="*50)
        
        # Load models
        self.load_models()
        
        # Prepare test data
        self.prepare_test_data()
        
        # Store all results
        all_results = {}
        
        # Evaluate each model
        for model_name in self.models.keys():
            print(f"\n{'='*30}")
            print(f"EVALUATING {model_name}")
            print(f"{'='*30}")
            
            try:
                # Evaluate model
                results = self.evaluate_model(model_name)
                all_results[model_name] = results
                
                # Print summary report
                self.print_summary_report(results)
                
                # Plot confusion matrix
                self.plot_confusion_matrix(results['confusion_matrix'], model_name)
                
                # Plot class accuracy
                self.plot_class_accuracy(results['classification_report'], model_name)
                
                # Save detailed results
                self.save_detailed_results(results, model_name)
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        # Compare models if multiple models exist
        if len(all_results) > 1:
            self.compare_models(all_results)
            
        print(f"\n{'='*50}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*50}")
        
        return all_results
    
    def compare_models(self, all_results):
        """Compare performance across multiple models"""
        print(f"\n{'='*40}")
        print("MODEL COMPARISON")
        print(f"{'='*40}")
        
        comparison_data = []
        for model_name, results in all_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Macro F1': results['classification_report']['macro avg']['f1-score'],
                'Weighted F1': results['classification_report']['weighted avg']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("\nModel comparison saved to model_comparison.csv")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
        for i, metric in enumerate(metrics):
            axes[i].bar(comparison_df['Model'], comparison_df[metric])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for j, v in enumerate(comparison_df[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    # Change to parent directory to access models and data
    os.chdir('/home/inh_ua/projects/01-phat-lv')
    
    # Initialize evaluator
    evaluator = CellDetectionEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    return results

if __name__ == "__main__":
    results = main()