#!/usr/bin/env python3
"""
Comprehensive Testing Script for GRAD-CAM Implementation
=======================================================

Test suite để verify GRAD-CAM implementation correctness và performance
Includes unit tests, integration tests, và visual verification

Author: Cell Image-Based Disease Detection Team
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import time
from typing import List, Dict, Any

# Import our GRAD-CAM implementation
from gradcam_densenet import DenseNetGradCAM

class GradCAMTester:
    """Comprehensive testing class cho GRAD-CAM implementation"""
    
    def __init__(self, model_path: str, test_images_dir: str):
        """
        Initialize tester
        
        Args:
            model_path: Path to trained model
            test_images_dir: Directory containing test images
        """
        self.model_path = model_path
        self.test_images_dir = test_images_dir
        self.gradcam = None
        self.test_results = {}
        
    def setup(self) -> bool:
        """Setup GRAD-CAM instance"""
        try:
            print("🔧 Setting up GRAD-CAM...")
            self.gradcam = DenseNetGradCAM(self.model_path)
            print(f"✅ Setup completed. Target layer: {self.gradcam.target_layer_name}")
            return True
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test model loading và basic properties"""
        print("\\n🧪 Testing model loading...")
        
        results = {
            'test_name': 'model_loading',
            'passed': False,
            'details': {}
        }
        
        try:
            # Check model exists
            results['details']['model_exists'] = os.path.exists(self.model_path)
            
            # Check model can be loaded
            results['details']['model_loadable'] = self.gradcam.model is not None
            
            # Check model structure
            results['details']['model_layers_count'] = len(self.gradcam.model.layers)
            results['details']['input_shape'] = self.gradcam.model.input_shape
            results['details']['output_shape'] = self.gradcam.model.output_shape
            
            # Check target layer
            results['details']['target_layer_found'] = self.gradcam.target_layer_name is not None
            results['details']['target_layer_name'] = self.gradcam.target_layer_name
            
            # Check gradient model
            results['details']['grad_model_created'] = self.gradcam.grad_model is not None
            
            results['passed'] = all([
                results['details']['model_exists'],
                results['details']['model_loadable'],
                results['details']['target_layer_found'],
                results['details']['grad_model_created']
            ])
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"   Model loading: {status}")
        
        return results
    
    def test_preprocessing(self) -> Dict[str, Any]:
        """Test image preprocessing functionality"""
        print("\\n🧪 Testing image preprocessing...")
        
        results = {
            'test_name': 'preprocessing',
            'passed': False,
            'details': {}
        }
        
        try:
            # Find a test image
            test_image = self._get_sample_image()
            if test_image is None:
                results['details']['error'] = "No test images found"
                return results
            
            # Test preprocessing
            img_array, original_img = self.gradcam.preprocess_image(test_image)
            
            # Verify shapes
            results['details']['preprocessed_shape'] = img_array.shape
            results['details']['original_shape'] = original_img.shape
            results['details']['batch_dimension_added'] = len(img_array.shape) == 4
            results['details']['correct_input_size'] = img_array.shape[1:3] == tuple(self.gradcam.input_size)
            
            # Verify data ranges
            results['details']['preprocessed_range'] = {
                'min': float(img_array.min()),
                'max': float(img_array.max())
            }
            results['details']['original_range'] = {
                'min': float(original_img.min()),
                'max': float(original_img.max())
            }
            
            results['passed'] = all([
                results['details']['batch_dimension_added'],
                results['details']['correct_input_size']
            ])
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"   Image preprocessing: {status}")
        
        return results
    
    def test_gradcam_computation(self) -> Dict[str, Any]:
        """Test core GRAD-CAM computation"""
        print("\\n🧪 Testing GRAD-CAM computation...")
        
        results = {
            'test_name': 'gradcam_computation',
            'passed': False,
            'details': {}
        }
        
        try:
            test_image = self._get_sample_image()
            if test_image is None:
                results['details']['error'] = "No test images found"
                return results
            
            # Preprocess image
            img_array, _ = self.gradcam.preprocess_image(test_image)
            
            # Compute GRAD-CAM
            start_time = time.time()
            heatmap, info = self.gradcam.compute_gradcam(img_array)
            computation_time = time.time() - start_time
            
            # Verify heatmap properties
            results['details']['heatmap_shape'] = heatmap.shape
            results['details']['heatmap_correct_shape'] = heatmap.shape == tuple(self.gradcam.input_size)
            results['details']['heatmap_range'] = {
                'min': float(heatmap.min()),
                'max': float(heatmap.max())
            }
            results['details']['heatmap_normalized'] = (heatmap.min() >= 0) and (heatmap.max() <= 1)
            results['details']['heatmap_not_empty'] = heatmap.max() > 0
            results['details']['computation_time_sec'] = computation_time
            
            # Verify prediction info
            results['details']['prediction_info'] = info
            results['details']['valid_confidence'] = 0 <= info['confidence'] <= 1
            results['details']['valid_class_index'] = 0 <= info['predicted_class'] < len(self.gradcam.class_names)
            results['details']['probabilities_sum'] = sum(info['all_probabilities'])
            results['details']['probabilities_valid'] = abs(sum(info['all_probabilities']) - 1.0) < 1e-6
            
            results['passed'] = all([
                results['details']['heatmap_correct_shape'],
                results['details']['heatmap_normalized'],
                results['details']['heatmap_not_empty'],
                results['details']['valid_confidence'],
                results['details']['valid_class_index'],
                results['details']['probabilities_valid']
            ])
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"   GRAD-CAM computation: {status}")
        
        return results
    
    def test_visualization(self) -> Dict[str, Any]:
        """Test visualization functionality"""
        print("\\n🧪 Testing visualization...")
        
        results = {
            'test_name': 'visualization',
            'passed': False,
            'details': {}
        }
        
        try:
            test_image = self._get_sample_image()
            if test_image is None:
                results['details']['error'] = "No test images found"
                return results
            
            # Test overlay creation
            img_array, original_img = self.gradcam.preprocess_image(test_image)
            heatmap, _ = self.gradcam.compute_gradcam(img_array)
            overlay = self.gradcam.create_heatmap_overlay(original_img, heatmap)
            
            # Verify overlay properties
            results['details']['overlay_shape'] = overlay.shape
            results['details']['overlay_same_size_as_original'] = overlay.shape == original_img.shape
            results['details']['overlay_range'] = {
                'min': int(overlay.min()),
                'max': int(overlay.max())
            }
            results['details']['overlay_valid_range'] = (overlay.min() >= 0) and (overlay.max() <= 255)
            
            # Test full visualization pipeline
            temp_output = "/tmp/test_gradcam_viz.png"
            viz_results = self.gradcam.visualize_gradcam(
                img_path=test_image,
                save_path=temp_output,
                figsize=(15, 5)
            )
            
            results['details']['visualization_completed'] = viz_results is not None
            results['details']['output_file_created'] = os.path.exists(temp_output)
            
            # Cleanup
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            results['passed'] = all([
                results['details']['overlay_same_size_as_original'],
                results['details']['overlay_valid_range'],
                results['details']['visualization_completed']
            ])
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"   Visualization: {status}")
        
        return results
    
    def test_batch_processing(self) -> Dict[str, Any]:
        """Test batch processing functionality"""
        print("\\n🧪 Testing batch processing...")
        
        results = {
            'test_name': 'batch_processing',
            'passed': False,
            'details': {}
        }
        
        try:
            # Get multiple test images
            test_images = self._get_multiple_sample_images(3)
            results['details']['test_images_count'] = len(test_images)
            
            if len(test_images) == 0:
                results['details']['error'] = "No test images found"
                return results
            
            # Test batch processing
            temp_output_dir = "/tmp/gradcam_batch_test"
            os.makedirs(temp_output_dir, exist_ok=True)
            
            start_time = time.time()
            batch_results = self.gradcam.batch_gradcam(
                image_paths=test_images,
                output_dir=temp_output_dir
            )
            processing_time = time.time() - start_time
            
            results['details']['batch_results_count'] = len(batch_results)
            results['details']['all_images_processed'] = len(batch_results) == len(test_images)
            results['details']['processing_time_sec'] = processing_time
            results['details']['avg_time_per_image'] = processing_time / len(test_images) if test_images else 0
            
            # Verify batch results structure
            if batch_results:
                sample_result = batch_results[0]
                results['details']['result_has_required_keys'] = all(
                    key in sample_result for key in ['image_path', 'heatmap', 'overlay', 'info', 'target_layer']
                )
            
            # Cleanup
            import shutil
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
            
            results['passed'] = all([
                results['details']['all_images_processed'],
                results['details'].get('result_has_required_keys', True),
                len(batch_results) > 0
            ])
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"   Batch processing: {status}")
        
        return results
    
    def test_verification_functions(self) -> Dict[str, Any]:
        """Test verification functionality"""
        print("\\n🧪 Testing verification functions...")
        
        results = {
            'test_name': 'verification_functions',
            'passed': False,
            'details': {}
        }
        
        try:
            test_image = self._get_sample_image()
            if test_image is None:
                results['details']['error'] = "No test images found"
                return results
            
            # Run verification
            verification = self.gradcam.verify_gradcam(test_image)
            
            results['details']['verification_completed'] = verification is not None
            results['details']['verification_results'] = verification
            results['details']['all_checks_defined'] = all(
                key in verification for key in [
                    'heatmap_shape_correct', 'heatmap_range_valid', 'heatmap_not_empty',
                    'confidence_valid', 'prediction_consistent', 'probabilities_sum_to_one',
                    'all_checks_passed'
                ]
            )
            
            results['passed'] = all([
                results['details']['verification_completed'],
                results['details']['all_checks_defined'],
                isinstance(verification.get('all_checks_passed'), bool)
            ])
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"   Verification functions: {status}")
        
        return results
    
    def run_performance_test(self) -> Dict[str, Any]:
        """Test performance characteristics"""
        print("\\n⚡ Running performance test...")
        
        results = {
            'test_name': 'performance',
            'passed': False,
            'details': {}
        }
        
        try:
            test_images = self._get_multiple_sample_images(5)
            
            if len(test_images) == 0:
                results['details']['error'] = "No test images found"
                return results
            
            # Time individual operations
            times = []
            
            for img_path in test_images:
                start_time = time.time()
                
                # Preprocessing
                preprocess_start = time.time()
                img_array, original_img = self.gradcam.preprocess_image(img_path)
                preprocess_time = time.time() - preprocess_start
                
                # GRAD-CAM computation
                gradcam_start = time.time()
                heatmap, info = self.gradcam.compute_gradcam(img_array)
                gradcam_time = time.time() - gradcam_start
                
                # Visualization
                viz_start = time.time()
                overlay = self.gradcam.create_heatmap_overlay(original_img, heatmap)
                viz_time = time.time() - viz_start
                
                total_time = time.time() - start_time
                
                times.append({
                    'preprocess': preprocess_time,
                    'gradcam': gradcam_time,
                    'visualization': viz_time,
                    'total': total_time
                })
            
            # Compute statistics
            avg_times = {
                'preprocess': np.mean([t['preprocess'] for t in times]),
                'gradcam': np.mean([t['gradcam'] for t in times]),
                'visualization': np.mean([t['visualization'] for t in times]),
                'total': np.mean([t['total'] for t in times])
            }
            
            results['details']['individual_times'] = times
            results['details']['average_times'] = avg_times
            results['details']['images_processed'] = len(test_images)
            
            # Performance benchmarks (reasonable thresholds)
            results['details']['preprocess_fast_enough'] = avg_times['preprocess'] < 1.0  # < 1 second
            results['details']['gradcam_fast_enough'] = avg_times['gradcam'] < 5.0       # < 5 seconds  
            results['details']['visualization_fast_enough'] = avg_times['visualization'] < 1.0  # < 1 second
            results['details']['total_fast_enough'] = avg_times['total'] < 7.0           # < 7 seconds total
            
            results['passed'] = all([
                results['details']['preprocess_fast_enough'],
                results['details']['gradcam_fast_enough'],
                results['details']['visualization_fast_enough'],
                results['details']['total_fast_enough']
            ])
            
        except Exception as e:
            results['details']['error'] = str(e)
        
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"   Performance test: {status}")
        print(f"   Average total time: {results['details'].get('average_times', {}).get('total', 0):.2f}s per image")
        
        return results
    
    def run_all_tests(self, save_results: bool = True) -> Dict[str, Any]:
        """Run complete test suite"""
        print("🚀 Running GRAD-CAM Test Suite")
        print("=" * 50)
        
        if not self.setup():
            return {'status': 'setup_failed'}
        
        # Run all tests
        test_methods = [
            self.test_model_loading,
            self.test_preprocessing,
            self.test_gradcam_computation,
            self.test_visualization,
            self.test_batch_processing,
            self.test_verification_functions,
            self.run_performance_test
        ]
        
        all_results = []
        
        for test_method in test_methods:
            try:
                result = test_method()
                all_results.append(result)
            except Exception as e:
                all_results.append({
                    'test_name': test_method.__name__,
                    'passed': False,
                    'details': {'error': str(e)}
                })
        
        # Summary
        passed_tests = sum(1 for r in all_results if r.get('passed', False))
        total_tests = len(all_results)
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'test_images_dir': self.test_images_dir,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'overall_status': 'PASS' if passed_tests == total_tests else 'FAIL',
            'individual_results': all_results
        }
        
        print("\\n📊 TEST SUMMARY")
        print("=" * 30)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Overall status: {summary['overall_status']}")
        
        # Save results
        if save_results:
            results_path = "gradcam_test_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"\\n💾 Test results saved to: {results_path}")
        
        return summary
    
    def _get_sample_image(self) -> str:
        """Get a single sample image for testing"""
        test_images = self._get_multiple_sample_images(1)
        return test_images[0] if test_images else None
    
    def _get_multiple_sample_images(self, max_count: int = 5) -> List[str]:
        """Get multiple sample images for testing"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = []
        
        if os.path.isdir(self.test_images_dir):
            for ext in image_extensions:
                test_images.extend(Path(self.test_images_dir).glob(f"**/*{ext}"))
                test_images.extend(Path(self.test_images_dir).glob(f"**/*{ext.upper()}"))
                if len(test_images) >= max_count:
                    break
        
        return [str(p) for p in test_images[:max_count]]


def main():
    """Main testing function"""
    # Configuration
    MODEL_NAME = "DenseNet121_Transfer_4Class" 
    model_path = f"./models/final_{MODEL_NAME}.h5"
    test_images_dir = "./data/test"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using densenet.py")
        return
    
    # Check if test images exist
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        print("Please provide a valid test images directory")
        return
    
    # Run tests
    tester = GradCAMTester(model_path, test_images_dir)
    results = tester.run_all_tests(save_results=True)
    
    # Exit with appropriate code
    if results.get('overall_status') == 'PASS':
        print("\\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()