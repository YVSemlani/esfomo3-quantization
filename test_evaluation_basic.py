#!/usr/bin/env python3
"""
Basic test script for evaluation framework
Tests evaluation setup without running full benchmarks
"""

import torch
from evaluation_framework import EvaluationConfig, ModelEvaluator


def test_evaluation_config():
    """Test evaluation configuration"""
    print("🔍 Testing evaluation config...")
    
    try:
        config = EvaluationConfig(
            model_path="microsoft/DialoGPT-small",
            model_type="original",
            batch_size=1,
            tasks=["arc_easy"]  # Just one task for testing
        )
        
        print(f"✅ Config created successfully!")
        print(f"   Model: {config.model_path}")
        print(f"   Type: {config.model_type}")
        print(f"   Tasks: {config.tasks}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_model_loading():
    """Test model loading functionality"""
    print("\n🔍 Testing model loading...")
    
    try:
        evaluator = ModelEvaluator(output_dir="./test_eval_output")
        
        config = EvaluationConfig(
            model_path="microsoft/DialoGPT-small",
            model_type="original",
            batch_size=1
        )
        
        model, tokenizer = evaluator.load_model(config)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_memory_measurement():
    """Test memory measurement"""
    print("\n🔍 Testing memory measurement...")
    
    try:
        evaluator = ModelEvaluator()
        memory_usage = evaluator.measure_memory_usage()
        
        print(f"✅ Memory measurement works!")
        print(f"   Current memory usage: {memory_usage:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory measurement failed: {e}")
        return False


def test_inference_speed():
    """Test inference speed measurement"""
    print("\n🔍 Testing inference speed measurement...")
    
    try:
        evaluator = ModelEvaluator(output_dir="./test_eval_output")
        
        config = EvaluationConfig(
            model_path="microsoft/DialoGPT-small",
            model_type="original",
            batch_size=1
        )
        
        model, tokenizer = evaluator.load_model(config)
        
        # Test with reduced samples for speed
        speed = evaluator.measure_inference_speed(model, tokenizer, num_samples=2)
        
        print(f"✅ Inference speed measurement works!")
        print(f"   Speed: {speed:.2f} ms/token")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference speed test failed: {e}")
        return False


def test_hardware_info():
    """Test hardware information collection"""
    print("\n🔍 Testing hardware info collection...")
    
    try:
        evaluator = ModelEvaluator()
        hardware_info = evaluator.get_hardware_info()
        
        print(f"✅ Hardware info collection works!")
        print(f"   CPU count: {hardware_info['cpu_count']}")
        print(f"   GPU available: {hardware_info['gpu_available']}")
        if hardware_info['gpu_available']:
            print(f"   GPU name: {hardware_info['gpu_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware info test failed: {e}")
        return False


def main():
    """Run basic evaluation tests"""
    print("🎯 Evaluation Framework Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Evaluation Config", test_evaluation_config),
        ("Memory Measurement", test_memory_measurement),
        ("Hardware Info", test_hardware_info),
        ("Model Loading", test_model_loading),
        ("Inference Speed", test_inference_speed),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All evaluation tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    exit(main()) 