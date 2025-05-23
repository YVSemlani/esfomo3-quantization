#!/usr/bin/env python3
"""
Basic test script for Phase 1 implementation
Tests model loading and analysis without full quantization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from phase_1_quantization import Phase1Quantizer
from analyze_nemotron_model import NemotronModelAnalyzer


def test_analyzer_functionality():
    """Test the existing analyzer with a smaller model for validation"""
    print("🔍 Testing analyzer functionality...")
    
    try:
        # Use a smaller model for testing
        analyzer = NemotronModelAnalyzer("microsoft/DialoGPT-small")
        
        # Load model
        print("Loading DialoGPT-small for testing...")
        analyzer.load_model(load_in_4bit=False)
        
        # Run quick analysis
        print("Running analysis...")
        analyzer.analyze_all_layers()
        
        # Print summary
        analyzer.print_summary()
        
        print("✅ Analyzer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        return False


def test_phase1_initialization():
    """Test Phase 1 quantizer initialization"""
    print("\n🔍 Testing Phase 1 quantizer initialization...")
    
    try:
        quantizer = Phase1Quantizer(
            model_name="microsoft/DialoGPT-small",  # Use smaller model for testing
            output_dir="./test_output"
        )
        
        print(f"✅ Phase 1 quantizer initialized successfully!")
        print(f"   Model: {quantizer.model_name}")
        print(f"   Output dir: {quantizer.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 initialization failed: {e}")
        return False


def test_memory_estimation():
    """Test memory estimation functionality"""
    print("\n🔍 Testing memory estimation...")
    
    try:
        # Create a simple dummy model for testing
        dummy_model = torch.nn.Linear(100, 50)
        
        quantizer = Phase1Quantizer()
        memory_mb = quantizer._estimate_model_memory(dummy_model)
        
        print(f"✅ Memory estimation works!")
        print(f"   Dummy model memory: {memory_mb:.4f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory estimation failed: {e}")
        return False


def main():
    """Run basic tests"""
    print("🎯 Phase 1 Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        ("Memory Estimation", test_memory_estimation),
        ("Phase 1 Initialization", test_phase1_initialization),
        ("Analyzer Functionality", test_analyzer_functionality),
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
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    exit(main()) 