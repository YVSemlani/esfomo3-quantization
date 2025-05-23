#!/usr/bin/env python3
"""
Test script for running actual benchmark evaluation
Tests the complete evaluation pipeline with a real benchmark
"""

from evaluation_framework import EvaluationConfig, ModelEvaluator


def test_single_benchmark():
    """Test running a single benchmark on a small model"""
    print("🎯 Testing Real Benchmark Evaluation")
    print("=" * 50)
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(output_dir="./test_benchmark_output")
        
        # Configure for a single quick benchmark
        config = EvaluationConfig(
            model_path="microsoft/DialoGPT-small",
            model_type="original",
            batch_size=1,
            tasks=["arc_easy"],  # Single task for testing
        )
        
        print(f"📊 Running benchmark evaluation...")
        print(f"   Model: {config.model_path}")
        print(f"   Tasks: {config.tasks}")
        print(f"   This may take a few minutes...\n")
        
        # Run evaluation
        results = evaluator.evaluate_model(config)
        
        # Save results
        results_file = evaluator.save_results(results)
        
        # Print summary
        print(f"\n✅ Benchmark evaluation completed!")
        print(f"   Results saved to: {results_file}")
        print(f"   Memory usage: {results.memory_usage_mb:.1f} MB")
        print(f"   Inference speed: {results.inference_time_per_token_ms:.2f} ms/token")
        print(f"   Evaluation time: {results.total_evaluation_time_seconds:.1f} seconds")
        
        # Print benchmark results
        if results.task_results:
            print(f"\n📈 Benchmark Results:")
            for task, task_result in results.task_results.items():
                print(f"   {task}:")
                if isinstance(task_result, dict):
                    for metric, value in task_result.items():
                        if isinstance(value, (int, float)):
                            print(f"     {metric}: {value:.4f}")
                        else:
                            print(f"     {metric}: {value}")
                else:
                    print(f"     Result: {task_result}")
        
        # Check for errors
        if results.errors:
            print(f"\n⚠️  Errors encountered:")
            for error in results.errors:
                print(f"   - {error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run benchmark evaluation test"""
    print("🚀 Starting Benchmark Evaluation Test\n")
    
    try:
        success = test_single_benchmark()
        
        if success:
            print(f"\n🎉 Benchmark evaluation test completed successfully!")
            print(f"✅ Ready to proceed with Nemotron evaluation")
            return 0
        else:
            print(f"\n❌ Benchmark evaluation test failed")
            return 1
            
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 