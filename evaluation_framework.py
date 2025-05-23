#!/usr/bin/env python3
"""
Evaluation Framework for Nemotron Quantized Models

Comprehensive evaluation suite for benchmarking quantized models against original performance.
Supports arc_easy, arc_challenge, piqa, winogrande, and hellaswag benchmarks.

Author: esfomo3 research team
Date: 2024-12-19
"""

import torch
import json
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import pandas as pd

try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("⚠️  lm_eval not available. Install with: pip install lm-eval")
    evaluator = None
    HFLM = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""
    model_path: str
    model_type: str  # "original", "phase1", "phase2", "phase3"
    batch_size: int = 1
    device: str = "cuda"
    max_length: int = 2048
    tasks: List[str] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"]


@dataclass 
class EvaluationResults:
    """Results from evaluation run"""
    model_type: str
    model_path: str
    task_results: Dict[str, Dict[str, float]]
    memory_usage_mb: float
    inference_time_per_token_ms: float
    total_evaluation_time_seconds: float
    hardware_info: Dict[str, Any]
    errors: List[str]


class ModelEvaluator:
    """Evaluator for quantized Nemotron models"""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if lm_eval is available
        if evaluator is None:
            raise ImportError("lm_eval not available. Install with: pip install lm-eval")
    
    def load_model(self, config: EvaluationConfig) -> tuple:
        """Load model and tokenizer based on configuration"""
        print(f"🔍 Loading {config.model_type} model from {config.model_path}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path, 
                trust_remote_code=True
            )
            
            # Load model based on type
            if config.model_type == "original":
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif config.model_type.startswith("phase"):
                # Load quantized model
                model = AutoGPTQForCausalLM.from_quantized(
                    config.model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")
            
            print(f"✅ Model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def measure_memory_usage(self) -> float:
        """Measure current GPU memory usage in MB"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
            return memory_allocated
        else:
            # Fallback to system memory for CPU inference
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)
    
    def measure_inference_speed(self, model, tokenizer, num_samples: int = 10) -> float:
        """Measure inference speed in ms per token"""
        print("⏱️ Measuring inference speed...")
        
        test_prompts = [
            "The capital of France is",
            "Artificial intelligence is",
            "The future of technology",
            "Machine learning models",
            "Quantum computing will"
        ]
        
        total_time = 0
        total_tokens = 0
        
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for i in range(num_samples):
                prompt = test_prompts[i % len(test_prompts)]
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                start_time = time.time()
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                end_time = time.time()
                
                # Count new tokens generated
                new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                total_tokens += new_tokens
                total_time += (end_time - start_time)
        
        # Convert to ms per token
        ms_per_token = (total_time / total_tokens) * 1000
        print(f"✅ Inference speed: {ms_per_token:.2f} ms/token")
        return ms_per_token
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information for evaluation context"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        return info
    
    def run_lm_eval_benchmarks(self, config: EvaluationConfig) -> Dict[str, Dict[str, float]]:
        """Run lm-eval benchmarks on specified tasks"""
        print(f"🏃 Running lm-eval benchmarks on tasks: {config.tasks}")
        
        try:
            # Prepare model arguments for lm-eval
            if config.model_type == "original":
                model_args = f"pretrained={config.model_path},trust_remote_code=True,dtype=float16"
            elif config.model_type.startswith("phase"):
                # For quantized models, need to specify they're already quantized
                model_args = f"pretrained={config.model_path},trust_remote_code=True,use_quantized=True"
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")
            
            # Run evaluation
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=config.tasks,
                device=config.device,
                batch_size=config.batch_size,
                max_length=config.max_length,
                limit=None,  # Evaluate on full dataset
                cache_requests=True
            )
            
            # Extract task results
            task_results = {}
            for task in config.tasks:
                if task in results["results"]:
                    task_results[task] = results["results"][task]
                else:
                    print(f"⚠️ Results for task {task} not found")
            
            print(f"✅ Benchmarks completed for {len(task_results)} tasks")
            return task_results
            
        except Exception as e:
            print(f"❌ Benchmark evaluation failed: {e}")
            raise
    
    def evaluate_model(self, config: EvaluationConfig) -> EvaluationResults:
        """Run complete evaluation for a model"""
        print(f"\n🎯 Starting evaluation for {config.model_type} model")
        print("=" * 60)
        
        start_time = time.time()
        errors = []
        
        try:
            # Load model
            model, tokenizer = self.load_model(config)
            
            # Measure memory usage
            memory_usage = self.measure_memory_usage()
            
            # Measure inference speed
            inference_speed = self.measure_inference_speed(model, tokenizer)
            
            # Run benchmarks
            task_results = self.run_lm_eval_benchmarks(config)
            
            # Get hardware info
            hardware_info = self.get_hardware_info()
            
            total_time = time.time() - start_time
            
            # Create results object
            results = EvaluationResults(
                model_type=config.model_type,
                model_path=config.model_path,
                task_results=task_results,
                memory_usage_mb=memory_usage,
                inference_time_per_token_ms=inference_speed,
                total_evaluation_time_seconds=total_time,
                hardware_info=hardware_info,
                errors=errors
            )
            
            print(f"✅ Evaluation completed successfully in {total_time:.1f} seconds")
            return results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")
            
            # Return partial results
            return EvaluationResults(
                model_type=config.model_type,
                model_path=config.model_path,
                task_results={},
                memory_usage_mb=0,
                inference_time_per_token_ms=0,
                total_evaluation_time_seconds=time.time() - start_time,
                hardware_info=self.get_hardware_info(),
                errors=errors
            )
    
    def save_results(self, results: EvaluationResults, filename: Optional[str] = None) -> str:
        """Save evaluation results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{results.model_type}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to dictionary for JSON serialization
        results_dict = {
            "model_type": results.model_type,
            "model_path": results.model_path,
            "task_results": results.task_results,
            "memory_usage_mb": results.memory_usage_mb,
            "inference_time_per_token_ms": results.inference_time_per_token_ms,
            "total_evaluation_time_seconds": results.total_evaluation_time_seconds,
            "hardware_info": results.hardware_info,
            "errors": results.errors,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")
        return str(filepath)
    
    def compare_models(self, results_list: List[EvaluationResults]) -> pd.DataFrame:
        """Create comparison table of multiple model evaluations"""
        print("📊 Creating model comparison table...")
        
        comparison_data = []
        
        for results in results_list:
            row = {
                "Model Type": results.model_type,
                "Memory (MB)": f"{results.memory_usage_mb:.1f}",
                "Speed (ms/token)": f"{results.inference_time_per_token_ms:.2f}",
                "Eval Time (s)": f"{results.total_evaluation_time_seconds:.1f}",
                "Errors": len(results.errors)
            }
            
            # Add task results
            for task, task_result in results.task_results.items():
                # Get main metric for each task (usually accuracy)
                if isinstance(task_result, dict):
                    # Find main metric (acc, acc_norm, em, etc.)
                    main_metric = None
                    for metric in ["acc_norm", "acc", "em", "exact_match"]:
                        if metric in task_result:
                            main_metric = task_result[metric]
                            break
                    
                    if main_metric is not None:
                        row[f"{task}"] = f"{main_metric:.3f}"
                    else:
                        row[f"{task}"] = "N/A"
                else:
                    row[f"{task}"] = str(task_result)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print("✅ Comparison table created")
        return df


class BatchEvaluator:
    """Batch evaluator for comparing multiple models"""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.evaluator = ModelEvaluator(output_dir)
        self.output_dir = Path(output_dir)
    
    def evaluate_all_phases(self, 
                          original_model_path: str = "nvidia/Nemotron-H-8B-Base-8K",
                          phase1_path: Optional[str] = None,
                          phase2_path: Optional[str] = None,
                          phase3_path: Optional[str] = None) -> List[EvaluationResults]:
        """Evaluate all available model phases"""
        print("🎯 Starting batch evaluation of all model phases")
        print("=" * 70)
        
        results = []
        
        # Evaluate original model
        original_config = EvaluationConfig(
            model_path=original_model_path,
            model_type="original"
        )
        results.append(self.evaluator.evaluate_model(original_config))
        
        # Evaluate quantized phases if available
        if phase1_path and Path(phase1_path).exists():
            phase1_config = EvaluationConfig(
                model_path=phase1_path,
                model_type="phase1"
            )
            results.append(self.evaluator.evaluate_model(phase1_config))
        
        if phase2_path and Path(phase2_path).exists():
            phase2_config = EvaluationConfig(
                model_path=phase2_path,
                model_type="phase2"
            )
            results.append(self.evaluator.evaluate_model(phase2_config))
        
        if phase3_path and Path(phase3_path).exists():
            phase3_config = EvaluationConfig(
                model_path=phase3_path,
                model_type="phase3"
            )
            results.append(self.evaluator.evaluate_model(phase3_config))
        
        # Save all results
        for result in results:
            self.evaluator.save_results(result)
        
        # Create comparison table
        comparison_df = self.evaluator.compare_models(results)
        
        # Save comparison table
        comparison_file = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"📊 Comparison table saved to {comparison_file}")
        
        # Print summary
        print("\n📋 Evaluation Summary:")
        print(comparison_df.to_string(index=False))
        
        return results


def main():
    """Main function for running evaluations"""
    print("🎯 Nemotron Model Evaluation Framework")
    print("=" * 50)
    
    # Create batch evaluator
    batch_evaluator = BatchEvaluator()
    
    # Example: Evaluate original model
    print("Starting evaluation of original Nemotron model...")
    
    original_config = EvaluationConfig(
        model_path="nvidia/Nemotron-H-8B-Base-8K",
        model_type="original",
        batch_size=1
    )
    
    try:
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(original_config)
        evaluator.save_results(results)
        
        print(f"\n📊 Evaluation Results Summary:")
        print(f"   Memory usage: {results.memory_usage_mb:.1f} MB")
        print(f"   Inference speed: {results.inference_time_per_token_ms:.2f} ms/token")
        print(f"   Evaluation time: {results.total_evaluation_time_seconds:.1f} seconds")
        
        for task, task_result in results.task_results.items():
            if isinstance(task_result, dict) and "acc" in task_result:
                print(f"   {task}: {task_result['acc']:.3f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 