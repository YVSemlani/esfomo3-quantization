# Nemotron Hybrid Model Quantization Framework

Advanced quantization research framework for NVIDIA Nemotron hybrid models, specializing in state space model (SSM) quantization with multi-phase strategies.

## Project Overview

This framework implements sophisticated quantization techniques for Nemotron hybrid architectures that combine Mamba-2 state space models with transformer attention layers. The project focuses on developing phase-specific quantization strategies to achieve optimal memory reduction while preserving model accuracy.

### Key Features

- **Multi-Phase Quantization**: Progressive quantization strategies from conservative to aggressive
- **Hybrid Architecture Support**: Specialized handling for SSM + Transformer components  
- **MambaQuant Integration**: Advanced quantization for state space models
- **Comprehensive Evaluation**: Automated benchmarking on standard NLP tasks
- **Hardware Optimization**: Designed for NVIDIA GPU deployment

---

## Directory Structure

```
esfomo3/
├── Core Quantization Implementation
│   ├── phase_1_quantization.py          # W8A8 uniform quantization (conservative)
│   ├── phase_2_quantization.py          # W4A8 uniform quantization (aggressive) 
│   ├── phase_3_quantization.py          # Block-specific mixed precision (advanced)
│   └── analyze_nemotron_model.py        # Model architecture analysis tool
│
├── Evaluation Framework
│   ├── evaluation_framework.py          # Comprehensive benchmarking suite
│   ├── base_evals.py                   # Basic evaluation script
│   ├── quantized_evals.py              # Quantized model evaluation
│   └── run_analysis.py                 # Model analysis runner
│
├── Testing Suite
│   ├── test_phase1_basic_functionality.py    # Phase 1 functionality tests
│   ├── test_evaluation_framework_basic.py    # Evaluation framework tests
│   └── test_evaluation_real_benchmark.py     # Real benchmark validation
│
├── Documentation & Planning
│   ├── phase_a_development_log.md       # Detailed development tracking
│   ├── quantization_implementation_guide.md # Implementation guidelines
│   ├── nemotron_architecture_summary.md     # Architecture analysis
│   └── phased_quantization_strategy.py      # Strategy documentation
│
├── External Dependencies
│   ├── MambaQuant/                     # SSM-specific quantization framework
│   └── mamba/                          # Official Mamba implementation
│
├── Generated Outputs
│   ├── quantized_models/               # Saved quantized models
│   ├── evaluation_results/             # Benchmark results
│   ├── test_benchmark_output/          # Test outputs
│   ├── test_eval_output/               # Test evaluation results
│   └── test_output/                    # General test outputs
│
└── Utility Scripts
    ├── main.py                         # Basic model loading example
    ├── debug_layer_names.py            # Layer debugging utility
    └── phased_quantization_strategy.py # Strategy implementation
```

---

## File Functionality

### 🎯 Core Quantization Files

#### `phase_1_quantization.py` (13KB, 333 lines)
**Conservative W8A8 uniform quantization implementation**
- Target: 50% memory reduction with minimal accuracy loss
- Integrates MambaQuant for SSM layers, AutoGPTQ for MLP/Attention
- Complete pipeline: load → analyze → quantize → validate → save
- Memory estimation and performance tracking

#### `phase_2_quantization.py` (3.1KB, 91 lines) 
**Aggressive W4A8 uniform quantization (skeleton)**
- Target: 75% memory reduction with monitored accuracy trade-offs
- 4-bit weight quantization with enhanced calibration
- Status: Framework ready, implementation pending Phase 1 validation

#### `phase_3_quantization.py` (6.2KB, 161 lines)
**Block-specific mixed precision quantization (skeleton)**
- SSM blocks: W4A8, MLP/Attention blocks: W8A8
- Custom activation conversion layers for precision transitions
- Status: Advanced framework designed, implementation pending

#### `analyze_nemotron_model.py` (39KB, 872 lines)
**Comprehensive model architecture analyzer**
- Layer categorization: SSM, MLP, Attention, etc.
- Parameter counting and memory estimation
- Quantization suitability assessment
- Export capabilities for detailed analysis

### 📊 Evaluation Framework

#### `evaluation_framework.py` (16KB, 451 lines)
**Complete benchmarking and evaluation suite**
- Support for all major NLP benchmarks: arc_easy, arc_challenge, piqa, winogrande, hellaswag
- Memory usage and inference speed measurement
- Hardware detection and optimization
- Batch evaluation for model comparison
- Results export and visualization

#### `base_evals.py` / `quantized_evals.py` / `run_analysis.py`
**Specialized evaluation scripts**
- Baseline model evaluation
- Quantized model benchmarking  
- Analysis pipeline coordination

### 🧪 Testing Suite

#### `test_phase1_basic_functionality.py` (3.5KB, 125 lines)
**Phase 1 implementation validation**
- Memory estimation testing
- Quantizer initialization validation
- Model analyzer functionality verification

#### `test_evaluation_framework_basic.py` (4.8KB, 174 lines)
**Evaluation framework validation**
- Configuration testing
- Model loading verification
- Hardware detection validation
- Inference speed measurement testing

#### `test_evaluation_real_benchmark.py` (3.1KB, 96 lines)
**Real benchmark pipeline validation**
- Complete evaluation pipeline testing
- Actual benchmark execution (arc_easy)
- Results validation and export testing

### 📚 External Dependencies

#### `MambaQuant/` Submodule
**Advanced quantization framework for Mamba models**
- Variance-aligned rotation methods (KLT-Enhanced, Smooth-Fused)
- Hardware-aware quantization algorithms
- Support for VIM, Mamba2D, Mamba3D architectures

#### `mamba/` Submodule  
**Official Mamba state space model implementation**
- Reference implementation for SSM architectures
- Core algorithms and model definitions

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA support
- 16GB+ GPU memory (recommended for Nemotron-8B)

### Environment Setup

```bash
# Clone repository with submodules
git clone --recursive https://github.com/yourusername/esfomo3.git
cd esfomo3

# Create virtual environment  
python -m venv esfomo3-venv
source esfomo3-venv/bin/activate  # Linux/Mac
# or
esfomo3-venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers auto-gptq lm-eval accelerate pandas psutil
```

### Verify Installation

```bash
# Test basic functionality
python test_phase1_basic_functionality.py

# Test evaluation framework
python test_evaluation_framework_basic.py

# Test with real benchmark (takes ~3 minutes)
python test_evaluation_real_benchmark.py
```

---

## Basic Usage

### Phase 1 W8A8 Quantization

```python
from phase_1_quantization import Phase1Quantizer

# Initialize quantizer
quantizer = Phase1Quantizer(
    model_name="nvidia/Nemotron-H-8B-Base-8K",
    output_dir="./quantized_models/phase1_w8a8"
)

# Run complete quantization pipeline
results = quantizer.run_complete_phase1_pipeline()

print(f"Memory reduction: {results['memory_reduction_percent']:.1f}%")
```

### Model Evaluation

```python
from evaluation_framework import EvaluationConfig, ModelEvaluator

# Configure evaluation
config = EvaluationConfig(
    model_path="nvidia/Nemotron-H-8B-Base-8K",
    model_type="original",
    tasks=["arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"]
)

# Run evaluation
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(config)

# Save results
evaluator.save_results(results)
```

### Model Architecture Analysis

```python
from analyze_nemotron_model import NemotronModelAnalyzer

# Analyze model structure
analyzer = NemotronModelAnalyzer("nvidia/Nemotron-H-8B-Base-8K")
analyzer.load_model()
analyzer.analyze_all_layers()

# Print comprehensive analysis
analyzer.print_summary()
analyzer.print_quantization_plan()
```

---

## Development Status

### ✅ Phase A: Foundation Complete
- **Infrastructure**: All quantization and evaluation frameworks implemented
- **Testing**: Comprehensive test suite with real benchmark validation
- **Integration**: MambaQuant and AutoGPTQ frameworks ready
- **Hardware**: NVIDIA A40 GPU environment validated

### 🔄 Current Priority: Phase 1 W8A8 Implementation
- Framework: Complete and tested
- Next: Implement actual quantization logic
- Target: 50% memory reduction with <1% accuracy loss

### 📋 Future Phases
- **Phase 2**: W4A8 aggressive quantization (75% memory reduction)
- **Phase 3**: Block-specific mixed precision (optimal balance)

---

## Research Context

This framework addresses key challenges in quantizing hybrid SSM-Transformer architectures:

1. **State Space Model Quantization**: Specialized techniques for temporal state compression
2. **Hybrid Architecture Handling**: Different strategies for SSM vs Attention components  
3. **Long Context Efficiency**: Leveraging million-token capability for calibration
4. **Production Deployment**: Consumer RTX hardware optimization

### Target Models
- **Primary**: NVIDIA Nemotron-H-8B-Base-8K (hybrid architecture)
- **Architecture**: Mamba-2 SSM + selective attention layers
- **Context**: 8K-1M token support
- **Deployment**: RTX 4090/5090 consumer hardware

---

## Contributing

This is an active research project. Current development focuses on implementing Phase 1 W8A8 quantization with MambaQuant integration.

### Development Workflow
1. All changes tracked in `phase_a_development_log.md`
2. Commit early and often with detailed messages
3. Test all changes with provided test suite
4. Update documentation for new features

### Contact
**Yash Semlani** - yashvsemlani@gmail.com

---

## License

This project builds on several open-source frameworks:
- MambaQuant: [Original License]
- Mamba: Apache 2.0 
- AutoGPTQ: MIT License
- Transformers: Apache 2.0

Research portions follow academic fair use guidelines.
