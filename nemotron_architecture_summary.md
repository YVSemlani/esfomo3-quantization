# Nemotron-H-8B-Base-8K Architecture Analysis Summary

## Model Overview
- **Total Parameters**: 8.1B (8,100,852,736)
- **Total Memory**: 15.45 GB
- **Architecture**: Hybrid Mamba2-Transformer with selective attention
- **Context Length**: 8K tokens

## Complete Layer Breakdown

### 1. SSM (State Space Model) Components - 32.5% of model (2.63B params)

The SSM components form the core of the Mamba2 architecture and were initially missed in the analysis due to a technical oversight. Here's the complete breakdown:

#### SSM Input/Output Projections (Largest Components)
- **SSM Input Projections**: 24 layers, 1.82B parameters (69.3% of SSM)
  - Each: `backbone.layers.X.mixer.in_proj` (18560 × 4096 = 76M params)
  - **Quantization**: 4-bit (highest priority for memory savings)

- **SSM Output Projections**: 24 layers, 805M parameters (30.6% of SSM)  
  - Each: `backbone.layers.X.mixer.out_proj` (4096 × 8192 = 33.5M params)
  - **Quantization**: 4-bit (high priority)

#### SSM Auxiliary Components
- **SSM Convolution Layers**: 24 layers, 1.23M parameters
  - Each: `backbone.layers.X.mixer.conv1d` (Conv1d with kernel_size=4)
  - **Quantization**: 8-bit (medium priority)

- **SSM Normalization**: 24 layers, 197K parameters
  - Each: `backbone.layers.X.mixer.norm` (RMSNorm with 8192 dims)
  - **Quantization**: 8-bit (medium priority)

#### SSM Core Parameters (Very Small)
- **A_log (State Matrices)**: 24 parameters, 3,072 total params
  - Each: `backbone.layers.X.mixer.A_log` (128 dims)
  - **Quantization**: Skip (too small)

- **D (Skip Connections)**: 24 parameters, 3,072 total params
  - Each: `backbone.layers.X.mixer.D` (128 dims)  
  - **Quantization**: Skip (too small)

- **dt_bias (Time Step Bias)**: 24 parameters, 3,072 total params
  - Each: `backbone.layers.X.mixer.dt_bias` (128 dims)
  - **Quantization**: Skip (too small)

### 2. MLP Layers - 52.2% of model (4.23B params)
- **Count**: 48 layers (24 up_proj + 24 down_proj)
- **Pattern**: `backbone.layers.X.mixer.{up_proj,down_proj}`
- **Size**: Each pair ~176M parameters (21504 × 4096 each direction)
- **Quantization**: 8-bit standard quantization

### 3. Attention Layers - 2.1% of model (168M params)
- **Count**: 16 layers (4 attention blocks × 4 projections each)
- **Locations**: Layers 7, 18, 29, 40 only
- **Pattern**: `backbone.layers.{7,18,29,40}.mixer.{q,k,v,o}_proj`
- **Size**: Each projection ~10.5M parameters (4096 × 2560)
- **Quantization**: 8-bit standard transformer quantization

### 4. Other Components - 13.2% of model (1.07B params)
- **Embedding**: 537M parameters (131072 vocab × 4096 dims)
- **Output Head**: 537M parameters (same as embedding)
- **Normalization**: 217K parameters (53 LayerNorm layers)

## Why SSM Components Were Initially Missed

### Technical Issue
The original analysis had a critical flaw: it only examined `named_modules()` but SSM core parameters (A_log, D, dt_bias) are `nn.Parameter` objects, not modules. They're only accessible via `named_parameters()`.

### Analysis Method Problems
1. **Module-only analysis**: Only looked at `torch.nn.Module` objects
2. **Parameter categorization**: Failed to distinguish SSM-specific linear layers from general linear layers
3. **Incomplete architecture understanding**: Didn't account for Mamba2's specific parameter structure

### Resolution
The updated analysis now:
1. **Analyzes both modules AND parameters**: Captures all SSM components
2. **Proper categorization**: Distinguishes SSM projections from MLP layers
3. **Complete SSM breakdown**: Shows all 7 types of SSM components separately

## Quantization Strategy

### Recommended Approach
1. **SSM Input/Output Projections** (2.63B params): 4-bit MambaQuant
2. **MLP Layers** (4.23B params): 8-bit standard quantization  
3. **Attention Layers** (168M params): 8-bit transformer quantization
4. **SSM Auxiliary** (1.4M params): 8-bit quantization
5. **SSM Core Parameters** (9K params): Skip quantization
6. **Embeddings/Output** (1.07B params): Skip or 8-bit

### Expected Memory Savings
- **Current**: 15.45 GB
- **After quantization**: ~6.5 GB (58% reduction)
- **Primary savings**: SSM projections (4-bit) + MLP layers (8-bit)

## Architecture Pattern

The model follows this hybrid pattern across 52 layers:
```
M-M-M-M*-M-M-M-A-M*-M-M-M-M*-M-M-M-M*-M-A-M*-M-M-M-M*-M-M-M-M*-M-A-M*-M-M-M-M*-M-M-M-M*-M-A-M*-M-M-M-M*-M-M-M-M*-M
```

Where:
- **M**: Mamba2 layer (SSM components)
- **M***: MLP layer  
- **A**: Attention layer (every 11 layers: 7, 18, 29, 40)

This confirms the HuggingFace description: "primarily Mamba-2 and MLP layers combined with just four Attention layers." 