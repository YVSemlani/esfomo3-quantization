#!/usr/bin/env python3
"""
Debug script to examine actual layer names and types in Nemotron model.
This will help us understand the naming conventions and fix categorization.
"""

import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict

def examine_model_structure():
    """Load model and examine actual layer names and types"""
    print("Loading Nemotron model to examine layer structure...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/Nemotron-H-8B-Base-8K",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model config: {model.config}")
    
    print("\n" + "="*80)
    print("EXAMINING ALL LAYER NAMES AND TYPES")
    print("="*80)
    
    # Group by module type
    module_types = defaultdict(list)
    
    # Examine all named modules
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        module_types[module_type].append(name)
    
    # Print summary by module type
    print("\nMODULE TYPES SUMMARY:")
    print("-" * 50)
    for module_type, names in sorted(module_types.items()):
        print(f"{module_type}: {len(names)} instances")
        if len(names) <= 5:  # Show all if 5 or fewer
            for name in names:
                print(f"  - {name}")
        else:  # Show first few examples
            for name in names[:3]:
                print(f"  - {name}")
            print(f"  ... and {len(names)-3} more")
        print()
    
    print("\n" + "="*80)
    print("DETAILED LAYER ANALYSIS - ATTENTION PATTERNS")
    print("="*80)
    
    # Look for attention-related layers specifically
    attention_keywords = ['attn', 'attention', 'self_attn', 'cross_attn', 'multihead', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    print("Searching for attention-related layers:")
    attention_found = False
    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in attention_keywords):
            print(f"  ATTENTION: {name} ({module.__class__.__name__})")
            attention_found = True
    
    if not attention_found:
        print("  No obvious attention layers found with standard keywords")
    
    print("\n" + "="*80)
    print("DETAILED LAYER ANALYSIS - MAMBA PATTERNS")
    print("="*80)
    
    # Look for mamba-related layers
    mamba_keywords = ['mamba', 'ssm', 'selective', 'mixer', 'state_space', 'ssd']
    
    print("Searching for mamba-related layers:")
    mamba_found = False
    for name, module in model.named_modules():
        name_lower = name.lower()
        module_name = module.__class__.__name__.lower()
        if any(keyword in name_lower or keyword in module_name for keyword in mamba_keywords):
            print(f"  MAMBA: {name} ({module.__class__.__name__})")
            mamba_found = True
            if len([n for n in model.named_modules() if any(k in n[0].lower() or k in n[1].__class__.__name__.lower() for k in mamba_keywords)]) > 20:
                break  # Limit output if too many
    
    if not mamba_found:
        print("  No obvious mamba layers found with standard keywords")
    
    print("\n" + "="*80)
    print("DETAILED LAYER ANALYSIS - MLP PATTERNS")
    print("="*80)
    
    # Look for MLP-related layers
    mlp_keywords = ['mlp', 'ffn', 'feed_forward', 'gate_proj', 'up_proj', 'down_proj', 'fc']
    
    print("Searching for MLP-related layers:")
    mlp_found = False
    for name, module in model.named_modules():
        name_lower = name.lower()
        module_name = module.__class__.__name__.lower()
        if any(keyword in name_lower or keyword in module_name for keyword in mlp_keywords):
            print(f"  MLP: {name} ({module.__class__.__name__})")
            mlp_found = True
    
    if not mlp_found:
        print("  No obvious MLP layers found with standard keywords")
    
    print("\n" + "="*80)
    print("LAYER STRUCTURE HIERARCHY")
    print("="*80)
    
    # Examine the first few layers of the model hierarchy
    print("Model structure (first 2 levels):")
    def print_model_structure(module, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return
        
        for name, child in module.named_children():
            print(f"{prefix}{name}: {child.__class__.__name__}")
            if current_depth < max_depth - 1:
                print_model_structure(child, prefix + "  ", max_depth, current_depth + 1)
    
    print_model_structure(model)
    
    print("\n" + "="*80)
    print("QUANTIZABLE LAYERS ANALYSIS")
    print("="*80)
    
    # Focus on quantizable layers
    quantizable_types = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Embedding)
    
    quantizable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, quantizable_types) and len(list(module.children())) == 0:
            param_count = sum(p.numel() for p in module.parameters())
            quantizable_layers.append((name, module.__class__.__name__, param_count))
    
    # Sort by parameter count
    quantizable_layers.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Found {len(quantizable_layers)} quantizable layers")
    print("\nTop 20 largest quantizable layers:")
    for i, (name, module_type, param_count) in enumerate(quantizable_layers[:20]):
        print(f"{i+1:2d}. {name:<60} {module_type:<15} {param_count:>10,} params")
    
    print("\n" + "="*80)
    print("LOOKING FOR LAYER PATTERNS IN NAMES")
    print("="*80)
    
    # Analyze layer name patterns
    all_names = [name for name, _ in model.named_modules()]
    
    # Look for numbered patterns that might indicate layer structure
    import re
    
    # Look for patterns like "layers.0", "blocks.1", etc.
    layer_patterns = defaultdict(list)
    for name in all_names:
        # Extract patterns like layers.N, blocks.N, etc.
        pattern_matches = re.findall(r'(layers|blocks|stages|transformer)\.\d+', name)
        for pattern in pattern_matches:
            layer_patterns[pattern.split('.')[0]].append(name)
    
    print("Layer grouping patterns found:")
    for pattern, names in layer_patterns.items():
        print(f"{pattern}: {len(names)} layers")
        # Show a few examples
        unique_prefixes = set()
        for name in names[:10]:
            parts = name.split('.')
            if len(parts) >= 3:
                prefix = '.'.join(parts[:3])
                unique_prefixes.add(prefix)
        
        print(f"  Example prefixes: {sorted(list(unique_prefixes))[:5]}")
    
    return model

if __name__ == "__main__":
    model = examine_model_structure() 