import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import json
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any
import numpy as np
from pathlib import Path


class NemotronModelAnalyzer:
    """
    Comprehensive analysis tool for Nemotron hybrid models.
    Analyzes layer types, shapes, parameters, and quantization potential.
    """
    
    def __init__(self, model_name: str = "nvidia/Nemotron-H-8B-Base-8K"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.layer_analysis = []
        self.summary_stats = {}
        
    def load_model(self, load_in_4bit: bool = False):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if not load_in_4bit else None
        )
        
        print(f"Model loaded successfully!")
        print(f"Model class: {self.model.__class__.__name__}")
        print(f"Device map: {getattr(self.model, 'hf_device_map', 'Not available')}")
        
    def analyze_layer(self, name: str, module: torch.nn.Module) -> Dict[str, Any]:
        """Analyze a single layer and extract relevant information"""
        
        layer_info = {
            'name': name,
            'module_type': module.__class__.__name__,
            'module_path': module.__class__.__module__,
            'layer_category': self._categorize_layer(name, module),
            'is_quantizable': self._is_quantizable(module),
            'parameter_count': self._count_parameters(module),
            'parameter_details': self._get_parameter_details(module),
            'has_bias': self._has_bias(module),
            'activation_function': self._get_activation_function(module),
            'special_attributes': self._get_special_attributes(name, module),
            'quantization_suitability': self._assess_quantization_suitability(name, module),
            'memory_usage_mb': self._estimate_memory_usage(module),
        }
        
        return layer_info
    
    def _categorize_layer(self, name: str, module: torch.nn.Module) -> str:
        """Categorize layer type for quantization strategy"""
        name_lower = name.lower()
        module_name = module.__class__.__name__.lower()
        
        # Nemotron-specific module types (most accurate)
        if 'mamba2mixer' in module_name or 'nemotrohnmamba2mixer' in module_name:
            return 'mamba_mixer'
        elif 'attention' in module_name and 'nemotron' in module_name:
            return 'attention_module'
        elif 'mlp' in module_name and 'nemotron' in module_name:
            return 'mlp_module'
        
        # SSM-specific parameters (State Space Model components)
        if 'A_log' in name:
            return 'ssm_A_log'  # SSM state matrix (log)
        elif name.endswith('.D') and 'mixer' in name:
            return 'ssm_D'  # SSM skip connection
        elif 'dt_bias' in name:
            return 'ssm_dt_bias'  # SSM time step bias
        
        # SSM layer types
        if isinstance(module, torch.nn.Conv1d) and 'mixer' in name:
            return 'ssm_conv1d'  # SSM convolution layer
        elif 'mixer' in name and 'in_proj' in name:
            return 'ssm_in_proj'  # SSM input projection 
        elif 'mixer' in name and 'out_proj' in name:
            return 'ssm_out_proj'  # SSM output projection
        elif 'mixer' in name and 'norm' in name:
            return 'ssm_norm'  # SSM normalization
        
        # Attention patterns - look for specific Nemotron attention structure
        attention_patterns = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj'
        ]
        
        # Check if this linear layer is part of an attention block
        if isinstance(module, torch.nn.Linear) and any(pattern in name_lower for pattern in attention_patterns):
            # Additional check: see if parent is attention
            parent_path = '.'.join(name.split('.')[:-1])
            if 'layers.7.' in parent_path or 'layers.18.' in parent_path or 'layers.29.' in parent_path or 'layers.40.' in parent_path:
                return 'attention'
        
        # MLP patterns - look for up_proj/down_proj in MLP layers
        if isinstance(module, torch.nn.Linear):
            if 'up_proj' in name_lower or 'down_proj' in name_lower:
                # MLP layers can be within mixer modules in hybrid architectures
                return 'mlp'
        
        # Layer normalization (excluding SSM norms already categorized)
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)) or 'norm' in name_lower:
            if 'mixer' not in name:  # SSM norms already categorized above
                return 'norm'
        
        # Embedding layers
        if isinstance(module, torch.nn.Embedding):
            return 'embedding'
        
        # Output head
        if 'lm_head' in name_lower:
            return 'output_head'
        
        # Catch remaining linear layers
        if isinstance(module, torch.nn.Linear):
            return 'linear_other'
        
        # Conv layers (non-SSM)
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            return 'conv_other'
        
        # Default
        return 'other'
    
    def _is_quantizable(self, module: torch.nn.Module) -> bool:
        """Check if module is suitable for quantization"""
        quantizable_types = (
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.Embedding
        )
        return isinstance(module, quantizable_types)
    
    def _count_parameters(self, module: torch.nn.Module) -> Dict[str, int]:
        """Count parameters in the module"""
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def _get_parameter_details(self, module: torch.nn.Module) -> List[Dict[str, Any]]:
        """Get detailed information about each parameter"""
        param_details = []
        
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None:
                param_info = {
                    'name': param_name,
                    'shape': list(param.shape),
                    'dtype': str(param.dtype),
                    'requires_grad': param.requires_grad,
                    'numel': param.numel(),
                    'memory_mb': param.numel() * param.element_size() / (1024 * 1024),
                    'min_val': float(param.min().item()) if param.numel() > 0 else None,
                    'max_val': float(param.max().item()) if param.numel() > 0 else None,
                    'mean_val': float(param.mean().item()) if param.numel() > 0 else None,
                    'std_val': float(param.std().item()) if param.numel() > 0 else None,
                }
                param_details.append(param_info)
        
        return param_details
    
    def _has_bias(self, module: torch.nn.Module) -> bool:
        """Check if module has bias"""
        if hasattr(module, 'bias'):
            return module.bias is not None
        return False
    
    def _get_activation_function(self, module: torch.nn.Module) -> str:
        """Try to identify activation function"""
        if hasattr(module, 'activation'):
            return str(module.activation)
        elif hasattr(module, 'act'):
            return str(module.act)
        elif hasattr(module, 'act_fn'):
            return str(module.act_fn)
        else:
            return 'unknown'
    
    def _get_special_attributes(self, name: str, module: torch.nn.Module) -> Dict[str, Any]:
        """Extract special attributes relevant to quantization"""
        special_attrs = {}
        
        # Common attributes that affect quantization
        attr_names = [
            'in_features', 'out_features', 'kernel_size', 'stride', 'padding',
            'groups', 'dilation', 'num_embeddings', 'embedding_dim',
            'd_model', 'd_state', 'd_conv', 'expand', 'dt_rank',
            'num_heads', 'head_dim', 'hidden_size'
        ]
        
        for attr in attr_names:
            if hasattr(module, attr):
                value = getattr(module, attr)
                if value is not None:
                    special_attrs[attr] = value
        
        return special_attrs
    
    def _assess_quantization_suitability(self, name: str, module: torch.nn.Module) -> Dict[str, Any]:
        """Assess how suitable this layer is for different quantization techniques"""
        assessment = {
            'overall_suitability': 'low',
            'recommended_bits': 16,
            'recommended_method': 'none',
            'considerations': []
        }
        
        if not self._is_quantizable(module):
            assessment['considerations'].append('Not a quantizable layer type')
            return assessment
        
        param_count = self._count_parameters(module)['total']
        
        # High parameter count layers are good candidates
        if param_count > 1000000:  # 1M parameters
            assessment['overall_suitability'] = 'high'
            assessment['recommended_bits'] = 4
        elif param_count > 100000:  # 100K parameters
            assessment['overall_suitability'] = 'medium'
            assessment['recommended_bits'] = 8
        else:
            assessment['overall_suitability'] = 'low'
            assessment['recommended_bits'] = 16
        
        # Layer-specific recommendations
        layer_category = self._categorize_layer(name, module)
        
        if layer_category == 'mamba':
            assessment['recommended_method'] = 'mamba_quant'
            assessment['considerations'].append('Use specialized Mamba quantization techniques')
        elif layer_category == 'attention':
            assessment['recommended_method'] = 'standard_gptq'
            assessment['considerations'].append('Use standard transformer quantization')
        elif layer_category == 'linear':
            assessment['recommended_method'] = 'gptq'
            assessment['considerations'].append('Standard linear layer quantization')
        elif layer_category == 'embedding':
            assessment['recommended_method'] = 'embedding_quant'
            assessment['considerations'].append('Special handling for embeddings')
        
        return assessment
    
    def _estimate_memory_usage(self, module: torch.nn.Module) -> float:
        """Estimate memory usage in MB"""
        total_params = sum(p.numel() for p in module.parameters())
        # Assume float16 (2 bytes per parameter)
        bytes_per_param = 2
        total_bytes = total_params * bytes_per_param
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _analyze_standalone_parameters(self):
        """Analyze standalone nn.Parameter objects like A_log, D, dt_bias"""
        # Get module parameter names to avoid double counting
        module_param_names = set()
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                for param_name, _ in module.named_parameters():
                    full_param_name = f"{name}.{param_name}" if name else param_name
                    module_param_names.add(full_param_name)
        
        # Analyze parameters not already covered by modules
        for name, param in self.model.named_parameters():
            if name not in module_param_names:
                # These are standalone parameters
                param_info = self._analyze_parameter(name, param)
                if param_info:
                    self.layer_analysis.append(param_info)
    
    def _analyze_parameter(self, name: str, param: torch.nn.Parameter) -> Dict[str, Any]:
        """Analyze a standalone parameter"""
        param_count = param.numel()
        if param_count == 0:
            return None
            
        # Create parameter info similar to layer info
        param_info = {
            'name': name,
            'module_type': 'Parameter',
            'module_path': 'torch.nn.Parameter',
            'layer_category': self._categorize_parameter(name, param),
            'is_quantizable': self._is_parameter_quantizable(param),
            'parameter_count': {
                'total': param_count,
                'trainable': param_count if param.requires_grad else 0,
                'non_trainable': 0 if param.requires_grad else param_count
            },
            'parameter_details': [{
                'name': name.split('.')[-1],
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'requires_grad': param.requires_grad,
                'numel': param_count
            }],
            'has_bias': False,
            'activation_function': 'N/A',
            'special_attributes': self._get_parameter_attributes(name, param),
            'quantization_suitability': self._assess_parameter_quantization_suitability(name, param),
            'memory_usage_mb': self._estimate_parameter_memory_usage(param),
        }
        
        return param_info
    
    def _categorize_parameter(self, name: str, param: torch.nn.Parameter) -> str:
        """Categorize standalone parameters"""
        if 'A_log' in name:
            return 'ssm_A_log'
        elif name.endswith('.D') and 'mixer' in name:
            return 'ssm_D'
        elif 'dt_bias' in name:
            return 'ssm_dt_bias'
        else:
            return 'parameter_other'
    
    def _is_parameter_quantizable(self, param: torch.nn.Parameter) -> bool:
        """Check if a parameter is quantizable"""
        # Small parameters usually not worth quantizing
        return param.numel() > 100
    
    def _get_parameter_attributes(self, name: str, param: torch.nn.Parameter) -> Dict[str, Any]:
        """Get special attributes for a parameter"""
        return {
            'shape': list(param.shape),
            'dtype': str(param.dtype),
            'device': str(param.device),
            'requires_grad': param.requires_grad
        }
    
    def _assess_parameter_quantization_suitability(self, name: str, param: torch.nn.Parameter) -> Dict[str, Any]:
        """Assess quantization suitability for a parameter"""
        return {
            'recommended': param.numel() > 1000,  # Only recommend for larger parameters
            'reason': 'Too small' if param.numel() <= 1000 else 'Suitable for quantization',
            'priority': 'low' if param.numel() <= 1000 else 'medium'
        }
    
    def _estimate_parameter_memory_usage(self, param: torch.nn.Parameter) -> float:
        """Estimate memory usage for a parameter"""
        # Assume float16 (2 bytes per parameter)
        bytes_per_param = 2
        total_bytes = param.numel() * bytes_per_param
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def analyze_all_layers(self):
        """Analyze all layers and parameters in the model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Analyzing all layers...")
        self.layer_analysis = []
        
        # Analyze modules (layers)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                layer_info = self.analyze_layer(name, module)
                if layer_info['parameter_count']['total'] > 0:  # Only include if has parameters
                    self.layer_analysis.append(layer_info)
        
        # Analyze standalone parameters (SSM parameters like A_log, D, dt_bias)
        self._analyze_standalone_parameters()
        
        self._generate_summary_stats()
        print(f"Analyzed {len(self.layer_analysis)} layers")
    
    def _generate_summary_stats(self):
        """Generate summary statistics from analysis results"""
        if not self.layer_analysis:
            print("No analysis results to summarize")
            return
        
        # Group by category with comprehensive SSM breakdown
        category_stats = defaultdict(lambda: {
            'count': 0, 
            'total_params': 0, 
            'memory_mb': 0, 
            'layers': []
        })
        
        for layer_info in self.layer_analysis:
            category = layer_info['layer_category']
            params = layer_info['parameter_count']['total']
            memory = layer_info['memory_usage_mb']
            
            category_stats[category]['count'] += 1
            category_stats[category]['total_params'] += params
            category_stats[category]['memory_mb'] += memory
            category_stats[category]['layers'].append(layer_info['name'])
        
        # Group related SSM categories for main summary
        ssm_categories = [
            'ssm_in_proj', 'ssm_out_proj', 'ssm_conv1d', 'ssm_norm',
            'ssm_A_log', 'ssm_D', 'ssm_dt_bias'
        ]
        
        # Calculate grouped totals
        grouped_stats = {}
        
        # SSM layers (group all SSM components)
        ssm_total_params = sum(category_stats[cat]['total_params'] for cat in ssm_categories if cat in category_stats)
        ssm_total_memory = sum(category_stats[cat]['memory_mb'] for cat in ssm_categories if cat in category_stats)
        ssm_total_count = sum(category_stats[cat]['count'] for cat in ssm_categories if cat in category_stats)
        
        if ssm_total_params > 0:
            grouped_stats['ssm_layers'] = {
                'count': ssm_total_count,
                'total_params': ssm_total_params,
                'memory_mb': ssm_total_memory,
                'breakdown': {cat: category_stats[cat] for cat in ssm_categories if cat in category_stats}
            }
        
        # MLP layers
        mlp_total_params = category_stats['mlp']['total_params']
        mlp_total_memory = category_stats['mlp']['memory_mb']
        mlp_total_count = category_stats['mlp']['count']
        
        if mlp_total_params > 0:
            grouped_stats['mlp_layers'] = {
                'count': mlp_total_count,
                'total_params': mlp_total_params,
                'memory_mb': mlp_total_memory
            }
        
        # Attention layers
        attention_total_params = category_stats['attention']['total_params']
        attention_total_memory = category_stats['attention']['memory_mb']
        attention_total_count = category_stats['attention']['count']
        
        if attention_total_params > 0:
            grouped_stats['attention_layers'] = {
                'count': attention_total_count,
                'total_params': attention_total_params,
                'memory_mb': attention_total_memory
            }
        
        # Other categories
        other_categories = ['embedding', 'output_head', 'norm', 'linear_other', 'conv_other', 'other']
        for category in other_categories:
            if category in category_stats and category_stats[category]['total_params'] > 0:
                grouped_stats[category] = category_stats[category]
        
        # Calculate total
        total_params = sum(info['total_params'] for info in grouped_stats.values())
        total_memory = sum(info['memory_mb'] for info in grouped_stats.values())
        
        self.summary_stats = {
            'detailed_categories': dict(category_stats),
            'grouped_categories': grouped_stats,
            'total_params': total_params,
            'total_memory_mb': total_memory,
            'model_info': {
                'name': self.model_name,
                'total_layers': len(self.layer_analysis)
            }
        }
    
    def print_summary(self):
        """Print a comprehensive summary"""
        if not self.summary_stats:
            print("No analysis data available. Run analyze_all_layers() first.")
            return
        
        print("\n" + "="*80)
        print("NEMOTRON MODEL ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nModel: {self.model_name}")
        print(f"Total layers: {self.summary_stats['model_info']['total_layers']}")
        print(f"Total parameters: {self.summary_stats['total_params']:,}")
        print(f"Total memory usage: {self.summary_stats['total_memory_mb']:.2f} MB")
        
        print("\n" + "-"*60)
        print("LAYER CATEGORY BREAKDOWN")
        print("-"*60)
        
        categories = self.summary_stats['grouped_categories']
        
        print(f"{'Category':<15} {'Count':<8} {'Parameters':<15} {'Memory (MB)':<12} {'%Params':<8}")
        print("-" * 70)
        
        for category in sorted(categories.keys()):
            count = categories[category]['count']
            params = categories[category]['total_params']
            memory = categories[category]['memory_mb']
            percent = (params / self.summary_stats['total_params'] * 100) if self.summary_stats['total_params'] > 0 else 0
            
            print(f"{category:<15} {count:<8} {params:<15,} {memory:<12.2f} {percent:<8.1f}%")
        
        print("\n" + "-"*60)
        print("QUANTIZATION RECOMMENDATIONS")
        print("-"*60)
        
        # Categorize by quantization strategy using new SSM categories
        ssm_categories = ['ssm_in_proj', 'ssm_out_proj', 'ssm_conv1d', 'ssm_norm', 'ssm_A_log', 'ssm_D', 'ssm_dt_bias']
        ssm_layers = [l for l in self.layer_analysis if l['layer_category'] in ssm_categories and l['is_quantizable']]
        attention_layers = [l for l in self.layer_analysis if l['layer_category'] == 'attention' and l['is_quantizable']]
        mlp_layers = [l for l in self.layer_analysis if l['layer_category'] == 'mlp' and l['is_quantizable']]
        high_param_layers = [l for l in self.layer_analysis if l['parameter_count']['total'] > 1000000]
        
        print(f"SSM layers (use MambaQuant): {len(ssm_layers)}")
        print(f"MLP layers (use standard): {len(mlp_layers)}")
        print(f"Attention layers (use standard): {len(attention_layers)}")
        print(f"High-parameter layers (>1M params): {len(high_param_layers)}")
        
        # Memory savings estimation
        potential_savings = 0
        for layer in self.layer_analysis:
            if layer['is_quantizable'] and layer['parameter_count']['total'] > 100000:
                if layer['layer_category'] in ['ssm_in_proj', 'ssm_out_proj']:
                    potential_savings += layer['memory_usage_mb'] * 0.75  # 4-bit quantization for main SSM
                elif layer['layer_category'] in ['ssm_conv1d', 'ssm_norm']:
                    potential_savings += layer['memory_usage_mb'] * 0.5   # 8-bit quantization for SSM aux
                elif layer['layer_category'] == 'attention':
                    potential_savings += layer['memory_usage_mb'] * 0.5   # 8-bit quantization
                elif layer['layer_category'] == 'mlp':
                    potential_savings += layer['memory_usage_mb'] * 0.5   # 8-bit quantization
                else:
                    potential_savings += layer['memory_usage_mb'] * 0.5   # 8-bit quantization
        
        print(f"\nEstimated memory savings with quantization: {potential_savings:.2f} MB ({potential_savings/self.summary_stats['total_memory_mb']*100:.1f}%)")
    
    def export_detailed_analysis(self, output_dir: str = "./analysis_output"):
        """Export detailed analysis to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export layer details as CSV
        df_data = []
        for layer in self.layer_analysis:
            row = {
                'name': layer['name'],
                'module_type': layer['module_type'],
                'category': layer['layer_category'],
                'is_quantizable': layer['is_quantizable'],
                'total_params': layer['parameter_count']['total'],
                'memory_mb': layer['memory_usage_mb'],
                'has_bias': layer['has_bias'],
                'recommended_bits': layer['quantization_suitability']['recommended_bits'],
                'recommended_method': layer['quantization_suitability']['recommended_method'],
                'suitability': layer['quantization_suitability']['overall_suitability']
            }
            
            # Add special attributes
            for attr, value in layer['special_attributes'].items():
                row[f'attr_{attr}'] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path / "layer_analysis.csv", index=False)
        
        # Export full analysis as JSON
        with open(output_path / "full_analysis.json", 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'summary_stats': self.summary_stats,
                'layer_details': self.layer_analysis
            }, f, indent=2, default=str)
        
        # Export summary as text
        with open(output_path / "summary.txt", 'w') as f:
            # Redirect print to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            self.print_summary()
            sys.stdout = old_stdout
        
        print(f"\nDetailed analysis exported to: {output_path}")
        print(f"Files created:")
        print(f"  - layer_analysis.csv: Tabular layer data")
        print(f"  - full_analysis.json: Complete analysis data")
        print(f"  - summary.txt: Summary report")
    
    def filter_layers(self, category: str = None, min_params: int = None, 
                     quantizable_only: bool = False) -> List[Dict[str, Any]]:
        """Filter layers based on criteria"""
        filtered = self.layer_analysis.copy()
        
        if category:
            filtered = [l for l in filtered if l['layer_category'] == category]
        
        if min_params:
            filtered = [l for l in filtered if l['parameter_count']['total'] >= min_params]
        
        if quantizable_only:
            filtered = [l for l in filtered if l['is_quantizable']]
        
        return filtered
    
    def get_quantization_plan(self) -> Dict[str, Any]:
        """Generate a realistic quantization plan based on layer analysis"""
        if not self.layer_analysis:
            return {}
        
        # Count quantizable layers by type
        ssm_layers = [l for l in self.layer_analysis if l['layer_category'].startswith('ssm_') and l['is_quantizable']]
        mlp_layers = [l for l in self.layer_analysis if l['layer_category'] == 'mlp' and l['is_quantizable']]
        attention_layers = [l for l in self.layer_analysis if l['layer_category'] == 'attention' and l['is_quantizable']]
        
        # Calculate parameters by category
        ssm_params = sum(l['parameter_count']['total'] for l in ssm_layers)
        mlp_params = sum(l['parameter_count']['total'] for l in mlp_layers)
        attention_params = sum(l['parameter_count']['total'] for l in attention_layers)
        total_quantizable = ssm_params + mlp_params + attention_params
        
        quantization_plan = {
            'strategy': 'phased_approach',
            'phases': {
                'phase_1_w8a8_uniform': {
                    'description': 'Conservative uniform W8A8 quantization',
                    'target_layers': {
                        'ssm_layers': [l['name'] for l in ssm_layers],
                        'mlp_layers': [l['name'] for l in mlp_layers], 
                        'attention_layers': [l['name'] for l in attention_layers]
                    },
                    'quantization': 'W8A8 (8-bit weights, 8-bit activations)',
                    'framework': 'Standard GPTQ/AWQ + MambaQuant for SSM awareness',
                    'expected_memory_reduction': '50%',
                    'expected_size': '~7.7 GB',
                    'pros': ['Accuracy preservation', 'Stable implementation', 'Wide framework support'],
                    'cons': ['Less aggressive compression'],
                    'priority': 1
                },
                'phase_2_w4a8_uniform': {
                    'description': 'Aggressive uniform W4A8 quantization',
                    'target_layers': {
                        'ssm_layers': [l['name'] for l in ssm_layers],
                        'mlp_layers': [l['name'] for l in mlp_layers],
                        'attention_layers': [l['name'] for l in attention_layers]
                    },
                    'quantization': 'W4A8 (4-bit weights, 8-bit activations)', 
                    'framework': 'MambaQuant W4A8 + GPTQ 4-bit',
                    'expected_memory_reduction': '75%',
                    'expected_size': '~3.9 GB',
                    'pros': ['Maximum memory savings', 'Uniform activation handling'],
                    'cons': ['Potential accuracy loss', 'More aggressive compression'],
                    'priority': 2
                },
                'phase_3_block_specific': {
                    'description': 'Block-type specific quantization (advanced)',
                    'target_layers': {
                        'ssm_blocks_w4a8': [l['name'] for l in ssm_layers],
                        'mlp_blocks_w8a8': [l['name'] for l in mlp_layers],
                        'attention_blocks_w8a8': [l['name'] for l in attention_layers]
                    },
                    'quantization': 'SSM: W4A8, MLP/Attention: W8A8',
                    'framework': 'Custom hybrid implementation',
                    'expected_memory_reduction': '62%',
                    'expected_size': '~5.9 GB',
                    'pros': ['Optimized per block type', 'SSM-specific optimization'],
                    'cons': ['Complex implementation', 'Activation format conversion overhead'],
                    'priority': 3,
                    'challenges': [
                        'Activation format conversion between blocks',
                        'Custom kernel implementation required',
                        'Memory bandwidth optimization needed'
                    ]
                }
            },
            'skip_quantization': [
                l['name'] for l in self.layer_analysis 
                if not l['is_quantizable'] or l['parameter_count']['total'] < 1000
            ],
            'layer_statistics': {
                'total_quantizable_layers': len(ssm_layers) + len(mlp_layers) + len(attention_layers),
                'ssm_layers': {'count': len(ssm_layers), 'params': ssm_params, 'percent': f"{ssm_params/total_quantizable*100:.1f}%"},
                'mlp_layers': {'count': len(mlp_layers), 'params': mlp_params, 'percent': f"{mlp_params/total_quantizable*100:.1f}%"},
                'attention_layers': {'count': len(attention_layers), 'params': attention_params, 'percent': f"{attention_params/total_quantizable*100:.1f}%"},
                'total_quantizable_params': total_quantizable
            }
        }
        
        return quantization_plan
    
    def print_ssm_breakdown(self):
        """Print detailed breakdown of SSM components"""
        if not self.summary_stats or 'grouped_categories' not in self.summary_stats:
            print("No summary statistics available. Run analyze_all_layers() first.")
            return
        
        print("\n" + "="*80)
        print("DETAILED SSM (STATE SPACE MODEL) BREAKDOWN")
        print("="*80)
        
        ssm_info = self.summary_stats['grouped_categories'].get('ssm_layers')
        if not ssm_info:
            print("No SSM layers found")
            return
        
        print(f"Total SSM Parameters: {ssm_info['total_params']:,}")
        print(f"Total SSM Memory: {ssm_info['memory_mb']:.2f} MB")
        print(f"Total SSM Components: {ssm_info['count']}")
        
        total_model_params = self.summary_stats['total_params']
        ssm_percentage = (ssm_info['total_params'] / total_model_params * 100) if total_model_params > 0 else 0
        print(f"SSM as % of model: {ssm_percentage:.1f}%")
        
        print(f"\n{'SSM Component':<20} {'Count':<8} {'Parameters':<15} {'Memory (MB)':<12} {'Description':<25}")
        print("-" * 90)
        
        ssm_descriptions = {
            'ssm_in_proj': 'Input projections',
            'ssm_out_proj': 'Output projections', 
            'ssm_conv1d': 'Convolution layers',
            'ssm_norm': 'Normalization layers',
            'ssm_A_log': 'State matrices (log)',
            'ssm_D': 'Skip connections',
            'ssm_dt_bias': 'Time step biases'
        }
        
        breakdown = ssm_info.get('breakdown', {})
        for component, description in ssm_descriptions.items():
            if component in breakdown:
                comp_info = breakdown[component]
                print(f"{component:<20} {comp_info['count']:<8} {comp_info['total_params']:<15,} "
                      f"{comp_info['memory_mb']:<12.2f} {description:<25}")
        
        print(f"\n{'Quantization Recommendations:'}")
        print(f"- SSM input/output projections: 4-bit quantization (largest components)")
        print(f"- SSM conv1d & normalization: 8-bit quantization (medium impact)")
        print(f"- SSM parameters (A_log, D, dt_bias): Skip quantization (very small)")
        
        return ssm_info
    
    def print_quantization_plan(self):
        """Print detailed phased quantization plan"""
        plan = self.get_quantization_plan()
        if not plan:
            print("No quantization plan available. Run analyze_all_layers() first.")
            return
        
        print("\n" + "="*80)
        print("PHASED QUANTIZATION STRATEGY")
        print("="*80)
        
        # Print layer statistics
        stats = plan['layer_statistics']
        print(f"📊 **Quantizable Layer Statistics:**")
        print(f"   • Total quantizable layers: {stats['total_quantizable_layers']}")
        print(f"   • SSM layers: {stats['ssm_layers']['count']} ({stats['ssm_layers']['percent']})")
        print(f"   • MLP layers: {stats['mlp_layers']['count']} ({stats['mlp_layers']['percent']})")
        print(f"   • Attention layers: {stats['attention_layers']['count']} ({stats['attention_layers']['percent']})")
        print(f"   • Total quantizable parameters: {stats['total_quantizable_params']:,}")
        
        # Print each phase
        for phase_name, phase_info in plan['phases'].items():
            priority = phase_info['priority']
            print(f"\n🎯 **PHASE {priority}: {phase_info['description'].upper()}**")
            print("-" * 60)
            
            print(f"**Quantization Scheme:** {phase_info['quantization']}")
            print(f"**Framework:** {phase_info['framework']}")
            print(f"**Expected Memory Reduction:** {phase_info['expected_memory_reduction']}")
            print(f"**Expected Model Size:** {phase_info['expected_size']}")
            
            # Target layers summary
            target_layers = phase_info['target_layers']
            for layer_type, layers in target_layers.items():
                if layers:
                    print(f"   • {layer_type}: {len(layers)} layers")
            
            # Pros and cons
            print(f"**Pros:** {', '.join(phase_info['pros'])}")
            print(f"**Cons:** {', '.join(phase_info['cons'])}")
            
            # Challenges for advanced phases
            if 'challenges' in phase_info:
                print(f"**Implementation Challenges:**")
                for challenge in phase_info['challenges']:
                    print(f"   - {challenge}")
        
        # Skip quantization summary
        skip_layers = plan['skip_quantization']
        print(f"\n⏭️  **SKIP QUANTIZATION:** {len(skip_layers)} layers")
        print("   (Small parameters, embeddings, normalization)")
        
        print(f"\n📋 **RECOMMENDED IMPLEMENTATION ORDER:**")
        print("   1. Start with Phase 1 (W8A8) for stability and accuracy baseline")
        print("   2. Evaluate Phase 2 (W4A8) for memory-constrained scenarios")
        print("   3. Develop Phase 3 (Block-specific) for optimal performance")
        
        return plan


def main():
    """Main analysis function"""
    print("Starting Nemotron Model Analysis...")
    
    # Initialize analyzer
    analyzer = NemotronModelAnalyzer()
    
    # Load model (you can set load_in_4bit=True to see quantized version)
    analyzer.load_model(load_in_4bit=False)
    
    # Analyze all layers
    analyzer.analyze_all_layers()
    
    # Print summary
    analyzer.print_summary()
    
    # Export detailed analysis
    analyzer.export_detailed_analysis()
    
    # Show some specific analyses
    print("\n" + "="*60)
    print("SPECIFIC LAYER ANALYSES")
    print("="*60)
    
    # Show Mamba layers
    mamba_layers = analyzer.filter_layers(category='mamba', quantizable_only=True)
    print(f"\nMamba layers ({len(mamba_layers)}):")
    for layer in mamba_layers[:5]:  # Show first 5
        print(f"  {layer['name']}: {layer['parameter_count']['total']:,} params, {layer['memory_usage_mb']:.2f} MB")
    
    # Show attention layers
    attention_layers = analyzer.filter_layers(category='attention', quantizable_only=True)
    print(f"\nAttention layers ({len(attention_layers)}):")
    for layer in attention_layers[:5]:  # Show first 5
        print(f"  {layer['name']}: {layer['parameter_count']['total']:,} params, {layer['memory_usage_mb']:.2f} MB")
    
    # Show largest layers
    large_layers = analyzer.filter_layers(min_params=1000000)
    print(f"\nLargest layers (>1M params, {len(large_layers)}):")
    for layer in sorted(large_layers, key=lambda x: x['parameter_count']['total'], reverse=True)[:10]:
        print(f"  {layer['name']}: {layer['parameter_count']['total']:,} params, {layer['layer_category']}")
    
    # Generate quantization plan
    plan = analyzer.get_quantization_plan()
    print(f"\n" + "="*60)
    print("QUANTIZATION PLAN")
    print("="*60)
    print(f"Mamba layers (4-bit): {len(plan['phases']['phase_1_w8a8_uniform']['target_layers']['ssm_layers'])}")
    print(f"Attention layers (8-bit): {len(plan['phases']['phase_1_w8a8_uniform']['target_layers']['attention_layers'])}")
    print(f"Linear layers (8-bit): {len(plan['phases']['phase_1_w8a8_uniform']['target_layers']['mlp_layers'])}")
    print(f"Skip quantization: {len(plan['skip_quantization'])}")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main() 