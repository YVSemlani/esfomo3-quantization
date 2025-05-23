#!/usr/bin/env python3
"""
Quick runner script for Nemotron model analysis.
Usage examples for different analysis scenarios.
"""

from analyze_nemotron_model import NemotronModelAnalyzer
import argparse
import sys


def quick_analysis():
    """Run a quick analysis with basic output"""
    print("=== QUICK ANALYSIS ===")
    
    analyzer = NemotronModelAnalyzer()
    analyzer.load_model(load_in_4bit=False)
    analyzer.analyze_all_layers()
    analyzer.print_summary()
    
    return analyzer


def detailed_analysis():
    """Run detailed analysis with exports"""
    print("=== DETAILED ANALYSIS ===")
    
    analyzer = NemotronModelAnalyzer()
    analyzer.load_model(load_in_4bit=False)
    analyzer.analyze_all_layers()
    
    # Print summary
    analyzer.print_summary()
    
    # Export all results
    analyzer.export_detailed_analysis("./detailed_analysis_output")
    
    # Show detailed breakdowns
    print("\n" + "="*80)
    print("DETAILED LAYER BREAKDOWNS")
    print("="*80)
    
    # Mamba layers detailed
    mamba_layers = analyzer.filter_layers(category='mamba', quantizable_only=True)
    if mamba_layers:
        print(f"\n=== MAMBA LAYERS ({len(mamba_layers)}) ===")
        for layer in mamba_layers:
            print(f"{layer['name']}:")
            print(f"  Type: {layer['module_type']}")
            print(f"  Params: {layer['parameter_count']['total']:,}")
            print(f"  Memory: {layer['memory_usage_mb']:.2f} MB")
            print(f"  Attributes: {layer['special_attributes']}")
            print()
    
    # Attention layers detailed  
    attention_layers = analyzer.filter_layers(category='attention', quantizable_only=True)
    if attention_layers:
        print(f"\n=== ATTENTION LAYERS ({len(attention_layers)}) ===")
        for layer in attention_layers:
            print(f"{layer['name']}:")
            print(f"  Type: {layer['module_type']}")
            print(f"  Params: {layer['parameter_count']['total']:,}")
            print(f"  Memory: {layer['memory_usage_mb']:.2f} MB")
            print(f"  Attributes: {layer['special_attributes']}")
            print()
    
    return analyzer


def compare_quantized():
    """Compare fp16 vs 4-bit quantized versions"""
    print("=== QUANTIZATION COMPARISON ===")
    
    # Analyze FP16 version
    print("\nAnalyzing FP16 version...")
    analyzer_fp16 = NemotronModelAnalyzer()
    analyzer_fp16.load_model(load_in_4bit=False)
    analyzer_fp16.analyze_all_layers()
    
    # Analyze 4-bit version
    print("\nAnalyzing 4-bit quantized version...")
    analyzer_4bit = NemotronModelAnalyzer()
    analyzer_4bit.load_model(load_in_4bit=True)
    analyzer_4bit.analyze_all_layers()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    fp16_memory = analyzer_fp16.summary_stats['total_memory_mb']
    bit4_memory = analyzer_4bit.summary_stats['total_memory_mb']
    
    print(f"FP16 Memory Usage: {fp16_memory:.2f} MB")
    print(f"4-bit Memory Usage: {bit4_memory:.2f} MB")
    print(f"Memory Reduction: {((fp16_memory - bit4_memory) / fp16_memory * 100):.1f}%")
    
    return analyzer_fp16, analyzer_4bit


def layer_analysis_by_type():
    """Analyze layers grouped by type"""
    print("=== LAYER TYPE ANALYSIS ===")
    
    analyzer = NemotronModelAnalyzer()
    analyzer.load_model(load_in_4bit=False)
    analyzer.analyze_all_layers()
    
    # Group by category
    categories = ['mamba', 'attention', 'linear', 'mlp', 'normalization', 'embedding']
    
    for category in categories:
        layers = analyzer.filter_layers(category=category)
        if not layers:
            continue
            
        print(f"\n=== {category.upper()} LAYERS ===")
        print(f"Count: {len(layers)}")
        
        total_params = sum(l['parameter_count']['total'] for l in layers)
        total_memory = sum(l['memory_usage_mb'] for l in layers)
        quantizable = [l for l in layers if l['is_quantizable']]
        
        print(f"Total parameters: {total_params:,}")
        print(f"Total memory: {total_memory:.2f} MB")
        print(f"Quantizable layers: {len(quantizable)}")
        
        if quantizable:
            print("Largest quantizable layers:")
            sorted_layers = sorted(quantizable, 
                                 key=lambda x: x['parameter_count']['total'], 
                                 reverse=True)
            for layer in sorted_layers[:3]:
                print(f"  {layer['name']}: {layer['parameter_count']['total']:,} params")
    
    return analyzer


def main():
    """Main function with command line options"""
    parser = argparse.ArgumentParser(description="Nemotron Model Analysis Tool")
    parser.add_argument('--mode', choices=['quick', 'detailed', 'compare', 'by-type'], 
                       default='quick', help='Analysis mode')
    parser.add_argument('--export', action='store_true', 
                       help='Export detailed results to files')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'quick':
            analyzer = quick_analysis()
        elif args.mode == 'detailed':
            analyzer = detailed_analysis()
        elif args.mode == 'compare':
            analyzer = compare_quantized()
        elif args.mode == 'by-type':
            analyzer = layer_analysis_by_type()
        
        if args.export and hasattr(analyzer, 'export_detailed_analysis'):
            analyzer.export_detailed_analysis(f"./{args.mode}_analysis_output")
            print(f"\nResults exported to ./{args.mode}_analysis_output/")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 