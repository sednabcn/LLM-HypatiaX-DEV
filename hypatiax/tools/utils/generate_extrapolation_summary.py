#!/usr/bin/env python3
"""
Generate Extrapolation Summary for Paper
=========================================

Reads extrapolation test results JSON files and generates:
1. LaTeX table for paper (Table 1)
2. Summary statistics
3. Comparison charts

Usage:
    python generate_extrapolation_summary.py --results results/*.json --output paper_table_1.tex
    python generate_extrapolation_summary.py --results results/arrhenius*.json --format markdown
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List
import numpy as np


class ExtrapolationSummaryGenerator:
    """Generate summary tables and stats from extrapolation test results"""
    
    def __init__(self):
        self.results = []
        self.methods = set()
        self.test_cases = set()
    
    def load_results(self, result_files: List[str]):
        """Load all result JSON files"""
        print(f"\n📂 Loading {len(result_files)} result files...")
        
        for filepath in result_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Handle different JSON formats
                    if isinstance(data, dict):
                        if 'llm' in data and 'hybrid' in data:
                            # Format from test_arrhenius_extrapolation()
                            self.results.append({
                                'test': Path(filepath).stem,
                                'methods': data
                            })
                        elif 'method' in data:
                            # Single method result
                            self.results.append({
                                'test': data.get('test_name', Path(filepath).stem),
                                'methods': {data['method']: data}
                            })
                        else:
                            print(f"⚠️  Unknown format in {filepath}")
                    
                    print(f"  ✅ Loaded: {filepath}")
                    
            except Exception as e:
                print(f"  ❌ Error loading {filepath}: {e}")
        
        # Extract unique methods and test cases
        for result in self.results:
            self.methods.update(result['methods'].keys())
            self.test_cases.add(result['test'])
        
        print(f"\n📊 Summary:")
        print(f"   Tests: {len(self.test_cases)}")
        print(f"   Methods: {len(self.methods)}")
        print(f"   Methods found: {', '.join(sorted(self.methods))}")
    
    def generate_latex_table(self, regime: str = 'medium') -> str:
        """
        Generate LaTeX table for paper
        
        Args:
            regime: Which extrapolation regime to show ('near', 'medium', 'far')
        """
        
        # Aggregate results by method
        method_stats = {}
        
        for result in self.results:
            for method_name, method_data in result['methods'].items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'train_r2': [],
                        'train_rmse': [],
                        'extrap_near': [],
                        'extrap_medium': [],
                        'extrap_far': [],
                        'correct_forms': 0,
                        'total_tests': 0
                    }
                
                stats = method_stats[method_name]
                
                # Training metrics
                train = method_data.get('training', {})
                stats['train_r2'].append(train.get('r2', 0))
                stats['train_rmse'].append(train.get('rmse', 0))
                
                # Extrapolation errors
                regimes = method_data.get('regimes', {})
                stats['extrap_near'].append(
                    regimes.get('near', {}).get('extrapolation_error_pct', 0)
                )
                stats['extrap_medium'].append(
                    regimes.get('medium', {}).get('extrapolation_error_pct', 0)
                )
                stats['extrap_far'].append(
                    regimes.get('far', {}).get('extrapolation_error_pct', 0)
                )
                
                stats['total_tests'] += 1
        
        # Build LaTeX table
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Expression Discovery Performance with Extrapolation Analysis}")
        latex.append(r"\label{tab:main_results_extrap}")
        latex.append(r"\begin{tabular}{lcccccc}")
        latex.append(r"\toprule")
        latex.append(r"\multirow{2}{*}{Method} & \multicolumn{2}{c}{Training} & \multicolumn{3}{c}{Extrapolation Error (\%)} & \multirow{2}{*}{Tests} \\")
        latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-6}")
        latex.append(r" & $R^2$ & RMSE & Near (1.2×) & Medium (2×) & Far (5×) & \\")
        latex.append(r"\midrule")
        
        # Sort methods by average medium extrapolation error
        sorted_methods = sorted(
            method_stats.items(),
            key=lambda x: np.mean(x[1]['extrap_medium']) if x[1]['extrap_medium'] else float('inf')
        )
        
        for method_name, stats in sorted_methods:
            # Calculate averages
            avg_r2 = np.mean(stats['train_r2']) if stats['train_r2'] else 0
            avg_rmse = np.mean(stats['train_rmse']) if stats['train_rmse'] else 0
            avg_near = np.mean(stats['extrap_near']) if stats['extrap_near'] else 0
            avg_medium = np.mean(stats['extrap_medium']) if stats['extrap_medium'] else 0
            avg_far = np.mean(stats['extrap_far']) if stats['extrap_far'] else 0
            
            # Check if this is the best method (lowest medium error)
            is_best = (method_name == sorted_methods[0][0])
            
            # Format row
            if is_best:
                row = f"\\textbf{{{method_name}}} & "
                row += f"\\textbf{{{avg_r2:.2f}}} & "
                row += f"\\textbf{{{avg_rmse:.3f}}} & "
                row += f"\\textbf{{{avg_near:.0f}}} & "
                row += f"\\textbf{{{avg_medium:.0f}}} & "
                row += f"\\textbf{{{avg_far:.0f}}} & "
                row += f"\\textbf{{{stats['total_tests']}}}"
            else:
                row = f"{method_name} & "
                row += f"{avg_r2:.2f} & "
                row += f"{avg_rmse:.3f} & "
                row += f"{avg_near:.0f} & "
                row += f"{avg_medium:.0f} & "
                row += f"{avg_far:.0f} & "
                row += f"{stats['total_tests']}"
            
            row += " \\\\"
            latex.append(row)
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        return "\n".join(latex)
    
    def generate_markdown_table(self) -> str:
        """Generate Markdown table for README or documentation"""
        
        # Aggregate results
        method_stats = {}
        
        for result in self.results:
            for method_name, method_data in result['methods'].items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'train_r2': [],
                        'extrap_medium': [],
                    }
                
                train = method_data.get('training', {})
                method_stats[method_name]['train_r2'].append(train.get('r2', 0))
                
                regimes = method_data.get('regimes', {})
                method_stats[method_name]['extrap_medium'].append(
                    regimes.get('medium', {}).get('extrapolation_error_pct', 0)
                )
        
        # Build markdown
        lines = []
        lines.append("# Extrapolation Test Results")
        lines.append("")
        lines.append("| Method | Avg R² | Med. Extrap. Error (%) | Status |")
        lines.append("|--------|--------|------------------------|--------|")
        
        # Sort by extrapolation error
        sorted_methods = sorted(
            method_stats.items(),
            key=lambda x: np.mean(x[1]['extrap_medium']) if x[1]['extrap_medium'] else float('inf')
        )
        
        for method_name, stats in sorted_methods:
            avg_r2 = np.mean(stats['train_r2']) if stats['train_r2'] else 0
            avg_extrap = np.mean(stats['extrap_medium']) if stats['extrap_medium'] else 0
            
            # Determine status
            if avg_extrap < 50:
                status = "✅ Excellent"
            elif avg_extrap < 100:
                status = "✓ Good"
            elif avg_extrap < 200:
                status = "⚠️ Moderate"
            elif avg_extrap < 500:
                status = "❌ Poor"
            else:
                status = "💥 Catastrophic"
            
            lines.append(f"| {method_name} | {avg_r2:.3f} | {avg_extrap:.1f} | {status} |")
        
        return "\n".join(lines)
    
    def generate_summary_stats(self) -> str:
        """Generate detailed summary statistics"""
        
        lines = []
        lines.append("="*80)
        lines.append("EXTRAPOLATION TEST SUMMARY")
        lines.append("="*80)
        lines.append("")
        lines.append(f"Total test cases: {len(self.test_cases)}")
        lines.append(f"Methods compared: {len(self.methods)}")
        lines.append("")
        
        # Aggregate by method
        method_stats = {}
        
        for result in self.results:
            for method_name, method_data in result['methods'].items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'train_r2': [],
                        'train_rmse': [],
                        'extrap_near': [],
                        'extrap_medium': [],
                        'extrap_far': [],
                    }
                
                stats = method_stats[method_name]
                train = method_data.get('training', {})
                regimes = method_data.get('regimes', {})
                
                stats['train_r2'].append(train.get('r2', 0))
                stats['train_rmse'].append(train.get('rmse', 0))
                stats['extrap_near'].append(regimes.get('near', {}).get('extrapolation_error_pct', 0))
                stats['extrap_medium'].append(regimes.get('medium', {}).get('extrapolation_error_pct', 0))
                stats['extrap_far'].append(regimes.get('far', {}).get('extrapolation_error_pct', 0))
        
        # Print stats by method
        for method_name in sorted(method_stats.keys()):
            stats = method_stats[method_name]
            
            lines.append(f"\n{method_name}")
            lines.append("-" * 40)
            
            if stats['train_r2']:
                lines.append(f"Training R²:           {np.mean(stats['train_r2']):.4f} ± {np.std(stats['train_r2']):.4f}")
                lines.append(f"Training RMSE:         {np.mean(stats['train_rmse']):.6f}")
                lines.append(f"Near Extrap (1.2×):    {np.mean(stats['extrap_near']):.1f}%")
                lines.append(f"Medium Extrap (2×):    {np.mean(stats['extrap_medium']):.1f}%")
                lines.append(f"Far Extrap (5×):       {np.mean(stats['extrap_far']):.1f}%")
            else:
                lines.append("No results available")
        
        lines.append("")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def save_output(self, content: str, filepath: str):
        """Save generated content to file"""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"\n💾 Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Extrapolation Summary Tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate LaTeX table for paper
  python generate_extrapolation_summary.py --results results/*.json --output paper_table_1.tex
  
  # Generate Markdown summary
  python generate_extrapolation_summary.py --results results/*.json --format markdown --output summary.md
  
  # Print statistics to console
  python generate_extrapolation_summary.py --results results/*.json --format stats
        """
    )
    
    parser.add_argument('--results', nargs='+', required=True,
                       help='Result JSON files (supports wildcards)')
    parser.add_argument('--output', type=str,
                       help='Output file path')
    parser.add_argument('--format', type=str, default='latex',
                       choices=['latex', 'markdown', 'stats'],
                       help='Output format (default: latex)')
    parser.add_argument('--regime', type=str, default='medium',
                       choices=['near', 'medium', 'far'],
                       help='Extrapolation regime for LaTeX table (default: medium)')
    
    args = parser.parse_args()
    
    # Expand wildcards in result files
    result_files = []
    for pattern in args.results:
        result_files.extend(glob.glob(pattern))
    
    if not result_files:
        print("❌ No result files found!")
        print(f"   Searched for: {args.results}")
        return 1
    
    # Generate summary
    generator = ExtrapolationSummaryGenerator()
    generator.load_results(result_files)
    
    # Generate content based on format
    if args.format == 'latex':
        content = generator.generate_latex_table(args.regime)
        print("\n" + "="*80)
        print("GENERATED LATEX TABLE")
        print("="*80)
        print(content)
        
    elif args.format == 'markdown':
        content = generator.generate_markdown_table()
        print("\n" + content)
        
    elif args.format == 'stats':
        content = generator.generate_summary_stats()
        print("\n" + content)
    
    # Save to file if requested
    if args.output:
        generator.save_output(content, args.output)
        print(f"\n✅ Output saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
