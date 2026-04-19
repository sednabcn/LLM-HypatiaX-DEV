#!/usr/bin/env python3
"""
Generate plots for the paper
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

def load_data():
    """Load the shared dataset"""
    data_file = DATA_DIR / "all_systems_merged.json"
    if data_file.exists():
        with open(data_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: {data_file} not found. Using sample data.")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demonstration"""
    return {
        'systems': ['System A', 'System B', 'System C'],
        'scores': [0.85, 0.78, 0.92],
        'tests': 127
    }

def plot_results():
    """Create main results figure"""
    data = load_data()
    
    plt.figure(figsize=(10, 6))
    
    if 'systems' in data and 'scores' in data:
        systems = data['systems']
        scores = data['scores']
        
        plt.bar(systems, scores, color=['#2E86AB', '#A23B72', '#F18F01'])
        plt.ylabel('Performance Score', fontsize=12)
        plt.xlabel('System', fontsize=12)
        plt.title('System Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
    else:
        # Fallback plot
        x = np.linspace(0, 10, 100)
        plt.plot(x, np.sin(x), label='Sample Data')
        plt.legend()
        plt.title('Placeholder Figure')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'results.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {FIG_DIR / 'results.pdf'}")

if __name__ == '__main__':
    print("Generating figures...")
    plot_results()
    print("Done!")
