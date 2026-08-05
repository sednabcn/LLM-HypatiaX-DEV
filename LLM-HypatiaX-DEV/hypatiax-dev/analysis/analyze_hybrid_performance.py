#!/usr/bin/env python3
"""
analyze_hybrid_performance.py

Comprehensive performance analysis for the Hybrid DeFi System.
Analyzes results from hybrid_defi_full.py and generates detailed reports.

Features:
- Overall performance metrics
- Domain-specific analysis
- Decision strategy evaluation
- Extrapolation performance
- Comparative analysis (LLM vs Ensemble vs NN)
- Failure analysis
- Visualization generation
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class HybridPerformanceAnalyzer:
    """Analyze performance of hybrid DeFi system"""

    def __init__(self, results_dir: str = "hypatiax/data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.hybrid_results = []
        self.extrapolation_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_latest_results(self) -> bool:
        """Load the most recent results files"""

        # Find latest hybrid results.
        # Step 1 outputs to hybrid_pysr/defi/ (RESULT_SUBDIR); fall back to
        # results_dir root for backwards compatibility with older runs.
        defi_subdir = self.results_dir / "hybrid_pysr" / "defi"
        hybrid_files = sorted(defi_subdir.glob("hybrid_defi_*.json"))
        if not hybrid_files:
            # Fallback: check root results dir (pre-fix runs)
            hybrid_files = sorted(self.results_dir.glob("hybrid_defi_*.json"))
        if not hybrid_files:
            print("❌ No hybrid results found in hybrid_pysr/defi/ or results root")
            return False

        latest_hybrid = hybrid_files[-1]
        print(f"📂 Loading hybrid results: {latest_hybrid.name}")

        with open(latest_hybrid) as f:
            data = json.load(f)
        # Step 1 wraps results in {"timestamp":..., "total":..., "results":[...]}
        self.hybrid_results = data.get("results", data) if isinstance(data, dict) else data

        print(f"✅ Loaded {len(self.hybrid_results)} hybrid test cases")

        # Find latest extrapolation results (optional)
        extrap_files = sorted(self.results_dir.glob("extrapolation_*.csv"))
        if extrap_files:
            latest_extrap = extrap_files[-1]
            print(f"📂 Loading extrapolation results: {latest_extrap.name}")
            self.extrapolation_results = pd.read_csv(latest_extrap).to_dict('records')
            print(f"✅ Loaded {len(self.extrapolation_results)} extrapolation cases")
        else:
            print("⚠️  No extrapolation results found (skipping)")

        return True

    def analyze_overall_performance(self) -> dict[str, Any]:
        """Analyze overall system performance"""

        print("\n" + "=" * 80)
        print("📊 OVERALL PERFORMANCE ANALYSIS")
        print("=" * 80)

        total_cases = len(self.hybrid_results)

        # Extract R² scores
        r2_scores = []
        success_count = 0

        for result in self.hybrid_results:
            eval_metrics = result.get('evaluation', {})
            r2 = eval_metrics.get('r2')

            if r2 is not None:
                r2_scores.append(r2)
                if r2 > 0.9:  # Consider success if R² > 0.9
                    success_count += 1

        # Calculate statistics
        if r2_scores:
            stats = {
                'total_cases': total_cases,
                'valid_r2_count': len(r2_scores),
                'success_count': success_count,
                'success_rate': success_count / len(r2_scores),
                'mean_r2': np.mean(r2_scores),
                'median_r2': np.median(r2_scores),
                'std_r2': np.std(r2_scores),
                'min_r2': np.min(r2_scores),
                'max_r2': np.max(r2_scores),
                'q25_r2': np.percentile(r2_scores, 25),
                'q75_r2': np.percentile(r2_scores, 75),
            }

            # Performance tiers
            excellent = sum(1 for r2 in r2_scores if r2 > 0.99)
            good = sum(1 for r2 in r2_scores if 0.95 <= r2 <= 0.99)
            acceptable = sum(1 for r2 in r2_scores if 0.90 <= r2 < 0.95)
            poor = sum(1 for r2 in r2_scores if r2 < 0.90)

            stats.update({
                'excellent_count': excellent,  # R² > 0.99
                'good_count': good,            # 0.95 <= R² <= 0.99
                'acceptable_count': acceptable, # 0.90 <= R² < 0.95
                'poor_count': poor,            # R² < 0.90
            })
        else:
            stats = {'error': 'No valid R² scores found'}

        # Print summary
        print("\n📈 Summary Statistics:")
        print(f"   Total cases: {stats['total_cases']}")
        print(f"   Valid R² scores: {stats['valid_r2_count']}")
        print(f"   Success rate: {stats['success_rate'] * 100:.1f}% (R² > 0.9)")
        print("\n📊 R² Distribution:")
        print(f"   Mean: {stats['mean_r2']:.6f}")
        print(f"   Median: {stats['median_r2']:.6f}")
        print(f"   Std Dev: {stats['std_r2']:.6f}")
        print(f"   Range: [{stats['min_r2']:.6f}, {stats['max_r2']:.6f}]")
        print(f"   Q25-Q75: [{stats['q25_r2']:.6f}, {stats['q75_r2']:.6f}]")
        print("\n🎯 Performance Tiers:")
        valid = stats['valid_r2_count']
        print(f"   Excellent (R² > 0.99): {stats['excellent_count']} ({stats['excellent_count']/valid*100:.1f}%)")
        print(f"   Good (0.95-0.99): {stats['good_count']} ({stats['good_count']/valid*100:.1f}%)")
        print(f"   Acceptable (0.90-0.95): {stats['acceptable_count']} ({stats['acceptable_count']/valid*100:.1f}%)")
        print(f"   Poor (< 0.90): {stats['poor_count']} ({stats['poor_count']/valid*100:.1f}%)")

        return stats

    def analyze_by_domain(self) -> dict[str, dict[str, Any]]:
        """Analyze performance by DeFi domain"""

        print("\n" + "=" * 80)
        print("🏢 DOMAIN-SPECIFIC ANALYSIS")
        print("=" * 80)

        domains = {}

        for result in self.hybrid_results:
            domain = result.get('domain', 'Unknown')

            if domain not in domains:
                domains[domain] = {
                    'cases': [],
                    'r2_scores': [],
                }

            domains[domain]['cases'].append(result)

            r2 = result.get('evaluation', {}).get('r2')
            if r2 is not None:
                domains[domain]['r2_scores'].append(r2)

        # Calculate domain statistics
        domain_stats = {}

        for domain, data in domains.items():
            r2_scores = data['r2_scores']

            if r2_scores:
                domain_stats[domain] = {
                    'total': len(data['cases']),
                    'mean_r2': np.mean(r2_scores),
                    'median_r2': np.median(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'min_r2': np.min(r2_scores),
                    'max_r2': np.max(r2_scores),
                    'success_rate': sum(1 for r2 in r2_scores if r2 > 0.9) / len(r2_scores),
                }

        # Print domain performance
        print("\n📊 Performance by Domain:")
        print(f"{'Domain':<25} {'Cases':<8} {'Mean R²':<12} {'Success Rate':<15} {'Range'}")
        print("-" * 80)

        for domain in sorted(domain_stats.keys()):
            stats = domain_stats[domain]
            print(f"{domain:<25} {stats['total']:<8} {stats['mean_r2']:<12.6f} "
                  f"{stats['success_rate']*100:>6.1f}%        "
                  f"[{stats['min_r2']:.3f}, {stats['max_r2']:.3f}]")

        return domain_stats

    def analyze_decision_strategy(self) -> dict[str, Any]:
        """Analyze performance of different decision strategies"""

        print("\n" + "=" * 80)
        print("🎯 DECISION STRATEGY ANALYSIS")
        print("=" * 80)

        strategies = {
            'llm': [],
            'ensemble': [],
            'nn': [],
            'unknown': []
        }

        for result in self.hybrid_results:
            decision = result.get('decision', 'unknown')
            r2 = result.get('evaluation', {}).get('r2')

            if decision in strategies and r2 is not None:
                strategies[decision].append({
                    'r2': r2,
                    'description': result.get('description', ''),
                    'domain': result.get('domain', ''),
                })

        # Calculate strategy statistics
        strategy_stats = {}

        for strategy, results in strategies.items():
            if results:
                r2_scores = [r['r2'] for r in results]
                strategy_stats[strategy] = {
                    'count': len(results),
                    'mean_r2': np.mean(r2_scores),
                    'median_r2': np.median(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'min_r2': np.min(r2_scores),
                    'max_r2': np.max(r2_scores),
                    'success_rate': sum(1 for r2 in r2_scores if r2 > 0.9) / len(r2_scores),
                }

        # Print strategy performance
        total = sum(s['count'] for s in strategy_stats.values())

        print("\n📊 Strategy Distribution:")
        print(f"   Total decisions: {total}")
        for strategy, stats in strategy_stats.items():
            pct = stats['count'] / total * 100 if total > 0 else 0
            print(f"   {strategy.upper():<10}: {stats['count']:>4} ({pct:>5.1f}%)")

        print("\n📈 Strategy Performance:")
        print(f"{'Strategy':<12} {'Count':<8} {'Mean R²':<12} {'Success Rate':<15} {'Range'}")
        print("-" * 80)

        for strategy in ['llm', 'ensemble', 'nn']:
            if strategy in strategy_stats:
                stats = strategy_stats[strategy]
                print(f"{strategy.upper():<12} {stats['count']:<8} {stats['mean_r2']:<12.6f} "
                      f"{stats['success_rate']*100:>6.1f}%        "
                      f"[{stats['min_r2']:.3f}, {stats['max_r2']:.3f}]")

        # Comparative analysis
        print("\n🔍 Comparative Insights:")

        if 'llm' in strategy_stats and 'ensemble' in strategy_stats:
            llm_mean = strategy_stats['llm']['mean_r2']
            ens_mean = strategy_stats['ensemble']['mean_r2']
            diff = llm_mean - ens_mean

            if diff > 0.01:
                print(f"   • LLM outperforms Ensemble by {diff:.4f} R² on average")
            elif diff < -0.01:
                print(f"   • Ensemble outperforms LLM by {abs(diff):.4f} R² on average")
            else:
                print(f"   • LLM and Ensemble perform similarly (Δ = {diff:.4f})")

        if 'nn' in strategy_stats:
            nn_success = strategy_stats['nn']['success_rate']
            print(f"   • Neural Network success rate: {nn_success*100:.1f}%")

        return strategy_stats

    def analyze_component_performance(self) -> dict[str, Any]:
        """Analyze individual component performance (LLM, Ensemble, NN)"""

        print("\n" + "=" * 80)
        print("🔧 COMPONENT PERFORMANCE ANALYSIS")
        print("=" * 80)

        component_stats = {
            'llm': {'r2_scores': [], 'success': [], 'failures': []},
            'ensemble': {'r2_scores': [], 'success': [], 'failures': []},
            'nn': {'r2_scores': [], 'success': [], 'failures': []}
        }

        for result in self.hybrid_results:
            # LLM results
            llm_result = result.get('llm_result', {})
            llm_metrics = llm_result.get('metrics', {})
            llm_r2 = llm_metrics.get('r2')

            if llm_r2 is not None:
                component_stats['llm']['r2_scores'].append(llm_r2)
                if llm_metrics.get('success', False):
                    component_stats['llm']['success'].append(result)
                else:
                    component_stats['llm']['failures'].append(result)

            # Ensemble results
            ensemble_result = result.get('ensemble_result', {})
            ensemble_metrics = ensemble_result.get('metrics', {})
            ensemble_r2 = ensemble_metrics.get('r2')

            if ensemble_r2 is not None:
                component_stats['ensemble']['r2_scores'].append(ensemble_r2)
                if ensemble_metrics.get('success', False):
                    component_stats['ensemble']['success'].append(result)
                else:
                    component_stats['ensemble']['failures'].append(result)

            # NN results
            nn_result = result.get('nn_result', {})
            nn_metrics = nn_result.get('metrics', {})
            nn_r2 = nn_metrics.get('r2')

            if nn_r2 is not None:
                component_stats['nn']['r2_scores'].append(nn_r2)
                if nn_metrics.get('success', False):
                    component_stats['nn']['success'].append(result)
                else:
                    component_stats['nn']['failures'].append(result)

        # Print component statistics
        print("\n📊 Component Statistics:")
        print(f"{'Component':<12} {'Attempts':<10} {'Successes':<12} {'Failures':<10} {'Mean R²'}")
        print("-" * 80)

        summary = {}

        for component, data in component_stats.items():
            attempts = len(data['r2_scores'])
            successes = len(data['success'])
            failures = len(data['failures'])
            mean_r2 = np.mean(data['r2_scores']) if data['r2_scores'] else 0

            summary[component] = {
                'attempts': attempts,
                'successes': successes,
                'failures': failures,
                'success_rate': successes / attempts if attempts > 0 else 0,
                'mean_r2': mean_r2,
            }

            print(f"{component.upper():<12} {attempts:<10} {successes:<12} {failures:<10} {mean_r2:.6f}")

        return summary

    def analyze_failures(self) -> list[dict[str, Any]]:
        """Analyze cases where the system performed poorly"""

        print("\n" + "=" * 80)
        print("❌ FAILURE ANALYSIS")
        print("=" * 80)

        failures = []

        for result in self.hybrid_results:
            r2 = result.get('evaluation', {}).get('r2')

            if r2 is not None and r2 < 0.90:
                failures.append({
                    'description': result.get('description', 'Unknown'),
                    'domain': result.get('domain', 'Unknown'),
                    'decision': result.get('decision', 'unknown'),
                    'r2': r2,
                    'llm_r2': result.get('llm_result', {}).get('metrics', {}).get('r2'),
                    'ensemble_r2': result.get('ensemble_result', {}).get('metrics', {}).get('r2'),
                    'nn_r2': result.get('nn_result', {}).get('metrics', {}).get('r2'),
                })

        print(f"\n⚠️  Found {len(failures)} cases with R² < 0.90")

        if failures:
            # Sort by R² (worst first)
            failures.sort(key=lambda x: x['r2'])

            print("\n🔍 Top 10 Worst Cases:")
            print(f"{'Description':<45} {'Domain':<15} {'Decision':<10} {'R²'}")
            print("-" * 90)

            for i, failure in enumerate(failures[:10], 1):
                desc = failure['description'][:43] + '...' if len(failure['description']) > 45 else failure['description']
                print(f"{desc:<45} {failure['domain']:<15} {failure['decision']:<10} {failure['r2']:.6f}")

            # Analyze failure patterns
            print("\n📊 Failure Patterns:")

            # By domain
            failure_domains = {}
            for f in failures:
                domain = f['domain']
                failure_domains[domain] = failure_domains.get(domain, 0) + 1

            print("\n   By Domain:")
            for domain, count in sorted(failure_domains.items(), key=lambda x: x[1], reverse=True):
                print(f"      {domain}: {count}")

            # By decision
            failure_decisions = {}
            for f in failures:
                decision = f['decision']
                failure_decisions[decision] = failure_decisions.get(decision, 0) + 1

            print("\n   By Decision:")
            for decision, count in sorted(failure_decisions.items(), key=lambda x: x[1], reverse=True):
                print(f"      {decision.upper()}: {count}")

        return failures

    def analyze_extrapolation(self) -> dict[str, Any]:
        """Analyze extrapolation performance"""

        if not self.extrapolation_results:
            print("\n⚠️  No extrapolation results available")
            return {}

        print("\n" + "=" * 80)
        print("🔮 EXTRAPOLATION ANALYSIS")
        print("=" * 80)

        r2_scores = []
        success_count = 0

        for result in self.extrapolation_results:
            if isinstance(result, dict):
                r2 = result.get('r2')
                success = result.get('success', False)
            else:
                continue

            if r2 is not None:
                r2_scores.append(r2)
                if success:
                    success_count += 1

        if r2_scores:
            stats = {
                'total_cases': len(self.extrapolation_results),
                'valid_r2_count': len(r2_scores),
                'success_count': success_count,
                'success_rate': success_count / len(r2_scores),
                'mean_r2': np.mean(r2_scores),
                'median_r2': np.median(r2_scores),
                'std_r2': np.std(r2_scores),
                'min_r2': np.min(r2_scores),
                'max_r2': np.max(r2_scores),
            }

            print("\n📈 Extrapolation Statistics:")
            print(f"   Total cases: {stats['total_cases']}")
            print(f"   Success rate: {stats['success_rate'] * 100:.1f}%")
            print(f"   Mean R²: {stats['mean_r2']:.6f}")
            print(f"   Median R²: {stats['median_r2']:.6f}")
            print(f"   Range: [{stats['min_r2']:.6f}, {stats['max_r2']:.6f}]")

            return stats
        else:
            print("❌ No valid extrapolation R² scores")
            return {}

    def generate_recommendations(self, overall_stats: dict, domain_stats: dict,
                                strategy_stats: dict, failures: list) -> list[str]:
        """Generate actionable recommendations based on analysis"""

        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)

        recommendations = []

        # Overall performance
        if overall_stats.get('mean_r2', 0) > 0.95:
            recommendations.append("✅ Overall performance is excellent (Mean R² > 0.95)")
        elif overall_stats.get('mean_r2', 0) > 0.90:
            recommendations.append("✅ Overall performance is good (Mean R² > 0.90)")
        else:
            recommendations.append("⚠️  Overall performance needs improvement (Mean R² < 0.90)")
            recommendations.append("   → Consider tuning hyperparameters or improving prompts")

        # Domain-specific
        if domain_stats:
            worst_domain = min(domain_stats.items(), key=lambda x: x[1]['mean_r2'])
            if worst_domain[1]['mean_r2'] < 0.85:
                recommendations.append(f"⚠️  Domain '{worst_domain[0]}' underperforms (Mean R² = {worst_domain[1]['mean_r2']:.4f})")
                recommendations.append(f"   → Add domain-specific training data for {worst_domain[0]}")

        # Strategy comparison
        if 'llm' in strategy_stats and 'ensemble' in strategy_stats:
            llm_mean = strategy_stats['llm']['mean_r2']
            ens_mean = strategy_stats['ensemble']['mean_r2']

            if llm_mean > ens_mean + 0.05:
                recommendations.append("✅ LLM significantly outperforms Ensemble")
                recommendations.append("   → Consider using LLM more frequently")
            elif ens_mean > llm_mean + 0.05:
                recommendations.append("✅ Ensemble significantly outperforms LLM")
                recommendations.append("   → Consider using Ensemble more frequently")

        # Failures
        if failures:
            if len(failures) > len(self.hybrid_results) * 0.1:
                recommendations.append(f"⚠️  High failure rate: {len(failures)} cases with R² < 0.90")
                recommendations.append("   → Review failure patterns and add targeted improvements")

            # Common failure domains
            failure_domains = {}
            for f in failures:
                domain = f['domain']
                failure_domains[domain] = failure_domains.get(domain, 0) + 1

            if failure_domains:
                worst_fail_domain = max(failure_domains.items(), key=lambda x: x[1])
                recommendations.append(f"⚠️  Most failures in '{worst_fail_domain[0]}' ({worst_fail_domain[1]} cases)")
                recommendations.append(f"   → Focus improvement efforts on {worst_fail_domain[0]} domain")

        # Print recommendations
        print()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:>2}. {rec}")

        return recommendations

    def save_report(self, overall_stats: dict, domain_stats: dict,
                   strategy_stats: dict, component_stats: dict,
                   failures: list, recommendations: list):
        """Save comprehensive report to JSON"""

        report = {
            'timestamp': self.timestamp,
            'overall': overall_stats,
            'by_domain': domain_stats,
            'by_strategy': strategy_stats,
            'component_performance': component_stats,
            'failures': failures,
            'recommendations': recommendations,
            'extrapolation_tests': self.extrapolation_results if self.extrapolation_results else [],
        }

        output_file = self.results_dir / f"report_hybrid_{self.timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Report saved to: {output_file}")

        return output_file

    def run_full_analysis(self):
        """Run complete analysis pipeline"""

        print("\n" + "=" * 80)
        print("🔍 HYBRID SYSTEM PERFORMANCE ANALYSIS")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print("=" * 80)

        # Load results
        if not self.load_latest_results():
            return

        # Run analyses
        overall_stats = self.analyze_overall_performance()
        domain_stats = self.analyze_by_domain()
        strategy_stats = self.analyze_decision_strategy()
        component_stats = self.analyze_component_performance()
        failures = self.analyze_failures()

        # Extrapolation (if available)
        self.analyze_extrapolation()

        # Generate recommendations
        recommendations = self.generate_recommendations(
            overall_stats, domain_stats, strategy_stats, failures
        )

        # Save report
        report_file = self.save_report(
            overall_stats, domain_stats, strategy_stats,
            component_stats, failures, recommendations
        )

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print("\n📊 Summary:")
        print(f"   Total cases analyzed: {overall_stats.get('total_cases', 0)}")
        print(f"   Mean R²: {overall_stats.get('mean_r2', 0):.6f}")
        print(f"   Success rate: {overall_stats.get('success_rate', 0) * 100:.1f}%")
        print(f"   Failures (R² < 0.90): {len(failures)}")
        print(f"\n📁 Report: {report_file}")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description="Analyze Hybrid DeFi System Performance")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='hypatiax/data/results',
        help='Directory containing results files'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = HybridPerformanceAnalyzer(results_dir=args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
