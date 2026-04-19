#!/usr/bin/env python3
"""
Merge All 5 Systems Data for Comprehensive Statistical Analysis
================================================================
Combines:
  - Pure LLM, Neural Network, Hybrid v40 (from extrapolation file)
  - System 2 Symbolic, System 3 LLM+Fallback (from systems_2_3_data.json)

Creates unified dataset for n=30 per system analysis.

Author: Ruperto Bonet Chaple
Date: January 2026
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Reproducibility seeds (added for JMLR submission)
random.seed(42)
np.random.seed(42)


class ComprehensiveSystemMerger:
    """Merge data from multiple sources into unified format."""

    def __init__(
        self,
        extrap_file: str = "all_domains_extrap_v4_20260120_223747.json",
        interp_file: str = "standalone_real_methods_20260116_003311.json",
        systems_23_file: str = "systems_2_3_2_data.json",
    ):
        """
        Initialize merger with data files.

        Args:
            extrap_file: Extrapolation data (Hybrid v40, NN, Pure LLM)
            interp_file: Interpolation data (R² scores)
            systems_23_file: Systems 2 & 3 data
        """
        self.extrap_file = extrap_file
        self.interp_file = interp_file
        self.systems_23_file = systems_23_file

        self.data = {
            "Pure_LLM": {},
            "Neural_Network": {},
            "Hybrid_v40": {},
            "System_2_Symbolic": {},
            "System_3_LLM_Fallback": {},
        }

    def load_all_data(self):
        """Load all data files."""
        print("\n" + "=" * 80)
        print("LOADING DATA FILES")
        print("=" * 80)

        # Load extrapolation data
        with open(self.extrap_file, "r") as f:
            self.extrap_data = json.load(f)
        print(f"✅ Loaded: {self.extrap_file}")
        print(f"   Tests: {self.extrap_data['total_tests']}")

        # Load interpolation data
        with open(self.interp_file, "r") as f:
            self.interp_data = json.load(f)
        print(f"✅ Loaded: {self.interp_file}")
        print(f"   Tests: {self.interp_data['total_tests']}")

        # Load Systems 2 & 3 data
        with open(self.systems_23_file, "r") as f:
            self.systems_23_data = json.load(f)
        print(f"✅ Loaded: {self.systems_23_file}")
        print(f"   Tests: {self.systems_23_data['total_tests']}")

    def merge_data(self) -> Dict:
        """
        Merge all data sources into unified structure.

        Returns:
            Unified data dictionary
        """
        print("\n" + "=" * 80)
        print("MERGING DATA")
        print("=" * 80)

        unified = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "version": "Comprehensive 5-System Dataset v1.0",
            "methods": [
                "Pure LLM",
                "Neural Network",
                "Hybrid System v40",
                "System 2 Symbolic",
                "System 3 LLM+Fallback",
            ],
            "total_tests": 0,
            "tests": [],
        }

        # Create test name mapping
        test_map = {}

        # Step 1: Process extrapolation data (Systems: Pure LLM, NN, Hybrid v40)
        print("\n[1/3] Processing extrapolation data...")
        for test in self.extrap_data["tests"]:
            test_name = test["test_name"]
            domain = test["domain"]

            if test_name not in test_map:
                test_map[test_name] = {
                    "test_name": test_name,
                    "domain": domain,
                    "results": {},
                }

            # Add results for each method
            for method_key in ["Pure LLM", "Neural Network", "Hybrid System v40"]:
                if method_key in test["results"]:
                    result = test["results"][method_key]

                    # Map method name to unified naming
                    unified_method = self._map_method_name(method_key)

                    test_map[test_name]["results"][unified_method] = result

        print(f"   Added {len(test_map)} tests from extrapolation file")

        # Step 2: Add interpolation R² data from interp_file
        print("\n[2/3] Adding interpolation R² scores...")
        for test in self.interp_data["tests"]:
            test_name = test["test_name"]

            if test_name not in test_map:
                # Test exists in interp but not extrap - skip for now
                continue

            # Update R² scores from interpolation file (more accurate)
            for method_key in ["Pure LLM", "Neural Network", "Hybrid System v40"]:
                if method_key in test["results"]:
                    unified_method = self._map_method_name(method_key)

                    if unified_method in test_map[test_name]["results"]:
                        # Update R² from interpolation file
                        interp_r2 = test["results"][method_key].get("r2", np.nan)
                        test_map[test_name]["results"][unified_method]["r2"] = interp_r2

        print(f"   Updated R² scores for {len(test_map)} tests")

        # Step 3: Add Systems 2 & 3 data
        print("\n[3/3] Adding Systems 2 & 3 data...")
        systems_23_count = 0

        for test in self.systems_23_data["tests"]:
            test_name = test["test_name"]
            domain = test["domain"]

            # Handle different test naming conventions
            # e.g., "biology_allometric_scaling" vs "allometric_scaling"
            base_name = test_name.split("_", 1)[-1] if "_" in test_name else test_name

            # Try to find matching test
            matched_test = None
            for existing_test in test_map:
                if base_name in existing_test or existing_test in base_name:
                    matched_test = existing_test
                    break

            # If no match, create new entry
            if not matched_test:
                test_map[test_name] = {
                    "test_name": test_name,
                    "domain": domain,
                    "results": {},
                }
                matched_test = test_name

            # Add Systems 2 & 3 results
            for method_key in test["results"]:
                result = test["results"][method_key]

                # Map method name
                unified_method = self._map_method_name(method_key)

                test_map[matched_test]["results"][unified_method] = result
                systems_23_count += 1

        print(f"   Added {systems_23_count} system results")

        # Convert to list
        unified["tests"] = list(test_map.values())
        unified["total_tests"] = len(unified["tests"])

        # Summary
        print("\n" + "=" * 80)
        print("MERGE SUMMARY")
        print("=" * 80)
        print(f"Total unique tests: {unified['total_tests']}")

        # Count coverage per system
        coverage = {method: 0 for method in unified["methods"]}
        for test in unified["tests"]:
            for method in unified["methods"]:
                if method in test["results"]:
                    coverage[method] += 1

        print("\nCoverage per system:")
        for method, count in coverage.items():
            print(f"  • {method:30s}: {count:2d} tests")

        return unified

    def _map_method_name(self, method_name: str) -> str:
        """
        Map various method names to unified naming.

        Args:
            method_name: Original method name

        Returns:
            Unified method name
        """
        mapping = {
            "Pure LLM": "Pure LLM",
            "Neural Network": "Neural Network",
            "Hybrid System v40": "Hybrid System v40",
            "System 2 Symbolic": "System 2 Symbolic",
            "System 3 LLM Fallback": "System 3 LLM+Fallback",
            "System 3 LLM+Fallback": "System 3 LLM+Fallback",
        }

        return mapping.get(method_name, method_name)

    def save_merged_data(
        self, unified: Dict, output_file: str = "all_systems_comprehensive.json"
    ):
        """
        Save merged data to file.

        Args:
            unified: Unified data dictionary
            output_file: Output filename
        """
        with open(output_file, "w") as f:
            json.dump(unified, f, indent=2)

        print(f"\n✅ Saved merged data: {output_file}")

    def create_summary_table(self, unified: Dict) -> pd.DataFrame:
        """
        Create summary table showing data availability.

        Args:
            unified: Unified data dictionary

        Returns:
            Summary DataFrame
        """
        rows = []

        for test in unified["tests"]:
            test_name = test["test_name"]
            domain = test["domain"]

            row = {"test_name": test_name, "domain": domain}

            # Check each system
            for method in unified["methods"]:
                if method in test["results"]:
                    result = test["results"][method]
                    r2 = result.get("r2", np.nan)
                    has_extrap = "extrapolation_errors" in result

                    row[f"{method}_R2"] = r2
                    row[f"{method}_Extrap"] = "✓" if has_extrap else "✗"
                else:
                    row[f"{method}_R2"] = np.nan
                    row[f"{method}_Extrap"] = "✗"

            rows.append(row)

        return pd.DataFrame(rows)

    def identify_common_tests(self, unified: Dict) -> List[str]:
        """
        Identify tests that have ALL 5 systems tested.

        Args:
            unified: Unified data dictionary

        Returns:
            List of test names with complete coverage
        """
        complete_tests = []

        for test in unified["tests"]:
            if len(test["results"]) == 5:
                complete_tests.append(test["test_name"])

        return complete_tests


def main():
    """Main execution."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE 5-SYSTEM DATA MERGER")
    print("=" * 80)

    # Initialize merger
    merger = ComprehensiveSystemMerger()

    # Load data
    merger.load_all_data()

    # Merge data
    unified = merger.merge_data()

    # Save merged data
    merger.save_merged_data(unified, "all_systems_comprehensive.json")

    # Create summary table
    print("\n" + "=" * 80)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 80)
    summary = merger.create_summary_table(unified)
    summary.to_csv("data_availability_summary.csv", index=False)
    print("\n✅ Saved: data_availability_summary.csv")

    # Show sample
    print("\nSample (first 5 tests):")
    display_cols = ["test_name", "domain"] + [f"{m}_R2" for m in unified["methods"][:3]]
    print(summary[display_cols].head())

    # Identify complete tests
    print("\n" + "=" * 80)
    print("COMPLETE TEST COVERAGE")
    print("=" * 80)
    complete = merger.identify_common_tests(unified)
    print(f"Tests with ALL 5 systems: {len(complete)}")

    if complete:
        print("\nComplete tests:")
        for test_name in complete[:10]:
            print(f"  • {test_name}")
        if len(complete) > 10:
            print(f"  ... and {len(complete) - 10} more")

    # Final recommendations
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review data_availability_summary.csv to check coverage")
    print("2. Use all_systems_comprehensive.json for statistical analysis")
    print("3. Update statistical_analysis.py to handle 5 systems")
    print("\nRecommended analysis approach:")
    print("  • Focus on tests with complete 5-system coverage")
    print("  • Report partial coverage tests separately")
    print("  • Use n=15-30 depending on system coverage")

    # Create analysis-ready subset
    print("\n" + "=" * 80)
    print("CREATING ANALYSIS-READY SUBSET")
    print("=" * 80)

    # Create subset with only complete tests
    complete_subset = {
        "timestamp": unified["timestamp"],
        "version": "Complete Coverage Subset - All 5 Systems",
        "methods": unified["methods"],
        "total_tests": len(complete),
        "tests": [test for test in unified["tests"] if test["test_name"] in complete],
    }

    with open("all_systems_complete_only.json", "w") as f:
        json.dump(complete_subset, f, indent=2)

    print(f"✅ Saved complete-coverage subset: all_systems_complete_only.json")
    print(f"   Contains {len(complete)} tests with all 5 systems")

    print("\n" + "=" * 80)
    print("MERGE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
