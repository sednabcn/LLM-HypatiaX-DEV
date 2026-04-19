#!/usr/bin/env python3
"""
Extract System 2 (Symbolic Only) and System 3 (LLM+Symbolic Fallback) Data
===========================================================================
Processes 30 individual test JSON files and extracts interpolation + extrapolation
data for statistical analysis.

Author: Ruperto Bonet Chaple
Date: January 2026
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import glob


class SystemDataExtractor:
    """Extract data from individual test files for Systems 2 & 3."""

    def __init__(self, test_directory: str = "."):
        """
        Initialize extractor.

        Args:
            test_directory: Directory containing individual test JSON files
        """
        self.test_dir = Path(test_directory)
        self.test_files = []
        self.extracted_data = {"System_2_Symbolic": [], "System_3_LLM_Fallback": []}

    def find_test_files(self, pattern: str = "*test*.json") -> List[Path]:
        """
        Find all test files matching pattern.

        Args:
            pattern: Glob pattern to match test files

        Returns:
            List of Path objects for test files
        """
        files = list(self.test_dir.glob(pattern))
        self.test_files = sorted(files)

        print(f"Found {len(self.test_files)} test files:")
        for f in self.test_files[:5]:  # Show first 5
            print(f"  • {f.name}")
        if len(self.test_files) > 5:
            print(f"  ... and {len(self.test_files) - 5} more")

        return self.test_files

    def extract_from_file(self, filepath: Path) -> Optional[Dict]:
        """
        Extract relevant data from a single test file.

        Args:
            filepath: Path to test JSON file

        Returns:
            Dictionary with extracted data or None if invalid
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Determine which system this is
            system_type = self._identify_system(data)
            if not system_type:
                return None

            # Extract test metadata
            test_name = data.get("test_name", filepath.stem)
            domain = data.get("domain", "unknown")

            # Extract interpolation metrics (R²)
            r2_score = data.get("r2_score", np.nan)
            success = data.get("success", False)

            # Extract formula
            expression = data.get("expression", "UNKNOWN")

            # Extract validation score
            validation_score = data.get("validation_score", np.nan)

            # Extract timing
            total_time = data.get("timing", {}).get("total", np.nan)

            # Extract discovery metadata
            discovery = data.get("discovery", {})
            llm_mode = discovery.get("llm_mode", "unknown")
            discovery_engine = discovery.get("discovery_engine", "unknown")

            # Extract extrapolation errors (if available)
            extrap_errors = self._extract_extrapolation_errors(data)

            result = {
                "test_name": test_name,
                "domain": domain,
                "system_type": system_type,
                "success": success,
                "r2_score": r2_score,
                "expression": expression,
                "validation_score": validation_score,
                "total_time": total_time,
                "llm_mode": llm_mode,
                "discovery_engine": discovery_engine,
                "extrap_near": extrap_errors.get("near", np.nan),
                "extrap_medium": extrap_errors.get("medium", np.nan),
                "extrap_far": extrap_errors.get("far", np.nan),
                "extrap_r2_near": extrap_errors.get("r2_near", np.nan),
                "extrap_r2_medium": extrap_errors.get("r2_medium", np.nan),
                "extrap_r2_far": extrap_errors.get("r2_far", np.nan),
                "source_file": filepath.name,
            }

            return result

        except Exception as e:
            print(f"⚠️  Error reading {filepath.name}: {e}")
            return None

    def _identify_system(self, data: Dict) -> Optional[str]:
        """
        Identify which system generated this data.

        System 2 (Symbolic Only): Uses PySR with 4-layer validation
        System 3 (LLM+Fallback): LLM primary, symbolic fallback

        Returns:
            'System_2_Symbolic' or 'System_3_LLM_Fallback' or None
        """
        discovery = data.get("discovery", {})
        llm_mode = discovery.get("llm_mode", "")
        discovery_engine = discovery.get("discovery_engine", "")

        # Check for LLM modes that indicate System 3
        llm_fallback_modes = [
            "hybrid_llm_only",  # LLM succeeded
            "hybrid_pysr_better",  # Fell back to symbolic
            "hybrid_llm_better",  # LLM was better
        ]

        if llm_mode in llm_fallback_modes:
            return "System_3_LLM_Fallback"

        # Check for pure symbolic (System 2)
        if discovery_engine == "symbolic" and llm_mode == "unknown":
            return "System_2_Symbolic"

        # If validation layers exist with high scores, likely System 2
        validation = data.get("validation", {})
        layer_results = validation.get("layer_results", {})
        if len(layer_results) >= 4:  # 4-layer validation
            return "System_2_Symbolic"

        return None

    def _extract_extrapolation_errors(self, data: Dict) -> Dict:
        """
        Extract extrapolation errors if available.

        Returns:
            Dictionary with near/medium/far errors and R² values
        """
        extrap = {}

        # Check if extrapolation data exists
        discovery = data.get("discovery", {})

        # Method 1: Direct extrapolation_errors field
        if "extrapolation_errors" in discovery:
            errors = discovery["extrapolation_errors"]
            extrap["near"] = errors.get("near", np.nan)
            extrap["medium"] = errors.get("medium", np.nan)
            extrap["far"] = errors.get("far", np.nan)

        # Method 2: extrapolation_r2 field
        if "extrapolation_r2" in discovery:
            r2_vals = discovery["extrapolation_r2"]
            extrap["r2_near"] = r2_vals.get("near", np.nan)
            extrap["r2_medium"] = r2_vals.get("medium", np.nan)
            extrap["r2_far"] = r2_vals.get("far", np.nan)

        # Method 3: Top-level fields
        if "extrapolation_errors" in data:
            errors = data["extrapolation_errors"]
            extrap["near"] = errors.get("near", np.nan)
            extrap["medium"] = errors.get("medium", np.nan)
            extrap["far"] = errors.get("far", np.nan)

        return extrap

    def process_all_files(self, pattern: str = "*test*.json") -> pd.DataFrame:
        """
        Process all test files and create comprehensive dataset.

        Args:
            pattern: Glob pattern for test files

        Returns:
            DataFrame with all extracted data
        """
        print("\n" + "=" * 80)
        print("EXTRACTING DATA FROM TEST FILES")
        print("=" * 80)

        # Find files
        self.find_test_files(pattern)

        if not self.test_files:
            print("\n❌ No test files found!")
            print(f"   Looking for pattern: {pattern}")
            print(f"   In directory: {self.test_dir.absolute()}")
            return pd.DataFrame()

        # Process each file
        all_results = []
        system_2_count = 0
        system_3_count = 0

        print(f"\nProcessing {len(self.test_files)} files...")

        for filepath in self.test_files:
            result = self.extract_from_file(filepath)

            if result:
                all_results.append(result)

                if result["system_type"] == "System_2_Symbolic":
                    system_2_count += 1
                elif result["system_type"] == "System_3_LLM_Fallback":
                    system_3_count += 1

        # Create DataFrame
        df = pd.DataFrame(all_results)

        # Summary
        print(f"\n✅ Successfully extracted {len(all_results)} tests")
        print(f"   • System 2 (Symbolic Only): {system_2_count} tests")
        print(f"   • System 3 (LLM+Fallback): {system_3_count} tests")
        print(
            f"   • Unidentified: {len(all_results) - system_2_count - system_3_count} tests"
        )

        return df

    def create_interpolation_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interpolation (R²) comparison table.

        Args:
            df: DataFrame from process_all_files()

        Returns:
            Summary DataFrame with R² statistics
        """
        summary = (
            df.groupby("system_type")
            .agg(
                {
                    "r2_score": ["count", "mean", "std", "min", "max"],
                    "validation_score": ["mean", "std"],
                    "total_time": ["mean", "std"],
                }
            )
            .round(4)
        )

        return summary

    def create_extrapolation_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create extrapolation error comparison table.

        Args:
            df: DataFrame from process_all_files()

        Returns:
            Summary DataFrame with extrapolation statistics
        """
        # Filter out inf/nan values
        df_clean = df.copy()

        for col in ["extrap_near", "extrap_medium", "extrap_far"]:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

        summary = (
            df_clean.groupby("system_type")
            .agg(
                {
                    "extrap_near": ["count", "mean", "std"],
                    "extrap_medium": ["count", "mean", "std"],
                    "extrap_far": ["count", "mean", "std"],
                }
            )
            .round(2)
        )

        return summary

    def generate_json_for_statistical_analysis(
        self, df: pd.DataFrame, output_file: str = "systems_2_3_data.json"
    ):
        """
        Generate JSON file compatible with your statistical_analysis.py script.

        Args:
            df: DataFrame from process_all_files()
            output_file: Output JSON filename
        """
        # Create structure matching your existing JSON format
        output = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "version": "Systems 2 & 3 Extraction",
            "methods": ["System 2 Symbolic", "System 3 LLM+Fallback"],
            "total_tests": len(df),
            "tests": [],
        }

        # Group by test name
        for test_name, group in df.groupby("test_name"):
            test_entry = {
                "test_name": test_name,
                "domain": group.iloc[0]["domain"],
                "results": {},
            }

            # Add results for each system
            for _, row in group.iterrows():
                system_name = row["system_type"].replace("_", " ")

                result = {
                    "method": system_name,
                    "test_name": test_name,
                    "domain": row["domain"],
                    "success": bool(row["success"]),
                    "r2": (
                        float(row["r2_score"]) if not pd.isna(row["r2_score"]) else 0.0
                    ),
                    "rmse": np.inf,  # Not extracted, use infinity
                    "time": (
                        float(row["total_time"])
                        if not pd.isna(row["total_time"])
                        else 0.0
                    ),
                    "formula": str(row["expression"]),
                    "error": None,
                    "metadata": {
                        "validation_score": (
                            float(row["validation_score"])
                            if not pd.isna(row["validation_score"])
                            else 0.0
                        ),
                        "llm_mode": str(row["llm_mode"]),
                        "discovery_engine": str(row["discovery_engine"]),
                    },
                }

                # Add extrapolation errors if available
                if not pd.isna(row["extrap_near"]):
                    result["extrapolation_errors"] = {
                        "near": (
                            float(row["extrap_near"])
                            if row["extrap_near"] != np.inf
                            else np.inf
                        ),
                        "medium": (
                            float(row["extrap_medium"])
                            if row["extrap_medium"] != np.inf
                            else np.inf
                        ),
                        "far": (
                            float(row["extrap_far"])
                            if row["extrap_far"] != np.inf
                            else np.inf
                        ),
                    }

                test_entry["results"][system_name] = result

            output["tests"].append(test_entry)

        # Save to file
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n✅ Generated statistical analysis JSON: {output_file}")
        print(f"   • {len(output['tests'])} unique tests")
        print(f"   • Ready for statistical_analysis.py")


def main():
    """Main execution function."""

    print("\n" + "=" * 80)
    print("SYSTEM 2 & 3 DATA EXTRACTION TOOL")
    print("=" * 80)

    # Initialize extractor
    extractor = SystemDataExtractor(test_directory=".")

    # Option 1: Process files matching a pattern
    print("\nOption 1: Process all *test*.json files in current directory")
    print("Option 2: Specify custom pattern")
    print("Option 3: Specify directory path")

    # For automated use, just process all files
    df = extractor.process_all_files(pattern="*.json")

    if df.empty:
        print("\n❌ No data extracted. Check your file patterns and directory.")
        return

    # Display summary statistics
    print("\n" + "=" * 80)
    print("INTERPOLATION (R²) SUMMARY")
    print("=" * 80)
    interp_summary = extractor.create_interpolation_comparison(df)
    print(interp_summary)

    print("\n" + "=" * 80)
    print("EXTRAPOLATION ERROR SUMMARY")
    print("=" * 80)
    extrap_summary = extractor.create_extrapolation_comparison(df)
    print(extrap_summary)

    # Save detailed data
    df.to_csv("systems_2_3_detailed.csv", index=False)
    print("\n✅ Saved detailed data: systems_2_3_detailed.csv")

    # Generate JSON for statistical analysis
    extractor.generate_json_for_statistical_analysis(df)

    # Show sample of data
    print("\n" + "=" * 80)
    print("SAMPLE DATA (first 5 tests)")
    print("=" * 80)
    print(df[["test_name", "system_type", "r2_score", "extrap_medium"]].head())

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. systems_2_3_detailed.csv  - Full detailed data")
    print("  2. systems_2_3_data.json     - For statistical_analysis.py")
    print("\nNext steps:")
    print("  1. Review systems_2_3_detailed.csv for data quality")
    print("  2. Run statistical_analysis.py with new JSON file")
    print("  3. Compare all 5 systems (Pure LLM, NN, Hybrid v40, System 2, System 3)")


if __name__ == "__main__":
    main()
