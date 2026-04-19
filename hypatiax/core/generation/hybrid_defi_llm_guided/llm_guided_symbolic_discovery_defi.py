#!/usr/bin/env python3
"""
LLM-GUIDED SYMBOLIC DISCOVERY FOR DEFI v1.0
============================================
Specialized version for DeFi protocol test cases with enhanced reporting.

Based on llm_guided_symbolic_discovery_v12.py
Integrated with experiment_protocol_defi_20.py

Features:
✅ 20 DeFi test cases across 6 domains
✅ 6 extrapolation tests
✅ Hybrid LLM + PySR discovery
✅ Session management with resume
✅ Detailed results table with observations
✅ Complete JSON export
✅ Statistical analysis

Usage:
    # Run all DeFi tests with hybrid mode
    python llm_guided_symbolic_discovery_defi.py --batch --mode hybrid

    # Run single test (e.g., Nernst-like impermanent loss)
    python llm_guided_symbolic_discovery_defi.py --test impermanent_loss --mode hybrid

    # Resume interrupted run
    python llm_guided_symbolic_discovery_defi.py --batch --resume

    # Custom iterations
    python llm_guided_symbolic_discovery_defi.py --batch --iterations 50

Author: HypatiaX Team
Version: 1.0 DeFi Edition
Date: 2026-01-13
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# ============================================================================
# SETUP & PATHS
# ============================================================================

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

RESULTS_DIR = Path("hypatiax/data/results/llm_guided_defi")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from hypatiax.tools.symbolic.symbolic_engine import (
        DiscoveryConfig,
        LLMConfig,
        SymbolicEngineWithLLM,
    )
    from hypatiax.tools.symbolic.hybrid_system_v40 import (
        HybridDiscoverySystem,
        DiscoveryMode,
    )

    HAS_INTEGRATED_ENGINE = True
    print("✅ Using integrated SymbolicEngineWithLLM v20")
except ImportError:
    HAS_INTEGRATED_ENGINE = False
    print("⚠️  Integrated engine not available")
    sys.exit(1)

try:
    from hypatiax.tools.validation.ensemble_validator import EnsembleValidator

    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False
    print("⚠️  EnsembleValidator not available")

# Import DeFi Protocol
try:
    from hypatiax.protocols.experiment_protocol_defi_20 import DeFiExperimentProtocolExtended

    print("✅ Loaded DeFi Protocol v3.0")
except ImportError:
    print("❌ Error: experiment_protocol_defi_20.py not found")
    sys.exit(1)

# ============================================================================
# JSON SERIALIZATION
# ============================================================================


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return convert_to_json_serializable(obj.__dict__)
    return obj


# ============================================================================
# DEFI PROTOCOL TO TEST CASES CONVERTER
# ============================================================================


def convert_defi_protocol_to_test_cases(
    protocol: DeFiExperimentProtocolExtended, domains: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Convert DeFi protocol to test cases dictionary."""

    test_cases = {}
    all_domains = protocol.get_all_domains()
    domains_to_load = domains if domains else all_domains

    print(f"\n📥 Converting DeFi Protocol to test cases...")
    print(f"   Domains: {', '.join(domains_to_load)}")

    for domain in domains_to_load:
        if domain not in all_domains:
            print(f"⚠️  Domain '{domain}' not found, skipping...")
            continue

        protocol_tests = protocol.load_test_data(domain, num_samples=100)

        for desc, X_sample, y_sample, var_names, metadata in protocol_tests:
            eq_name = metadata.get("equation_name", "unknown")
            test_name = f"{domain}_{eq_name}"

            # Create data generator closure

            def make_generator(prot, dom, eq):
                def generator(n):
                    tests = prot.load_test_data(dom, num_samples=n)
                    for d, X, y, v, m in tests:
                        if m.get("equation_name") == eq:
                            # Protocol already computed y correctly
                            # Just return it wrapped in a function
                            y_copy = y.copy()

                            def y_func(X_input):
                                return y_copy

                            return X, y_func
                    raise ValueError(f"Test {eq} not found in domain {dom}")

                return generator

            # Extract metadata
            var_descriptions = {var: f"{var} in {desc}" for var in var_names}
            units = metadata.get("units", {var: "dimensionless" for var in var_names})

            test_cases[test_name] = {
                "domain": domain,
                "equation_name": eq_name,
                "name": metadata.get("equation_name", desc).replace("_", " ").title(),
                "description": desc,
                "ground_truth": metadata.get("ground_truth", ""),
                "variables": var_names,
                "variable_descriptions": var_descriptions,
                "variable_units": units,
                "variable_roles": metadata.get("variable_roles", {}),
                "generate_data": make_generator(protocol, domain, eq_name),
                "use_enhanced_config": metadata.get("use_enhanced_config", False),
                "extrapolation_test": metadata.get("extrapolation_test", False),
                "difficulty": metadata.get("difficulty", "medium"),
                "metadata": metadata,
            }

    print(f"✅ Converted {len(test_cases)} test cases from protocol")

    # Show extrapolation tests
    extrap_tests = [
        name for name, tc in test_cases.items() if tc.get("extrapolation_test")
    ]
    if extrap_tests:
        print(f"\n🚀 Extrapolation tests ({len(extrap_tests)}):")
        for name in extrap_tests:
            print(f"   - {name}")

    return test_cases


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================


class SessionManager:
    """Manages test sessions with checkpointing and complete results export."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = (
            session_id or f"llm_defi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.session_dir = RESULTS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.session_dir / "checkpoint.json"
        self.completed_tests = set()
        self.failed_tests = set()
        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    self.completed_tests = set(data.get("completed", []))
                    self.failed_tests = set(data.get("failed", []))
                    print(
                        f"\n📂 Checkpoint: {len(self.completed_tests)} completed, {len(self.failed_tests)} failed"
                    )
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(
                    {
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat(),
                        "completed": list(self.completed_tests),
                        "failed": list(self.failed_tests),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")

    def is_completed(self, test_name: str) -> bool:
        return test_name in self.completed_tests

    def save_test_result(self, test_name: str, result: Dict, passed: bool):
        """Save test result with proper JSON serialization."""
        test_file = self.session_dir / f"{test_name}.json"

        result["_metadata"] = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "passed": bool(passed),
            "test_name": test_name,
            "method": "llm_guided_defi_v1.0",
        }

        clean_result = convert_to_json_serializable(result)

        try:
            with open(test_file, "w") as f:
                json.dump(clean_result, f, indent=2, default=str)

            if passed:
                self.completed_tests.add(test_name)
            else:
                self.failed_tests.add(test_name)
            self._save_checkpoint()
            print(f"   💾 Saved: {test_file.name}")

        except Exception as e:
            print(f"   ❌ Failed to save {test_file.name}: {e}")
            if passed:
                self.completed_tests.add(test_name)
            else:
                self.failed_tests.add(test_name)
            self._save_checkpoint()

    def load_all_results(self) -> Dict[str, Dict]:
        results = {}
        for f in self.session_dir.glob("*.json"):
            if f.name not in [
                "checkpoint.json",
                "summary.json",
                "complete_results.json",
            ]:
                try:
                    with open(f, "r") as file:
                        results[f.stem] = json.load(file)
                except Exception as e:
                    print(f"⚠️  Failed to load {f.name}: {e}")
        return results

    def get_pending_tests(self, all_tests: List[str]) -> List[str]:
        return [t for t in all_tests if t not in self.completed_tests]

    def save_summary(self, summary: Dict):
        """Save summary with complete JSON export."""
        # Save standard summary
        summary_file = self.session_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n📊 Summary saved: {summary_file}")

        # Save complete results export
        complete_export = self._generate_complete_export(summary)
        complete_file = self.session_dir / "complete_results.json"
        with open(complete_file, "w") as f:
            json.dump(complete_export, f, indent=2, default=str)
        print(f"📦 Complete results: {complete_file}")
        print(f"   Size: {complete_file.stat().st_size / 1024:.1f} KB")

    def _generate_complete_export(self, summary: Dict) -> Dict:
        """Generate complete JSON export with all test outputs."""
        all_results = self.load_all_results()

        export = {
            "session_metadata": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "configuration": summary.get("configuration", {}),
                "total_tests": summary.get("total_tests", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "pass_rate": (
                    summary.get("passed", 0) / max(summary.get("total_tests", 1), 1)
                )
                * 100,
                "total_time_minutes": summary.get("total_time", 0) / 60,
            },
            "summary_statistics": {
                "by_domain": dict(summary.get("by_domain", {})),
                "by_difficulty": dict(summary.get("by_difficulty", {})),
                "extrapolation_tests": summary.get("extrapolation_results", []),
                "detailed_results_table": summary.get("detailed_results", []),
            },
            "individual_test_results": {},
        }

        # Add each test's complete output
        for test_name, result in all_results.items():
            test_export = {
                "metadata": {
                    "test_name": test_name,
                    "domain": result.get("domain", "unknown"),
                    "difficulty": result.get("difficulty", "unknown"),
                    "ground_truth": result.get("ground_truth", "N/A"),
                    "extrapolation_test": result.get("extrapolation_test", False),
                    "timestamp": result.get("timestamp", "unknown"),
                    "execution_time_seconds": result.get("timing", {}).get(
                        "total", 0.0
                    ),
                },
                "discovery": result.get("discovery", {}),
                "validation": result.get("validation", {}),
                "variables": {
                    "names": result.get("test_config", {}).get("variables", []),
                    "units": result.get("test_config", {}).get("variable_units", {}),
                    "descriptions": result.get("test_config", {}).get(
                        "variable_descriptions", {}
                    ),
                },
                "error": result.get("error"),
                "full_result": result,
            }

            export["individual_test_results"][test_name] = test_export

        return export


# ============================================================================
# INTEGRATED LLM-GUIDED DISCOVERY FOR DEFI
# ============================================================================


class IntegratedLLMDiscoveryDeFi:
    """LLM-guided discovery optimized for DeFi test cases."""

    def __init__(
        self,
        llm_mode: str = "hybrid",
        api_key: Optional[str] = None,
        niterations: int = 50,
    ):
        """Initialize integrated LLM discovery for DeFi."""
        if not HAS_INTEGRATED_ENGINE:
            raise ImportError("SymbolicEngineWithLLM not available")

        self.llm_mode = llm_mode
        self.niterations = niterations

        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()
            self.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key and llm_mode != "none":
            raise ValueError("ANTHROPIC_API_KEY required for LLM modes")

        print(f"✅ Integrated LLM Discovery initialized (DeFi)")
        print(f"   Mode: {llm_mode}")
        print(f"   Iterations: {niterations}")

    def discover(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: List[str],
        domain: str,
        description: str,
        variable_descriptions: Optional[Dict[str, str]] = None,
        variable_units: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Discover equation using integrated engine."""

        if verbose:
            print(f"\n{'='*80}")
            print(f"INTEGRATED LLM-GUIDED DISCOVERY - DEFI v1.0")
            print(f"{'='*80}")
            print(f"Domain: {domain}")
            print(f"Variables: {', '.join(variable_names)}")
            print(f"Samples: {len(y)}")
            print(f"Mode: {self.llm_mode}")

        start_time = time.time()

        try:
            # Create discovery config
            discovery_config = DiscoveryConfig(
                niterations=self.niterations,
                populations=50,
                enable_auto_configuration=True,
            )

            # Create LLM config if needed
            llm_config = None
            if self.llm_mode != "none":
                llm_config = LLMConfig(
                    enabled=True,
                    api_key=self.api_key,
                    n_candidates=5,
                    model="claude-sonnet-4-6",
                )

            # Create symbolic engine with LLM
            symbolic_engine = SymbolicEngineWithLLM(
                config=discovery_config,
                domain=domain,
                llm_config=llm_config,
                llm_mode=self.llm_mode,
            )

            # Create hybrid system
            hybrid = HybridDiscoverySystem(
                domain=domain,
                discovery_config=discovery_config,
                discovery_mode=DiscoveryMode.CALIBRATED,
                max_retries=3,
                enable_physics_fallback=False,
            )

            # Patch hybrid system to use LLM engine
            if self.llm_mode != "none" and llm_config:
                print(f"   🔧 Patching hybrid system with LLM engine")
                hybrid.symbolic_engine = symbolic_engine

            # Run discovery
            if verbose:
                print(f"\n🔬 Starting discovery...")

            result = hybrid.discover_validate_interpret(
                X=X,
                y=y,
                variable_names=variable_names,
                variable_descriptions=variable_descriptions or {},
                variable_units=variable_units or {},
                description=description,
                equation_name=description,
                validate_first=True,
            )

            # Extract results
            discovery = result.get("discovery", {})
            validation = result.get("validation", {})

            total_time = time.time() - start_time

            # Determine success (DeFi-specific criteria)
            r2 = discovery.get("r2_score", 0.0)
            val_score = validation.get("total_score", 0.0)

            success = (r2 > 0.99 and val_score > 30.0) or (
                r2 > 0.95 and val_score > 80.0
            )

            if verbose:
                print(f"\n{'='*80}")
                status = "✅ SUCCESS" if success else "⚠️  BELOW THRESHOLD"
                print(f"{status}")
                print(f"   Expression: {discovery.get('expression')}")
                print(f"   R² Score: {r2:.4f}")
                print(f"   Validation: {val_score:.1f}/100")
                print(f"   Total time: {total_time:.2f}s")

                llm_mode_used = discovery.get("llm_mode", "unknown")
                if llm_mode_used:
                    print(f"   LLM Mode: {llm_mode_used}")

            return {
                "success": success,
                "r2_score": r2,
                "validation_score": val_score,
                "expression": discovery.get("expression"),
                "discovery": discovery,
                "validation": validation,
                "timing": {"total": total_time},
                "llm_mode": discovery.get("llm_mode", self.llm_mode),
                "test_name": description,
                "timestamp": datetime.now().isoformat(),
                "ground_truth": "",
                "domain": domain,
            }

        except Exception as e:
            if verbose:
                print(f"\n❌ Error: {e}")
                import traceback

                traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "r2_score": 0.0,
                "validation_score": 0.0,
                "expression": None,
                "timing": {"total": time.time() - start_time},
            }


# ============================================================================
# TEST EXECUTION
# ============================================================================


def run_single_test_llm(
    test_name: str,
    test_cases: Dict,
    llm_mode: str = "hybrid",
    api_key: Optional[str] = None,
    niterations: int = 50,
    verbose: bool = True,
    session: Optional[SessionManager] = None,
) -> Dict:
    """Run single test with integrated LLM-guided discovery."""

    test_config = test_cases[test_name]

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {test_config['name']} | Domain: {test_config['domain']}")
        print(f"{'='*80}")
        print(f"Difficulty: {test_config.get('difficulty', 'unknown')}")
        if test_config.get("extrapolation_test"):
            print(f"🚀 EXTRAPOLATION TEST")

    start = time.time()

    try:
        # Generate data
        X, y_func = test_config["generate_data"](1000)
        y = y_func(X)

        # Discover
        discoverer = IntegratedLLMDiscoveryDeFi(
            llm_mode=llm_mode, api_key=api_key, niterations=niterations
        )

        result = discoverer.discover(
            X=X,
            y=y,
            variable_names=test_config["variables"],
            domain=test_config["domain"],
            description=test_config.get("name", test_name),
            variable_descriptions=test_config.get("variable_descriptions", {}),
            variable_units=test_config.get("variable_units", {}),
            verbose=verbose,
        )

        result.update(
            {
                "test_name": test_name,
                "ground_truth": test_config.get("ground_truth", ""),
                "difficulty": test_config.get("difficulty", "unknown"),
                "extrapolation_test": test_config.get("extrapolation_test", False),
                "test_config": test_config,
            }
        )

        passed = result["success"]

        if session:
            session.save_test_result(test_name, result, passed)

        return result

    except Exception as e:
        error_result = {
            "error": str(e),
            "test_name": test_name,
            "timing": {"total": time.time() - start},
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "difficulty": test_config.get("difficulty", "unknown"),
            "extrapolation_test": test_config.get("extrapolation_test", False),
        }
        if session:
            session.save_test_result(test_name, error_result, False)
        if verbose:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()
        return error_result


def run_single_test_by_name(
    test_name: str,
    test_cases: Dict,
    llm_mode: str = "hybrid",
    api_key: Optional[str] = None,
    niterations: int = 50,
) -> Dict:
    """Run a single named test."""

    # Find matching tests
    matching_tests = [t for t in test_cases.keys() if test_name.lower() in t.lower()]

    if not matching_tests:
        print(f"❌ Test '{test_name}' not found")
        print(f"\nAvailable tests:")
        for t in sorted(test_cases.keys()):
            print(f"  - {t}")
        return {}

    if len(matching_tests) > 1:
        print(f"⚠️  Multiple matches found:")
        for t in matching_tests:
            print(f"  - {t}")
        print(f"\nUsing: {matching_tests[0]}")

    target_test = matching_tests[0]
    print(f"\n🎯 Running single test: {target_test}")

    # Run the test
    result = run_single_test_llm(
        target_test,
        test_cases,
        llm_mode=llm_mode,
        api_key=api_key,
        niterations=niterations,
        verbose=True,
        session=None,
    )

    return result


def run_defi_suite(
    test_cases: Dict,
    llm_mode: str = "hybrid",
    api_key: Optional[str] = None,
    niterations: int = 50,
    resume: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Dict]:
    """Run full DeFi suite with integrated LLM-guided discovery."""

    # Session management
    if resume and Path(RESULTS_DIR / "current_session.json").exists():
        with open(RESULTS_DIR / "current_session.json", "r") as f:
            session_id = json.load(f).get("session_id")

    session = SessionManager(session_id)
    with open(RESULTS_DIR / "current_session.json", "w") as f:
        json.dump({"session_id": session.session_id}, f)

    print(f"\n{'='*80}")
    print(f"LLM-GUIDED DISCOVERY - DEFI SUITE v1.0")
    print(f"{'='*80}")
    print(f"Tests: {len(test_cases)}")
    print(f"Mode: {llm_mode}")
    print(f"Iterations: {niterations}")

    # Get pending tests
    pending = (
        session.get_pending_tests(list(test_cases.keys()))
        if resume
        else list(test_cases.keys())
    )

    if not pending:
        print("✅ All tests completed!")
        results = session.load_all_results()
        print_results_table(results, test_cases)
        return results

    print(f"Running: {len(pending)}/{len(test_cases)} tests")

    # Run tests
    for i, test_name in enumerate(pending, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(pending)}: {test_name}")
        print(f"{'='*80}")

        try:
            run_single_test_llm(
                test_name,
                test_cases,
                llm_mode,
                api_key,
                niterations,
                verbose=True,
                session=session,
            )
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted! Progress saved. Use --resume")
            break
        except Exception as e:
            print(f"❌ Test failed: {e}")
            continue

    results = session.load_all_results()

    # Generate and save summary
    summary = generate_summary(results, test_cases, llm_mode, niterations)
    session.save_summary(summary)

    print_results_table(results, test_cases)
    return results


# ============================================================================
# SUMMARY GENERATION
# ============================================================================


def generate_summary(
    results: Dict[str, Dict], test_cases: Dict, llm_mode: str, niterations: int
) -> Dict:
    """Generate comprehensive summary with statistics."""

    summary = {
        "total_tests": len(results),
        "passed": 0,
        "failed": 0,
        "by_domain": defaultdict(
            lambda: {"passed": 0, "failed": 0, "extrapolation": 0}
        ),
        "by_difficulty": defaultdict(lambda: {"passed": 0, "failed": 0}),
        "extrapolation_results": [],
        "detailed_results": [],
        "configuration": {
            "mode": llm_mode,
            "iterations": niterations,
        },
        "total_time": 0.0,
    }

    for test_name, result in results.items():
        metadata = result.get("_metadata", {})
        passed = metadata.get("passed", False)

        if passed:
            summary["passed"] += 1
        else:
            summary["failed"] += 1

        # By domain
        domain = result.get("domain", "unknown")
        if passed:
            summary["by_domain"][domain]["passed"] += 1
        else:
            summary["by_domain"][domain]["failed"] += 1

        # Track extrapolation
        if result.get("extrapolation_test"):
            summary["by_domain"][domain]["extrapolation"] += 1
            summary["extrapolation_results"].append(
                {
                    "test_name": test_name,
                    "domain": domain,
                    "passed": passed,
                    "r2": result.get("r2_score", 0.0),
                }
            )

        # By difficulty
        difficulty = result.get("difficulty", "unknown")
        if passed:
            summary["by_difficulty"][difficulty]["passed"] += 1
        else:
            summary["by_difficulty"][difficulty]["failed"] += 1

        # Timing
        summary["total_time"] += result.get("timing", {}).get("total", 0.0)

        # Detailed results
        discovery = result.get("discovery", {})
        validation = result.get("validation", {})

        detailed_entry = {
            "test_name": test_name,
            "domain": domain,
            "difficulty": difficulty,
            "r2": discovery.get("r2_score", 0.0),
            "validation_score": validation.get("total_score", 0.0),
            "time": result.get("timing", {}).get("total", 0.0),
            "passed": passed,
            "extrapolation": result.get("extrapolation_test", False),
            "expression": discovery.get("expression", "N/A"),
            "ground_truth": result.get("ground_truth", "N/A"),
            "error": result.get("error"),
        }

        summary["detailed_results"].append(detailed_entry)

    # Sort by domain, then test name
    summary["detailed_results"].sort(key=lambda x: (x["domain"], x["test_name"]))

    return summary


def print_results_table(results: Dict[str, Dict], test_cases: Dict):
    """Print detailed results table."""
    print(f"\n{'='*120}")
    print(f"LLM-GUIDED DISCOVERY - DEFI RESULTS".center(120))
    print(f"{'='*120}")
    print(
        f"{'Test Name':<40} | {'R²':>6} | {'Val':>5} | {'Time':>6} | {'Status':>6} | {'Observations':<40}"
    )
    print(f"{'-'*120}")

    sorted_tests = sorted(
        results.items(),
        key=lambda x: (test_cases.get(x[0], {}).get("domain", ""), x[0]),
    )

    current_domain = None
    for test_name, result in sorted_tests:
        domain = test_cases.get(test_name, {}).get("domain", "unknown")
        if domain != current_domain:
            if current_domain:
                print()
            print(f"{'─'*120}")
            print(f"{domain.upper()}")
            print(f"{'─'*120}")
            current_domain = domain

        test_name_short = test_name[:38]
        r2 = result.get("r2_score", 0.0)
        val = result.get("validation_score", 0.0)
        time_taken = result.get("timing", {}).get("total", 0.0)
        passed = result.get("_metadata", {}).get("passed", False)

        status = "✅ PASS" if passed else "❌ FAIL"

        # Generate observations
        observations = []
        if result.get("extrapolation_test"):
            observations.append("Extrap")
        if r2 >= 0.99:
            observations.append("Excellent fit")
        elif r2 >= 0.95:
            observations.append("Good fit")
        elif r2 >= 0.90:
            observations.append("Acceptable")
        else:
            observations.append("Poor fit")

        if val >= 85:
            observations.append("Strong val")
        elif val >= 70:
            observations.append("OK val")
        elif val >= 50:
            observations.append("Weak val")
        else:
            observations.append("Failed val")

        if result.get("error"):
            observations.append("ERROR")

        obs_str = ", ".join(observations)[:38]

        print(
            f"{test_name_short:<40} | {r2:>6.4f} | {val:>5.1f} | {time_taken:>5.1f}s | {status:>6} | {obs_str:<40}"
        )

    print(f"{'='*120}")

    # Summary statistics
    total = len(results)
    passed = sum(
        1 for r in results.values() if r.get("_metadata", {}).get("passed", False)
    )

    if total > 0:
        r2_values = [
            r.get("r2_score", 0) for r in results.values() if not r.get("error")
        ]
        val_values = [
            r.get("validation_score", 0) for r in results.values() if not r.get("error")
        ]

        print(f"\nSUMMARY: {passed}/{total} passed ({passed/total*100:.1f}%)")

        if r2_values:
            print(f"\n📈 R² Statistics:")
            print(f"   Mean: {np.mean(r2_values):.4f}")
            print(f"   Median: {np.median(r2_values):.4f}")
            print(f"   Std Dev: {np.std(r2_values):.4f}")
            print(f"   Min: {np.min(r2_values):.4f}")
            print(f"   Max: {np.max(r2_values):.4f}")

        if val_values:
            print(f"\n📊 Validation Statistics:")
            print(f"   Mean: {np.mean(val_values):.1f}/100")
            print(f"   Median: {np.median(val_values):.1f}/100")
            print(f"   Std Dev: {np.std(val_values):.1f}")
            print(f"   Min: {np.min(val_values):.1f}/100")
            print(f"   Max: {np.max(val_values):.1f}/100")

    print(f"{'='*120}\n")


# ============================================================================
# MAIN CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LLM-Guided Symbolic Discovery for DeFi v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all DeFi tests with hybrid mode
  python llm_guided_symbolic_discovery_defi.py --batch --mode hybrid

  # Run single test (impermanent loss)
  python llm_guided_symbolic_discovery_defi.py --test impermanent_loss --mode hybrid

  # Resume interrupted run
  python llm_guided_symbolic_discovery_defi.py --batch --resume

  # Custom iterations
  python llm_guided_symbolic_discovery_defi.py --batch --iterations 100

  # Specific domain
  python llm_guided_symbolic_discovery_defi.py --domain amm --batch
        """,
    )

    parser.add_argument("--batch", action="store_true", help="Run all tests")
    parser.add_argument("--test", type=str, help="Run single test by name")
    parser.add_argument(
        "--domain",
        type=str,
        choices=[
            "amm",
            "risk_var",
            "liquidity",
            "expected_shortfall",
            "liquidation",
            "staking",
        ],
        help="Run all tests in specific domain",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["none", "seed", "hybrid", "fallback"],
        help="LLM mode (default: hybrid)",
    )
    parser.add_argument("--api-key", type=str, help="Anthropic API key")
    parser.add_argument(
        "--niterations", type=int, default=50, help="PySR iterations (default: 50)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--list", action="store_true", help="List all tests")

    args = parser.parse_args()

    # Check dependencies
    if not HAS_INTEGRATED_ENGINE:
        print("❌ Error: SymbolicEngineWithLLM not available")
        return 1

    # API key handling
    api_key = args.api_key
    if not api_key and args.mode != "none":
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY not found")
            print("\nSetup options:")
            print("  1. export ANTHROPIC_API_KEY=your_key")
            print("  2. echo 'ANTHROPIC_API_KEY=your_key' > .env")
            print("  3. --api-key YOUR_KEY")
            return 1

    # Load DeFi protocol
    print(f"\n📄 Loading DeFi Protocol v3.0...")
    protocol = DeFiExperimentProtocolExtended()

    # Show statistics
    stats = protocol.get_protocol_statistics()
    print(f"\n{'='*80}")
    print(f"DEFI PROTOCOL STATISTICS".center(80))
    print(f"{'='*80}")
    print(f"Total tests: {stats['total_tests']}")
    print(f"Extrapolation tests: {stats['extrapolation_tests']}")
    print(f"Domains: {len(stats['domains'])}")
    print(
        f"Difficulty: Easy: {stats['difficulty']['easy']} | "
        f"Medium: {stats['difficulty']['medium']} | "
        f"Hard: {stats['difficulty']['hard']}"
    )
    print(f"{'='*80}")

    # Convert to test cases
    domains = [args.domain] if args.domain else None
    test_cases = convert_defi_protocol_to_test_cases(protocol, domains)

    if not test_cases:
        print("\n❌ No test cases loaded")
        return 1

    # Handle --list
    if args.list:
        print(f"\n{'='*80}")
        print(f"AVAILABLE DEFI TEST CASES".center(80))
        print(f"{'='*80}")
        for domain in protocol.get_all_domains():
            domain_tests = [
                name for name, tc in test_cases.items() if tc["domain"] == domain
            ]
            if domain_tests:
                print(f"\n{domain.upper()} ({len(domain_tests)} tests):")
                for name in sorted(domain_tests):
                    tc = test_cases[name]
                    diff = tc.get("difficulty", "?")
                    extrap = " 🚀" if tc.get("extrapolation_test") else ""
                    print(f"  [{diff:6s}] {name}{extrap}")
        print(f"\n{'='*80}")
        return 0

    # Run tests
    verbose = not args.quiet

    try:
        if args.test:
            # Single test
            result = run_single_test_by_name(
                args.test,
                test_cases,
                llm_mode=args.mode,
                api_key=api_key,
                niterations=args.niterations,
            )

            if result and result.get("success"):
                print(f"\n✅ Test passed!")
                return 0
            else:
                print(f"\n❌ Test failed!")
                return 1

        elif args.batch:
            # Batch mode
            run_defi_suite(
                test_cases=test_cases,
                llm_mode=args.mode,
                api_key=api_key,
                niterations=args.niterations,
                resume=args.resume,
            )
        else:
            parser.print_help()
            return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# Run all DeFi tests
python llm_guided_symbolic_discovery_defi.py --batch --mode hybrid

# Single test (e.g., impermanent loss - similar to Nernst)
python llm_guided_symbolic_discovery_defi.py --test impermanent_loss --mode hybrid

# Resume interrupted run
python llm_guided_symbolic_discovery_defi.py --batch --resume

# Specific domain
python llm_guided_symbolic_discovery_defi.py --domain amm --batch

# List all tests
python llm_guided_symbolic_discovery_defi.py --list
"""
