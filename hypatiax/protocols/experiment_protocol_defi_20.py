"""
experiment_protocol_defi_extended.py - 20 TEST CASES COMPLETE
==============================================================
Extended DeFi protocol with 20 comprehensive test cases across 6 domains

✅ 20 test cases total
✅ 6 extrapolation tests marked
✅ Full metadata alignment with Protocol B
✅ All domains complete

Domains:
- AMM (4 tests)
- Risk/VaR (4 tests)
- Liquidity (4 tests)
- Expected Shortfall (4 tests)
- Liquidation (4 tests)
- Staking (3 tests) ⭐ NEW

Author: HypatiaX Team
Version: 3.0 Extended - COMPLETE
Date: 2026-01-06
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import os
from pathlib import Path
from datetime import datetime, timezone

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42  # Fixed for reproducibility — matches paper
import random
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch; torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
except ImportError:
    pass
PYSR_KWARGS = {'random_state': SEED, 'deterministic': True}


def save_results(results: dict, out_base: str = 'data/results/') -> str:
    """Save campaign results to timestamped JSON — matches repo structure."""
    ts  = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(out_base) / 'hybrid_pysr' / 'defi'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'protocol_defi_{ts}.json'
    payload = {
        'campaign':  'Campaign 4 – DeFi Suite (23 tests)',
        'seed':      SEED,
        'timestamp': ts,
        'results':   results,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f'[saved] {path}')
    return str(path)


class DeFiExperimentProtocolExtended:
    """Extended DeFi experiment protocol with 20 test cases"""

    def __init__(self):
        self.domains = {
            "amm": self._generate_amm_tests,
            "risk_var": self._generate_var_tests,
            "liquidity": self._generate_liquidity_tests,
            "expected_shortfall": self._generate_es_tests,
            "liquidation": self._generate_liquidation_tests,
            "staking": self._generate_staking_tests,
        }

    def get_all_domains(self) -> List[str]:
        """Get all available domains"""
        return list(self.domains.keys())

    def load_test_data(self, domain: str, num_samples: int = 100) -> List[Tuple]:
        """Load test data for a domain"""
        if domain not in self.domains:
            raise ValueError(f"Unknown domain: {domain}")
        return self.domains[domain](num_samples)

    # ========================================================================
    # AMM DOMAIN - 4 TESTS
    # ========================================================================

    def _generate_amm_tests(self, n: int) -> List[Tuple]:
        """AMM test cases - 4 tests"""
        tests = []

        # Test 1: Impermanent loss (EXTRAPOLATION TEST)
        np.random.seed(42)
        price_ratio = np.concatenate(
            [np.linspace(0.5, 1.5, n // 2), np.linspace(1.6, 2.5, n // 2)]
        )
        np.random.shuffle(price_ratio)
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1

        tests.append(
            (
                "Impermanent loss in constant product AMM (Uniswap V2)",
                price_ratio.reshape(-1, 1),
                il,
                ["price_ratio"],
                {
                    "equation_name": "impermanent_loss",
                    "domain": "amm",
                    "difficulty": "hard",
                    "formula_type": "rational_sqrt",
                    "extrapolation_test": True,
                    "ground_truth": "2*sqrt(r)/(1+r) - 1",
                    "protocol": "Protocol_DeFi",
                    "train_range": "0.5-1.5",
                    "test_range": "1.6-2.5",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 2: IL Percentage
        price_ratio = np.linspace(0.5, 2.0, n)
        il_pct = (2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1) * 100
        tests.append(
            (
                "Impermanent loss percentage in AMM",
                price_ratio.reshape(-1, 1),
                il_pct,
                ["price_ratio"],
                {
                    "equation_name": "il_percentage",
                    "domain": "amm",
                    "difficulty": "medium",
                    "formula_type": "rational_sqrt",
                    "ground_truth": "(2*sqrt(r)/(1+r) - 1) * 100",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 3: Constant product
        reserve_x = np.linspace(100, 10000, n)
        invariant_k = np.random.uniform(1e6, 1e8, n)
        reserve_y = invariant_k / reserve_x
        tests.append(
            (
                "Constant product formula: reserve Y given reserve X and invariant k",
                np.column_stack([reserve_x, invariant_k]),
                reserve_y,
                ["reserve_x", "invariant_k"],
                {
                    "equation_name": "constant_product",
                    "domain": "amm",
                    "difficulty": "easy",
                    "formula_type": "rational",
                    "ground_truth": "k / x",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 4: Price impact
        reserve_x = np.linspace(10000, 100000, n)
        swap_amount = reserve_x * np.random.uniform(0.001, 0.1, n)
        price_impact = swap_amount / (reserve_x + swap_amount)
        tests.append(
            (
                "Price impact of swap in constant product AMM",
                np.column_stack([reserve_x, swap_amount]),
                price_impact,
                ["reserve_x", "swap_amount"],
                {
                    "equation_name": "price_impact",
                    "domain": "amm",
                    "difficulty": "easy",
                    "formula_type": "rational",
                    "ground_truth": "dx / (x + dx)",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        return tests

    # ========================================================================
    # RISK VAR DOMAIN - 4 TESTS
    # ========================================================================

    def _generate_var_tests(self, n: int) -> List[Tuple]:
        """Value at Risk test cases - 4 tests"""
        tests = []

        # Test 1: VaR 95% (EXTRAPOLATION TEST)
        np.random.seed(43)
        portfolio_value = np.linspace(10000, 1000000, n)
        daily_vol = np.concatenate(
            [np.linspace(0.01, 0.03, n // 2), np.linspace(0.035, 0.05, n // 2)]
        )
        np.random.shuffle(daily_vol)
        var_95 = portfolio_value * daily_vol * 1.645

        tests.append(
            (
                "Parametric Value at Risk at 95% confidence (1-day)",
                np.column_stack([portfolio_value, daily_vol]),
                var_95,
                ["portfolio_value", "daily_volatility"],
                {
                    "equation_name": "var_95",
                    "domain": "risk_var",
                    "difficulty": "medium",
                    "formula_type": "linear_multiplicative",
                    "extrapolation_test": True,
                    "ground_truth": "portfolio_value * volatility * 1.645",
                    "protocol": "Protocol_DeFi",
                    "train_range": "vol 0.01-0.03",
                    "test_range": "vol 0.035-0.05",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 2: VaR 99%
        portfolio_value = np.linspace(10000, 1000000, n)
        daily_vol = np.linspace(0.01, 0.05, n)
        var_99 = portfolio_value * daily_vol * 2.326
        tests.append(
            (
                "Parametric Value at Risk at 99% confidence (1-day)",
                np.column_stack([portfolio_value, daily_vol]),
                var_99,
                ["portfolio_value", "daily_volatility"],
                {
                    "equation_name": "var_99",
                    "domain": "risk_var",
                    "difficulty": "medium",
                    "formula_type": "linear_multiplicative",
                    "ground_truth": "portfolio_value * volatility * 2.326",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 3: Multi-day VaR
        var_1day = np.linspace(1000, 100000, n)
        time_horizon = np.random.choice([5, 10, 21, 30], n).astype(float)
        var_multiday = var_1day * np.sqrt(time_horizon)
        tests.append(
            (
                "Multi-day Value at Risk using square root of time rule",
                np.column_stack([var_1day, time_horizon]),
                var_multiday,
                ["var_1day", "time_horizon_days"],
                {
                    "equation_name": "var_multiday",
                    "domain": "risk_var",
                    "difficulty": "easy",
                    "formula_type": "power_law",
                    "ground_truth": "var_1day * sqrt(days)",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 4: Portfolio VaR
        var_asset1 = np.linspace(5000, 50000, n)
        var_asset2 = np.linspace(3000, 30000, n)
        correlation = np.linspace(-0.5, 0.9, n)
        var_portfolio = np.sqrt(
            var_asset1**2 + var_asset2**2 + 2 * correlation * var_asset1 * var_asset2
        )
        tests.append(
            (
                "Portfolio VaR for two correlated assets",
                np.column_stack([var_asset1, var_asset2, correlation]),
                var_portfolio,
                ["var_asset1", "var_asset2", "correlation"],
                {
                    "equation_name": "portfolio_var",
                    "domain": "risk_var",
                    "difficulty": "hard",
                    "formula_type": "power_law_complex",
                    "ground_truth": "sqrt(var1^2 + var2^2 + 2*rho*var1*var2)",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        return tests

    # ========================================================================
    # LIQUIDITY DOMAIN - 4 TESTS
    # ========================================================================

    def _generate_liquidity_tests(self, n: int) -> List[Tuple]:
        """Liquidity test cases - 4 tests"""
        tests = []

        # Test 1: Kelly Criterion (EXTRAPOLATION TEST)
        np.random.seed(44)
        expected_apy = np.concatenate(
            [np.linspace(0.05, 0.18, n // 2), np.linspace(0.22, 0.30, n // 2)]
        )
        np.random.shuffle(expected_apy)
        il_risk = np.linspace(0.05, 0.25, n)
        f_star = np.minimum(expected_apy / (2.0 * il_risk**2), 1.0)

        tests.append(
            (
                "Optimal LP position size using risk-adjusted Kelly criterion",
                np.column_stack([expected_apy, il_risk]),
                f_star,
                ["expected_fee_apy", "il_risk"],
                {
                    "equation_name": "kelly_criterion",
                    "domain": "liquidity",
                    "difficulty": "hard",
                    "formula_type": "rational_capped",
                    "extrapolation_test": True,
                    "ground_truth": "min(μ / (2 * σ²), 1.0)",
                    "protocol": "Protocol_DeFi",
                    "train_range": "apy 0.05-0.18",
                    "test_range": "apy 0.22-0.30",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 2: LP fee earnings
        liquidity_provided = np.linspace(10000, 1000000, n)
        pool_liquidity = liquidity_provided * np.random.uniform(10, 100, n)
        total_fees = np.random.uniform(1000, 50000, n)
        fee_share = (liquidity_provided / pool_liquidity) * total_fees
        tests.append(
            (
                "LP fee earnings based on liquidity share",
                np.column_stack([liquidity_provided, pool_liquidity, total_fees]),
                fee_share,
                ["liquidity_provided", "pool_liquidity", "total_fees"],
                {
                    "equation_name": "lp_fee_share",
                    "domain": "liquidity",
                    "difficulty": "easy",
                    "formula_type": "rational_multiplicative",
                    "ground_truth": "(liq_provided / pool_liq) * total_fees",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 3: Capital efficiency
        price_lower = np.linspace(1800, 2000, n)
        price_upper = np.linspace(2200, 2400, n)
        price_current = np.linspace(1900, 2300, n)
        efficiency = price_upper / (price_upper - price_lower)
        tests.append(
            (
                "Capital efficiency multiplier for concentrated liquidity position",
                np.column_stack([price_lower, price_upper, price_current]),
                efficiency,
                ["price_lower", "price_upper", "price_current"],
                {
                    "equation_name": "capital_efficiency",
                    "domain": "liquidity",
                    "difficulty": "medium",
                    "formula_type": "rational",
                    "ground_truth": "P_upper / (P_upper - P_lower)",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 4: APY from APR
        apr = np.linspace(0.05, 0.50, n)
        apy = (1 + apr / 365) ** 365 - 1
        tests.append(
            (
                "APY calculation from APR with daily compounding",
                apr.reshape(-1, 1),
                apy,
                ["apr"],
                {
                    "equation_name": "apy_from_apr",
                    "domain": "liquidity",
                    "difficulty": "medium",
                    "formula_type": "exponential",
                    "ground_truth": "(1 + apr/365)^365 - 1",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        return tests

    # ========================================================================
    # EXPECTED SHORTFALL DOMAIN - 4 TESTS
    # ========================================================================

    def _generate_es_tests(self, n: int) -> List[Tuple]:
        """Expected Shortfall test cases - 4 tests"""
        tests = []

        # Test 1: ES 95% (EXTRAPOLATION TEST)
        np.random.seed(45)
        portfolio_value = np.linspace(10000, 1000000, n)
        daily_vol = np.concatenate(
            [np.linspace(0.01, 0.03, n // 2), np.linspace(0.035, 0.05, n // 2)]
        )
        np.random.shuffle(daily_vol)
        es_95 = portfolio_value * daily_vol * 2.063

        tests.append(
            (
                "Expected Shortfall (CVaR) at 95% confidence for normal returns",
                np.column_stack([portfolio_value, daily_vol]),
                es_95,
                ["portfolio_value", "daily_volatility"],
                {
                    "equation_name": "es_95",
                    "domain": "expected_shortfall",
                    "difficulty": "medium",
                    "formula_type": "linear_multiplicative",
                    "extrapolation_test": True,
                    "ground_truth": "portfolio * volatility * 2.063",
                    "protocol": "Protocol_DeFi",
                    "train_range": "vol 0.01-0.03",
                    "test_range": "vol 0.035-0.05",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 2: ES 99%
        portfolio_value = np.linspace(10000, 1000000, n)
        daily_vol = np.linspace(0.01, 0.05, n)
        es_99 = portfolio_value * daily_vol * 2.665
        tests.append(
            (
                "Expected Shortfall (CVaR) at 99% confidence for normal returns",
                np.column_stack([portfolio_value, daily_vol]),
                es_99,
                ["portfolio_value", "daily_volatility"],
                {
                    "equation_name": "es_99",
                    "domain": "expected_shortfall",
                    "difficulty": "medium",
                    "formula_type": "linear_multiplicative",
                    "ground_truth": "portfolio * volatility * 2.665",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 3: ES from VaR
        var_95 = np.linspace(5000, 100000, n)
        es_from_var = var_95 * 1.254
        tests.append(
            (
                "Expected Shortfall from VaR using tail risk multiplier",
                var_95.reshape(-1, 1),
                es_from_var,
                ["var_95"],
                {
                    "equation_name": "es_from_var",
                    "domain": "expected_shortfall",
                    "difficulty": "easy",
                    "formula_type": "linear_multiplicative",
                    "ground_truth": "var_95 * 1.254",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 4: Portfolio ES
        pos1_es = np.linspace(10000, 100000, n)
        pos2_es = np.linspace(5000, 50000, n)
        correlation = np.linspace(-0.3, 0.8, n)
        portfolio_es = pos1_es + pos2_es + correlation * np.sqrt(pos1_es * pos2_es)
        tests.append(
            (
                "Portfolio Expected Shortfall for correlated positions",
                np.column_stack([pos1_es, pos2_es, correlation]),
                portfolio_es,
                ["position1_es", "position2_es", "correlation"],
                {
                    "equation_name": "portfolio_es",
                    "domain": "expected_shortfall",
                    "difficulty": "hard",
                    "formula_type": "complex_additive",
                    "ground_truth": "ES1 + ES2 + ρ*sqrt(ES1*ES2)",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        return tests

    # ========================================================================
    # LIQUIDATION DOMAIN - 4 TESTS
    # ========================================================================

    def _generate_liquidation_tests(self, n: int) -> List[Tuple]:
        """Liquidation test cases - 4 tests"""
        tests = []
        maintenance_margin = 0.8

        # Test 1: Liquidation LONG (EXTRAPOLATION TEST)
        np.random.seed(46)
        entry_price = np.linspace(30000, 50000, n)
        leverage = np.concatenate(
            [np.linspace(2, 5, n // 2), np.linspace(7, 10, n // 2)]
        )
        np.random.shuffle(leverage)
        liq_price_long = entry_price * (1 - 1 / (leverage * maintenance_margin))

        tests.append(
            (
                "Liquidation price for leveraged long position",
                np.column_stack([entry_price, leverage]),
                liq_price_long,
                ["entry_price", "leverage"],
                {
                    "equation_name": "liquidation_long",
                    "domain": "liquidation",
                    "difficulty": "hard",
                    "formula_type": "complex_rational",
                    "extrapolation_test": True,
                    "ground_truth": "entry_price * (1 - 1/(leverage * 0.8))",
                    "protocol": "Protocol_DeFi",
                    "train_range": "leverage 2-5",
                    "test_range": "leverage 7-10",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 2: Liquidation SHORT
        entry_price = np.linspace(30000, 50000, n)
        leverage = np.linspace(2, 10, n)
        liq_price_short = entry_price * (1 + 1 / (leverage * maintenance_margin))
        tests.append(
            (
                "Liquidation price for leveraged short position",
                np.column_stack([entry_price, leverage]),
                liq_price_short,
                ["entry_price", "leverage"],
                {
                    "equation_name": "liquidation_short",
                    "domain": "liquidation",
                    "difficulty": "hard",
                    "formula_type": "complex_rational",
                    "ground_truth": "entry_price * (1 + 1/(leverage * 0.8))",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 3: Max leverage
        entry_price = np.linspace(30000, 50000, n)
        acceptable_loss_pct = np.linspace(0.05, 0.20, n)
        max_leverage = 1 / (acceptable_loss_pct * maintenance_margin)
        tests.append(
            (
                "Maximum safe leverage for given acceptable loss tolerance",
                np.column_stack([entry_price, acceptable_loss_pct]),
                max_leverage,
                ["entry_price", "acceptable_loss_pct"],
                {
                    "equation_name": "max_leverage",
                    "domain": "liquidation",
                    "difficulty": "medium",
                    "formula_type": "rational",
                    "ground_truth": "1 / (loss_pct * 0.8)",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 4: Required collateral
        position_size = np.linspace(10000, 1000000, n)
        leverage = np.linspace(2, 10, n)
        collateral = position_size / leverage
        tests.append(
            (
                "Required collateral for leveraged position",
                np.column_stack([position_size, leverage]),
                collateral,
                ["position_size", "leverage"],
                {
                    "equation_name": "required_collateral",
                    "domain": "liquidation",
                    "difficulty": "easy",
                    "formula_type": "rational",
                    "ground_truth": "position_size / leverage",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        return tests

    # ========================================================================
    # STAKING DOMAIN - 3 TESTS ⭐ NEW!
    # ========================================================================

    def _generate_staking_tests(self, n: int) -> List[Tuple]:
        """Staking reward test cases - 3 tests"""
        tests = []

        # Test 1: Simple staking rewards
        stake_amount = np.linspace(1000, 100000, n)
        apr = np.linspace(0.05, 0.15, n)
        time_days = np.random.uniform(30, 365, n)
        rewards = stake_amount * apr * (time_days / 365)
        tests.append(
            (
                "Simple staking rewards calculation",
                np.column_stack([stake_amount, apr, time_days]),
                rewards,
                ["stake_amount", "apr", "time_days"],
                {
                    "equation_name": "staking_rewards",
                    "domain": "staking",
                    "difficulty": "easy",
                    "formula_type": "linear_multiplicative",
                    "ground_truth": "stake * apr * (days/365)",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 2: Dilution from emissions
        user_stake = np.linspace(1000, 50000, n)
        total_staked = user_stake * np.random.uniform(100, 1000, n)
        new_emissions = np.random.uniform(1000, 10000, n)
        share_after = user_stake / (total_staked + new_emissions)
        tests.append(
            (
                "Staker's share after token emissions dilution",
                np.column_stack([user_stake, total_staked, new_emissions]),
                share_after,
                ["user_stake", "total_staked", "new_emissions"],
                {
                    "equation_name": "staking_dilution",
                    "domain": "staking",
                    "difficulty": "medium",
                    "formula_type": "rational",
                    "ground_truth": "user_stake / (total_staked + new_emissions)",
                    "protocol": "Protocol_DeFi",
                    "noise_level": 0.0,
                },
            )
        )

        # Test 3: Compound staking (EXTRAPOLATION TEST)
        np.random.seed(47)
        principal = np.linspace(10000, 100000, n)
        apr = np.concatenate(
            [np.linspace(0.08, 0.12, n // 2), np.linspace(0.15, 0.20, n // 2)]
        )
        np.random.shuffle(apr)
        periods = np.random.choice([12, 52, 365], n).astype(float)
        time_years = np.random.uniform(0.5, 2.0, n)
        final_amount = principal * (1 + apr / periods) ** (periods * time_years)

        tests.append(
            (
                "Compound staking rewards with auto-restaking",
                np.column_stack([principal, apr, periods, time_years]),
                final_amount,
                ["principal", "apr", "periods_per_year", "time_years"],
                {
                    "equation_name": "compound_staking",
                    "domain": "staking",
                    "difficulty": "hard",
                    "formula_type": "exponential",
                    "extrapolation_test": True,
                    "ground_truth": "P * (1 + r/n)^(n*t)",
                    "protocol": "Protocol_DeFi",
                    "train_range": "apr 0.08-0.12",
                    "test_range": "apr 0.15-0.20",
                    "noise_level": 0.0,
                },
            )
        )

        return tests

    # ========================================================================
    # UTILITIES & REPORTING
    # ========================================================================

    @staticmethod
    def get_domain_description(domain: str) -> str:
        descriptions = {
            "amm": "AMM - impermanent loss, price impact, constant product",
            "risk_var": "VaR - parametric risk measures, portfolio correlations",
            "liquidity": "Liquidity - Kelly criterion, fees, capital efficiency",
            "expected_shortfall": "Expected Shortfall (CVaR) - tail risk measures",
            "liquidation": "Liquidation - leverage mechanics, collateral requirements",
            "staking": "Staking - rewards, dilution, compound returns",
        }
        return descriptions.get(domain, "Unknown domain")

    @staticmethod
    def get_protocol_statistics() -> Dict:
        return {
            "version": "3.0 Extended",
            "total_tests": 20,
            "domains": {
                "amm": 4,
                "risk_var": 4,
                "liquidity": 4,
                "expected_shortfall": 4,
                "liquidation": 4,
                "staking": 3,
            },
            "difficulty": {"easy": 7, "medium": 8, "hard": 5},
            "extrapolation_tests": 6,
        }

    def generate_experiment_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive experiment report"""
        successful = [
            r for r in results if r.get("evaluation", {}).get("success", False)
        ]
        r2_scores = [
            r["evaluation"]["r2"] for r in successful if "r2" in r.get("evaluation", {})
        ]

        by_domain = {}
        for r in results:
            domain = r.get("domain", r.get("metadata", {}).get("domain", "unknown"))
            if domain not in by_domain:
                by_domain[domain] = {"total": 0, "successful": 0, "r2_scores": []}
            by_domain[domain]["total"] += 1
            if r.get("evaluation", {}).get("success"):
                by_domain[domain]["successful"] += 1
                if "r2" in r.get("evaluation", {}):
                    by_domain[domain]["r2_scores"].append(r["evaluation"]["r2"])

        for domain in by_domain:
            scores = by_domain[domain]["r2_scores"]
            by_domain[domain]["mean_r2"] = np.mean(scores) if scores else None
            by_domain[domain]["median_r2"] = np.median(scores) if scores else None

        extrap_tests = [
            {
                "description": r.get("description", "N/A"),
                "domain": r.get("domain", "N/A"),
                "r2": r.get("evaluation", {}).get("r2"),
                "success": r.get("evaluation", {}).get("success", False),
            }
            for r in results
            if r.get("metadata", {}).get("extrapolation_test", False)
        ]

        return {
            "overall": {
                "total_cases": len(results),
                "successful": len(successful),
                "success_rate": len(successful) / len(results) if results else 0,
                "mean_r2": np.mean(r2_scores) if r2_scores else None,
                "median_r2": np.median(r2_scores) if r2_scores else None,
                "std_r2": np.std(r2_scores) if r2_scores else None,
            },
            "by_domain": by_domain,
            "extrapolation_tests": extrap_tests,
        }


if __name__ == "__main__":
    protocol = DeFiExperimentProtocolExtended()

    print("=" * 80)
    print("DEFI PROTOCOL v3.0 EXTENDED: 20 COMPLETE TEST CASES".center(80))
    print("=" * 80)
    print(f"Version: 3.0 Extended | Date: 2026-01-06")
    print("=" * 80)

    total_tests = 0
    extrap_count = 0

    for domain in protocol.get_all_domains():
        tests = protocol.load_test_data(domain, num_samples=10)
        total_tests += len(tests)

        print(f"\n{domain.upper()}: {len(tests)} tests")
        print(f"Description: {protocol.get_domain_description(domain)}")
        print("-" * 80)

        for desc, X, y, vars, meta in tests:
            if meta.get("extrapolation_test"):
                extrap_count += 1
            extrap = "🚀 EXTRAP" if meta.get("extrapolation_test") else "        "
            print(
                f"  {extrap} | {meta['equation_name']:25s} | {meta['difficulty']:6s} | {meta['ground_truth']}"
            )

    stats = protocol.get_protocol_statistics()
    print(f"\n{'=' * 80}")
    print(f"SUMMARY".center(80))
    print(f"{'=' * 80}")
    print(f"Total tests: {stats['total_tests']}")
    print(f"Extrapolation tests: {stats['extrapolation_tests']}")
    print(f"Domains: {len(stats['domains'])}")
    print(f"Difficulty breakdown:")
    print(f"  Easy: {stats['difficulty']['easy']}")
    print(f"  Medium: {stats['difficulty']['medium']}")
    print(f"  Hard: {stats['difficulty']['hard']}")
    print(f"{'=' * 80}")

    # ── Save protocol definition to JSON ────────────────────────────────────
    protocol_export = {
        'statistics':   stats,
        'domains':      {
            d: [
                {
                    'description': desc,
                    'variables':   vars_,
                    'metadata':    meta,
                }
                for desc, _X, _y, vars_, meta in protocol.load_test_data(d, num_samples=10)
            ]
            for d in protocol.get_all_domains()
        },
        'total_extrapolation_tests': extrap_count,
    }
    save_results(protocol_export)
