"""
experiment_protocol_defi.py - ENHANCED
Fixes for Kelly criterion, better test case generation, improved metadata
"""

import os as _os
import pathlib as _pathlib
import sys as _sys

# ── sys.path bootstrap ────────────────────────────────────────────────────
# Ensures hypatiax.* imports resolve whether this file is run directly
# or imported by run_all_checkpoint.py.
_PROTO_DIR  = _pathlib.Path(__file__).resolve().parent
_REPO_ROOT  = _pathlib.Path(_os.environ.get("REPRO_ROOT", str(_PROTO_DIR.parent)))
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "hypatiax")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _pathlib, _sys, _PROTO_DIR, _REPO_ROOT, _p


import numpy as np


class DeFiExperimentProtocol:
    """Enhanced DeFi experiment protocol with fixed formulas"""

    def __init__(self):
        self.domains = {
            # ── canonical names ──────────────────────────────────────────────
            "amm": self._generate_amm_tests,
            "risk_var": self._generate_var_tests,
            "liquidity": self._generate_liquidity_tests,
            "expected_shortfall": self._generate_es_tests,
            "liquidation": self._generate_liquidation_tests,
            # ── aliases used by test_enhanced_defi_extrapolation.py ──────────
            # "risk"        → combined VaR + ES tests
            "risk": self._generate_risk_alias_tests,
            # "lending"     → collateral-ratio / LTV tests
            "lending": self._generate_lending_tests,
            # "staking"     → APY / compounding tests
            "staking": self._generate_staking_tests,
            # "trading"     → leverage / liquidation-price tests
            "trading": self._generate_trading_tests,
            # "derivatives" → options / Black-Scholes tests
            "derivatives": self._generate_derivatives_tests,
        }

    @staticmethod
    def _apply_shard_ids(domains: list[str]) -> list[str]:
        """Filter *domains* to those listed in the SHARD_IDS environment variable.

        SHARD_IDS: space- or comma-separated domain names
        (e.g. ``"amm risk_var"`` or ``"amm,risk_var"``).
        Falls back to the full domain list when the variable is unset or empty
        (local / Colab runs).

        This mirrors the ``_apply_task_ids_defi`` two-stage pattern in
        hypatiax_defi_benchmark_v3c.py.  The fallback is intentionally SILENT:
        for exp1b, SHARD_IDS contains synthetic checkpoint-tracking IDs such as
        ``"portfolio_seed42"`` which match no domain name — that is correct
        behaviour, not a misconfiguration.  Emitting a RuntimeWarning here
        would be actively misleading, so the fallback is silent.

        For exp1 / suppA, SHARD_IDS = domain keys (e.g. ``"amm risk_var"``),
        which match exactly and produce the correct disjoint domain subset.
        """
        import os

        raw = os.environ.get("SHARD_IDS", "").replace(",", " ").split()
        if not raw:
            return domains           # unset → local / Colab run, no filtering
        allowed = set(raw)
        filtered = [d for d in domains if d in allowed]
        if not filtered:
            # No domain-key match — SHARD_IDS are synthetic checkpoint IDs
            # (exp1b pattern) or genuinely stale.  Silent fallback: actual
            # case filtering is handled by DEFI_TASK_FILTER in run_benchmark().
            return domains
        return filtered

    def get_all_domains(self) -> list[str]:
        """Get all available domains, filtered by SHARD_IDS if set."""
        return self._apply_shard_ids(list(self.domains.keys()))

    def load_test_data(self, domain: str, num_samples: int = 100) -> list[tuple]:
        """Load test data for a domain"""
        if domain not in self.domains:
            raise ValueError(f"Unknown domain: {domain}")
        return self.domains[domain](num_samples)

    # ========================================================================
    # AMM DOMAIN
    # ========================================================================

    def _generate_amm_tests(self, n: int) -> list[tuple]:
        """AMM test cases"""
        tests = []

        # Test 1: Impermanent loss (EXTRAPOLATION TEST)
        np.random.seed(42)
        price_ratio = np.concatenate(
            [
                np.linspace(0.5, 1.5, n // 2),  # Training range
                np.linspace(1.6, 2.5, n // 2),  # Extrapolation range
            ]
        )
        np.random.shuffle(price_ratio)

        # Correct IL formula: 2*sqrt(r)/(1+r) - 1
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1

        tests.append(
            (
                "Impermanent loss in constant product AMM (Uniswap V2)",
                price_ratio.reshape(-1, 1),
                il,
                ["price_ratio"],
                {
                    "domain": "amm",
                    "extrapolation_test": True,
                    "ground_truth": "2*sqrt(r)/(1+r) - 1",
                    "train_range": "0.5-1.5",
                    "test_range": "1.6-2.5",
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
                    "domain": "amm",
                    "ground_truth": "(2*sqrt(r)/(1+r) - 1) * 100",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 3: Constant product - reserve Y
        reserve_x = np.linspace(100, 10000, n)
        invariant_k = np.random.uniform(1e6, 1e8, n)
        reserve_y = invariant_k / reserve_x

        tests.append(
            (
                "Constant product formula: reserve Y given reserve X and invariant k",
                np.column_stack([reserve_x, invariant_k]),
                reserve_y,
                ["reserve_x", "invariant_k"],
                {"domain": "amm", "ground_truth": "k / x", "extrapolation_test": False},
            )
        )

        # Test 4: Price impact
        reserve_x = np.linspace(10000, 100000, n)
        swap_amount = reserve_x * np.random.uniform(0.001, 0.1, n)
        price_impact = swap_amount / (reserve_x + swap_amount)

        tests.append(
            (
                "Constant Product Price Impact of swap in constant product AMM",
                np.column_stack([reserve_x, swap_amount]),
                price_impact,
                ["reserve_x", "swap_amount"],
                {
                    "domain": "amm",
                    "ground_truth": "dx / (x + dx)",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 5: Reserve Ratio
        # FIXED: independent log-uniform sampling so ratio spans many decades
        # and the NN sees genuine variance in the target.
        rng = np.random.default_rng(7)
        reserve_x = np.exp(rng.uniform(np.log(100), np.log(100000), n))
        reserve_y = np.exp(rng.uniform(np.log(100), np.log(100000), n))
        reserve_ratio = reserve_x / reserve_y

        tests.append(
            (
                "Reserve Ratio of token reserves in liquidity pool",
                np.column_stack([reserve_x, reserve_y]),
                reserve_ratio,
                ["reserve_x", "reserve_y"],
                {
                    "domain": "amm",
                    "ground_truth": "reserve_x / reserve_y",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 6: Convexity Adjustment (effective price accounting for slippage)
        # effective_price = spot * (1 + 0.5 * trade_size / pool_depth)
        spot_price = np.linspace(1000, 5000, n)
        trade_size = np.linspace(100, 50000, n)
        pool_depth = np.linspace(100000, 10000000, n)
        convexity_adjusted_price = spot_price * (
            1 + 0.5 * trade_size / (pool_depth + 1e-10)
        )

        tests.append(
            (
                "Convexity Adjustment effective price in AMM with slippage",
                np.column_stack([spot_price, trade_size, pool_depth]),
                convexity_adjusted_price,
                ["spot_price", "trade_size", "pool_depth"],
                {
                    "domain": "amm",
                    "ground_truth": "spot * (1 + 0.5 * trade_size / pool_depth)",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 7: Spot price from AMM reserves
        # FIXED: independent log-uniform sampling so the ratio reserve_b/reserve_a
        # has real variance (was near-constant 1.0 with identical linspace ranges).
        rng7 = np.random.default_rng(13)
        reserve_a = np.exp(rng7.uniform(np.log(100), np.log(50000), n))
        reserve_b = np.exp(rng7.uniform(np.log(100), np.log(50000), n))
        spot_price = reserve_b / reserve_a

        tests.append((
            "Spot price from AMM reserve ratio (token A / token B)",
            np.column_stack([reserve_a, reserve_b]),
            spot_price,
            ["reserve_a", "reserve_b"],
            {"domain": "amm", "ground_truth": "reserve_b / reserve_a",
             "extrapolation_test": False},
        ))

        # Test 8: Output amount from constant-product swap (dy = y*dx/(x+dx))
        reserve_x2 = np.linspace(10000, 500000, n)
        reserve_y2 = np.linspace(10000, 500000, n)
        dx = reserve_x2 * np.random.uniform(0.001, 0.05, n)
        dy = reserve_y2 * dx / (reserve_x2 + dx + 1e-10)

        tests.append((
            "AMM output amount from constant product swap formula",
            np.column_stack([reserve_x2, reserve_y2, dx]),
            dy,
            ["reserve_x", "reserve_y", "dx"],
            {"domain": "amm", "ground_truth": "y*dx/(x+dx)",
             "extrapolation_test": False},
        ))

        # Test 9: LP token share percentage after deposit
        lp_deposit = np.linspace(1000, 100000, n)
        pool_total = lp_deposit * np.random.uniform(5, 50, n)
        lp_share_pct = lp_deposit / (pool_total + 1e-10) * 100

        tests.append((
            "LP share percentage after deposit relative to pool total",
            np.column_stack([lp_deposit, pool_total]),
            lp_share_pct,
            ["lp_deposit", "pool_total"],
            {"domain": "amm", "ground_truth": "deposit / pool_total * 100",
             "extrapolation_test": False},
        ))

        # Test 10: Price slippage percentage  = 100 * dx / (2 * reserve_x + dx)
        reserve_x3 = np.linspace(50000, 2000000, n)
        trade_dx = reserve_x3 * np.random.uniform(0.001, 0.10, n)
        slippage_pct = 100.0 * trade_dx / (2 * reserve_x3 + trade_dx + 1e-10)

        tests.append((
            "Price slippage percentage in AMM swap from trade size",
            np.column_stack([reserve_x3, trade_dx]),
            slippage_pct,
            ["reserve_x", "trade_dx"],
            {"domain": "amm", "ground_truth": "100 * dx / (2*x + dx)",
             "extrapolation_test": False},
        ))

        # Test 11: Arbitrage profit = (external_price - amm_price) * amount / external_price
        amm_price = np.linspace(1000, 5000, n)
        ext_price = amm_price * np.random.uniform(1.01, 1.10, n)
        arb_amount = np.random.uniform(100, 10000, n)
        arb_profit = (ext_price - amm_price) * arb_amount / (ext_price + 1e-10)

        tests.append((
            "AMM arbitrage profit from external price spread and trade size",
            np.column_stack([amm_price, ext_price, arb_amount]),
            arb_profit,
            ["amm_price", "ext_price", "arb_amount"],
            {"domain": "amm",
             "ground_truth": "(ext_price - amm_price) * amount / ext_price",
             "extrapolation_test": False},
        ))

        # Test 12: Uniswap V3 virtual liquidity L = amount / (sqrt(P_upper) - sqrt(P_lower))
        p_lower = np.linspace(1500, 1900, n)
        p_upper = p_lower * np.random.uniform(1.05, 1.30, n)
        token_amount = np.random.uniform(1000, 100000, n)
        v3_liquidity = token_amount / (np.sqrt(p_upper) - np.sqrt(p_lower) + 1e-10)

        tests.append((
            "Uniswap V3 virtual liquidity L for concentrated position",
            np.column_stack([p_lower, p_upper, token_amount]),
            v3_liquidity,
            ["price_lower", "price_upper", "token_amount"],
            {"domain": "amm",
             "ground_truth": "amount / (sqrt(P_upper) - sqrt(P_lower))",
             "extrapolation_test": False},
        ))

        # Test 13: Impermanent loss breakeven fee rate
        # For a given price-ratio move r, the minimum fee rate an LP must earn
        # to break even is exactly the absolute IL:
        #   fee_breakeven(r) = 1 - 2*sqrt(r) / (1 + r)
        # This is smooth and fully explicit — no implicit solving required.
        # Wide price-ratio range ensures the aggressive split (train r<1.86,
        # test r>1.86) stays in the non-degenerate region of the U-shaped curve.
        rng_ilb = np.random.default_rng(17)
        # Independent log-uniform sampling avoids collinearity with other vars
        price_ratio_ilb = np.exp(rng_ilb.uniform(np.log(0.25), np.log(4.0), n))
        fee_breakeven = 1.0 - 2.0 * np.sqrt(price_ratio_ilb) / (1.0 + price_ratio_ilb)

        tests.append((
            "Impermanent loss breakeven fee rate for LP profitability",
            price_ratio_ilb.reshape(-1, 1),
            fee_breakeven,
            ["price_ratio"],
            {"domain": "amm",
             "ground_truth": "1 - 2*sqrt(r) / (1 + r)",
             "extrapolation_test": False,
             "note": "Explicit form of IL-breakeven; smooth U-shape min at r=1"},
        ))

        # Test 14: Constant product formula (multivariate) — 3-token pool: z = k / (x * y)
        # Represents a generalised constant-product invariant for three reserves.
        # Extrapolation: split on reserve_x (idx 0) so the NN sees x values it
        # hasn't trained on while y and k remain in-distribution.
        rng_mv = np.random.default_rng(23)
        reserve_mx = np.exp(rng_mv.uniform(np.log(1000), np.log(100000), n))
        reserve_my = np.exp(rng_mv.uniform(np.log(1000), np.log(100000), n))
        # invariant_mv = x * y * z  where z is in [1000, 100000] independently
        reserve_mz_true = np.exp(rng_mv.uniform(np.log(1000), np.log(100000), n))
        invariant_mv = reserve_mx * reserve_my * reserve_mz_true
        # Target: z = k / (x * y)
        reserve_mz = invariant_mv / (reserve_mx * reserve_my + 1e-10)

        tests.append((
            "Constant product formula (multivariate) for three-token pool reserve z given k and x and y",
            np.column_stack([reserve_mx, reserve_my, invariant_mv]),
            reserve_mz,
            ["reserve_x", "reserve_y", "invariant_k"],
            {
                "domain": "amm",
                "ground_truth": "k / (x * y)",
                "extrapolation_test": False,
                "note": "Generalized 3-asset constant-product; z = k / (x * y)",
            },
        ))

        return tests

    def _generate_var_tests(self, n: int) -> list[tuple]:
        """Value at Risk test cases"""
        tests = []

        # Test 1: VaR 95% (EXTRAPOLATION TEST)
        np.random.seed(43)
        portfolio_value = np.linspace(10000, 1000000, n)
        daily_vol = np.concatenate(
            [
                np.linspace(0.01, 0.03, n // 2),  # Training
                np.linspace(0.035, 0.05, n // 2),  # Extrapolation
            ]
        )
        np.random.shuffle(daily_vol)

        z_95 = 1.645
        var_95 = portfolio_value * daily_vol * z_95

        tests.append(
            (
                "Parametric Value at Risk at 95% confidence (1-day)",
                np.column_stack([portfolio_value, daily_vol]),
                var_95,
                ["portfolio_value", "daily_volatility"],
                {
                    "domain": "risk_var",
                    "extrapolation_test": True,
                    "ground_truth": "portfolio_value * volatility * 1.645",
                    "constants": {"z_score_95": 1.645},
                    "train_range": "vol 0.01-0.03",
                    "test_range": "vol 0.035-0.05",
                },
            )
        )

        # Test 2: VaR 99%
        portfolio_value = np.linspace(10000, 1000000, n)
        daily_vol = np.linspace(0.01, 0.05, n)
        z_99 = 2.326
        var_99 = portfolio_value * daily_vol * z_99

        tests.append(
            (
                "Parametric Value at Risk at 99% confidence (1-day)",
                np.column_stack([portfolio_value, daily_vol]),
                var_99,
                ["portfolio_value", "daily_volatility"],
                {
                    "domain": "risk_var",
                    "ground_truth": "portfolio_value * volatility * 2.326",
                    "constants": {"z_score_99": 2.326},
                    "extrapolation_test": False,
                },
            )
        )

        # Test 3: Multi-day VaR
        var_1day = np.linspace(1000, 100000, n)
        time_horizon = np.random.choice([5, 10, 21, 30], n)
        var_multiday = var_1day * np.sqrt(time_horizon)

        tests.append(
            (
                "Multi-day Value at Risk using square root of time rule",
                np.column_stack([var_1day, time_horizon]),
                var_multiday,
                ["var_1day", "time_horizon_days"],
                {
                    "domain": "risk_var",
                    "ground_truth": "var_1day * sqrt(days)",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 4: Portfolio VaR with correlation
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
                    "domain": "risk_var",
                    "ground_truth": "sqrt(var1^2 + var2^2 + 2*rho*var1*var2)",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 5: Annualised VaR from daily using sqrt-of-time
        daily_var = np.linspace(1000, 100000, n)
        trading_days = np.full(n, 252.0)
        annual_var = daily_var * np.sqrt(trading_days)

        tests.append((
            "Annualised VaR from daily VaR using square-root-of-time rule",
            daily_var.reshape(-1, 1),
            annual_var,
            ["daily_var"],
            {"domain": "risk_var", "ground_truth": "daily_var * sqrt(252)",
             "constants": {"trading_days": 252}, "extrapolation_test": False},
        ))

        # Test 6: Information ratio = active_return / tracking_error
        active_return = np.linspace(-0.05, 0.15, n)
        tracking_error = np.linspace(0.02, 0.20, n)
        information_ratio = active_return / (tracking_error + 1e-10)

        tests.append((
            "Information ratio from active return over tracking error",
            np.column_stack([active_return, tracking_error]),
            information_ratio,
            ["active_return", "tracking_error"],
            {"domain": "risk_var", "ground_truth": "active_return / tracking_error",
             "extrapolation_test": False},
        ))

        # Test 7: Portfolio tracking error volatility (annualised)
        daily_te = np.linspace(0.002, 0.020, n)
        ann_te = daily_te * np.sqrt(252)

        tests.append((
            "Annualised Portfolio tracking error volatility from daily TE",
            daily_te.reshape(-1, 1),
            ann_te,
            ["daily_tracking_error"],
            {"domain": "risk_var", "ground_truth": "daily_te * sqrt(252)",
             "extrapolation_test": False},
        ))

        # Test 8: Incremental VaR = portfolio_var_new - portfolio_var_old
        port_var_old = np.linspace(10000, 500000, n)
        new_position_var = np.linspace(1000, 100000, n)
        corr_new = np.linspace(-0.2, 0.6, n)
        port_var_new = np.sqrt(port_var_old**2 + new_position_var**2
                               + 2 * corr_new * port_var_old * new_position_var)
        incremental_var = port_var_new - port_var_old

        tests.append((
            "Incremental VaR from adding new position to portfolio",
            np.column_stack([port_var_old, new_position_var, corr_new]),
            incremental_var,
            ["portfolio_var", "position_var", "correlation"],
            {"domain": "risk_var",
             "ground_truth": "sqrt(v1^2+v2^2+2*rho*v1*v2) - v1",
             "extrapolation_test": False},
        ))

        return tests

    # ========================================================================
    # LIQUIDITY DOMAIN - FIXED KELLY
    # ========================================================================

    def _generate_liquidity_tests(self, n: int) -> list[tuple]:
        """Liquidity test cases with FIXED Kelly criterion"""
        tests = []

        # Test 1: FIXED Kelly Criterion (EXTRAPOLATION TEST)
        np.random.seed(44)
        expected_apy = np.concatenate(
            [
                np.linspace(0.05, 0.18, n // 2),  # Training
                np.linspace(0.22, 0.30, n // 2),  # Extrapolation
            ]
        )
        np.random.shuffle(expected_apy)

        il_risk = np.linspace(0.05, 0.25, n)

        # CORRECT Kelly formula: f* = min(μ / (λ * σ²), 1.0)
        # where λ = 2.0 (risk aversion)
        risk_aversion = 2.0
        f_star = expected_apy / (risk_aversion * il_risk**2)
        f_star = np.minimum(f_star, 1.0)  # Cap at 100%

        tests.append(
            (
                "Optimal LP Position (Kelly) using risk-adjusted Kelly criterion",
                np.column_stack([expected_apy, il_risk]),
                f_star,
                ["expected_fee_apy", "il_risk"],
                {
                    "domain": "liquidity",
                    "extrapolation_test": True,
                    "ground_truth": "min(μ / (2 * σ²), 1.0)",
                    "constants": {"risk_aversion": 2.0},
                    "train_range": "apy 0.05-0.18",
                    "test_range": "apy 0.22-0.30",
                    "note": "Risk-adjusted Kelly with cap at 100%",
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
                    "domain": "liquidity",
                    "ground_truth": "(liq_provided / pool_liq) * total_fees",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 3: Capital efficiency (concentrated liquidity)
        price_lower = np.linspace(1800, 2000, n)
        price_upper = np.linspace(2200, 2400, n)
        price_current = np.linspace(1900, 2300, n)

        # Simplified capital efficiency
        efficiency = price_upper / (price_upper - price_lower)

        tests.append(
            (
                "Capital efficiency multiplier for concentrated liquidity position",
                np.column_stack([price_lower, price_upper, price_current]),
                efficiency,
                ["price_lower", "price_upper", "price_current"],
                {
                    "domain": "liquidity",
                    "ground_truth": "P_upper / (P_upper - P_lower)",
                    "extrapolation_test": False,
                    "note": "Simplified efficiency measure",
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
                    "domain": "liquidity",
                    "ground_truth": "(1 + apr/365)^365 - 1",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 5: Fee APY from trading volume for LP position
        daily_volume = np.linspace(100000, 50000000, n)
        pool_tvl = np.linspace(1000000, 200000000, n)
        fee_rate = np.full(n, 0.003)  # 0.3% Uniswap V2 fee
        fee_apy = (daily_volume * fee_rate / (pool_tvl + 1e-10)) * 365

        tests.append((
            "Fee APY from trading volume and pool liquidity for LP position",
            np.column_stack([daily_volume, pool_tvl]),
            fee_apy,
            ["daily_volume", "pool_tvl"],
            {"domain": "liquidity",
             "ground_truth": "(volume * 0.003 / tvl) * 365",
             "constants": {"fee_rate": 0.003},
             "extrapolation_test": False},
        ))

        # Test 6: Concentrated liquidity position width (upper / lower ratio)
        p_low = np.linspace(1000, 3000, n)
        p_high = p_low * np.random.uniform(1.05, 2.0, n)
        width_ratio = p_high / (p_low + 1e-10)

        tests.append((
            "Concentrated liquidity position width from upper and lower price bounds",
            np.column_stack([p_low, p_high]),
            width_ratio,
            ["price_lower", "price_upper"],
            {"domain": "liquidity", "ground_truth": "P_upper / P_lower",
             "extrapolation_test": False},
        ))

        # Test 7: LP position rebalancing cost = trade_size * fee * 2
        rebal_size = np.linspace(1000, 500000, n)
        trade_fee = np.full(n, 0.003)
        rebal_cost = rebal_size * trade_fee * 2  # round trip

        tests.append((
            "LP position rebalancing cost from fee rate and position size",
            rebal_size.reshape(-1, 1),
            rebal_cost,
            ["rebalance_size"],
            {"domain": "liquidity",
             "ground_truth": "rebalance_size * fee * 2",
             "constants": {"fee": 0.003},
             "extrapolation_test": False},
        ))

        # Test 8: Impermanent loss breakeven fee rate
        # Fee must exceed |IL| to be profitable: fee_break = |IL| / (volume_share)
        il_pct = np.linspace(0.001, 0.15, n)
        volume_share = np.linspace(0.001, 0.10, n)
        breakeven_fee = il_pct / (volume_share + 1e-10)

        tests.append((
            "Impermanent loss breakeven fee rate for IL sustainability",
            np.column_stack([il_pct, volume_share]),
            breakeven_fee,
            ["il_pct", "volume_share"],
            {"domain": "liquidity", "ground_truth": "il_pct / volume_share",
             "extrapolation_test": False},
        ))

        # Test 9: Concentrated liquidity position width (v2) — sqrt-price difference
        # Distinct from v1 (P_upper / P_lower ratio): this is the Uniswap V3
        # sqrt-price span sqrt(P_upper) - sqrt(P_lower), which scales the
        # virtual token amounts in a concentrated position.
        rng_v2 = np.random.default_rng(41)
        p_low_v2 = np.sort(rng_v2.uniform(800, 3000, n))   # split var — sorted for aggressive split
        p_high_v2 = p_low_v2 * rng_v2.uniform(1.05, 2.5, n)
        sqrt_width_v2 = np.sqrt(p_high_v2) - np.sqrt(p_low_v2)

        tests.append((
            "Concentrated liquidity position width (v2) using square-root price bounds",
            np.column_stack([p_low_v2, p_high_v2]),
            sqrt_width_v2,
            ["price_lower", "price_upper"],
            {
                "domain": "liquidity",
                "ground_truth": "sqrt(P_upper) - sqrt(P_lower)",
                "extrapolation_test": False,
                "note": "v2: sqrt-price span, distinct from v1 ratio P_upper/P_lower",
            },
        ))

        return tests
    # ========================================================================

    def _generate_es_tests(self, n: int) -> list[tuple]:
        """Expected Shortfall test cases"""
        tests = []

        # Test 1: ES 95% (EXTRAPOLATION TEST)
        np.random.seed(45)
        portfolio_value = np.linspace(10000, 1000000, n)
        daily_vol = np.concatenate(
            [np.linspace(0.01, 0.03, n // 2), np.linspace(0.035, 0.05, n // 2)]
        )
        np.random.shuffle(daily_vol)

        # ES = portfolio * vol * 2.063 (for 95% confidence, normal dist)
        es_95 = portfolio_value * daily_vol * 2.063

        tests.append(
            (
                "Expected Shortfall at 95% confidence (CVaR) for normal returns",
                np.column_stack([portfolio_value, daily_vol]),
                es_95,
                ["portfolio_value", "daily_volatility"],
                {
                    "domain": "expected_shortfall",
                    "extrapolation_test": True,
                    "ground_truth": "portfolio * volatility * 2.063",
                    "constants": {"es_multiplier_95": 2.063},
                    "train_range": "vol 0.01-0.03",
                    "test_range": "vol 0.035-0.05",
                },
            )
        )

        # Test 2: ES 99%
        portfolio_value = np.linspace(10000, 1000000, n)
        daily_vol = np.linspace(0.01, 0.05, n)
        es_99 = portfolio_value * daily_vol * 2.665

        tests.append(
            (
                "Expected Shortfall at 99% confidence for normal returns",
                np.column_stack([portfolio_value, daily_vol]),
                es_99,
                ["portfolio_value", "daily_volatility"],
                {
                    "domain": "expected_shortfall",
                    "ground_truth": "portfolio * volatility * 2.665",
                    "constants": {"es_multiplier_99": 2.665},
                    "extrapolation_test": False,
                },
            )
        )

        # Test 3: ES from VaR
        var_95 = np.linspace(5000, 100000, n)
        es_from_var = var_95 * 1.254  # Tail multiplier for normal dist

        tests.append(
            (
                "Expected Shortfall from VaR using tail risk multiplier",
                var_95.reshape(-1, 1),
                es_from_var,
                ["var_95"],
                {
                    "domain": "expected_shortfall",
                    "ground_truth": "var_95 * 1.254",
                    "constants": {"tail_multiplier": 1.254},
                    "extrapolation_test": False,
                },
            )
        )

        # Test 4: Portfolio ES with correlation
        pos1_es = np.linspace(10000, 100000, n)
        pos2_es = np.linspace(5000, 50000, n)
        correlation = np.linspace(-0.3, 0.8, n)

        # Simplified portfolio ES (linear with correlation term)
        portfolio_es = pos1_es + pos2_es + correlation * np.sqrt(pos1_es * pos2_es)

        tests.append(
            (
                "Portfolio Expected Shortfall for correlated positions",
                np.column_stack([pos1_es, pos2_es, correlation]),
                portfolio_es,
                ["position1_es", "position2_es", "correlation"],
                {
                    "domain": "expected_shortfall",
                    "ground_truth": "ES1 + ES2 + ρ*sqrt(ES1*ES2)",
                    "extrapolation_test": False,
                    "note": "Simplified correlation adjustment",
                },
            )
        )

        # Test 5: ES scaled by holding period (square-root-of-time)
        es_1day = np.linspace(2000, 200000, n)
        holding_days = np.random.choice([5, 10, 21], n)
        es_multiday = es_1day * np.sqrt(holding_days)

        tests.append((
            "ES scaling for multi-day holding period using square-root rule",
            np.column_stack([es_1day, holding_days]),
            es_multiday,
            ["es_1day", "holding_days"],
            {"domain": "expected_shortfall",
             "ground_truth": "es_1day * sqrt(holding_days)",
             "extrapolation_test": False},
        ))

        # Test 6: Component ES from weight and correlation
        # FIX: original used linspace for all 3 vars → perfectly correlated by index
        # → test set (high weight) coincidentally had high individual_es AND high corr,
        #   making it a triple extrapolation that no model could handle.
        # FIX: sample each variable independently so the split on weight (idx 0)
        #   only extrapolates weight, leaving individual_es and corr in-distribution.
        rng_ces = np.random.default_rng(31)
        weight       = np.sort(rng_ces.uniform(0.05, 0.95, n))   # split var — sorted so aggressive split works
        individual_es = np.exp(rng_ces.uniform(np.log(5000), np.log(100000), n))  # log-uniform, independent
        corr_comp    = rng_ces.uniform(0.0, 0.8, n)               # independent
        component_es = weight * individual_es * (1 + corr_comp)

        tests.append((
            "Component ES for portfolio position from weight and correlation",
            np.column_stack([weight, individual_es, corr_comp]),
            component_es,
            ["weight", "individual_es", "correlation"],
            {"domain": "expected_shortfall",
             "ground_truth": "weight * ES * (1 + correlation)",
             "extrapolation_test": False},
        ))

        # Test 7: Historical ES as mean of worst losses
        # Simulated: mean of tail = VaR * (1 + tail_factor)
        var_input = np.linspace(1000, 100000, n)
        tail_factor = np.full(n, 0.25)
        historical_es = var_input * (1 + tail_factor)

        tests.append((
            "Historical ES from tail loss mean above VaR threshold",
            var_input.reshape(-1, 1),
            historical_es,
            ["var"],
            {"domain": "expected_shortfall",
             "ground_truth": "var * 1.25",
             "constants": {"tail_factor": 0.25},
             "extrapolation_test": False},
        ))

        return tests

    # ========================================================================
    # LIQUIDATION DOMAIN - FIXED
    # ========================================================================

    def _generate_liquidation_tests(self, n: int) -> list[tuple]:
        """Liquidation test cases with CORRECT formulas"""
        tests = []

        # Test 1: Liquidation price LONG (EXTRAPOLATION TEST)
        np.random.seed(46)
        entry_price = np.linspace(30000, 50000, n)
        leverage = np.concatenate(
            [
                np.linspace(2, 5, n // 2),  # Training
                np.linspace(7, 10, n // 2),  # Extrapolation
            ]
        )
        np.random.shuffle(leverage)

        # CORRECT: P_liq = P_entry * (1 - 1/(L * 0.8))
        maintenance_margin = 0.8
        liq_price_long = entry_price * (1 - 1 / (leverage * maintenance_margin))

        tests.append(
            (
                "Liquidation price for leveraged long position",
                np.column_stack([entry_price, leverage]),
                liq_price_long,
                ["entry_price", "leverage"],
                {
                    "domain": "liquidation",
                    "extrapolation_test": True,
                    "ground_truth": "entry_price * (1 - 1/(leverage * 0.8))",
                    "constants": {"maintenance_margin": 0.8},
                    "train_range": "leverage 2-5",
                    "test_range": "leverage 7-10",
                },
            )
        )

        # Test 2: Liquidation price SHORT
        entry_price = np.linspace(30000, 50000, n)
        leverage = np.linspace(2, 10, n)

        # CORRECT: P_liq = P_entry * (1 + 1/(L * 0.8))
        liq_price_short = entry_price * (1 + 1 / (leverage * maintenance_margin))

        tests.append(
            (
                "Liquidation price for leveraged short position",
                np.column_stack([entry_price, leverage]),
                liq_price_short,
                ["entry_price", "leverage"],
                {
                    "domain": "liquidation",
                    "ground_truth": "entry_price * (1 + 1/(leverage * 0.8))",
                    "constants": {"maintenance_margin": 0.8},
                    "extrapolation_test": False,
                },
            )
        )

        # Test 3: Maximum safe leverage
        # FIX: entry_price was included as a feature column but does NOT appear
        #   in the formula (ground_truth = "1 / (loss_pct * 0.8)").  Including
        #   it misled LLMs that read variable names and introduced a spurious
        #   correlation for NNs.  Removed — feature is acceptable_loss_pct only.
        acceptable_loss_pct = np.linspace(0.05, 0.20, n)

        # CORRECT: L_max = 1 / (loss * 0.8)
        max_leverage = 1 / (acceptable_loss_pct * maintenance_margin)

        tests.append(
            (
                "Maximum safe leverage for given acceptable loss tolerance",
                acceptable_loss_pct.reshape(-1, 1),
                max_leverage,
                ["acceptable_loss_pct"],
                {
                    "domain": "liquidation",
                    "ground_truth": "1 / (loss_pct * 0.8)",
                    "constants": {"maintenance_margin": 0.8},
                    "extrapolation_test": False,
                },
            )
        )

        # Test 4: Required collateral
        position_size = np.linspace(10000, 1000000, n)
        leverage = np.linspace(2, 10, n)

        # Simple inverse relationship
        collateral = position_size / leverage

        tests.append(
            (
                "Required collateral for leveraged position",
                np.column_stack([position_size, leverage]),
                collateral,
                ["position_size", "leverage"],
                {
                    "domain": "liquidation",
                    "ground_truth": "position_size / leverage",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 5: Margin ratio at current mark price
        collateral_margin = np.linspace(1000, 200000, n)
        unrealised_pnl = np.linspace(-50000, 50000, n)
        notional = np.linspace(10000, 1000000, n)
        margin_ratio = (collateral_margin + unrealised_pnl) / (notional + 1e-10)

        tests.append((
            "Position margin ratio at mark price including unrealised PnL",
            np.column_stack([collateral_margin, unrealised_pnl, notional]),
            margin_ratio,
            ["collateral", "unrealised_pnl", "notional"],
            {"domain": "liquidation",
             "ground_truth": "(collateral + upnl) / notional",
             "extrapolation_test": False},
        ))

        # Test 6: Partial liquidation amount to restore margin
        deficit = np.linspace(100, 50000, n)
        liquidation_fee = np.full(n, 0.005)
        partial_liq = deficit / (1 - liquidation_fee)

        tests.append((
            "Partial liquidation amount to restore undercollateralised position",
            np.column_stack([deficit, liquidation_fee]),
            partial_liq,
            ["deficit", "liquidation_fee"],
            {"domain": "liquidation",
             "ground_truth": "deficit / (1 - fee)",
             "extrapolation_test": False},
        ))

        # Test 7: Cross-margin available balance = total_margin - used_margin
        total_margin = np.linspace(10000, 500000, n)
        used_margin = total_margin * np.random.uniform(0.1, 0.8, n)
        available = total_margin - used_margin

        tests.append((
            "Cross-margin available balance after subtracting used margin",
            np.column_stack([total_margin, used_margin]),
            available,
            ["total_margin", "used_margin"],
            {"domain": "liquidation",
             "ground_truth": "total_margin - used_margin",
             "extrapolation_test": False},
        ))

        # Test 8: Realised PnL on long close = (exit - entry) * size
        entry_p = np.linspace(1000, 50000, n)
        exit_p = entry_p * np.random.uniform(0.8, 1.3, n)
        size = np.random.uniform(0.1, 10.0, n)
        realised_pnl = (exit_p - entry_p) * size

        tests.append((
            "Realized PnL for long position on close at exit price",
            np.column_stack([entry_p, exit_p, size]),
            realised_pnl,
            ["entry_price", "exit_price", "size"],
            {"domain": "liquidation",
             "ground_truth": "(exit_price - entry_price) * size",
             "extrapolation_test": False},
        ))

        return tests

    # ========================================================================
    # ALIAS / NEW DOMAIN GENERATORS
    # Added to support the domain names used in test_enhanced_defi_extrapolation
    # ========================================================================

    def _generate_risk_alias_tests(self, n: int) -> list[tuple]:
        """
        'risk' domain — combines VaR and Expected-Shortfall test cases so that
        the extrapolation test can look up cases like:
          "Value at Risk at 95%"  → maps to VaR 95% (first case in risk_var)
          "Expected Shortfall at 95%" → maps to ES 95% (first case in expected_shortfall)
          "Portfolio Sharpe Ratio" → added below
          "Correlated Portfolio VaR" → maps to portfolio VaR with correlation
        """
        tests = []
        tests.extend(self._generate_var_tests(n))
        tests.extend(self._generate_es_tests(n))

        # Sharpe ratio: (return - risk_free) / volatility
        np.random.seed(50)
        portfolio_return = np.linspace(0.05, 0.40, n)
        risk_free = np.full(n, 0.04)
        volatility = np.linspace(0.05, 0.35, n)
        sharpe = (portfolio_return - risk_free) / (volatility + 1e-10)

        tests.append(
            (
                "Portfolio Sharpe Ratio for risk-adjusted return",
                np.column_stack([portfolio_return, volatility]),
                sharpe,
                ["portfolio_return", "volatility"],
                {
                    "domain": "risk",
                    "ground_truth": "(return - 0.04) / volatility",
                    "constants": {"risk_free_rate": 0.04},
                    "extrapolation_test": False,
                },
            )
        )

        # Correlated portfolio VaR (already in risk_var but rename domain tag)
        var_asset1 = np.linspace(5000, 50000, n)
        var_asset2 = np.linspace(3000, 30000, n)
        correlation = np.linspace(-0.5, 0.9, n)
        var_portfolio = np.sqrt(
            var_asset1 ** 2 + var_asset2 ** 2
            + 2 * correlation * var_asset1 * var_asset2
        )

        tests.append(
            (
                "Correlated Portfolio VaR for two correlated assets",
                np.column_stack([var_asset1, var_asset2, correlation]),
                var_portfolio,
                ["var_asset1", "var_asset2", "correlation"],
                {
                    "domain": "risk",
                    "ground_truth": "sqrt(var1^2 + var2^2 + 2*rho*var1*var2)",
                    "extrapolation_test": False,
                },
            )
        )

        return tests

    def _generate_lending_tests(self, n: int) -> list[tuple]:
        """
        'lending' domain — collateral ratio, LTV, borrowing interest, health factor.
        Covers extrapolation test cases:
          "Collateral Ratio", "Borrowing Interest", "Multi-Collateral LTV"
        """
        tests = []
        np.random.seed(51)

        # Test 1: Collateral ratio
        collateral = np.linspace(10000, 1000000, n)
        borrowed = collateral * np.random.uniform(0.1, 0.9, n)
        collateral_ratio = collateral / (borrowed + 1e-10)

        tests.append(
            (
                "Collateral Ratio for lending position",
                np.column_stack([collateral, borrowed]),
                collateral_ratio,
                ["collateral_value", "borrowed_value"],
                {
                    "domain": "lending",
                    "ground_truth": "collateral / borrowed",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 2: Borrowing interest (compound)
        principal = np.linspace(1000, 100000, n)
        rate = np.concatenate(
            [np.linspace(0.02, 0.10, n // 2), np.linspace(0.12, 0.20, n // 2)]
        )
        np.random.shuffle(rate)
        time_years = np.random.uniform(0.5, 3.0, n)
        accrued_interest = principal * (np.exp(rate * time_years) - 1)

        tests.append(
            (
                "Borrowing Interest accrued on borrowed amount (continuous compounding)",
                np.column_stack([principal, rate, time_years]),  # FIX: include time_years — formula depends on it
                accrued_interest,
                ["principal", "interest_rate", "time_years"],
                {
                    "domain": "lending",
                    "ground_truth": "principal * (exp(rate * t) - 1)",
                    "extrapolation_test": True,
                    "train_range": "rate 0.02-0.10",
                    "test_range": "rate 0.12-0.20",
                    "note": "FIX: time_years added to features; was causing ~0.31 R² ceiling",
                },
            )
        )

        # Test 3: Loan-to-value (LTV)
        loan_amount = np.linspace(5000, 500000, n)
        asset_value = loan_amount / np.random.uniform(0.3, 0.85, n)
        ltv = loan_amount / (asset_value + 1e-10)

        tests.append(
            (
                "Loan-to-Value ratio for collateralised lending",
                np.column_stack([loan_amount, asset_value]),
                ltv,
                ["loan_amount", "asset_value"],
                {
                    "domain": "lending",
                    "ground_truth": "loan / asset_value",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 4: Multi-collateral LTV (weighted)
        collat_a = np.linspace(5000, 200000, n)
        collat_b = np.linspace(3000, 100000, n)
        weight_a = np.full(n, 0.85)  # LTV factor for asset A
        weight_b = np.full(n, 0.70)  # LTV factor for asset B
        borrowed_mc = np.random.uniform(3000, 100000, n)
        effective_collateral = collat_a * weight_a + collat_b * weight_b
        multi_ltv = borrowed_mc / (effective_collateral + 1e-10)

        tests.append(
            (
                "Multi-Collateral LTV with weighted collateral factors",
                np.column_stack([collat_a, collat_b, borrowed_mc]),
                multi_ltv,
                ["collateral_a", "collateral_b", "borrowed"],
                {
                    "domain": "lending",
                    "ground_truth": "borrowed / (0.85*collat_a + 0.70*collat_b)",
                    "constants": {"ltv_factor_a": 0.85, "ltv_factor_b": 0.70},
                    "extrapolation_test": False,
                },
            )
        )

        # Test 5: Utilization rate of DeFi lending pool
        total_borrowed = np.linspace(100000, 9000000, n)
        total_supplied = total_borrowed / np.random.uniform(0.3, 0.95, n)
        utilization = total_borrowed / (total_supplied + 1e-10)

        tests.append((
            "Utilization rate of DeFi lending pool (borrowed / supplied)",
            np.column_stack([total_borrowed, total_supplied]),
            utilization,
            ["total_borrowed", "total_supplied"],
            {"domain": "lending", "ground_truth": "borrowed / supplied",
             "extrapolation_test": False},
        ))

        # Test 6: Borrow APY from utilization (linear model)
        util_rate = np.linspace(0.0, 0.95, n)
        base_rate = 0.02
        slope = 0.20
        borrow_apy = base_rate + slope * util_rate

        tests.append((
            "Borrow APY from utilization rate using linear interest model",
            util_rate.reshape(-1, 1),
            borrow_apy,
            ["utilization_rate"],
            {"domain": "lending",
             "ground_truth": "0.02 + 0.20 * utilization",
             "constants": {"base_rate": 0.02, "slope": 0.20},
             "extrapolation_test": False},
        ))

        # Test 7: Supply APY from borrow APY, utilization and reserve factor
        borrow_rate = np.linspace(0.02, 0.30, n)
        util2 = np.linspace(0.10, 0.95, n)
        reserve_factor = np.full(n, 0.10)
        supply_apy = borrow_rate * util2 * (1 - reserve_factor)

        tests.append((
            "Supply APY from borrow utilization and reserve factor adjustment",
            np.column_stack([borrow_rate, util2]),
            supply_apy,
            ["borrow_rate", "utilization"],
            {"domain": "lending",
             "ground_truth": "borrow_rate * util * (1 - 0.10)",
             "constants": {"reserve_factor": 0.10},
             "extrapolation_test": False},
        ))

        # Test 8: Health factor for collateralised borrowing position
        collat_value = np.linspace(10000, 1000000, n)
        liq_threshold = np.full(n, 0.85)
        debt = collat_value * np.random.uniform(0.3, 0.80, n)
        health_factor = (collat_value * liq_threshold) / (debt + 1e-10)

        tests.append((
            "Health factor for collateralised borrowing position safety",
            np.column_stack([collat_value, debt]),
            health_factor,
            ["collateral_value", "debt"],
            {"domain": "lending",
             "ground_truth": "(collateral * 0.85) / debt",
             "constants": {"liq_threshold": 0.85},
             "extrapolation_test": False},
        ))

        # Test 9: Liquidation bonus amount for liquidator
        seized_collateral = np.linspace(1000, 200000, n)
        bonus_pct = np.full(n, 0.08)  # 8% bonus
        liquidation_bonus = seized_collateral * bonus_pct

        tests.append((
            "Liquidation bonus received by liquidator for collateral seizure",
            seized_collateral.reshape(-1, 1),
            liquidation_bonus,
            ["seized_collateral"],
            {"domain": "lending",
             "ground_truth": "seized_collateral * 0.08",
             "constants": {"bonus_pct": 0.08},
             "extrapolation_test": False},
        ))

        # Test 10: Protocol reserve accumulation from lending interest
        interest_earned = np.linspace(100, 1000000, n)
        reserve_frac = np.full(n, 0.10)
        protocol_reserve = interest_earned * reserve_frac

        tests.append((
            "Protocol reserve accumulation from lending interest fees",
            interest_earned.reshape(-1, 1),
            protocol_reserve,
            ["interest_earned"],
            {"domain": "lending",
             "ground_truth": "interest_earned * 0.10",
             "constants": {"reserve_factor": 0.10},
             "extrapolation_test": False},
        ))

        return tests

    def _generate_staking_tests(self, n: int) -> list[tuple]:
        """
        'staking' domain — APY / compounding reward tests.
        Covers: "Simple Staking APY", "Compounding Staking Returns"
        """
        tests = []
        np.random.seed(52)

        # Test 1: Simple staking APY (annualised)
        staked_amount = np.linspace(100, 100000, n)
        reward_rate = np.concatenate(
            [np.linspace(0.03, 0.10, n // 2), np.linspace(0.12, 0.20, n // 2)]
        )
        np.random.shuffle(reward_rate)
        annual_reward = staked_amount * reward_rate

        tests.append(
            (
                "Simple Staking APY annual reward for staked tokens",
                np.column_stack([staked_amount, reward_rate]),
                annual_reward,
                ["staked_amount", "reward_rate"],
                {
                    "domain": "staking",
                    "ground_truth": "staked_amount * reward_rate",
                    "extrapolation_test": True,
                    "train_range": "rate 0.03-0.10",
                    "test_range": "rate 0.12-0.20",
                },
            )
        )

        # Test 2: Auto-compounding staking returns
        # FIX: original feature matrix was [principal, apr] only, but the target
        #   principal * (1 + apr/n)^n depends on compounds_per_year (n=12/52/365).
        #   With n hidden, any (principal, apr) pair maps to three different targets
        #   → stochastically noisy, effectively intractable for both LLM and NN.
        # FIX: include compounds_per_year as a third feature so the formula is
        #   fully specified and deterministically learnable.
        principal = np.linspace(1000, 100000, n)
        apr = np.linspace(0.05, 0.50, n)
        compounds_per_year = np.random.choice([12, 52, 365], n).astype(float)
        compounded_value = principal * (1 + apr / compounds_per_year) ** compounds_per_year

        tests.append(
            (
                "Compounding Staking Returns with auto-compounding rewards",
                np.column_stack([principal, apr, compounds_per_year]),
                compounded_value,
                ["principal", "apr", "compounds_per_year"],
                {
                    "domain": "staking",
                    "ground_truth": "principal * (1 + apr/n)^n",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 3: Staking rewards with lock-up
        staked = np.linspace(1000, 50000, n)
        apy = np.linspace(0.05, 0.30, n)
        lock_days = np.random.choice([30, 60, 90, 180, 365], n)
        reward = staked * apy * lock_days / 365

        tests.append(
            (
                "Staking reward for fixed lock-up period",
                np.column_stack([staked, apy, lock_days]),
                reward,
                ["staked_amount", "apy", "lock_days"],
                {
                    "domain": "staking",
                    "ground_truth": "staked * apy * days / 365",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 4: Validator commission adjusted net reward
        gross_reward = np.linspace(100, 10000, n)
        commission_rate = np.linspace(0.0, 0.20, n)
        net_reward = gross_reward * (1 - commission_rate)

        tests.append((
            "Validator commission adjusted net reward after commission deduction",
            np.column_stack([gross_reward, commission_rate]),
            net_reward,
            ["gross_reward", "commission_rate"],
            {"domain": "staking",
             "ground_truth": "gross_reward * (1 - commission)",
             "extrapolation_test": False},
        ))

        # Test 5: Slashing penalty proportional to staked amount
        staked_s = np.linspace(1000, 500000, n)
        slash_pct = np.linspace(0.005, 0.10, n)
        slash_amount = staked_s * slash_pct

        tests.append((
            "Slashing penalty for validator misbehavior proportional to stake",
            np.column_stack([staked_s, slash_pct]),
            slash_amount,
            ["staked_amount", "slash_pct"],
            {"domain": "staking",
             "ground_truth": "staked * slash_pct",
             "extrapolation_test": False},
        ))

        # Test 6: Inflation-adjusted real staking APY
        nominal_apy = np.linspace(0.05, 0.30, n)
        inflation = np.linspace(0.02, 0.12, n)
        real_apy = (1 + nominal_apy) / (1 + inflation) - 1

        tests.append((
            "Inflation-adjusted staking real APY net of inflation rate",
            np.column_stack([nominal_apy, inflation]),
            real_apy,
            ["nominal_apy", "inflation_rate"],
            {"domain": "staking",
             "ground_truth": "(1 + nominal) / (1 + inflation) - 1",
             "extrapolation_test": False},
        ))

        # Test 7: Compound growth staking position over years
        initial_stake = np.linspace(1000, 100000, n)
        stk_apy = np.linspace(0.05, 0.35, n)
        years = np.random.choice([1, 2, 3, 5], n).astype(float)
        final_value = initial_stake * (1 + stk_apy) ** years

        tests.append((
            "Staking position annual growth with compound APY over years",
            np.column_stack([initial_stake, stk_apy, years]),
            final_value,
            ["initial_stake", "apy", "years"],
            {"domain": "staking",
             "ground_truth": "initial * (1 + apy)^years",
             "extrapolation_test": False},
        ))

        # Test 8: Epoch reward per delegator based on stake proportion
        total_epoch_reward = np.random.uniform(10000, 1000000, n)
        delegator_stake = np.linspace(100, 100000, n)
        total_stake = delegator_stake / np.random.uniform(0.001, 0.10, n)
        epoch_reward = total_epoch_reward * delegator_stake / (total_stake + 1e-10)

        tests.append((
            "Epoch reward per delegator proportional to stake share",
            np.column_stack([total_epoch_reward, delegator_stake, total_stake]),
            epoch_reward,
            ["epoch_reward", "delegator_stake", "total_stake"],
            {"domain": "staking",
             "ground_truth": "epoch_reward * delegator_stake / total_stake",
             "extrapolation_test": False},
        ))

        return tests

    def _generate_trading_tests(self, n: int) -> list[tuple]:
        """
        'trading' domain — leveraged trading formulas.
        Covers: "Liquidation Price Long/Short", "Effective Leverage"
        These re-use the same data as _generate_liquidation_tests but tagged
        with domain='trading' so the lookup succeeds.
        """
        tests = []
        np.random.seed(53)
        maintenance_margin = 0.8

        # Test 1: Liquidation price long
        entry_price = np.linspace(1000, 50000, n)
        leverage = np.concatenate(
            [np.linspace(2, 5, n // 2), np.linspace(7, 10, n // 2)]
        )
        np.random.shuffle(leverage)
        liq_long = entry_price * (1 - 1 / (leverage * maintenance_margin))

        tests.append(
            (
                "Liquidation Price Long for leveraged long trading position",
                np.column_stack([entry_price, leverage]),
                liq_long,
                ["entry_price", "leverage"],
                {
                    "domain": "trading",
                    "ground_truth": "entry_price * (1 - 1/(leverage * 0.8))",
                    "extrapolation_test": True,
                    "train_range": "leverage 2-5",
                    "test_range": "leverage 7-10",
                },
            )
        )

        # Test 2: Liquidation price short
        entry_price = np.linspace(1000, 50000, n)
        leverage = np.linspace(2, 10, n)
        liq_short = entry_price * (1 + 1 / (leverage * maintenance_margin))

        tests.append(
            (
                "Liquidation Price Short for leveraged short trading position",
                np.column_stack([entry_price, leverage]),
                liq_short,
                ["entry_price", "leverage"],
                {
                    "domain": "trading",
                    "ground_truth": "entry_price * (1 + 1/(leverage * 0.8))",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 3: Effective leverage after price movement
        initial_leverage = np.linspace(2, 10, n)
        price_change_pct = np.linspace(-0.20, 0.20, n)
        # Effective leverage = L / (1 + L * price_change)
        effective_leverage = initial_leverage / (
            1 + initial_leverage * price_change_pct + 1e-10
        )

        tests.append(
            (
                "Effective Leverage accounting for price movement",
                np.column_stack([initial_leverage, price_change_pct]),
                effective_leverage,
                ["initial_leverage", "price_change_pct"],
                {
                    "domain": "trading",
                    "ground_truth": "L / (1 + L * delta_p)",
                    "extrapolation_test": False,
                },
            )
        )

        # Test 4: Unrealised long PnL = (mark_price - entry) * size
        entry4 = np.linspace(1000, 50000, n)
        mark4 = entry4 * np.random.uniform(0.85, 1.30, n)
        size4 = np.random.uniform(0.01, 5.0, n)
        upnl_long = (mark4 - entry4) * size4

        tests.append((
            "Long position unrealized PnL from mark price and entry",
            np.column_stack([entry4, mark4, size4]),
            upnl_long,
            ["entry_price", "mark_price", "size"],
            {"domain": "trading",
             "ground_truth": "(mark_price - entry_price) * size",
             "extrapolation_test": False},
        ))

        # Test 5: Unrealised short PnL = (entry - mark_price) * size
        entry5 = np.linspace(1000, 50000, n)
        mark5 = entry5 * np.random.uniform(0.80, 1.25, n)
        size5 = np.random.uniform(0.01, 5.0, n)
        upnl_short = (entry5 - mark5) * size5

        tests.append((
            "Short position unrealized PnL from entry and mark price",
            np.column_stack([entry5, mark5, size5]),
            upnl_short,
            ["entry_price", "mark_price", "size"],
            {"domain": "trading",
             "ground_truth": "(entry_price - mark_price) * size",
             "extrapolation_test": False},
        ))

        # Test 6: Funding rate cost for long position over period
        position_notional = np.linspace(1000, 1000000, n)
        funding_rate = np.linspace(-0.001, 0.003, n)
        funding_periods = np.random.choice([1, 3, 8, 24], n).astype(float)
        funding_cost = position_notional * funding_rate * funding_periods

        tests.append((
            "Funding rate cost for long position over funding periods",
            np.column_stack([position_notional, funding_rate, funding_periods]),
            funding_cost,
            ["notional", "funding_rate", "periods"],
            {"domain": "trading",
             "ground_truth": "notional * funding_rate * periods",
             "extrapolation_test": False},
        ))

        # Test 7: Leveraged position notional from collateral and leverage
        collateral_t = np.linspace(500, 200000, n)
        leverage_t = np.linspace(2, 20, n)
        notional_t = collateral_t * leverage_t

        tests.append((
            "Leveraged position notional from collateral and leverage multiplier",
            np.column_stack([collateral_t, leverage_t]),
            notional_t,
            ["collateral", "leverage"],
            {"domain": "trading",
             "ground_truth": "collateral * leverage",
             "extrapolation_test": False},
        ))

        # Test 8: VaR scaling by confidence level using z-score ratio
        var_95 = np.linspace(1000, 100000, n)
        z_ratio = np.full(n, 2.326 / 1.645)  # z_99 / z_95
        var_99_scaled = var_95 * z_ratio

        tests.append((
            "VaR scaling by confidence level z-score ratio from 95 to 99",
            var_95.reshape(-1, 1),
            var_99_scaled,
            ["var_95"],
            {"domain": "trading",
             "ground_truth": "var_95 * (2.326 / 1.645)",
             "constants": {"z_99": 2.326, "z_95": 1.645},
             "extrapolation_test": False},
        ))

        # Test 9: Funding rate cost (extended) — mark/index premium model
        # Extends the basic funding rate by deriving the rate from the
        # mark-price / index-price premium rather than taking it as given:
        #   funding_rate = (mark_price - index_price) / index_price
        #   cost = notional * funding_rate * periods
        # This is the standard perpetual-futures funding mechanism.
        rng_ext = np.random.default_rng(37)
        notional_ext = np.exp(rng_ext.uniform(np.log(1000), np.log(1000000), n))
        index_price_ext = np.exp(rng_ext.uniform(np.log(100), np.log(5000), n))
        premium_pct = rng_ext.uniform(-0.005, 0.015, n)  # -0.5% to +1.5% typical range
        mark_price_ext = index_price_ext * (1 + premium_pct)
        periods_ext = rng_ext.choice([1, 3, 8, 24], n).astype(float)
        funding_cost_ext = notional_ext * (mark_price_ext - index_price_ext) / (index_price_ext + 1e-10) * periods_ext

        tests.append((
            "Funding rate cost (extended) from mark-index premium over funding periods",
            np.column_stack([notional_ext, mark_price_ext, index_price_ext, periods_ext]),
            funding_cost_ext,
            ["notional", "mark_price", "index_price", "periods"],
            {
                "domain": "trading",
                "ground_truth": "notional * (mark - index) / index * periods",
                "extrapolation_test": False,
                "note": "Extended: funding rate derived from mark/index premium",
            },
        ))

        return tests

    def _generate_derivatives_tests(self, n: int) -> list[tuple]:
        """
        'derivatives' domain — options pricing tests.
        Covers: "Options Delta", "Black-Scholes Call Price", "Volatility Smile Skew"
        Uses scipy.stats.norm for Black-Scholes; falls back gracefully if absent.
        """
        tests = []
        np.random.seed(54)

        try:
            from scipy.stats import norm
        except ImportError:
            norm = None

        # Test 1: Options Delta — FIXED: use linear proxy to avoid NaN from degenerate K values
        # delta ≈ 0.5 + moneyness / (2 * vol * sqrt(T))
        moneyness = np.linspace(-0.5, 0.5, n)
        implied_vol = np.linspace(0.10, 0.60, n)
        time_to_expiry = np.abs(np.random.uniform(0.05, 1.0, n)) + 0.05  # ensure > 0

        delta = 0.5 + moneyness / (2 * implied_vol * np.sqrt(time_to_expiry) + 1e-10)
        delta = np.clip(delta, 0.0, 1.0)

        tests.append(
            (
                "Options Delta rate of change of option price relative to underlying",
                np.column_stack([moneyness, implied_vol]),
                delta,
                ["moneyness", "implied_volatility"],
                {
                    "domain": "derivatives",
                    "ground_truth": "clip(0.5 + m/(2*vol*sqrt(T)), 0, 1)",
                    "extrapolation_test": False,
                    "note": "Linear delta approximation — avoids NaN from full BS",
                },
            )
        )

        # Test 2: Black-Scholes Call Price
        S = np.linspace(80, 120, n)
        K = np.full(n, 100.0)
        T = np.random.uniform(0.1, 1.0, n)
        sigma = np.concatenate(
            [np.linspace(0.10, 0.25, n // 2), np.linspace(0.30, 0.50, n // 2)]
        )
        np.random.shuffle(sigma)
        r = 0.04

        if norm is not None:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-10)
            d2 = d1 - sigma * np.sqrt(T)
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            # Rough proxy: intrinsic value + time value
            call_price = np.maximum(S - K, 0) + S * sigma * np.sqrt(T) * 0.4

        tests.append(
            (
                "Black-Scholes Call Price for European call option",
                np.column_stack([S, sigma]),
                call_price,
                ["spot_price", "implied_volatility"],
                {
                    "domain": "derivatives",
                    "ground_truth": "S*N(d1) - K*exp(-rT)*N(d2)",
                    "extrapolation_test": True,
                    "train_range": "sigma 0.10-0.25",
                    "test_range": "sigma 0.30-0.50",
                },
            )
        )

        # Test 3: Volatility smile skew (polynomial approximation)
        # IV(K) = ATM_vol + skew * (K - S)/S + curvature * ((K - S)/S)^2
        atm_vol = np.full(n, 0.20)
        skew = np.full(n, -0.10)
        curvature = np.full(n, 0.05)
        moneyness_skew = np.linspace(-0.30, 0.30, n)
        iv_smile = atm_vol + skew * moneyness_skew + curvature * moneyness_skew ** 2

        tests.append(
            (
                "Volatility Smile Skew implied volatility skew adjustment",
                moneyness_skew.reshape(-1, 1),
                iv_smile,
                ["moneyness"],
                {
                    "domain": "derivatives",
                    "ground_truth": "ATM_vol + skew*m + curvature*m^2",
                    "constants": {"atm_vol": 0.20, "skew": -0.10, "curvature": 0.05},
                    "extrapolation_test": False,
                },
            )
        )

        # Test 4: Forward price with cost-of-carry and dividend yield
        spot = np.linspace(50, 5000, n)
        risk_free_r = np.linspace(0.01, 0.10, n)
        div_yield = np.linspace(0.0, 0.05, n)
        maturity_t = np.random.uniform(0.1, 2.0, n)
        forward = spot * np.exp((risk_free_r - div_yield) * maturity_t)

        tests.append((
            "Forward price for derivative contract with cost-of-carry",
            np.column_stack([spot, risk_free_r, div_yield, maturity_t]),
            forward,
            ["spot", "risk_free_rate", "div_yield", "maturity"],
            {"domain": "derivatives",
             "ground_truth": "spot * exp((r - q) * T)",
             "extrapolation_test": False},
        ))

        # Test 5: Put option intrinsic value at expiry = max(K - S, 0)
        # FIX: original used linspace for both strike and spot_at_expiry with the
        #   same index range → perfectly correlated → splitting on strike (idx 0)
        #   placed all test-set spots above all test-set strikes (100% OTM, all zeros).
        # FIX: sample strike and spot independently so every split of strike contains
        #   a representative mix of ITM and OTM observations.
        rng_put = np.random.default_rng(53)
        strike       = np.sort(rng_put.uniform(50, 500, n))          # split var — sorted
        spot_exp     = rng_put.uniform(30, 600, n)                    # independent of strike
        put_intrinsic = np.maximum(strike - spot_exp, 0.0)

        tests.append((
            "Put option intrinsic value at expiry max(K minus S, 0)",
            np.column_stack([strike, spot_exp]),
            put_intrinsic,
            ["strike", "spot_at_expiry"],
            {"domain": "derivatives",
             "ground_truth": "max(K - S, 0)",
             "extrapolation_test": False},
        ))

        # Test 6: Call option intrinsic value at expiry = max(S - K, 0)
        # FIX: original used linspace for both strike6 and spot_exp6 with the
        #   same index range → perfectly correlated (same bug as Test 5 / put).
        #   Splitting on strike (idx 0) placed all test-set spots above all
        #   test-set strikes (deep ITM test, mixed train) → non-representative.
        # FIX: sample spot independently so every split of strike contains a
        #   representative mix of ITM and OTM observations.
        rng_call = np.random.default_rng(59)
        strike6   = np.sort(rng_call.uniform(50, 500, n))   # split var — sorted
        spot_exp6 = rng_call.uniform(30, 600, n)             # independent of strike
        call_intrinsic = np.maximum(spot_exp6 - strike6, 0.0)

        tests.append((
            "Call option intrinsic value at expiry max(S minus K, 0)",
            np.column_stack([strike6, spot_exp6]),
            call_intrinsic,
            ["strike", "spot_at_expiry"],
            {"domain": "derivatives",
             "ground_truth": "max(S - K, 0)",
             "extrapolation_test": False},
        ))

        # Test 7: Put-call parity relationship: C - P = S - K*exp(-rT)
        s_pcp = np.linspace(80, 150, n)
        k_pcp = np.full(n, 100.0)
        r_pcp = 0.04
        t_pcp = np.random.uniform(0.1, 1.0, n)
        pcp_diff = s_pcp - k_pcp * np.exp(-r_pcp * t_pcp)

        tests.append((
            "Put-call parity relationship C minus P equals S minus K discounted",
            np.column_stack([s_pcp, t_pcp]),
            pcp_diff,
            ["spot", "maturity"],
            {"domain": "derivatives",
             "ground_truth": "S - K * exp(-r * T)",
             "constants": {"K": 100.0, "r": 0.04},
             "extrapolation_test": False},
        ))

        # Test 8: Simple options moneyness percentage = (S - K) / K * 100
        s_mon = np.linspace(70, 150, n)
        k_mon = np.full(n, 100.0)
        moneyness_pct = (s_mon - k_mon) / k_mon * 100

        tests.append((
            "Simple options moneyness percentage relative to strike price",
            s_mon.reshape(-1, 1),
            moneyness_pct,
            ["spot_price"],
            {"domain": "derivatives",
             "ground_truth": "(S - K) / K * 100",
             "constants": {"K": 100.0},
             "extrapolation_test": False},
        ))


        # Test 9: Black-Scholes Put Price = K*exp(-rT)*N(-d2) - S*N(-d1)
        S_put = np.linspace(80, 120, n)
        K_put = np.full(n, 100.0)
        T_put = np.random.uniform(0.1, 1.0, n)
        sigma_put = np.concatenate(
            [np.linspace(0.10, 0.25, n // 2), np.linspace(0.30, 0.50, n // 2)]
        )
        np.random.shuffle(sigma_put)
        r_put = 0.04

        if norm is not None:
            d1_p = (np.log(S_put / K_put) + (r_put + 0.5 * sigma_put**2) * T_put) / (sigma_put * np.sqrt(T_put) + 1e-10)
            d2_p = d1_p - sigma_put * np.sqrt(T_put)
            put_price = K_put * np.exp(-r_put * T_put) * norm.cdf(-d2_p) - S_put * norm.cdf(-d1_p)
        else:
            put_price = np.maximum(K_put - S_put, 0) + S_put * sigma_put * np.sqrt(T_put) * 0.4

        tests.append((
            "Black-Scholes Put Price for European put option",
            np.column_stack([S_put, sigma_put]),
            put_price,
            ["spot_price", "implied_volatility"],
            {"domain": "derivatives",
             "ground_truth": "K*exp(-rT)*N(-d2) - S*N(-d1)",
             "extrapolation_test": True,
             "train_range": "sigma 0.10-0.25",
             "test_range": "sigma 0.30-0.50"},
        ))

        # Test 10: Gamma = N'(d1) / (S * sigma * sqrt(T))
        S_gam = np.linspace(80, 120, n)
        sigma_gam = np.linspace(0.10, 0.50, n)
        T_gam = np.abs(np.random.uniform(0.1, 1.0, n)) + 0.05
        K_gam = np.full(n, 100.0)
        r_gam = 0.04
        d1_gam = (np.log(S_gam / K_gam) + (r_gam + 0.5 * sigma_gam**2) * T_gam) / (sigma_gam * np.sqrt(T_gam) + 1e-10)

        if norm is not None:
            gamma = norm.pdf(d1_gam) / (S_gam * sigma_gam * np.sqrt(T_gam) + 1e-10)
        else:
            gamma = np.exp(-0.5 * d1_gam**2) / (S_gam * sigma_gam * np.sqrt(T_gam) * np.sqrt(2 * np.pi) + 1e-10)

        tests.append((
            "Gamma of option rate of change of delta with respect to spot",
            np.column_stack([S_gam, sigma_gam, T_gam]),
            gamma,
            ["spot_price", "implied_volatility", "time_to_expiry"],
            {"domain": "derivatives",
             "ground_truth": "N\'(d1) / (S * sigma * sqrt(T))",
             "extrapolation_test": False},
        ))

        # Test 11: Vega = S * N'(d1) * sqrt(T)
        S_veg = np.linspace(80, 120, n)
        sigma_veg = np.linspace(0.10, 0.50, n)
        T_veg = np.abs(np.random.uniform(0.1, 1.0, n)) + 0.05
        K_veg = np.full(n, 100.0)
        r_veg = 0.04
        d1_veg = (np.log(S_veg / K_veg) + (r_veg + 0.5 * sigma_veg**2) * T_veg) / (sigma_veg * np.sqrt(T_veg) + 1e-10)

        if norm is not None:
            vega = S_veg * norm.pdf(d1_veg) * np.sqrt(T_veg)
        else:
            vega = S_veg * np.exp(-0.5 * d1_veg**2) / np.sqrt(2 * np.pi) * np.sqrt(T_veg)

        tests.append((
            "Vega of option sensitivity of price to implied volatility",
            np.column_stack([S_veg, sigma_veg, T_veg]),
            vega,
            ["spot_price", "implied_volatility", "time_to_expiry"],
            {"domain": "derivatives",
             "ground_truth": "S * N\'(d1) * sqrt(T)",
             "extrapolation_test": False},
        ))

        # Test 12: Theta ≈ -(S*N'(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-rT)*N(d2)
        S_th = np.linspace(80, 120, n)
        sigma_th = np.linspace(0.10, 0.50, n)
        T_th = np.abs(np.random.uniform(0.1, 1.0, n)) + 0.05
        K_th = np.full(n, 100.0)
        r_th = 0.04
        d1_th = (np.log(S_th / K_th) + (r_th + 0.5 * sigma_th**2) * T_th) / (sigma_th * np.sqrt(T_th) + 1e-10)
        d2_th = d1_th - sigma_th * np.sqrt(T_th)

        if norm is not None:
            theta = (-(S_th * norm.pdf(d1_th) * sigma_th) / (2 * np.sqrt(T_th) + 1e-10)
                     - r_th * K_th * np.exp(-r_th * T_th) * norm.cdf(d2_th))
        else:
            theta = (-(S_th * np.exp(-0.5 * d1_th**2) / np.sqrt(2 * np.pi) * sigma_th) / (2 * np.sqrt(T_th) + 1e-10)
                     - r_th * K_th * np.exp(-r_th * T_th) * 0.5)

        tests.append((
            "Theta of option time decay of call option price per day",
            np.column_stack([S_th, sigma_th, T_th]),
            theta,
            ["spot_price", "implied_volatility", "time_to_expiry"],
            {"domain": "derivatives",
             "ground_truth": "-(S*N\'(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-rT)*N(d2)",
             "extrapolation_test": False},
        ))

        return tests

    # ========================================================================
    # REPORTING
    # ========================================================================

    def generate_experiment_report(self, results: list[dict]) -> dict:
        """Generate comprehensive experiment report"""

        successful = [
            r for r in results if r.get("evaluation", {}).get("success", False)
        ]

        r2_scores = [
            r["evaluation"]["r2"] for r in successful if "r2" in r.get("evaluation", {})
        ]

        # By domain
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

        # Calculate domain stats
        for domain in by_domain:
            scores = by_domain[domain]["r2_scores"]
            by_domain[domain]["mean_r2"] = np.mean(scores) if scores else None
            by_domain[domain]["median_r2"] = np.median(scores) if scores else None

        # Extrapolation tests
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
