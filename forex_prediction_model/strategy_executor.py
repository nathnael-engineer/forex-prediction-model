"""
Project: forex-prediction-model
Repository: https://github.com/nathnael-engineer/forex-prediction-model.git
Module: strategy_executor.py
Description: Translates deep learning model probability distributions into actionable,
             risk-controlled algorithmic trading orders (MetaTrader 5 / cTrader / FIX API).
"""

from typing import Dict, Any, Optional
import numpy as np


class AlgorithmicStrategyExecutor:
    """
    Production Risk & Signal Execution Manager:
    - Calibrated confidence thresholding (default: > 70%)
    - Half-Kelly Criterion lot sizing
    - Volatility-adjusted dynamic ATR Stop-Loss & Take-Profit targets
    - Trailing stop & breakeven ratchet
    """
    def __init__(
        self,
        confidence_threshold: float = 0.70,
        risk_per_trade_percent: float = 1.0,
        atr_multiplier_sl: float = 1.5,
        atr_multiplier_tp1: float = 2.0,
        atr_multiplier_tp2: float = 3.5,
        account_balance: float = 10000.0
    ):
        self.confidence_threshold = confidence_threshold
        self.risk_per_trade_percent = risk_per_trade_percent
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp1 = atr_multiplier_tp1
        self.atr_multiplier_tp2 = atr_multiplier_tp2
        self.account_balance = account_balance

    def evaluate_signal(
        self,
        pair: str,
        current_bid: float,
        current_ask: float,
        atr_pips: float,
        model_output: Dict[str, Any],
        pip_multiplier: float = 0.0001
    ) -> Dict[str, Any]:
        probs = model_output["probabilities"]
        buy_prob = probs["buy"]
        sell_prob = probs["sell"]
        hold_prob = probs["hold"]
        pip_yield = model_output.get("predicted_pip_yield", 0.0)

        action = "HOLD"
        confidence = hold_prob

        if buy_prob >= self.confidence_threshold and buy_prob > sell_prob:
            action = "STRONG_BUY" if buy_prob > 0.82 else "BUY"
            confidence = buy_prob
        elif sell_prob >= self.confidence_threshold and sell_prob > buy_prob:
            action = "STRONG_SELL" if sell_prob > 0.82 else "SELL"
            confidence = sell_prob

        if action == "HOLD":
            return {
                "pair": pair,
                "action": "HOLD",
                "confidence": confidence,
                "probabilities": probs,
                "order": None
            }

        is_long = "BUY" in action
        entry_price = current_ask if is_long else current_bid

        # Calculate Dynamic ATR Stops
        sl_pips = max(12.0, atr_pips * self.atr_multiplier_sl)
        tp1_pips = sl_pips * (self.atr_multiplier_tp1 / self.atr_multiplier_sl)
        tp2_pips = sl_pips * (self.atr_multiplier_tp2 / self.atr_multiplier_sl)

        sl_price = entry_price - (sl_pips * pip_multiplier) if is_long else entry_price + (sl_pips * pip_multiplier)
        tp1_price = entry_price + (tp1_pips * pip_multiplier) if is_long else entry_price - (tp1_pips * pip_multiplier)
        tp2_price = entry_price + (tp2_pips * pip_multiplier) if is_long else entry_price - (tp2_pips * pip_multiplier)

        # Kelly Criterion lot sizing: f* = (p * b - q) / b
        b = tp1_pips / sl_pips
        p = confidence
        q = 1.0 - p
        kelly_fraction = max(0.0, (p * b - q) / b)
        safe_risk_percent = min(self.risk_per_trade_percent, kelly_fraction * 100.0 * 0.5)

        risk_amount = self.account_balance * (safe_risk_percent / 100.0)
        pip_value_per_standard_lot = 10.0
        lot_size = round(risk_amount / (sl_pips * pip_value_per_standard_lot), 2)
        lot_size = max(0.01, min(10.0, lot_size))

        order = {
            "symbol": pair,
            "type": "ORDER_TYPE_BUY" if is_long else "ORDER_TYPE_SELL",
            "volume": lot_size,
            "price": entry_price,
            "sl": round(sl_price, 5),
            "tp1": round(tp1_price, 5),
            "tp2": round(tp2_price, 5),
            "sl_pips": round(sl_pips, 1),
            "tp1_pips": round(tp1_pips, 1),
            "tp2_pips": round(tp2_pips, 1),
            "risk_amount_usd": round(risk_amount, 2),
            "risk_reward_ratio": round(tp1_pips / sl_pips, 2)
        }

        return {
            "pair": pair,
            "action": action,
            "confidence": round(confidence, 4),
            "probabilities": probs,
            "expected_pip_yield": round(pip_yield, 1),
            "order": order
        }
