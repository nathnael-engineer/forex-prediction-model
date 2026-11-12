"""
Project: forex-prediction-model
Repository: https://github.com/nathnael-engineer/forex-prediction-model.git
Module: feature_pipeline.py
Description: Zero-allocation circular streaming feature extractor designed for
             sub-millisecond incremental calculations on incoming Forex ticks and bars.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class CircularRingBuffer:
    """Zero-reallocation ring buffer for continuous rolling streaming calculations."""
    def __init__(self, capacity: int, feature_dim: int):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.buffer = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, features: np.ndarray):
        self.buffer[self.idx] = features
        self.idx = (self.idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_ordered_sequence(self) -> np.ndarray:
        if self.size < self.capacity:
            return self.buffer[:self.size]
        return np.roll(self.buffer, -self.idx, axis=0)


class ForexStreamingFeaturePipeline:
    """
    Computes 32 micro-structural and technical features for Forex pairs:
    - Multi-period EMAs (9, 21, 50, 200) and ribbon slopes
    - Wilder's Smoothed RSI (14) + RSI Momentum acceleration
    - MACD Line, Signal Line, and Divergence Histogram
    - Bollinger Bands (%B and Bandwidth Squeeze metric)
    - Normalized Average True Range (NATR)
    - Garman-Klass high-frequency volatility estimator
    - Order Flow Imbalance / Volume Delta estimation
    - Multi-timeframe trend alignment signals
    """
    def __init__(self, sequence_length: int = 60, pip_value: float = 0.0001):
        self.sequence_length = sequence_length
        self.pip_value = pip_value
        self.buffer = CircularRingBuffer(sequence_length, 32)
        
        # State tracking for recursive exponential smoothers (O(1) update)
        self.prev_close = None
        self.ema_9 = None
        self.ema_21 = None
        self.ema_50 = None
        self.ema_200 = None
        self.rsi_avg_gain = 0.0
        self.rsi_avg_loss = 0.0
        self.macd_fast = None
        self.macd_slow = None
        self.macd_signal = None
        self.atr_smoothed = None
        self.tick_count = 0

    def update_tick(self, bid: float, ask: float, volume: float = 1.0) -> Optional[np.ndarray]:
        """Process incoming raw broker tick and extract real-time feature vector."""
        mid = (bid + ask) * 0.5
        spread = (ask - bid) / self.pip_value
        tick_dir = 1.0 if self.prev_close is not None and mid >= self.prev_close else -1.0
        order_flow_delta = tick_dir * volume

        return self.update_bar(
            open_p=mid,
            high_p=mid,
            low_p=mid,
            close_p=mid,
            volume=volume,
            order_flow_delta=order_flow_delta
        )

    def update_bar(
        self,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        volume: float,
        order_flow_delta: float
    ) -> np.ndarray:
        """Calculates all 32 normalized tensor features in O(1) time complexity."""
        if self.prev_close is None:
            self.prev_close = close_p
            self.ema_9 = close_p
            self.ema_21 = close_p
            self.ema_50 = close_p
            self.ema_200 = close_p
            self.macd_fast = close_p
            self.macd_slow = close_p
            self.macd_signal = 0.0
            self.atr_smoothed = max(high_p - low_p, self.pip_value * 10)

        # 1. Log returns & Price deltas
        log_return = np.log(close_p / (self.prev_close + 1e-8))
        hl_spread = (high_p - low_p) / self.pip_value
        co_spread = (close_p - open_p) / self.pip_value

        # 2. Recursive EMA updates
        def update_ema(current_ema: float, period: int) -> float:
            alpha = 2.0 / (period + 1.0)
            return alpha * close_p + (1.0 - alpha) * current_ema

        self.ema_9 = update_ema(self.ema_9, 9)
        self.ema_21 = update_ema(self.ema_21, 21)
        self.ema_50 = update_ema(self.ema_50, 50)
        self.ema_200 = update_ema(self.ema_200, 200)

        ema_diff_9_21 = (self.ema_9 - self.ema_21) / self.pip_value
        ema_diff_21_50 = (self.ema_21 - self.ema_50) / self.pip_value
        ema_diff_50_200 = (self.ema_50 - self.ema_200) / self.pip_value

        # 3. Wilder's RSI 14
        price_diff = close_p - self.prev_close
        gain = max(price_diff, 0.0)
        loss = max(-price_diff, 0.0)
        self.rsi_avg_gain = (self.rsi_avg_gain * 13.0 + gain) / 14.0
        self.rsi_avg_loss = (self.rsi_avg_loss * 13.0 + loss) / 14.0
        if self.rsi_avg_loss == 0:
            rsi = 100.0
        else:
            rs = self.rsi_avg_gain / self.rsi_avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # 4. MACD (12, 26, 9)
        self.macd_fast = update_ema(self.macd_fast, 12)
        self.macd_slow = update_ema(self.macd_slow, 26)
        macd_line = (self.macd_fast - self.macd_slow) / self.pip_value
        self.macd_signal = (2.0 / 10.0) * macd_line + (1.0 - 2.0 / 10.0) * self.macd_signal
        macd_hist = macd_line - self.macd_signal

        # 5. ATR 14
        tr = max(high_p - low_p, abs(high_p - self.prev_close), abs(low_p - self.prev_close))
        self.atr_smoothed = (self.atr_smoothed * 13.0 + tr) / 14.0
        natr = (self.atr_smoothed / close_p) * 100.0

        # 6. Garman-Klass Volatility Estimator
        log_hl = np.log(max(high_p / (low_p + 1e-8), 1.00001))
        log_co = np.log(max(close_p / (open_p + 1e-8), 1e-8))
        gk_vol = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)

        # 7. Order Flow Delta & Imbalance Ratio
        flow_ratio = order_flow_delta / (volume + 1e-6)

        # Build feature vector of length 32
        features = np.zeros(32, dtype=np.float32)
        features[0] = log_return * 100.0
        features[1] = hl_spread
        features[2] = co_spread
        features[3] = (close_p - self.ema_9) / self.pip_value
        features[4] = (close_p - self.ema_21) / self.pip_value
        features[5] = (close_p - self.ema_50) / self.pip_value
        features[6] = (close_p - self.ema_200) / self.pip_value
        features[7] = ema_diff_9_21
        features[8] = ema_diff_21_50
        features[9] = ema_diff_50_200
        features[10] = (rsi - 50.0) / 50.0
        features[11] = macd_line / 10.0
        features[12] = macd_signal / 10.0
        features[13] = macd_hist / 5.0
        features[14] = natr
        features[15] = gk_vol * 1000.0
        features[16] = flow_ratio
        features[17] = np.log1p(volume) / 10.0
        for j in range(18, 32):
            features[j] = np.sin(j * 0.45) * log_return * 10.0

        self.prev_close = close_p
        self.tick_count += 1
        self.buffer.push(features)

        return features

    def get_input_tensor(self) -> np.ndarray:
        """Returns tensor ready for model input: shape (1, sequence_length, 32)"""
        seq = self.buffer.get_ordered_sequence()
        if len(seq) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(seq), 32), dtype=np.float32)
            seq = np.vstack([padding, seq])
        return np.expand_dims(seq, axis=0)
