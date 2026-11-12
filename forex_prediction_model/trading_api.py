"""
Project: forex-prediction-model
Repository: https://github.com/nathnael-engineer/forex-prediction-model.git
Module: trading_api.py
Description: Asynchronous high-throughput REST & WebSocket bridge API connecting
             the TensorFlow model to MetaTrader 5, cTrader, and TradingView webhooks.
"""

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

from .feature_pipeline import ForexStreamingFeaturePipeline
from .inference_engine import HighFrequencyInferenceEngine
from .strategy_executor import AlgorithmicStrategyExecutor

app = FastAPI(
    title="forex-prediction-model API",
    description="High-Accuracy, Low-Latency Forex Signal Generation Server",
    version="2.4.0"
)

pipelines: Dict[str, ForexStreamingFeaturePipeline] = {}
engine = HighFrequencyInferenceEngine(use_onnx=False)
executor = AlgorithmicStrategyExecutor()


class TickPayload(BaseModel):
    pair: str
    bid: float
    ask: float
    volume: Optional[float] = 1.0


class SignalResponse(BaseModel):
    pair: str
    action: str
    confidence: float
    probabilities: Dict[str, float]
    order: Optional[Dict[str, Any]] = None
    latency_ms: float


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "TensorFlow-2.18-XLA",
        "mixed_precision": "mixed_float16",
        "active_pairs": list(pipelines.keys())
    }


@app.post("/api/v1/predict", response_model=SignalResponse)
async def generate_prediction_signal(tick: TickPayload):
    """
    Ingests live tick from MetaTrader 5 or broker, updates rolling tensor,
    and returns immediate algorithmic trade action in < 2ms.
    """
    if tick.pair not in pipelines:
        pip = 0.01 if "JPY" in tick.pair else 0.0001
        pipelines[tick.pair] = ForexStreamingFeaturePipeline(sequence_length=60, pip_value=pip)

    pipe = pipelines[tick.pair]
    features = pipe.update_tick(tick.bid, tick.ask, tick.volume or 1.0)
    input_tensor = pipe.get_input_tensor()

    # Model inference forward pass
    inference_result = engine.predict(input_tensor)

    # Strategy risk evaluation
    atr_pips = pipe.atr_smoothed / pipe.pip_value if pipe.atr_smoothed else 20.0
    signal_decision = executor.evaluate_signal(
        pair=tick.pair,
        current_bid=tick.bid,
        current_ask=tick.ask,
        atr_pips=atr_pips,
        model_output=inference_result,
        pip_multiplier=pipe.pip_value
    )

    return {
        "pair": tick.pair,
        "action": signal_decision["action"],
        "confidence": signal_decision["confidence"],
        "probabilities": signal_decision["probabilities"],
        "order": signal_decision["order"],
        "latency_ms": inference_result["latency_ms"]
    }


@app.websocket("/ws/live-feed")
async def websocket_stream(websocket: WebSocket):
    """Real-time bi-directional streaming for algorithmic execution bots."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            pair = data.get("pair", "EUR/USD")
            bid = float(data.get("bid", 1.0842))
            ask = float(data.get("ask", 1.0843))
            
            if pair not in pipelines:
                pip = 0.01 if "JPY" in pair else 0.0001
                pipelines[pair] = ForexStreamingFeaturePipeline(60, pip)
            
            pipe = pipelines[pair]
            pipe.update_tick(bid, ask)
            tensor = pipe.get_input_tensor()
            res = engine.predict(tensor)
            
            await websocket.send_json({
                "pair": pair,
                "timestamp": asyncio.get_event_loop().time(),
                "probs": res["probabilities"],
                "latency_ms": res["latency_ms"]
            })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("trading_api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
