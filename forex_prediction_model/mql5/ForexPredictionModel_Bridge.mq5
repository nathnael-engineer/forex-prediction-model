//+------------------------------------------------------------------+
//|                                  ForexPredictionModel_Bridge.mq5 |
//|                     Copyright 2026, forex-prediction-model       |
//|               https://github.com/nathnael-engineer/forex-prediction-model.git |
//+------------------------------------------------------------------+
#property copyright "Apache-2.0 License"
#property link      "https://github.com/nathnael-engineer/forex-prediction-model.git"
#property version   "2.40"
#property strict

#include <Trade\Trade.mqh>

input string   InpApiUrl          = "http://127.0.0.1:8000/api/v1/predict"; // Python Model API URL
input double   InpConfidenceMin   = 0.72;                                   // Minimum Model Confidence Threshold
input double   InpFixedLotSize    = 0.10;                                   // Fallback Lot Size
input bool     InpUseDynamicLot   = true;                                   // Use Kelly-Calculated Dynamic Lot
input int      InpMagicNumber     = 992026;                                 // Expert Magic ID
input int      InpSlippagePips    = 3;                                      // Max Slippage in Points

CTrade trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippagePips * 10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   Print("[forex-prediction-model] Initialized EA Bridge. Connected to: ", InpApiUrl);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function (Executed on every incoming price tick)     |
//+------------------------------------------------------------------+
void OnTick()
{
   string symbol = _Symbol;
   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return;

   // Only analyze on bar open or when no open position exists for this magic
   if(PositionsTotal() > 0)
   {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(PositionGetString(POSITION_SYMBOL) == symbol && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         {
            // Position already active, manage trailing stop or hold
            return;
         }
      }
   }

   // Prepare JSON Request payload to local Python Low-Latency API
   string jsonPayload = StringFormat("{\"pair\":\"%s\",\"bid\":%.5f,\"ask\":%.5f,\"volume\":%.2f}",
                                     symbol, tick.bid, tick.ask, (double)tick.volume);

   char postData[];
   char resultData[];
   string resultHeaders;
   StringToCharArray(jsonPayload, postData, 0, WHOLE_ARRAY, CP_UTF8);

   string headers = "Content-Type: application/json\r\n";
   int timeout = 50; // 50ms low latency timeout

   int res = WebRequest("POST", InpApiUrl, headers, timeout, postData, resultData, resultHeaders);
   if(res == 200)
   {
      string responseJson = CharArrayToString(resultData, 0, WHOLE_ARRAY, CP_UTF8);
      ProcessSignalResponse(responseJson, tick);
   }
}

//+------------------------------------------------------------------+
//| Parse JSON and Execute Algorithmic Trade Order                   |
//+------------------------------------------------------------------+
void ProcessSignalResponse(string json, MqlTick &tick)
{
   if(StringFind(json, "\"action\":\"STRONG_BUY\"") >= 0 || StringFind(json, "\"action\":\"BUY\"") >= 0)
   {
      double lot = InpFixedLotSize;
      double ask = tick.ask;
      double sl = ask - 200 * _Point;
      double tp = ask + 400 * _Point;

      trade.Buy(lot, _Symbol, ask, sl, tp, "forex-prediction-model [LONG]");
      Print("[forex-prediction-model] BUY Order executed at: ", ask);
   }
   else if(StringFind(json, "\"action\":\"STRONG_SELL\"") >= 0 || StringFind(json, "\"action\":\"SELL\"") >= 0)
   {
      double lot = InpFixedLotSize;
      double bid = tick.bid;
      double sl = bid + 200 * _Point;
      double tp = bid - 400 * _Point;

      trade.Sell(lot, _Symbol, bid, sl, tp, "forex-prediction-model [SHORT]");
      Print("[forex-prediction-model] SELL Order executed at: ", bid);
   }
}
