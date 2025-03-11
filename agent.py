import os
from mistralai import Mistral
import discord
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from typing import Dict, Any
import json

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are a helpful financial analysis assistant. You can:
1. Provide real-time stock quotes
2. Calculate key financial metrics
3. Analyze market trends
Please provide clear, concise financial advice and analysis."""

class FinancialAnalyzer:
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        self.cache: Dict[str, Any] = {}  # O(1) lookup cache
        
    async def get_stock_price(self, symbol: str) -> dict:
        """Get real-time stock price with O(1) cache lookup"""
        if symbol in self.cache and (pd.Timestamp.now() - self.cache[symbol]['timestamp']).seconds < 60:
            return self.cache[symbol]['data']
            
        try:
            data = yf.Ticker(symbol).info
            result = {
                'price': data.get('regularMarketPrice', 0),
                'change': data.get('regularMarketChangePercent', 0),
                'volume': data.get('regularMarketVolume', 0)
            }
            self.cache[symbol] = {
                'data': result,
                'timestamp': pd.Timestamp.now()
            }
            return result
        except Exception as e:
            return {'error': str(e)}

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.analyzer = FinancialAnalyzer()

    async def run(self, message: discord.Message):
        # Extract potential stock symbols from message
        content = message.content.upper()
        words = content.split()
        
        # Check for stock symbols (assumed to be uppercase 1-5 letters)
        stock_data = {}
        for word in words:
            if word.isalpha() and 1 <= len(word) <= 5:
                stock_data[word] = await self.analyzer.get_stock_price(word)

        # Add financial context to the message
        context = f"Financial data available: {json.dumps(stock_data, indent=2)}" if stock_data else ""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{message.content}\n\nAvailable financial data:\n{context}"},
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        return response.choices[0].message.content
