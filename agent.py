import os
from mistralai import Mistral
import discord
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from typing import Dict, Any
import json
import matplotlib.pyplot as plt
import io
import re

MAX_DISCORD_MESSAGE_LENGTH = 2000

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are a helpful financial analysis assistant. You can:
1. Provide real-time stock quotes
2. Calculate key financial metrics
3. Analyze market trends
If multiple symbols are provided, conduct a summary for each of them based on the financial data you are given. You may also discuss portfolio trends if TICKERSYMBOL:QUANTITY strings are given asking for a portfolio recommendation.
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
        self.ask_prices = {}  # Format: { "AAPL": [ask1, ask2, ...], "TSLA": [ask1, ask2, ...] }

    async def run(self, message: discord.Message):
        # Extract potential stock symbols from message
        content = message.content.upper()
        words = content.split()
        
        # Check for stock symbols (assumed to be uppercase 1-5 letters)
        stock_data = {}
        stock_qty = {}
        ticker_only = {}
        for wrd in words:
            if len(self.parse_ticker_quantity(wrd))>0:
                word = self.parse_ticker_quantity(wrd)[0][0]    
                qty = self.parse_ticker_quantity(wrd)[0][1]            
#if word.isalpha() and 1 <= len(word) <= 5:
                try:
                    price_data = await self.analyzer.get_stock_price(word)
                    if isinstance(price_data, dict) and 'price' in price_data:
                        stock_data[word] = price_data['price']  # Extract price list only
                        ticker_only[word] = price_data['price']
                        self.store_ask_price(word,price_data['price'])
                        stock_qty[word] = qty
                    else:
                        print(f"Invalid stock data received for {word}: {price_data}")
                except Exception as e:
                    print(f"Error fetching stock price for {word}: {e}")
            elif wrd.isalpha() and 1 <= len(wrd) <= 5:
                try:
                    stock_info = await self.analyzer.get_stock_price(wrd)  # Must return {'price': [...], 'ask': float}
                    
                    if isinstance(stock_info, dict) and 'price' in stock_info:
                        ticker_only[word] = stock_info['price']  # Store historical prices
                        self.store_ask_price(word, stock_info['price'])  # Store asks
                    
                    else:
                        print(f"Invalid stock data for {wrd}: {stock_info}")

                except Exception as e:
                    print(f"Error fetching stock data for {wrd}: {e}")


        # Add financial context to the message
        context = f"Financial data available: {json.dumps(ticker_only, indent=2)}" if ticker_only else ""
        portfolio_context = f"Portfolio info available: {json.dumps(stock_qty,indent=2)}" if stock_qty else ""
        historical_context = f"User may have asked about this before, reference change in prices since previous ask if there are multiple historical prices recorded for a symbol where Current Financial Data is also available. Historical prices were previously {json.dumps(self.ask_prices, indent=2)}" if self.ask_prices else "" 
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{message.content}\n\nAvailable financial data:\n{context}\n\nPortfolio analysis requested if quantities provided:\n{portfolio_context}\n\nPreviously asked stocks and prices: \n{historical_context}\n\n"}
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        response_content = response.choices[0].message.content

        # Generate a graph if stock data exists
        if stock_data:
            graph_image = self.generate_portfolio_pie(stock_data, stock_qty)
#            await message.channel.send(response_content, file=discord.File(graph_image, "stock_graph.png"))
            await self.send_long_message(message.channel, response_content)
            await message.channel.send(file=discord.File(graph_image, "stock_graph.png"))

        else:
            await message.channel.send(response_content[:1999])

    async def send_long_message(self, channel, content):
    
        chunks = [content[i:i + 2000] for i in range(0, len(content), 2000)]
        for chunk in chunks:
            await channel.send(chunk)
    
    def store_ask_price(self, stock, ask_price):
        #Stores the latest ask price for a stock -> allows you to compare and decide if it's up or down relative to the previous asks.
        if stock not in self.ask_prices:
            self.ask_prices[stock] = []
        self.ask_prices[stock].append(ask_price)

        # Limit to last 10 ask prices per stock
        self.ask_prices[stock] = self.ask_prices[stock][-10:]


    def parse_ticker_quantity(self,text):
        """Extracts ticker and quantity from a string."""
        pattern = r"\b([A-Z]{3,5}):(\d+)\b"
        matches = re.findall(pattern, text)
    
        # Convert quantity to int and return as list of tuples
        return [(ticker, int(quantity)) for ticker, quantity in matches]

 
    def generate_portfolio_pie(self, stock_data, stock_qty):
        """Generates a pie chart of the portfolio based on stock values (price * quantity)."""
        portfolio_values = {}

        for stock, price_list in stock_data.items():
            if stock in stock_qty:
                #latest_price = price_list[-1] if price_list else 0  # Get last available price
                total_value = price_list * stock_qty[stock]  # Compute investment value
                portfolio_values[stock] = total_value

        if not portfolio_values:
            return None  # No valid data to plot

     # Create a pie chart
        plt.figure(figsize=(7, 7))
        plt.pie(
         portfolio_values.values(),
         labels=portfolio_values.keys(),
         autopct="%1.1f%%",
         startangle=140,
         colors=plt.cm.Paired.colors
        )
        plt.title("Portfolio Distribution")

     # Save the image to a buffer
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format="png")
        plt.close()
        image_buffer.seek(0)

        return image_buffer  # Return image buffer for sending in Discord


    def generate_graph(self, stock_data, stock_qty):
        #Gen a portfolio representation graph and save as image
        plt.figure(figsize=(8, 5))

        for stock, data in stock_data.items():
            print('stock ', stock)
            print('prices', data)
            if isinstance(data, list) and all(isinstance(i, (int, float)) for i in data):
                plt.plot(range(len(data)), data, label=stock)
            

        plt.xlabel("Time")
        plt.ylabel("Stock relative price")
        plt.title("Stock Price Trends")
        plt.legend()
        plt.grid()

        # Save image to in-memory buffer
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format="png")
        plt.close()
        image_buffer.seek(0)

        return image_buffer
