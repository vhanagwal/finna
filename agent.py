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
import numpy as np

MAX_DISCORD_MESSAGE_LENGTH = 2000

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are an advanced financial analysis assistant with sophisticated capabilities:

1. Portfolio Analysis:
   - Real-time stock quotes and historical data
   - Risk assessment (Beta, Volatility)
   - Portfolio diversification metrics
   - Sector allocation analysis
   - Performance benchmarking

2. Technical Analysis:
   - Moving averages (SMA, EMA)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Support and resistance levels

3. Fundamental Analysis:
   - Key financial ratios (P/E, P/B, ROE)
   - Company financials analysis
   - Industry comparison
   - Growth metrics

4. Investment Recommendations:
   - Portfolio rebalancing suggestions
   - Risk-adjusted return analysis
   - Market sentiment analysis
   - Investment horizon considerations

For portfolio analysis (TICKERSYMBOL:QUANTITY), I provide:
- Portfolio diversification score
- Risk-adjusted returns
- Sector exposure analysis
- Rebalancing recommendations

I aim to provide clear, actionable financial advice based on comprehensive data analysis."""

class FinancialMetrics:
    """Advanced financial metrics calculator"""
    
    @staticmethod
    def calculate_beta(stock_returns, market_returns):
        """Calculate stock beta relative to market"""
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 1

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
        """Calculate Sharpe ratio for risk-adjusted returns"""
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if len(returns) > 0 else 0

    @staticmethod
    def calculate_volatility(returns):
        """Calculate annualized volatility"""
        return np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

class FinancialAnalyzer:
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        self.cache: Dict[str, Any] = {}  # O(1) lookup cache
        self.metrics = FinancialMetrics()
        self.sector_cache = {}
        
    async def get_stock_price(self, symbol: str) -> dict:
        """Get real-time stock price with O(1) cache lookup"""
        if symbol in self.cache and (pd.Timestamp.now() - self.cache[symbol]['timestamp']).seconds < 60:
            return self.cache[symbol]['data']
            
        try:
            data = yf.Ticker(symbol).info
            # Enhanced data collection
            result = {
                'price': data.get('regularMarketPrice', 0),
                'change': data.get('regularMarketChangePercent', 0),
                'volume': data.get('regularMarketVolume', 0),
                'pe_ratio': data.get('forwardPE', 0),
                'market_cap': data.get('marketCap', 0),
                'sector': data.get('sector', 'Unknown'),
                'beta': data.get('beta', 1.0),
                'dividend_yield': data.get('dividendYield', 0),
                '52_week_high': data.get('fiftyTwoWeekHigh', 0),
                '52_week_low': data.get('fiftyTwoWeekLow', 0)
            }
            
            # Cache the sector information
            if result['sector'] != 'Unknown':
                self.sector_cache[symbol] = result['sector']
            
            self.cache[symbol] = {
                'data': result,
                'timestamp': pd.Timestamp.now()
            }
            return result
        except Exception as e:
            return {'error': str(e)}
            
    async def get_historical_data(self, symbol: str, period='1y') -> pd.DataFrame:
        """Get historical price data for technical analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            return pd.DataFrame()
            
    def calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate technical indicators for a stock"""
        if df.empty:
            return {}
            
        try:
            # Calculate moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            latest = df.iloc[-1]
            return {
                'sma_20': latest['SMA_20'],
                'sma_50': latest['SMA_50'],
                'rsi': latest['RSI'],
                'macd': latest['MACD'],
                'macd_signal': latest['Signal_Line']
            }
        except Exception as e:
            return {}

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.analyzer = FinancialAnalyzer()
        self.ask_prices = {}  # Format: { "AAPL": [ask1, ask2, ...], "TSLA": [ask1, ask2, ...] }
        self.portfolio_history = {}  # Track portfolio performance over time
        
    async def analyze_portfolio(self, stock_data, stock_qty):
        """Comprehensive portfolio analysis"""
        analysis = {
            'total_value': 0,
            'sector_allocation': {},
            'risk_metrics': {},
            'technical_signals': {},
            'recommendations': []
        }
        
        # Calculate total portfolio value and sector allocation
        for symbol, price in stock_data.items():
            if symbol in stock_qty:
                value = price * stock_qty[symbol]
                analysis['total_value'] += value
                
                # Sector allocation
                sector = self.analyzer.sector_cache.get(symbol, 'Unknown')
                if sector not in analysis['sector_allocation']:
                    analysis['sector_allocation'][sector] = 0
                analysis['sector_allocation'][sector] += value
                
                # Get technical indicators
                hist_data = await self.analyzer.get_historical_data(symbol)
                if not hist_data.empty:
                    returns = hist_data['Close'].pct_change().dropna()
                    market_data = await self.analyzer.get_historical_data('^GSPC')  # S&P 500
                    market_returns = market_data['Close'].pct_change().dropna()
                    
                    # Calculate risk metrics
                    analysis['risk_metrics'][symbol] = {
                        'beta': self.analyzer.metrics.calculate_beta(returns, market_returns),
                        'volatility': self.analyzer.metrics.calculate_volatility(returns),
                        'sharpe_ratio': self.analyzer.metrics.calculate_sharpe_ratio(returns)
                    }
                    
                    # Get technical signals
                    analysis['technical_signals'][symbol] = self.analyzer.calculate_technical_indicators(hist_data)
        
        # Convert sector allocation to percentages
        for sector in analysis['sector_allocation']:
            analysis['sector_allocation'][sector] = (analysis['sector_allocation'][sector] / analysis['total_value']) * 100
            
        # Generate recommendations
        analysis['recommendations'] = self.generate_recommendations(analysis)
        
        return analysis
    
    def generate_recommendations(self, analysis):
        """Generate portfolio recommendations based on analysis"""
        recommendations = []
        
        # Check sector diversification
        max_sector_allocation = max(analysis['sector_allocation'].values())
        if max_sector_allocation > 30:
            recommendations.append("Consider diversifying across more sectors - current maximum sector exposure is {:.1f}%".format(max_sector_allocation))
            
        # Check individual stock risks
        for symbol, metrics in analysis['risk_metrics'].items():
            if metrics['volatility'] > 0.3:  # 30% annualized volatility
                recommendations.append(f"High volatility detected in {symbol} - consider reducing position")
            if metrics['beta'] > 1.5:
                recommendations.append(f"{symbol} shows high market sensitivity (β={metrics['beta']:.2f}) - review risk tolerance")
                
        # Technical analysis recommendations
        for symbol, signals in analysis['technical_signals'].items():
            if signals.get('rsi', 50) > 70:
                recommendations.append(f"{symbol} may be overbought (RSI={signals['rsi']:.1f})")
            elif signals.get('rsi', 50) < 30:
                recommendations.append(f"{symbol} may be oversold (RSI={signals['rsi']:.1f})")
                
        return recommendations

    async def run(self, message: discord.Message):
        # Extract potential stock symbols from message
        content = message.content.upper()
        words = content.split()
        
        # Check for stock symbols and quantities
        stock_data = {}
        stock_qty = {}
        ticker_only = {}
        
        for wrd in words:
            ticker_qty = self.parse_ticker_quantity(wrd)
            if ticker_qty:
                symbol, qty = ticker_qty[0]
                try:
                    price_data = await self.analyzer.get_stock_price(symbol)
                    if isinstance(price_data, dict) and 'price' in price_data:
                        stock_data[symbol] = price_data['price']
                        ticker_only[symbol] = price_data
                        self.store_ask_price(symbol, price_data['price'])
                        stock_qty[symbol] = qty
                    else:
                        await message.channel.send(f"Error: Invalid data for {symbol}")
                except Exception as e:
                    await message.channel.send(f"Error fetching data for {symbol}: {str(e)}")
            elif wrd.isalpha() and 1 <= len(wrd) <= 5:
                try:
                    price_data = await self.analyzer.get_stock_price(wrd)
                    if isinstance(price_data, dict) and 'price' in price_data:
                        ticker_only[wrd] = price_data
                        self.store_ask_price(wrd, price_data['price'])
                except Exception as e:
                    await message.channel.send(f"Error fetching data for {wrd}: {str(e)}")

        # Perform portfolio analysis if quantities provided
        portfolio_analysis = None
        if stock_qty:
            portfolio_analysis = await self.analyze_portfolio(stock_data, stock_qty)

        # Prepare context for LLM
        context = {
            'stocks': ticker_only,
            'portfolio': portfolio_analysis,
            'historical_prices': self.ask_prices if self.ask_prices else None
        }

        # Generate LLM response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{message.content}\n\nAnalysis Context: {json.dumps(context, indent=2)}"}
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        response_content = response.choices[0].message.content

        # Generate visualizations
        if stock_data:
            # Portfolio composition pie chart
            pie_chart = self.generate_portfolio_pie(stock_data, stock_qty)
            
            # Sector allocation chart if portfolio analysis exists
            sector_chart = None
            if portfolio_analysis:
                sector_chart = self.generate_sector_chart(portfolio_analysis['sector_allocation'])
            
            # Technical analysis chart for individual stocks
            tech_charts = {}
            for symbol in stock_data.keys():
                hist_data = await self.analyzer.get_historical_data(symbol)
                if not hist_data.empty:
                    chart = self.generate_technical_chart(hist_data, symbol)
                    if chart is not None:
                        tech_charts[symbol] = chart

            # Send response with all visualizations
            await self.send_long_message(message.channel, response_content)
            if pie_chart:
                await message.channel.send(file=discord.File(pie_chart, "portfolio_composition.png"))
            if sector_chart:
                await message.channel.send(file=discord.File(sector_chart, "sector_allocation.png"))
            for symbol, chart in tech_charts.items():
                await message.channel.send(file=discord.File(chart, f"{symbol}_technical.png"))
            
            return None  # Return None since we've already sent the messages
        else:
            return response_content[:1999]  # Return truncated response for Discord's limit

    def generate_sector_chart(self, sector_allocation):
        """Generate sector allocation pie chart"""
        plt.figure(figsize=(10, 7))
        sectors = list(sector_allocation.keys())
        values = list(sector_allocation.values())
        
        plt.pie(values, labels=sectors, autopct='%1.1f%%', startangle=90)
        plt.title("Sector Allocation")
        
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format="png", bbox_inches='tight')
        plt.close()
        image_buffer.seek(0)
        return image_buffer

    def generate_technical_chart(self, data, symbol):
        """Generate technical analysis chart"""
        try:
            if data.empty:
                return None
                
            if 'Close' not in data.columns:
                return None

            # Calculate technical indicators first
            # Calculate moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Create the plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), height_ratios=[3, 1, 1])
            
            # Price and Moving Averages
            ax1.plot(data.index, data['Close'], label='Price')
            ax1.plot(data.index, data['SMA_20'], label='20-day SMA')
            ax1.plot(data.index, data['SMA_50'], label='50-day SMA')
            ax1.set_title(f'{symbol} Technical Analysis')
            ax1.legend()
            ax1.grid(True)
            
            # MACD
            ax2.plot(data.index, data['MACD'], label='MACD')
            ax2.plot(data.index, data['Signal_Line'], label='Signal Line')
            ax2.legend()
            ax2.grid(True)
            
            # RSI
            ax3.plot(data.index, data['RSI'], label='RSI')
            ax3.axhline(y=70, color='r', linestyle='--')
            ax3.axhline(y=30, color='g', linestyle='--')
            ax3.legend()
            ax3.grid(True)
            
            plt.tight_layout()
            
            image_buffer = io.BytesIO()
            plt.savefig(image_buffer, format="png", bbox_inches='tight')
            plt.close()
            image_buffer.seek(0)
            return image_buffer
        except Exception as e:
            print(f"Error generating technical chart for {symbol}: {str(e)}")
            plt.close()  # Ensure figure is closed even if there's an error
            return None

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
