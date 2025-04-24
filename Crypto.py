import praw
import requests
import pandas as pd
from newsapi import NewsApiClient
import datetime
import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import tf_keras as keras

sys.stdout.reconfigure(encoding='utf-8')

### Reddit API Credentials (Replace with your own keys)
reddit = praw.Reddit(
    client_id="enter your id here",
    client_secret="enter your id here",
    user_agent="enter your id here"
)

### News API Credentials (Replace with your own key)
NEWS_API_KEY = "enter your id here"



# ===========================
# 🔹 Reddit Data Collection (Last 14 Days)
# ===========================
def fetch_reddit_posts(subreddit="Bitcoin", limit=100):
    start_timestamp = int((datetime.now(timezone.utc) - timedelta(days=14)).timestamp())  # Get 14-day old timestamp
    posts = []

    for post in reddit.subreddit(subreddit).new(limit=500):  # Increase limit to fetch more posts
        if post.created_utc >= start_timestamp:  # Only keep posts from last 14 days
            posts.append({
                "timestamp": datetime.fromtimestamp(post.created_utc, timezone.utc),
                "text": post.title, 
                "source": "Reddit"
            })

    return pd.DataFrame(posts)

# ===========================
# 🔹 News Data Collection (Last 14 Days)
# ===========================
def fetch_news(query="Bitcoin", num_articles=100):
    start_date = (datetime.now(timezone.utc) - timedelta(days=14)).strftime('%Y-%m-%d')

    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    articles = newsapi.get_everything(
        q=query,
        from_param=start_date,  # Start from 14 days ago
        language="en",
        sort_by="publishedAt",
        page_size=num_articles
    )
    
    news_data = []
    for article in articles.get("articles", []):
        news_data.append({
            "timestamp": article["publishedAt"], 
            "text": article["title"] + " - " + (article.get("description") or ""), 
            "source": "News"
        })

    return pd.DataFrame(news_data)

# ===========================
# 🔹 Blockchain Data Collection (On-Chain Sentiment)
# ===========================
def fetch_blockchain_data():
    url = "https://blockchain.info/unconfirmed-transactions?format=json"  # Free on-chain BTC data
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Error fetching blockchain data")
        return pd.DataFrame()
    
    transactions = response.json().get("txs", [])
    
    tx_data = []
    for tx in transactions[:100]:  # Limit to 100 transactions
        tx_data.append({
            # FIXED: Correct UTC timestamp conversion
            "timestamp": datetime.fromtimestamp(tx["time"], timezone.utc),  
            "value": sum(output["value"] for output in tx["out"]) / 1e8,  # BTC amount
            "source": "Blockchain"
        })
    
    return pd.DataFrame(tx_data)

# ===========================
# 🔹 Combine All Data Sources (Fix Timestamp Issue)
# ===========================
def combine_data():
    # Fetch data from all sources
    blockchain_data = fetch_blockchain_data()
    reddit_data = fetch_reddit_posts()
    news_data = fetch_news()
    
    # Concatenate all data into a single DataFrame
    all_data = pd.concat([blockchain_data, reddit_data, news_data], ignore_index=True)
    
    # Convert timestamp column to datetime format & remove timezone
    all_data['timestamp'] = pd.to_datetime(all_data['timestamp'], utc=True).dt.tz_localize(None)
    
    # Sort data by timestamp
    all_data = all_data.sort_values(by='timestamp', ascending=True)
    
    return all_data

# Run the function and save the data to a CSV file
data = combine_data()
data.to_csv("sentiment_data.csv", index=False)
print("Data saved to sentiment_data.csv")



import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

# Download VADER lexicon
nltk.download("vader_lexicon")

# Initialize sentiment analyzers
sia = SentimentIntensityAnalyzer()

# Load collected data
data = pd.read_csv("sentiment_data.csv")

# ==============================
# 🔹 Sentiment Analysis Function
# ==============================
def get_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0  # Neutral if no text available

    # VADER Sentiment Score
    vader_score = sia.polarity_scores(text)["compound"]

    # TextBlob Sentiment Score
    blob_score = TextBlob(text).sentiment.polarity  # Ranges from -1 to 1

    # Combine scores (weighted)
    final_score = (vader_score * 0.6) + (blob_score * 0.4)  

    return final_score

# Apply sentiment analysis
data["sentiment_score"] = data["text"].apply(get_sentiment)

# ==============================
# 🔹 Advanced Sentiment (Optional: BERT-based)
# ==============================
use_transformer = False  # Set to True to enable BERT sentiment scoring

if use_transformer:
    sentiment_pipeline = pipeline("sentiment-analysis")

    def get_transformer_sentiment(text):
        if not isinstance(text, str) or text.strip() == "":
            return 0

        sentiment_result = sentiment_pipeline(text[:512])  # Limit to 512 tokens
        score = sentiment_result[0]["score"]
        return score if sentiment_result[0]["label"] == "POSITIVE" else -score

    data["bert_sentiment"] = data["text"].apply(get_transformer_sentiment)
    data["final_sentiment"] = (data["sentiment_score"] * 0.5) + (data["bert_sentiment"] * 0.5)  # Weighted
else:
    data["final_sentiment"] = data["sentiment_score"]


# ==============================
# 🔹 Compute Rolling Average Sentiment
# ==============================
data["rolling_sentiment"] = data["final_sentiment"].rolling(window=10, min_periods=1).mean()

# ==============================
# 🔹 Generate Buy/Sell Signals
# ==============================
BUY_THRESHOLD = 0.7
SELL_THRESHOLD = 0.3

def generate_signal(sentiment):
    if sentiment > BUY_THRESHOLD:
        return "BUY"
    elif sentiment < SELL_THRESHOLD:
        return "SELL"
    else:
        return "HOLD"

data["trading_signal"] = data["rolling_sentiment"].apply(generate_signal)

# Save processed data
data.to_csv("sentiment_trading_data.csv", index=False)
print("Sentiment analysis completed and saved to sentiment_trading_data.csv")




import backtrader as bt
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 🔹 Fetch Bitcoin Price Data (Fix 'timestamp' Issue)
# ==============================
def fetch_bitcoin_prices():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=14&interval=hourly"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error if request fails
        
        prices = response.json().get("prices", [])
        if not prices:
            print("⚠️ No price data received from API.")
            return pd.DataFrame()

        price_data = pd.DataFrame(prices, columns=["timestamp", "close"])
        price_data["timestamp"] = pd.to_datetime(price_data["timestamp"], unit="ms")  # Convert timestamp
        return price_data

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching Bitcoin price data: {e}")
        return pd.DataFrame()

# Load sentiment trading data
data = pd.read_csv("sentiment_trading_data.csv")

# Ensure timestamp exists in sentiment data
if "timestamp" not in data.columns:
    raise KeyError("❌ Missing 'timestamp' column in sentiment_trading_data.csv")

data["timestamp"] = pd.to_datetime(data["timestamp"])  # Convert to datetime

# Fetch BTC price data and merge with sentiment data
btc_price_data = fetch_bitcoin_prices()

if btc_price_data.empty:
    raise ValueError("❌ No Bitcoin price data available. Check API response.")

# Merge with BTC price data
data = pd.merge_asof(data.sort_values("timestamp"), btc_price_data.sort_values("timestamp"), on="timestamp")

# Ensure 'close' column exists
if "close" not in data.columns:
    raise ValueError("❌ Missing 'close' column after merging with Bitcoin price data!")

# Compute rolling sentiment score (to smooth out noise)
data["rolling_sentiment"] = data["sentiment"].rolling(window=5, min_periods=1).mean()

# Set timestamp as index
data.set_index("timestamp", inplace=True)

# ==============================
# 🔹 Define Backtrader Strategy
# ==============================
class SentimentStrategy(bt.Strategy):
    params = (("buy_threshold", 0.7), ("sell_threshold", 0.3), ("risk_per_trade", 0.02))

    def __init__(self):
        self.dataclose = self.data.close
        self.sentiment = self.data.sentiment

    def next(self):
        if self.position:  # If we have an open position
            if self.sentiment[0] < self.params.sell_threshold:
                self.close()  # Sell
        elif self.sentiment[0] > self.params.buy_threshold:
            self.buy(size=self.broker.get_cash() * self.params.risk_per_trade / self.dataclose[0])  # Buy

# Convert DataFrame to Backtrader feed
class PandasData(bt.feeds.PandasData):
    lines = ("sentiment", "rolling_sentiment")
    params = (("sentiment", -1), ("rolling_sentiment", -1))

data_feed = PandasData(dataname=data)

# Run Backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(SentimentStrategy)
cerebro.adddata(data_feed)
cerebro.broker.set_cash(10000)
cerebro.run()

print("💰 Final Portfolio Value:", cerebro.broker.getvalue())

# ==============================
# 🔹 Machine Learning for Signal Improvement
# ==============================
# Feature Engineering
data["price_change"] = data["close"].pct_change().shift(-1)
data["target"] = np.where(data["price_change"] > 0, 1, 0)  # 1 = Buy, 0 = Sell

features = ["sentiment", "rolling_sentiment"]
X = data[features].dropna()
y = data["target"].dropna()

# Split data into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost Model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ==============================
# 🔹 Sharpe Ratio Optimization
# ==============================
def sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

best_threshold = None
best_sharpe = -np.inf

for threshold in np.arange(0.5, 0.9, 0.05):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SentimentStrategy, buy_threshold=threshold)
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(10000)
    results = cerebro.run()
    
    portfolio_values = np.array([x.broker.getvalue() for x in results])
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    sharpe = sharpe_ratio(portfolio_returns)

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_threshold = threshold

print("Optimized Buy Threshold:", best_threshold)
print(" Best Sharpe Ratio:", best_sharpe)





