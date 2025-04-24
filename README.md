
# 🧠 CryptoSentimentTrader 📊

A smart crypto trading pipeline that combines **social sentiment**, **news analysis**, **on-chain metrics**, and **machine learning** to generate and backtest trading signals for Bitcoin.

## 🚀 Features

- 🔎 **Multi-source Data Collection**:
  - Reddit posts (`r/Bitcoin`)
  - News articles (via NewsAPI)
  - Blockchain on-chain transaction data

- 📈 **Sentiment Analysis**:
  - VADER and TextBlob sentiment scores
  - Optional BERT-based transformer sentiment
  - Rolling sentiment smoothing

- 💡 **Trading Signal Generation**:
  - Simple rules based on sentiment thresholds
  - Buy/Hold/Sell labeling

- 🧪 **Backtesting with Backtrader**:
  - Strategy simulation on real-time BTC prices
  - Final portfolio value calculation

- 🤖 **Machine Learning Classifier**:
  - XGBoost model trained on historical sentiment/price data
  - Accuracy and classification reporting

- ⚙️ **Sharpe Ratio Optimization**:
  - Automated threshold tuning for maximum risk-adjusted returns

## 📂 File Structure

```
Crypto.py                # Main script
sentiment_data.csv       # Intermediate file with collected raw sentiment data
sentiment_trading_data.csv # Processed file with sentiment + signals
```

## 🛠️ Setup & Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/CryptoSentimentTrader.git
   cd CryptoSentimentTrader
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   praw
   requests
   pandas
   newsapi-python
   nltk
   textblob
   transformers
   backtrader
   xgboost
   scikit-learn
   tf-keras
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download("vader_lexicon")
   ```

4. Replace the following with **your own API keys**:
   - `client_id`, `client_secret` in the Reddit section
   - `NEWS_API_KEY` for the NewsAPI

## 📊 Example Output

```
✅ Accuracy: 0.78
📈 Best Sharpe Ratio: 1.42
🚀 Optimized Buy Threshold: 0.75
💰 Final Portfolio Value: $11,326.52
```

## 📌 Notes

- **API rate limits** may affect large data pulls.
- **Transformer sentiment** is optional but can be enabled for deeper NLP insight.
- Model assumes short-term prediction—do not use for long-term financial advice.

## 🤝 Contribution

Feel free to fork, improve, or contribute PRs. Suggestions welcome!

## 📜 License

MIT License
