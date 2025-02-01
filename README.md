# 🚀 Sentiment Analysis of Elon Musk’s Tweets & Tesla Stock Price Movements

## 📌 Project Overview
This project explores whether the **sentiment of Elon Musk’s tweets** influences **Tesla's stock price movements**. Using **Natural Language Processing (NLP)** and **machine learning**, we analyze tweet sentiment and investigate its correlation with Tesla's daily stock prices.

## 🎯 Objectives
- Perform **sentiment analysis** on Elon Musk’s tweets using the **TextBlob** library.
- Investigate the correlation between **tweet sentiment** and **Tesla stock price** fluctuations.
- Apply **machine learning models** to classify sentiment and analyze stock price trends over different **time lags**.

---

## 📂 Data Collection
This study is based on two primary datasets:

### 📄 1. Elon Musk’s Tweets
- A dataset containing **tweets from Elon Musk's Twitter account**.
- Includes **tweet text, timestamp, and metadata**.
- **Source:** [Kaggle Dataset](https://www.kaggle.com/code/gpreda/collect-elon-musk-tweets)

### 📄 2. Tesla Stock Prices
- Daily stock price data for **Tesla, Inc.**, including:
  - **Opening, Closing, High, Low prices, and Volume**.
- Collected using a **web scraper** from [GeeksforGeeks](https://www.geeksforgeeks.org/web-scraping-for-stock-prices-in-python/).

---

## ⚙️ Methodology
This project is carried out in multiple steps:

### 🛠 1. Data Preprocessing
- **Clean tweets** by removing URLs, usernames, special characters, and numbers.
- **Tokenization & Stopword Removal** to refine text data.

### 🎭 2. Sentiment Analysis
- **TextBlob** is used to classify tweets into:
  - **Positive** (1), **Neutral** (0), **Negative** (-1).

### 🤖 3. Machine Learning for Sentiment Classification
- Tweets are vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- Three machine learning models applied:
  1. **Naïve Bayes**
  2. **K-Nearest Neighbors (KNN)**
  3. **Support Vector Machine (SVM)**
- **Hyperparameter tuning** using **5-fold cross-validation**.

### 📈 4. Correlation & Time Lag Analysis
- **Correlation Analysis:** Sentiment scores vs. Tesla’s stock prices.
- **Time Lag Analysis:** Effects over different time lags (**1, 2, 3, 5, 7 days**).

---

## 📊 Results & Findings
✅ **Sentiment Analysis:**
- Majority of Musk’s tweets have a **positive sentiment**.

✅ **Machine Learning Performance:**
- **SVM achieved highest accuracy (91.04%)**.
- **Naïve Bayes (85.76%)** and **KNN (84.14%)** performed slightly lower.

✅ **Correlation Analysis:**
- **Very weak negative correlation** found between tweet sentiment and Tesla’s stock prices.

✅ **Time Lag Analysis:**
- No significant **delayed impact** of tweets on stock prices.

---

## 🏁 Conclusion
- **Elon Musk’s tweets carry distinct sentiments** but do **not significantly influence Tesla’s stock price** in the short term.
- Stock movements are driven by **other external factors** beyond Twitter sentiment.
- **Future Work:**
  - **Deep learning models** for improved analysis.
  - **Longer time lags** to detect potential delayed impacts.
  - **Incorporation of financial indicators** like trading volume.

---

## 🛠️ Tech Stack & Tools
- **Python**
- **TextBlob** (Sentiment Analysis)
- **Scikit-Learn** (Machine Learning)
- **Pandas, NumPy** (Data Processing)
- **Matplotlib, Seaborn** (Data Visualization)

---

## 📂 Repository Structure
📂 A-NLP-project-with-sentiment-analysis │── 📄 README.md # Project documentation │── 📄 elon_tweets.csv # Elon Musk's tweets dataset │── 📄 tesla_stock_prices.csv # Tesla stock prices dataset │── 📂 notebooks │ ├── 📄 data_preprocessing.ipynb # Data cleaning & processing │ ├── 📄 sentiment_analysis.ipynb # Sentiment classification │ ├── 📄 machine_learning.ipynb # Model training & evaluation │ ├── 📄 correlation_analysis.ipynb # Sentiment vs stock price analysis │── 📂 models │ ├── 📄 naive_bayes.pkl │ ├── 📄 knn.pkl │ ├── 📄 svm.pkl │── 📂 reports │ ├── 📄 results_analysis.pdf # Findings & insights │── 📄 requirements.txt # Dependencies


---

## 🔧 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ericylc23/A-NLP-project-with-sentiment-analysis.git
cd A-NLP-project-with-sentiment-analysis

📝 License
This project is open-source and available under the MIT License.

📬 Contact
For questions or collaborations, reach out via:

📧 Email: ericylc@bu.edu
🔗 LinkedIn: https://www.linkedin.com/in/eric-yuanlc/
