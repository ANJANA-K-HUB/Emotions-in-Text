# 🐦 Twitter Sentiment Analysis

An NLP project that classifies tweets as **Positive**, **Negative**, or **Neutral** using TF-IDF vectorization and a Support Vector Machine (SVM) classifier.

---

## 📌 Overview

Sentiment analysis on social media data helps understand public opinion on a wide range of topics. This project builds an automated machine learning pipeline to analyze 13,870 tweets extracted via the Twitter API — cleaning noisy text, extracting meaningful features, and classifying sentiment with an SVM model.

---

## 🏷️ Sentiment Categories

| Label | Sentiment | Value |
|-------|-----------|-------|
| 😊 Positive | Favorable opinion | `1` |
| 😐 Neutral | No clear opinion | `0` |
| 😠 Negative | Unfavorable opinion | `-1` |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | EDA visualizations |
| NLTK | Stopword removal & lemmatization |
| scikit-learn | TF-IDF, SVM, train/test split, metrics |
| Regex (`re`) | Tweet text cleaning |

---

## 📁 Project Structure

```
📦 twitter-sentiment-analysis
 ┣ 📓 PROJECT__Twitter_Sentiment_Analysis_.ipynb   # Main notebook
 ┗ 📄 twitter.csv                                   # Dataset (13,870 tweets)
```

---

## ⚙️ Pipeline

```
Load Data → Select Columns → Encode Labels → Clean & Preprocess
    → TF-IDF Vectorization → Train/Test Split → SVM → Evaluate
```

1. **Load** — Read `twitter.csv` and retain only `text` and `sentiment` columns
2. **Encode** — Map `Positive → 1`, `Neutral → 0`, `Negative → -1`
3. **Clean** — Remove duplicates and null values
4. **EDA** — Countplot and pie chart of sentiment distribution
5. **Preprocess** — Remove special characters, lowercase, remove stopwords, lemmatize
6. **Vectorize** — Transform tweets using `TfidfVectorizer` with bigrams (`ngram_range=(2,2)`)
7. **Split** — 70/30 train-test split
8. **Train** — Fit a Support Vector Classifier (`SVC`)
9. **Evaluate** — Accuracy score, confusion matrix heatmap, classification report

---

## 🔧 Text Preprocessing

The `preprocess()` function cleans each tweet through the following steps:

- Remove special characters using regex
- Convert to lowercase
- Remove English stopwords
- Lemmatize words using `WordNetLemmatizer` (verb mode)

```python
def preprocess(sentence):
    text = re.sub(r"[^a-zA-Z0-9]", " ", sentence).split()
    words = [x.lower() for x in text if x not in stopwords.words('english')]
    lemma = WordNetLemmatizer()
    word = [lemma.lemmatize(word, 'v') for word in words]
    return ' '.join(word)
```

---




## 📊 Evaluation

The model is evaluated using:

- **Accuracy Score** — Overall classification accuracy on the test set
- **Confusion Matrix** — Seaborn heatmap of predicted vs actual sentiments
- **Classification Report** — Per-class precision, recall, and F1-score

---

## 💡 Applications

- 📣 Brand reputation monitoring on social media
- 🏛️ Public opinion analysis on political events
- 📈 Market sentiment tracking for financial insights
- 🎯 Targeted marketing and campaign analysis
- 🛒 Customer feedback classification for e-commerce

---

