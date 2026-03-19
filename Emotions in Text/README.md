# 😊 Text-Based Emotion Detection

A Natural Language Processing (NLP) project that detects emotions from text using text preprocessing, Bag of Words vectorization, and a Random Forest classifier.

---

## 📌 Overview

Emotion detection is a branch of sentiment analysis focused on extracting and classifying emotions expressed in text. This project builds a multi-class emotion classifier trained on labeled text data, capable of identifying emotions such as happiness, sadness, surprise, fear, and more.

---

## 🏷️ Emotion Categories

The dataset contains text samples labeled with the following emotions:

| Emotion | Description |
|---------|-------------|
| 😊 Happy | Joyful, positive expressions |
| 😢 Sadness | Grief, sorrow, disappointment |
| 😲 Surprise | Unexpected or shocking events |
| 😨 Fear | Anxiety, dread, worry |
| 😡 Anger | Frustration, rage |
| 🤢 Disgust | Repulsion or strong dislike |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| NLTK | Stopword removal & lemmatization |
| scikit-learn | Vectorization, model training & evaluation |
| Regex (`re`) | Text cleaning |

---

## 📁 Project Structure

```
📦 emotion-detection
 ┣ 📓 PROJECT__EMOTIONS_IN_TEXT_.ipynb   # Main notebook
 ┗ 📄 Emotion_final.csv                  # Labeled text dataset
```

---

## ⚙️ Pipeline

```
Load Data → EDA → Clean & Preprocess → Vectorize → Train/Test Split
        → Random Forest → Evaluate
```

1. **Load** — Read `Emotion_final.csv` into a Pandas DataFrame
2. **EDA** — Explore unique emotions, view sample texts, visualize class distribution
3. **Clean** — Remove duplicates and null values
4. **Preprocess** — Lowercase, remove special characters, remove stopwords, lemmatize
5. **Encode** — Convert emotion labels to integers using `LabelEncoder`
6. **Vectorize** — Transform text into feature vectors using `CountVectorizer` (Bag of Words)
7. **Split** — 70/30 train-test split
8. **Train** — Fit a `RandomForestClassifier` with 50 estimators
9. **Evaluate** — Accuracy score, confusion matrix heatmap, classification report

---

## 🔧 Text Preprocessing

The custom `preprocess()` function applies the following steps to each text entry:

- Remove special characters using regex
- Convert to lowercase
- Remove English stopwords (`nltk.corpus.stopwords`)
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
- **Confusion Matrix** — Seaborn heatmap showing per-class predictions
- **Classification Report** — Per-class precision, recall, and F1-score

---

## 💡 Applications

- 💬 Sentiment-aware chatbots and virtual assistants
- 📱 Social media emotion monitoring
- 🛒 Customer feedback and review analysis
- 🧠 Mental health and well-being tracking tools
- 📞 Call centre and support ticket prioritization

---

