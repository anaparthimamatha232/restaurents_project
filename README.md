# Restaurant Rating Prediction and Sentiment Analysis

A Natural Language Processing (NLP) project that classifies restaurant review sentiment (Positive/Negative) using a Bidirectional LSTM deep learning model. Includes an interactive Streamlit web app for real-time predictions.

## 📌 Project Overview

This project analyzes restaurant reviews to **classify sentiment as Positive or Negative**, directly from raw review text, using a Bidirectional LSTM neural network trained end-to-end in TensorFlow/Keras.

## 📊 Dataset

- **Source:** [10,000 Restaurant Reviews (Zomato)](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews) — Kaggle
- **Size:** 10,000 rows, 8 columns
- **Columns used:** `Review` (text), `Rating` (1-5, including half-stars)
- **Target:** Binary sentiment — `Rating > 3` → Positive, else Negative
- **After cleaning:** ~9,954 rows

## 🧠 Model & Result

| Model | Task | Accuracy |
|---|---|---|
| **Bidirectional LSTM** | **Binary sentiment classification** | **85.0%** |

**Classification report:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Positive | 0.85 | 0.91 | 0.88 |
| Negative | 0.84 | 0.76 | 0.80 |

This is the only model in the current codebase — it is trained, evaluated, saved, and deployed by the notebook and app in this repository.

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP Preprocessing | NLTK (stopwords), spaCy (`en_core_web_sm` — lemmatization) |
| Deep Learning | TensorFlow / Keras (Bidirectional LSTM) |
| Frontend | Streamlit |
| Development | Jupyter Notebook (Anaconda), VS Code |

## 📁 Project Structure

```
restaurant-review-analysis/
│
├── data/
│   └── Restaurant_reviews_csv.csv        # Raw dataset
│
├── notebook/
│   └── restaurant_analysis.ipynb         # Full pipeline: cleaning, EDA,
│                                          # preprocessing, LSTM training, evaluation
│
├── app.py                                # Streamlit frontend application
├── restaurant_sentiment_model.keras      # Saved trained LSTM model
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## ⚙️ Setup & Installation

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd restaurant-review-analysis
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn tensorflow-cpu nltk spacy streamlit
python -m spacy download en_core_web_sm
```

### 3. Download NLTK stopwords (one-time — runs automatically in the notebook too)
```python
import nltk
nltk.download('stopwords')
```

## 🚀 How to Run

### Step 1 — Train the model (Jupyter Notebook)
```bash
jupyter notebook
```
Open `notebook/restaurant_analysis.ipynb` and run all cells:
**Kernel → Restart Kernel and Run All Cells**

This will, in order:
1. Load and clean the dataset (`data/Restaurant_reviews_csv.csv`)
2. Preprocess review text (lowercase → remove punctuation → remove stopwords → lemmatize)
3. Create the binary sentiment label (`Rating > 3` → Positive)
4. Build and train the Bidirectional LSTM model
5. Evaluate on a non-shuffled validation split (confusion matrix, classification report)
6. Save the trained model as `restaurant_sentiment_model.keras`

**Expected training time:** ~15–25 minutes on CPU (5–8 epochs with early stopping). No GPU required.

### Step 2 — Run the web app (Streamlit)
Once the model file (`restaurant_sentiment_model.keras`) exists in the project folder:
```bash
streamlit run app.py
```
This opens a browser window where you can type a review and get a real-time Positive/Negative prediction with a confidence score.

**Note:** The app only loads the saved model — it does **not** require the notebook to be running. You can run `streamlit run app.py` any time, as long as `restaurant_sentiment_model.keras` is present in the same folder.

## 📈 Methodology Summary

1. **Data Cleaning** — removed the invalid `"Like"` rating entry, duplicates, and nulls; rounded half-star ratings
2. **Text Preprocessing** — lowercasing, punctuation removal, stopword removal (NLTK), lemmatization (spaCy)
3. **Target Creation** — binary label: `Rating > 3` → Positive (0), else → Negative (1)
4. **Feature Extraction** — Keras `TextVectorization` (10,000-token vocabulary) + trainable Embedding layer
5. **Model Architecture** — Embedding → 2 stacked Bidirectional LSTM layers (128, 64 units) → Dense(32, ReLU) → Dense(1)
6. **Training** — Adam optimizer, Binary Crossentropy loss, early stopping on `val_loss`
7. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix (on a **non-shuffled** validation set — see bug note below)
8. **Deployment** — model saved in native Keras format, served via Streamlit

## 🐛 Notable Engineering Challenges Solved

- **Validation misalignment bug:** the validation `tf.data.Dataset` had `.shuffle()` applied, so predictions and true labels were computed in different shuffle orders across two separate passes — producing a falsely low accuracy (~54%) instead of the real ~85%. Fixed by evaluating on a dedicated non-shuffled `val_eval` dataset.
- **Keras 3 dtype error:** `numpy.array()` on review text created an incompatible fixed-width string dtype — fixed with `dtype=object`.
- **Inverted confidence score in the app:** the sigmoid output (probability of the *Negative* class) was initially displayed as Positive-class confidence — corrected to properly map the sigmoid output to both class probabilities before display.
- **Windows GPU limitation:** TensorFlow dropped native Windows GPU/CUDA support after v2.10 — DirectML was tried but was incompatible with the Python/TensorFlow version used, and WSL2+CUDA setup was judged not worth the time investment for this dataset size. Training proceeds on CPU.

## 📝 Models Explored During Development (Removed from Current Code)

Earlier in development, three additional models were built and compared for **5-class star rating prediction** (predicting the exact 1–5 star rating rather than binary sentiment). These were later removed from the codebase in favor of the single, better-performing binary sentiment LSTM above — they are documented here only for reference/history, not as part of the current pipeline:

| Model (no longer in code) | Task | Accuracy |
|---|---|---|
| Logistic Regression (TF-IDF) | 5-class rating prediction | 62.1% |
| Naive Bayes (TF-IDF) | 5-class rating prediction | 59.0% |
| Bidirectional LSTM | 5-class rating prediction | 61.2% |

**Why they were dropped:** the 5-class rating task was significantly harder than binary sentiment, largely due to (1) more classes to separate and (2) a skewed class distribution (2-star ratings were only ~7.5% of the data). The binary sentiment LSTM alone achieves 85% accuracy, is simpler to maintain, and is the model actually deployed in `app.py` — so the codebase was simplified to that single pipeline.

## 🔮 Future Improvements

- Extend binary sentiment to 3-class (Positive/Neutral/Negative)
- Address class imbalance (63% Positive / 37% Negative) with class-weighting or resampling
- Add aspect-based sentiment analysis (food, service, ambience separately)
- Experiment with pretrained transformers (e.g., DistilBERT)
- GPU-accelerated training via WSL2/CUDA for larger-scale experimentation

## 📄 License

This project was developed as part of the NIELIT AI Development Associate certification program, for educational purposes.

## 🙏 Acknowledgements

- Dataset: [Kaggle - joebeachcapital/restaurant-reviews](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews)
- Built as part of the NIELIT Chennai PG Program in Data Engineering
- Project Guide: Mr. Mayank Singh Gautam
