# 📡 BrandPulse AI
### Real-Time Social Media Sentiment Monitoring

An end-to-end NLP sentiment analysis pipeline built on the Twitter US Airline Sentiment dataset — from data cleaning to a live deployed API.

## 🚀 Live Links
- 🌐 [Streamlit Dashboard](https://ethin69-brandpulse-ai.streamlit.app/)
- ⚡ [Live API](https://brandpulse-ai-ys29.onrender.com)
- 📖 [API Docs](https://brandpulse-ai-ys29.onrender.com/docs)

## 📌 Features
- Live tweet sentiment prediction with confidence score
- Hourly sentiment trend chart
- Word clouds by sentiment
- Model performance metrics & confusion matrix
- Classical vs Deep Learning model comparison
- Batch tweet analysis with CSV export
- REST API with FastAPI
- Dockerized and deployed on Render

## 🛠️ Tech Stack
- **Data:** Twitter US Airline Sentiment (14,640 tweets)
- **Classical ML:** TF-IDF + Logistic Regression (77.5%) + Naive Bayes (72.4%)
- **Deep Learning:** Bidirectional LSTM (78.0%)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Containerization:** Docker
- **Deployment:** Render (API) + Streamlit Cloud (Dashboard)

## 📁 Project Structure
brandpulse-ai/
├── app/
│   ├── init.py
│   └── main.py          ← FastAPI backend
├── model/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
├── frontend.py          ← Streamlit frontend
├── app.py               ← Full dashboard
├── Dockerfile
├── requirements.txt
├── 01_Preprocessing_and_Classical.ipynb
├── 02_Classical_Models.ipynb
└── 03_Deep_Learning_LSTM.ipynb

## 🔌 API Usage
**Health Check:**
GET https://brandpulse-ai-ys29.onrender.com/health

**Predict Sentiment:**
POST https://brandpulse-ai-ys29.onrender.com/predict
Content-Type: application/json
{
"text": "The flight was amazing!"
}

**Response:**
```json
{
  "sentiment": "positive",
  "emoji": "😊",
  "confidence": "92.4%",
  "input_text": "The flight was amazing!"
}
```

## 👨‍💻 Developed by
**Ethin Issac Gerald**
