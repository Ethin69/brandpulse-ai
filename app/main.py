import joblib
import re
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Load model & vectorizer ───────────────────────────────────
model      = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# ── Clean function ────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="BrandPulse AI", description="Tweet Sentiment Analysis API")

# ── Input schema ──────────────────────────────────────────────
class TweetInput(BaseModel):
    text: str

# ── Health endpoint ───────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "OK"}

# ── Predict endpoint ──────────────────────────────────────────
@app.post("/predict")
def predict(data: TweetInput):
    cleaned    = clean_tweet(data.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probs      = model.predict_proba(vectorized)[0]
    confidence = float(probs.max() * 100)

    emoji_map = {'positive': '😊', 'negative': '😡', 'neutral': '😐'}

    return {
        "sentiment":   prediction,
        "emoji":       emoji_map[prediction],
        "confidence":  f"{confidence:.1f}%",
        "input_text":  data.text
    }