import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="BrandPulse AI", page_icon="📡", layout="wide")

st.markdown("""
<style>
.big-title { font-size: 2.5rem; font-weight: 800; color: #1a1a2e; }
.subtitle  { font-size: 1.1rem; color: #666; margin-bottom: 1rem; }
.card      { background: #f8f9fa; border-radius: 12px; padding: 1.5rem;
             text-align: center; border: 1px solid #e0e0e0; }
.pos { color: #2ecc71; font-size: 2rem; font-weight: 800; }
.neg { color: #e74c3c; font-size: 2rem; font-weight: 800; }
.neu { color: #f39c12; font-size: 2rem; font-weight: 800; }
.footer { text-align:center; color:#888; padding: 1.5rem 0; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

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

# ── Train model ───────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('cleaned_tweets.csv').dropna(subset=['clean_text'])
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    cm  = confusion_matrix(y_test, y_pred,
                           labels=['negative', 'neutral', 'positive'])
    return model, vectorizer, df, acc, f1, cm, y_test, y_pred

# ── Load trend data ───────────────────────────────────────────
@st.cache_data
def load_trend_data():
    raw = pd.read_csv('Tweets.csv', usecols=['tweet_created', 'airline_sentiment'])
    raw['tweet_created'] = pd.to_datetime(raw['tweet_created'], utc=True)
    raw['hour'] = raw['tweet_created'].dt.floor('h')
    trend = (
        raw.groupby(['hour', 'airline_sentiment'])
           .size()
           .reset_index(name='count')
    )
    return trend

model, vectorizer, df, acc, f1, cm, y_test, y_pred = train_model()
trend_df = load_trend_data()

# ── Session state for history ─────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="big-title">📡 BrandPulse AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time Social Media Sentiment Monitoring</p>',
            unsafe_allow_html=True)
st.divider()

# ── Tweet Input ───────────────────────────────────────────────
st.subheader("🔍 Analyze a Tweet")
tweet = st.text_area("Enter a tweet:", height=100,
                     placeholder="e.g. The flight was amazing and staff were super helpful!")

if st.button("Analyze Sentiment", type="primary"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet first!")
    else:
        cleaned    = clean_tweet(tweet)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probs      = model.predict_proba(vectorized)[0]
        classes    = model.classes_
        confidence = probs.max() * 100

        # Save to history
        emoji_map = {'positive': '😊', 'negative': '😡', 'neutral': '😐'}
        st.session_state.history.append({
            'Tweet':      tweet[:60] + '...' if len(tweet) > 60 else tweet,
            'Sentiment':  f"{emoji_map[prediction]} {prediction.capitalize()}",
            'Confidence': f"{confidence:.1f}%"
        })

        # Result card
        st.divider()
        st.subheader("📊 Prediction Result")
        col1, col2 = st.columns([1, 2])

        with col1:
            if prediction == 'positive':
                st.markdown('<div class="card"><p class="pos">😊 Positive</p></div>',
                            unsafe_allow_html=True)
            elif prediction == 'negative':
                st.markdown('<div class="card"><p class="neg">😡 Negative</p></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="card"><p class="neu">😐 Neutral</p></div>',
                            unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.1f}%")

        with col2:
            st.markdown("**Confidence breakdown:**")
            for cls, prob in zip(classes, probs):
                st.markdown(f"**{cls.capitalize()}**")
                st.progress(float(prob), text=f"{prob*100:.1f}%")

        st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Live Analytics",
    "📈 Sentiment Trend",
    "☁️ Word Clouds",
    "🤖 Model Performance",
    "⚖️ Model Comparison",
    "📋 Batch Analysis",
    "🕐 Recent Predictions"
])

# ── Tab 1: Live Analytics ─────────────────────────────────────
with tab1:
    st.subheader("Sentiment Distribution")
    counts = df['airline_sentiment'].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("😡 Negative", counts['negative'])
    col2.metric("😐 Neutral",  counts['neutral'])
    col3.metric("😊 Positive", counts['positive'])

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Bar Chart**")
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        ax.barh(['Negative', 'Neutral', 'Positive'],
                [counts['negative'], counts['neutral'], counts['positive']],
                color=colors)
        for i, v in enumerate([counts['negative'], counts['neutral'], counts['positive']]):
            ax.text(v + 50, i, str(v), va='center', fontweight='bold', fontsize=9)
        ax.set_xlabel("Number of Tweets")
        ax.set_title("Sentiment Distribution")
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with chart_col2:
        st.markdown("**Pie Chart**")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.pie(counts, labels=counts.index, autopct='%1.1f%%',
                colors=['#e74c3c', '#f39c12', '#2ecc71'], startangle=90)
        ax2.set_title("Sentiment Share")
        plt.tight_layout()
        st.pyplot(fig2)

# ── Tab 2: Sentiment Trend ────────────────────────────────────
with tab2:
    st.subheader("📈 Sentiment Trend Over Time")
    st.markdown("Hourly tweet volume by sentiment across the dataset period.")

    # Filter selector
    all_dates = trend_df['hour'].dt.date.unique()
    all_dates_sorted = sorted(all_dates)
    date_labels = [str(d) for d in all_dates_sorted]

    selected_date = st.selectbox(
        "Filter by date:",
        options=["All dates"] + date_labels
    )

    if selected_date == "All dates":
        filtered = trend_df.copy()
    else:
        filtered = trend_df[trend_df['hour'].dt.date.astype(str) == selected_date]

    # Pivot for plotting
    pivot = filtered.pivot(index='hour', columns='airline_sentiment', values='count').fillna(0)

    # Ensure all 3 columns exist
    for col in ['negative', 'neutral', 'positive']:
        if col not in pivot.columns:
            pivot[col] = 0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pivot.index, pivot['negative'], color='#e74c3c',
            linewidth=2, marker='o', markersize=3, label='😡 Negative')
    ax.plot(pivot.index, pivot['neutral'],  color='#f39c12',
            linewidth=2, marker='o', markersize=3, label='😐 Neutral')
    ax.plot(pivot.index, pivot['positive'], color='#2ecc71',
            linewidth=2, marker='o', markersize=3, label='😊 Positive')

    ax.fill_between(pivot.index, pivot['negative'], alpha=0.08, color='#e74c3c')
    ax.fill_between(pivot.index, pivot['neutral'],  alpha=0.08, color='#f39c12')
    ax.fill_between(pivot.index, pivot['positive'], alpha=0.08, color='#2ecc71')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Sentiment Trend Over Time (Hourly)")
    ax.legend(loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Peak insight
    st.markdown("#### 🔍 Key Insights")
    peak_neg_hour = pivot['negative'].idxmax()
    peak_pos_hour = pivot['positive'].idxmax()
    col1, col2 = st.columns(2)
    col1.info(f"😡 Peak negative activity: **{peak_neg_hour.strftime('%b %d, %H:%M')}**")
    col2.success(f"😊 Peak positive activity: **{peak_pos_hour.strftime('%b %d, %H:%M')}**")

# ── Tab 3: Word Clouds ────────────────────────────────────────
with tab3:
    st.subheader("☁️ Most Frequent Words by Sentiment")

    for sentiment, color, emoji in [
        ('positive', 'Greens', '😊'),
        ('negative', 'Reds',   '😡'),
        ('neutral',  'YlOrBr', '😐')
    ]:
        text = ' '.join(df[df['airline_sentiment'] == sentiment]['clean_text'].dropna())
        wc = WordCloud(width=700, height=300, background_color='white',
                       colormap=color, max_words=80).generate(text)
        st.markdown(f"**{emoji} {sentiment.capitalize()} Tweets**")
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

# ── Tab 4: Model Performance ──────────────────────────────────
with tab4:
    st.subheader("🤖 Model Performance — Logistic Regression")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc*100:.1f}%")
    col2.metric("F1 Score (weighted)", f"{f1*100:.1f}%")

    st.markdown("#### Confusion Matrix")
    col_cm, col_gap = st.columns([1, 1])
    with col_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm,
                   display_labels=['Neg', 'Neu', 'Pos'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title("Confusion Matrix — Logistic Regression", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

# ── Tab 5: Model Comparison ───────────────────────────────────
with tab5:
    st.subheader("⚖️ Classical NLP vs Deep Learning")

    comparison = pd.DataFrame({
        'Model':    ['Logistic Regression', 'Naive Bayes', 'Bi-LSTM'],
        'Accuracy': ['77.5%', '72.4%', '78.0%'],
        'F1 Score': ['76%',   '68%',   '76%'],
        'Speed':    ['Fast ⚡', 'Fastest ⚡⚡', 'Slow 🐢'],
        'Winner':   ['✅ Speed+Accuracy', '', '🏆 Best Accuracy']
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        models   = ['Logistic\nRegression', 'Naive\nBayes', 'Bi-LSTM']
        accuracy = [77.5, 72.4, 78.0]
        colors   = ['#3498db', '#95a5a6', '#e74c3c']
        bars = ax.bar(models, accuracy, color=colors, width=0.5)
        ax.set_ylim(60, 85)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")
        for bar, val in zip(bars, accuracy):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val}%', ha='center', fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    st.success("""
    🏆 **Winner: Logistic Regression** — selected based on accuracy-performance tradeoff and inference speed.
    Bi-LSTM achieves 78.0% accuracy vs 77.5% — only a 0.5% difference — but is ~10x slower to run.
    For real-time tweet prediction, Logistic Regression is the practical winner.
    """)

# ── Tab 6: Batch Analysis ─────────────────────────────────────
with tab6:
    st.subheader("📋 Batch Tweet Analysis")
    st.markdown("Paste multiple tweets below — one per line.")

    batch_input = st.text_area("Enter tweets (one per line):", height=200,
                               placeholder="Flight was amazing!\nWorst delay ever.\nFlight lands at 6pm.")

    if st.button("Analyze All", type="primary"):
        if batch_input.strip() == "":
            st.warning("Please enter at least one tweet!")
        else:
            tweets_list = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]
            results = []
            emoji_map = {'positive': '😊', 'negative': '😡', 'neutral': '😐'}

            for t in tweets_list:
                cleaned = clean_tweet(t)
                vec     = vectorizer.transform([cleaned])
                pred    = model.predict(vec)[0]
                conf    = model.predict_proba(vec).max() * 100
                results.append({
                    'Tweet':      t[:60] + '...' if len(t) > 60 else t,
                    'Sentiment':  f"{emoji_map[pred]} {pred.capitalize()}",
                    'Confidence': f"{conf:.1f}%"
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            st.markdown("#### Summary")
            sentiments = [r['Sentiment'] for r in results]
            col1, col2, col3 = st.columns(3)
            col1.metric("😡 Negative", sum('Negative' in s for s in sentiments))
            col2.metric("😐 Neutral",  sum('Neutral'  in s for s in sentiments))
            col3.metric("😊 Positive", sum('Positive' in s for s in sentiments))

            summary_counts = {
                'Negative': sum('Negative' in s for s in sentiments),
                'Neutral':  sum('Neutral'  in s for s in sentiments),
                'Positive': sum('Positive' in s for s in sentiments),
            }
            non_zero = {k: v for k, v in summary_counts.items() if v > 0}
            if non_zero:
                col_pie, col_sp = st.columns([1, 1])
                fig, ax = plt.subplots(figsize=(3, 2.5))
                color_map = {'Negative': '#e74c3c', 'Neutral': '#f39c12', 'Positive': '#2ecc71'}
                ax.pie(non_zero.values(), labels=non_zero.keys(),
                       autopct='%1.1f%%',
                       colors=[color_map[k] for k in non_zero.keys()],
                       startangle=90)
                ax.set_title("Batch Sentiment Breakdown")
                plt.tight_layout()
                with col_pie:
                    st.pyplot(fig, use_container_width=False)

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='sentiment_results.csv',
                mime='text/csv'
            )

# ── Tab 7: Recent Predictions ─────────────────────────────────
with tab7:
    st.subheader("🕐 Recent Predictions")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet — analyze a tweet above to see history here!")
    else:
        history_df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Total analyzed this session:** {len(st.session_state.history)}")

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="footer">
    <strong>📡 BrandPulse AI</strong><br>
    Twitter Sentiment Analysis using Classical NLP and Deep Learning<br>
    <small>Developed by Ethin Issac Gerald</small>
</div>
""", unsafe_allow_html=True)