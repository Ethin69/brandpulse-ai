import streamlit as st
import requests
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BrandPulse AI",
    page_icon="📡",
    layout="centered"
)

st.markdown("""
<style>
.big-title { font-size: 2.5rem; font-weight: 800; color: #1a1a2e; }
.subtitle  { font-size: 1.1rem; color: #666; margin-bottom: 1rem; }
.card      { background: #f8f9fa; border-radius: 12px; padding: 2rem;
             text-align: center; border: 1px solid #e0e0e0; margin: 1rem 0; }
.pos { color: #2ecc71; font-size: 2.5rem; font-weight: 800; }
.neg { color: #e74c3c; font-size: 2.5rem; font-weight: 800; }
.neu { color: #f39c12; font-size: 2.5rem; font-weight: 800; }
.footer { text-align:center; color:#888; padding: 1.5rem 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── API URL ───────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

# ── Session state for history ─────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="big-title">📡 BrandPulse AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time Tweet Sentiment Analysis — Powered by FastAPI</p>',
            unsafe_allow_html=True)
st.divider()

# ── Health check ──────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.status_code == 200:
        st.success("🟢 API is online and ready!")
    else:
        st.error("🔴 API is not responding correctly.")
except:
    st.error("🔴 Cannot connect to API. Make sure FastAPI is running.")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Tweet",
    "📋 Batch Analysis",
    "🕐 Recent Predictions"
])

# ── Tab 1: Single Tweet ───────────────────────────────────────
with tab1:
    st.subheader("🔍 Analyze a Tweet")
    tweet = st.text_area("Enter a tweet:", height=120,
                         placeholder="e.g. The flight was amazing and staff were super helpful!")

    if st.button("Analyze Sentiment", type="primary"):
        if tweet.strip() == "":
            st.warning("Please enter a tweet first!")
        else:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"text": tweet},
                        timeout=10
                    )

                    if response.status_code == 200:
                        result     = response.json()
                        sentiment  = result['sentiment']
                        emoji      = result['emoji']
                        confidence = result['confidence']
                        conf_float = float(confidence.replace('%', '')) / 100

                        # Save to history
                        st.session_state.history.append({
                            'Tweet':      tweet[:60] + '...' if len(tweet) > 60 else tweet,
                            'Sentiment':  f"{emoji} {sentiment.capitalize()}",
                            'Confidence': confidence
                        })

                        # Result card
                        st.divider()
                        st.subheader("📊 Result")

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            if sentiment == 'positive':
                                st.markdown(
                                    f'<div class="card"><p class="pos">{emoji} Positive</p></div>',
                                    unsafe_allow_html=True)
                            elif sentiment == 'negative':
                                st.markdown(
                                    f'<div class="card"><p class="neg">{emoji} Negative</p></div>',
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="card"><p class="neu">{emoji} Neutral</p></div>',
                                    unsafe_allow_html=True)
                            st.metric("Confidence", confidence)

                        with col2:
                            st.markdown("**Confidence Breakdown:**")
                            # Request proba breakdown by calling API
                            # We simulate 3 bars using the main confidence
                            labels = ['negative', 'neutral', 'positive']
                            for label in labels:
                                if label == sentiment:
                                    st.markdown(f"**{label.capitalize()}**")
                                    st.progress(conf_float, text=confidence)
                                else:
                                    remaining = (1 - conf_float) / 2
                                    st.markdown(f"**{label.capitalize()}**")
                                    st.progress(remaining, text=f"{remaining*100:.1f}%")

                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure FastAPI is running on port 8000.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ── Tab 2: Batch Analysis ─────────────────────────────────────
with tab2:
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

            progress = st.progress(0, text="Analyzing tweets...")

            for i, t in enumerate(tweets_list):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"text": t},
                        timeout=10
                    )
                    if response.status_code == 200:
                        r = response.json()
                        results.append({
                            'Tweet':      t[:60] + '...' if len(t) > 60 else t,
                            'Sentiment':  f"{r['emoji']} {r['sentiment'].capitalize()}",
                            'Confidence': r['confidence']
                        })
                except:
                    results.append({
                        'Tweet':      t[:60] + '...' if len(t) > 60 else t,
                        'Sentiment':  '❓ Error',
                        'Confidence': 'N/A'
                    })

                progress.progress((i + 1) / len(tweets_list),
                                  text=f"Analyzing {i+1}/{len(tweets_list)} tweets...")

            progress.empty()

            # Results table
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Summary
            st.markdown("#### Summary")
            sentiments = [r['Sentiment'] for r in results]
            col1, col2, col3 = st.columns(3)
            col1.metric("😡 Negative", sum('Negative' in s for s in sentiments))
            col2.metric("😐 Neutral",  sum('Neutral'  in s for s in sentiments))
            col3.metric("😊 Positive", sum('Positive' in s for s in sentiments))

            # Download
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='batch_sentiment_results.csv',
                mime='text/csv'
            )

# ── Tab 3: Recent Predictions ─────────────────────────────────
with tab3:
    st.subheader("🕐 Recent Predictions")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet — analyze a tweet in the Single Tweet tab!")
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
    Powered by FastAPI + Logistic Regression<br>
    <small>Developed by Ethin Issac Gerald</small>
</div>
""", unsafe_allow_html=True)