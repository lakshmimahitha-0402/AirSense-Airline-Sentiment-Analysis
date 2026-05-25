import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
import json
import warnings
warnings.filterwarnings('ignore')
 
st.set_page_config(
    page_title="AirSense — Airline Sentiment Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #1a1d24; border-right: 1px solid #2d3139; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="metric-container"] {
        background-color: #1a1d24; border: 1px solid #2d3139;
        border-radius: 8px; padding: 16px;
    }
    [data-testid="metric-container"] label { color: #9ca3af !important; font-size: 13px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 26px !important; font-weight: 700 !important;
    }
    h1, h2, h3 { color: #ffffff !important; }
    hr { border-color: #2d3139 !important; }
    .section-label {
        font-size: 15px; font-weight: 700; color: #ffffff;
        margin: 20px 0 10px 0; padding-bottom: 6px;
        border-bottom: 1px solid #2d3139;
    }
    .page-title { font-size: 36px; font-weight: 800; color: #ffffff; margin-bottom: 4px; }
    .page-subtitle { font-size: 14px; color: #9ca3af; margin-bottom: 20px; }
    .ai-box {
        background: linear-gradient(135deg, #1a1d24, #0e1117);
        border: 1px solid #c9a84c; border-radius: 12px;
        padding: 20px; margin: 10px 0;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
# ── Gemini API ────────────────────────────────────────────
try:
    GEMINI_API_KEY = st.secrets["AIzaSyCdDeDDKZBzcUbYafbfEyFAbmF2_4Z5QHY"]
except:
    GEMINI_API_KEY = os.getenv("AIzaSyCdDeDDKZBzcUbYafbfEyFAbmF2_4Z5QHY")
 
def ask_gemini(prompt):
    """Call Gemini 2.5 Flash via REST API"""
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024}
        }
        response = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif response.status_code == 429:
            return "⚠️ Rate limit reached. Please wait 1 minute and try again."
        elif response.status_code == 403:
            return "⚠️ API key invalid. Please check at aistudio.google.com"
        elif response.status_code == 400:
            err = response.json().get("error", {}).get("message", "Unknown")
            return f"⚠️ Bad request: {err}"
        else:
            return f"⚠️ API Error {response.status_code}: {response.text[:200]}"
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Check your internet connection."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
 
# ── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Tweets.csv")
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], utc=True)
    df['date']  = df['tweet_created'].dt.date
    df['hour']  = df['tweet_created'].dt.hour
    df['negativereason'] = df['negativereason'].fillna('N/A')
    return df
 
df = load_data()
 
# ── Dark chart style ──────────────────────────────────────
def dark_style():
    plt.rcParams.update({
        "figure.facecolor":"#1a1d24","axes.facecolor":"#1a1d24",
        "axes.edgecolor":"#2d3139","axes.labelcolor":"#9ca3af",
        "axes.titlecolor":"#ffffff","xtick.color":"#9ca3af",
        "ytick.color":"#9ca3af","text.color":"#ffffff",
        "grid.color":"#2d3139","grid.linestyle":"--",
        "grid.alpha":0.5,"axes.grid":True,
        "font.size":11,"axes.titlesize":13,"axes.labelsize":11,
    })
dark_style()
 
def insight(text):
    st.info(f"💡 **Insight:** {text}")
 
def recommendation(text):
    st.success(f"✅ **Recommendation:** {text}")
 
# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px 0;'>
        <div style='font-size:24px; font-weight:900; letter-spacing:3px; color:#ffffff;'>
            AIR<span style='color:#c9a84c;'>SENSE</span>
        </div>
        <div style='font-size:10px; color:#c9a84c; letter-spacing:2px; margin-top:2px;'>
            AIRLINE SENTIMENT INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
 
    st.markdown("**✈️ Airline**")
    airlines_list = ["All"] + sorted(df['airline'].unique().tolist())
    selected_airline = st.selectbox("Select Airline", airlines_list)
 
    st.markdown("**😊 Sentiment**")
    sentiments_list = ["All", "negative", "neutral", "positive"]
    selected_sentiment = st.selectbox("Select Sentiment", sentiments_list)
 
    st.markdown("**⭐ Confidence**")
    conf_min = st.slider("Min Confidence Score", 0.0, 1.0, 0.0, 0.05)
 
    st.markdown("---")
    st.markdown("**📊 Dashboard Module**")
    page = st.radio("", options=[
        "Sentiment Overview",
        "Airline Deep Dive",
        "GenAI Insights",
        "Tweet Explorer",
    ])
    st.markdown("---")
    total_tweets = len(df)
    st.markdown(f"<div style='font-size:11px;color:#9ca3af;'>Total Tweets: {total_tweets:,}</div>",
                unsafe_allow_html=True)
 
# ── Apply Filters ─────────────────────────────────────────
fdf = df.copy()
if selected_airline   != "All": fdf = fdf[fdf['airline'] == selected_airline]
if selected_sentiment != "All": fdf = fdf[fdf['airline_sentiment'] == selected_sentiment]
fdf = fdf[fdf['airline_sentiment_confidence'] >= conf_min]
 
if len(fdf) == 0:
    st.warning("⚠️ No tweets match current filters. Please adjust your selections.")
    st.stop()
 
churn_neg  = (fdf['airline_sentiment'] == 'negative').mean() * 100
churn_pos  = (fdf['airline_sentiment'] == 'positive').mean() * 100
churn_neu  = (fdf['airline_sentiment'] == 'neutral').mean() * 100
avg_conf   = fdf['airline_sentiment_confidence'].mean()
 
# ── LOGO HEADER ───────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:10px 0 20px 0;'>
    <div style='font-size:40px; font-weight:900; letter-spacing:6px; color:#ffffff;'>
        AIR<span style='color:#c9a84c;'>SENSE</span>
    </div>
    <div style='font-size:12px; color:#c9a84c; letter-spacing:4px; margin-top:4px;'>
        AIRLINE CUSTOMER SENTIMENT & GENAI INSIGHTS DASHBOARD
    </div>
</div>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════
# PAGE 1 — SENTIMENT OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Sentiment Overview":
    st.markdown('<div class="page-title">Sentiment Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Overall sentiment distribution across all airlines and tweets</div>', unsafe_allow_html=True)
 
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🐦 Total Tweets",   f"{len(fdf):,}")
    c2.metric("😡 Negative",       f"{churn_neg:.1f}%")
    c3.metric("😐 Neutral",        f"{churn_neu:.1f}%")
    c4.metric("😊 Positive",       f"{churn_pos:.1f}%")
    c5.metric("⭐ Avg Confidence", f"{avg_conf:.2f}")
    st.markdown("---")
 
    col_l, col_r = st.columns(2)
 
    with col_l:
        st.markdown('<div class="section-label">🍩 Overall Sentiment Distribution</div>', unsafe_allow_html=True)
        neg_c = (fdf['airline_sentiment']=='negative').sum()
        neu_c = (fdf['airline_sentiment']=='neutral').sum()
        pos_c = (fdf['airline_sentiment']=='positive').sum()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(
            [neg_c, neu_c, pos_c],
            labels=[f'Negative\n({neg_c:,})', f'Neutral\n({neu_c:,})', f'Positive\n({pos_c:,})'],
            colors=['#e74c3c','#9ca3af','#2ecc71'],
            autopct='%1.1f%%', startangle=90,
            textprops={'color':'white','fontsize':11},
            wedgeprops={'edgecolor':'#1a1d24','linewidth':2}
        )
        ax.set_title('Sentiment Distribution', color='#ffffff', fontsize=14, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
    with col_r:
        st.markdown('<div class="section-label">✈️ Sentiment % by Airline</div>', unsafe_allow_html=True)
        airline_sent = fdf.groupby(['airline','airline_sentiment']).size().unstack(fill_value=0)
        airline_sent['total'] = airline_sent.sum(axis=1)
        for s in ['negative','neutral','positive']:
            if s not in airline_sent.columns:
                airline_sent[s] = 0
        airline_sent['neg_pct'] = (airline_sent['negative'] / airline_sent['total'] * 100).round(1)
        airline_sent['pos_pct'] = (airline_sent['positive'] / airline_sent['total'] * 100).round(1)
        airline_sent = airline_sent.sort_values('neg_pct', ascending=False)
 
        x = np.arange(len(airline_sent))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(x - 0.2, airline_sent['neg_pct'], 0.35, color='#e74c3c', label='Negative %', edgecolor='#1a1d24')
        ax.bar(x + 0.2, airline_sent['pos_pct'], 0.35, color='#2ecc71', label='Positive %', edgecolor='#1a1d24')
        ax.set_xticks(x)
        ax.set_xticklabels(airline_sent.index, rotation=15, fontsize=10)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Negative vs Positive % by Airline', color='#ffffff', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        for i, (neg, pos) in enumerate(zip(airline_sent['neg_pct'], airline_sent['pos_pct'])):
            ax.text(i-0.2, neg+0.5, f'{neg:.0f}%', ha='center', fontsize=9, color='#ffffff', fontweight='bold')
            ax.text(i+0.2, pos+0.5, f'{pos:.0f}%', ha='center', fontsize=9, color='#ffffff', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
    insight(f"US Airways has the highest negative sentiment at 77.7% while Virgin America is the best performer at 35.9% negative. The overall dataset shows {churn_neg:.1f}% negative sentiment — airlines have significant customer satisfaction challenges.")
    recommendation("Airlines with negative sentiment above 60% (US Airways, United, American) should immediately audit their customer service, flight punctuality, and baggage handling processes.")
 
    st.markdown("---")
    col_l2, col_r2 = st.columns(2)
 
    with col_l2:
        st.markdown('<div class="section-label">😤 Top Negative Complaint Reasons</div>', unsafe_allow_html=True)
        neg_df = fdf[fdf['airline_sentiment'] == 'negative']
        neg_reasons = neg_df['negativereason'].value_counts().head(8)
        neg_reasons = neg_reasons[neg_reasons.index != 'N/A']
        colors_r = ['#e74c3c','#e67e22','#f39c12','#c0392b','#e74c3c','#e67e22','#f39c12','#c0392b']
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(neg_reasons.index, neg_reasons.values,
            color=colors_r[:len(neg_reasons)], edgecolor='#1a1d24', height=0.6)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Tweets')
        ax.set_title('Top Negative Complaint Reasons', color='#ffffff', fontsize=13, fontweight='bold')
        for bar, val in zip(bars, neg_reasons.values):
            ax.text(val+10, bar.get_y()+bar.get_height()/2,
                f'{val:,}', va='center', fontsize=10, color='#ffffff', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight("Customer Service Issues (2,910 tweets) is the #1 complaint — nearly double Late Flights (1,665). This shows operational improvements alone won't fix sentiment — service quality is the root problem.")
 
    with col_r2:
        st.markdown('<div class="section-label">📅 Daily Tweet Volume</div>', unsafe_allow_html=True)
        daily = fdf.groupby('date').size()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(len(daily)), daily.values, marker='o', color='#3498db',
            linewidth=2.5, markersize=7)
        ax.fill_between(range(len(daily)), daily.values, alpha=0.2, color='#3498db')
        ax.set_xticks(range(len(daily)))
        ax.set_xticklabels([str(d) for d in daily.index], rotation=45, fontsize=9)
        ax.set_ylabel('Number of Tweets')
        ax.set_title('Daily Tweet Volume (Feb 2015)', color='#ffffff', fontsize=13, fontweight='bold')
        for i, val in enumerate(daily.values):
            ax.text(i, val+10, f'{val}', ha='center', fontsize=8, color='#c9a84c', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
# ══════════════════════════════════════════════════════════
# PAGE 2 — AIRLINE DEEP DIVE
# ══════════════════════════════════════════════════════════
elif page == "Airline Deep Dive":
    st.markdown('<div class="page-title">Airline Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Detailed performance analysis per airline</div>', unsafe_allow_html=True)
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🐦 Tweets Analysed",  f"{len(fdf):,}")
    c2.metric("😡 Negative Rate",    f"{churn_neg:.1f}%")
    c3.metric("😊 Positive Rate",    f"{churn_pos:.1f}%")
    c4.metric("⭐ Avg Confidence",   f"{avg_conf:.2f}")
    st.markdown("---")
 
    col_l, col_r = st.columns(2)
 
    with col_l:
        st.markdown('<div class="section-label">📊 Stacked Tweet Count by Airline</div>', unsafe_allow_html=True)
        airline_counts = fdf.groupby(['airline','airline_sentiment']).size().unstack(fill_value=0)
        for s in ['negative','neutral','positive']:
            if s not in airline_counts.columns:
                airline_counts[s] = 0
        airline_counts['total'] = airline_counts.sum(axis=1)
        airline_counts = airline_counts.sort_values('total', ascending=False)
        x = np.arange(len(airline_counts))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x, airline_counts['negative'], color='#e74c3c', edgecolor='#1a1d24', label='Negative')
        ax.bar(x, airline_counts['neutral'], bottom=airline_counts['negative'],
            color='#9ca3af', edgecolor='#1a1d24', label='Neutral')
        ax.bar(x, airline_counts['positive'],
            bottom=airline_counts['negative']+airline_counts['neutral'],
            color='#2ecc71', edgecolor='#1a1d24', label='Positive')
        ax.set_xticks(x)
        ax.set_xticklabels(airline_counts.index, rotation=15, fontsize=11)
        ax.set_ylabel('Number of Tweets')
        ax.set_title('Tweet Volume by Airline & Sentiment', color='#ffffff', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        for i, total in enumerate(airline_counts['total']):
            ax.text(i, total+30, f'{total:,}', ha='center', fontsize=9, color='#c9a84c', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
    with col_r:
        st.markdown('<div class="section-label">⭐ Confidence Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_conf = {'negative':'#e74c3c','neutral':'#9ca3af','positive':'#2ecc71'}
        for sent, color in colors_conf.items():
            subset = fdf[fdf['airline_sentiment']==sent]['airline_sentiment_confidence']
            if len(subset) > 0:
                ax.hist(subset, bins=20, alpha=0.7, color=color,
                    label=sent.capitalize(), edgecolor='#1a1d24')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Number of Tweets')
        ax.set_title('Sentiment Confidence Distribution', color='#ffffff', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()
 
    st.markdown("---")
 
    # Per-airline scorecard
    st.markdown('<div class="section-label">🏆 Airline Performance Scorecard</div>', unsafe_allow_html=True)
    airline_score = fdf.groupby('airline').agg(
        Total_Tweets  = ('tweet_id', 'count'),
        Negative_Pct  = ('airline_sentiment', lambda x: (x=='negative').mean()*100),
        Positive_Pct  = ('airline_sentiment', lambda x: (x=='positive').mean()*100),
        Neutral_Pct   = ('airline_sentiment', lambda x: (x=='neutral').mean()*100),
        Avg_Confidence= ('airline_sentiment_confidence', 'mean'),
    ).round(2).sort_values('Negative_Pct')
 
    airline_score['Sentiment Score'] = (100 - airline_score['Negative_Pct']).round(1)
    airline_score['Rank'] = range(1, len(airline_score)+1)
    st.dataframe(airline_score, use_container_width=True)
 
    best  = airline_score.index[0]
    worst = airline_score.index[-1]
    insight(f"🥇 Best airline: {best} (Sentiment Score: {airline_score.loc[best,'Sentiment Score']}) | "
            f"❌ Worst airline: {worst} (Sentiment Score: {airline_score.loc[worst,'Sentiment Score']})")
    recommendation(f"{worst} should study {best}'s customer service model. "
                   "Key difference: Virgin America focuses on fewer routes with higher service quality vs "
                   "US Airways' high-volume, low-satisfaction approach.")
 
# ══════════════════════════════════════════════════════════
# PAGE 3 — GENAI INSIGHTS
# ══════════════════════════════════════════════════════════
elif page == "GenAI Insights":
    st.markdown('<div class="page-title">🤖 GenAI Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Powered by Google Gemini 1.5 Flash — AI-generated analysis of airline sentiment data</div>', unsafe_allow_html=True)
 
    # Prepare data summary for Gemini
    neg_reasons = df[df['airline_sentiment']=='negative']['negativereason'].value_counts().head(5)
    airline_neg = df.groupby('airline')['airline_sentiment'].apply(
        lambda x: (x=='negative').mean()*100).round(1).sort_values(ascending=False)
 
    data_summary = f"""
    Airline Twitter Sentiment Dataset Summary:
    - Total tweets: {len(df):,}
    - Negative: {(df['airline_sentiment']=='negative').sum():,} ({(df['airline_sentiment']=='negative').mean()*100:.1f}%)
    - Neutral: {(df['airline_sentiment']=='neutral').sum():,} ({(df['airline_sentiment']=='neutral').mean()*100:.1f}%)
    - Positive: {(df['airline_sentiment']=='positive').sum():,} ({(df['airline_sentiment']=='positive').mean()*100:.1f}%)
    - Airlines: {', '.join(df['airline'].unique())}
    - Airline negative rates: {airline_neg.to_dict()}
    - Top complaint reasons: {neg_reasons.to_dict()}
    - Date range: Feb 16-24, 2015
    """
 
    st.markdown("---")
 
    # Auto insights section
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown('<div class="section-label">📋 Executive Summary</div>', unsafe_allow_html=True)
        if st.button("🤖 Generate Executive Summary", key="exec_sum"):
            with st.spinner("Gemini is analysing the data..."):
                prompt = f"""You are a data analyst. Based on this airline Twitter sentiment data, write a professional executive summary in 4-5 sentences covering: overall sentiment health, worst and best performing airlines, top complaints, and business impact. Be specific with numbers.
 
{data_summary}"""
                result = ask_gemini(prompt)
                st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)
 
    with col2:
        st.markdown('<div class="section-label">🎯 Strategic Recommendations</div>', unsafe_allow_html=True)
        if st.button("🤖 Generate Recommendations", key="recs"):
            with st.spinner("Gemini is thinking..."):
                prompt = f"""You are a business strategy consultant. Based on this airline sentiment data, provide 5 specific, actionable recommendations for airline management to improve customer sentiment. Number each recommendation. Be concise and specific.
 
{data_summary}"""
                result = ask_gemini(prompt)
                st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)
 
    st.markdown("---")
 
    col3, col4 = st.columns(2)
 
    with col3:
        st.markdown('<div class="section-label">✈️ Airline-Specific Analysis</div>', unsafe_allow_html=True)
        selected_for_ai = st.selectbox("Choose an airline to analyse:",
            sorted(df['airline'].unique().tolist()))
        if st.button(f"🤖 Analyse {selected_for_ai}", key="airline_ai"):
            airline_data = df[df['airline']==selected_for_ai]
            airline_neg_reasons = airline_data[airline_data['airline_sentiment']=='negative']['negativereason'].value_counts().head(5)
            airline_neg_pct = (airline_data['airline_sentiment']=='negative').mean()*100
            airline_pos_pct = (airline_data['airline_sentiment']=='positive').mean()*100
            with st.spinner(f"Analysing {selected_for_ai}..."):
                prompt = f"""Analyse {selected_for_ai} airline performance based on this Twitter data:
- Total tweets: {len(airline_data):,}
- Negative: {airline_neg_pct:.1f}%
- Positive: {airline_pos_pct:.1f}%
- Top complaints: {airline_neg_reasons.to_dict()}
 
Provide: 1) Key strengths, 2) Critical weaknesses, 3) Three specific improvement actions. Be concise."""
                result = ask_gemini(prompt)
                st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)
 
    with col4:
        st.markdown('<div class="section-label">❓ Ask Gemini Anything</div>', unsafe_allow_html=True)
        user_question = st.text_area(
            "Ask a question about the airline sentiment data:",
            placeholder="e.g. Which airline should I choose for best experience?\nWhy do customers hate US Airways?\nWhat time of day has most complaints?",
            height=100
        )
        if st.button("🤖 Ask Gemini", key="custom_q"):
            if user_question.strip():
                with st.spinner("Gemini is answering..."):
                    prompt = f"""You are a data analyst expert. Answer this question using the airline Twitter sentiment dataset below. Be specific, insightful, and use the actual numbers from the data.
 
Question: {user_question}
 
Data Context:
{data_summary}"""
                    result = ask_gemini(prompt)
                    st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please type a question first!")
 
    st.markdown("---")
    st.markdown('<div class="section-label">🔍 Complaint Pattern Analysis</div>', unsafe_allow_html=True)
    if st.button("🤖 Analyse Complaint Patterns by Airline", key="complaint_pattern"):
        airline_complaint_data = df[df['airline_sentiment']=='negative'].groupby(
            ['airline','negativereason']).size().unstack(fill_value=0).to_dict()
        with st.spinner("Analysing complaint patterns..."):
            prompt = f"""Analyse complaint patterns across airlines from this Twitter data. 
Top complaints overall: {neg_reasons.to_dict()}
Airline negative rates: {airline_neg.to_dict()}
 
Identify: 1) Which complaints are industry-wide vs airline-specific, 2) Which airline has the most unique complaint pattern, 3) The single most impactful issue to fix across the industry. Use bullet points."""
            result = ask_gemini(prompt)
            st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════
# PAGE 4 — TWEET EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "Tweet Explorer":
    st.markdown('<div class="page-title">Tweet Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Search, filter and explore individual tweets</div>', unsafe_allow_html=True)
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🐦 Filtered Tweets",  f"{len(fdf):,}")
    c2.metric("😡 Negative",         f"{(fdf['airline_sentiment']=='negative').sum():,}")
    c3.metric("😊 Positive",         f"{(fdf['airline_sentiment']=='positive').sum():,}")
    c4.metric("⭐ Avg Confidence",   f"{avg_conf:.2f}")
    st.markdown("---")
 
    # Keyword search
    st.markdown('<div class="section-label">🔍 Keyword Search</div>', unsafe_allow_html=True)
    keyword = st.text_input("Search in tweet text:", placeholder="e.g. delay, luggage, service, cancel...")
    if keyword.strip():
        fdf = fdf[fdf['text'].str.contains(keyword, case=False, na=False)]
        st.info(f"Found **{len(fdf):,}** tweets containing '{keyword}'")
 
    st.markdown("---")
    st.markdown('<div class="section-label">📋 Tweet Details</div>', unsafe_allow_html=True)
    st.markdown(f"Showing **{min(500, len(fdf)):,}** of **{len(fdf):,}** tweets")
 
    display_cols = ['airline', 'airline_sentiment', 'airline_sentiment_confidence',
                    'negativereason', 'text', 'retweet_count', 'date']
    display_df = fdf[display_cols].head(500).copy()
    display_df.columns = ['Airline', 'Sentiment', 'Confidence',
                          'Negative Reason', 'Tweet Text', 'Retweets', 'Date']
 
    st.dataframe(display_df, use_container_width=True, height=400)
 
    csv = fdf.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Filtered Tweets as CSV",
        csv, "filtered_tweets.csv", "text/csv"
    )
 
    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-label">🔁 Most Retweeted Positive Tweets</div>', unsafe_allow_html=True)
        top_pos = fdf[fdf['airline_sentiment']=='positive'].nlargest(5, 'retweet_count')[['airline','text','retweet_count']]
        for _, row in top_pos.iterrows():
            st.markdown(f"""
            <div style='background:#1a2e1a; border-left:3px solid #2ecc71;
            padding:10px 14px; border-radius:6px; margin:6px 0; font-size:13px;'>
            ✈️ <b>{row['airline']}</b> · 🔁 {row['retweet_count']} retweets<br>
            {row['text']}
            </div>""", unsafe_allow_html=True)
 
    with col_r:
        st.markdown('<div class="section-label">🔁 Most Retweeted Negative Tweets</div>', unsafe_allow_html=True)
        top_neg = fdf[fdf['airline_sentiment']=='negative'].nlargest(5, 'retweet_count')[['airline','text','retweet_count']]
        for _, row in top_neg.iterrows():
            st.markdown(f"""
            <div style='background:#2e1a1a; border-left:3px solid #e74c3c;
            padding:10px 14px; border-radius:6px; margin:6px 0; font-size:13px;'>
            ✈️ <b>{row['airline']}</b> · 🔁 {row['retweet_count']} retweets<br>
            {row['text']}
            </div>""", unsafe_allow_html=True)
 
# ── FOOTER ────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("✈️ **AirSense** — Airline Sentiment & GenAI Insights Dashboard")
with col2:
    st.caption("Built by [Lakshmi Mahitha Noudu](https://www.linkedin.com/in/lakshmi-mahitha-noudu-490160268)")
with col3:
    st.caption("🤖 Powered by Google Gemini 2.5 Flash  |  Version 1.0  |  May 2026")
