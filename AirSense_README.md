# ✈️ AirSense — Airline Customer Sentiment & GenAI Insights Dashboard

## 📌 Project Overview

AirSense is an AI-powered sentiment analytics dashboard that analyses Twitter customer feedback for 6 major US airlines. Instead of relying on static reports, this project combines traditional data analytics with Google Gemini 2.5 Flash generative AI to produce real-time executive summaries, airline-specific recommendations, and natural language Q&A from sentiment data.

The solution combines Python-based analytics with an interactive Streamlit dashboard designed for airline executives and customer experience teams. This is an independent personal project built to demonstrate GenAI integration skills.

---

## 🎯 Business Objectives

- Identify which airlines have the highest and lowest customer satisfaction
- Detect the primary complaint categories driving negative sentiment
- Compare airline performance using sentiment scores and confidence metrics
- Generate AI-powered actionable recommendations using Google Gemini 2.5 Flash
- Enable natural language Q&A over airline sentiment data for non-technical users

---

## 🗂 Dataset Description

The dataset contains 14,640 tweets with the following key attributes:

- Sentiment label (positive / neutral / negative) with confidence score
- Airline name (United, US Airways, American, Southwest, Delta, Virgin America)
- Negative reason category (Customer Service Issue, Late Flight, Cancelled Flight, etc.)
- Tweet text, timestamp, retweet count
- Date range: February 17–24, 2015

Each row represents one tweet with human-annotated sentiment labels enabling granular airline performance analysis.

---

## 📊 Key Metrics

- **Overall Negative Rate** = Negative Tweets ÷ Total Tweets
- **Airline Sentiment Score** = % Positive / % Negative per airline
- **Complaint Category Share** = Reason Count ÷ Total Negative Tweets
- **AI Confidence Score** = Human annotator certainty (0–1 scale)
- **Peak Activity Hour** = Hour with maximum tweet volume (UTC)

---

## 🔑 Key Findings

- ✈️ **14,640 tweets** analysed across **6 US airlines**
- 😡 Overall negative rate: **62.69%** — nearly 4x the positive rate (16.14%)
- 🔴 **Worst airline:** US Airways at **77.7% negative** rate
- 🟢 **Best airline:** Virgin America at **35.9% negative** rate (30.2% positive)
- 📋 **#1 Complaint:** Customer Service Issues (**2,910 tweets** — top complaint for 5/6 airlines)
- 📋 **#2 Complaint:** Late Flight (**1,665 tweets**)
- 🕐 Peak complaint activity: **21:00 UTC** (US evening hours)
- 🤖 Avg AI sentiment confidence score: **90.0%**

---

## 🤖 GenAI Features — Google Gemini 2.5 Flash

| Feature | Description |
|---------|-------------|
| **Auto Executive Summary** | Click button → Gemini generates 200-word business summary of filtered data |
| **Per-Airline Recommendations** | Select airline → Gemini provides 3 specific improvement strategies |
| **Custom Q&A** | Type any question → Gemini answers using dataset context |

---

## 🖥 Streamlit Dashboard Features

### Dashboard Pages

- **Sentiment Overview** — Pie chart, negative rate by airline, daily volume trend, hourly activity
- **Airline Deep Dive** — Grouped bar chart, complaint heatmap, top reasons, confidence distribution
- **GenAI Insights** — Gemini executive summary, per-airline recommendations, custom Q&A
- **Tweet Explorer** — Keyword search, 500-row tweet table, CSV download

### Sidebar Filters

- Airline selector
- Sentiment filter (positive / neutral / negative)
- Minimum confidence score slider

---

## 📁 Repository Structure

```
airsense-airline-sentiment/
│
├── airsense_app.py              # Streamlit dashboard (main file)
├── Tweets.csv                   # Source dataset
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── notebooks/
│   └── AirSense_Analysis.ipynb  # EDA & analysis notebook
│
└── reports/
    ├── AirSense_Research_Paper.docx    # Full research report
    └── AirSense_Executive_Summary.docx # Executive summary
```

---

## ⚙️ How to Run the App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Gemini API key in airsense_app.py
# Replace: GEMINI_API_KEY = "your_key_here"
# Get free key at: https://aistudio.google.com

# 3. Run the dashboard
streamlit run airsense_app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core programming language |
| pandas | Data loading, cleaning, feature engineering |
| matplotlib | All charts and visualizations |
| seaborn | Heatmap visualization |
| Streamlit | Interactive web dashboard |
| Google Gemini 2.5 Flash | AI insights, summaries, recommendations |
| requests | REST API calls to Gemini |

---

## 🧠 Business Value

This project shifts airline customer experience management from reactive complaint handling to proactive AI-driven strategy, enabling:

- Real-time identification of which airlines need urgent intervention
- Automatic generation of targeted improvement recommendations
- Natural language access to sentiment insights for non-technical executives
- Scalable complaint categorisation and routing using generative AI

---

## 📌 Author

**Personal Project — GenAI & NLP Analytics**  
Lakshmi Mahitha Noudu | Data Analyst | Hyderabad, India | May 2026  
🔗 [LinkedIn](https://www.linkedin.com/in/lakshmi-mahitha-noudu-490160268) | [GitHub](https://github.com/yourusername)
