# twitter_sentiment_dashboard_enhanced.py

import streamlit as st
import pandas as pd
import random
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🐦"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1 {
        color: #1DA1F2;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Step 1: Simulate dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    positive_tweets = [
        "I absolutely love the new electric vehicles! They're the future.",
        "AI is transforming industries in incredible ways!",
        "The match was fantastic, what a win!",
        "Renewable energy is making great progress every year.",
        "I'm so happy to see technology helping the environment.",
        "Tesla's new model looks amazing and super efficient.",
        "Innovation in clean energy gives me hope for the future.",
        "Healthcare is improving so much thanks to AI.",
        "Electric bikes are so convenient and eco-friendly.",
        "It's a great time to invest in green technology.",
    ]

    negative_tweets = [
        "The battery life on these EVs is disappointing.",
        "Terrible experience with the latest update.",
        "The stock market is crashing again, awful times!",
        "I'm frustrated with how expensive electric cars are.",
        "Charging stations are still too few and unreliable.",
        "Customer support was useless, wasted my time.",
        "Pollution levels are rising again, so sad.",
        "The match result was unfair and frustrating.",
        "This movie was such a disappointment.",
        "The company's decision was terrible for employees.",
    ]

    neutral_tweets = [
        "Electric vehicles are becoming more common.",
        "The news today was quite informative.",
        "I read an article about AI and automation.",
        "The government announced a new EV policy.",
        "It looks like rain later in the evening.",
        "I might upgrade my phone next month.",
        "Just finished lunch, back to work now.",
        "Many companies are investing in green energy.",
        "Traffic seems lighter today than usual.",
        "The weather forecast looks normal this week.",
    ]

    all_tweets = []
    for _ in range(250):
        category = random.choice(["positive", "negative", "neutral"])
        if category == "positive":
            text = random.choice(positive_tweets)
        elif category == "negative":
            text = random.choice(negative_tweets)
        else:
            text = random.choice(neutral_tweets)
        all_tweets.append(text)

    df = pd.DataFrame({
        "date": pd.date_range("2025-09-01", periods=len(all_tweets), freq="D").tolist(),
        "user": [f"user{i}" for i in range(len(all_tweets))],
        "tweet": all_tweets
    })
    
    return df

df = load_data()

# -------------------------------------------------
# Step 2: Clean tweets
# -------------------------------------------------
def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

df["clean_tweet"] = df["tweet"].apply(clean_tweet)

# -------------------------------------------------
# Step 3: Sentiment analysis
# -------------------------------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def get_polarity_score(text):
    return TextBlob(text).sentiment.polarity

df["sentiment"] = df["clean_tweet"].apply(get_sentiment)
df["polarity_score"] = df["clean_tweet"].apply(get_polarity_score)

# -------------------------------------------------
# Header
# -------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🐦 Twitter Sentiment Analysis Dashboard")
    st.markdown("**Real-time insights** from simulated Twitter data with advanced analytics")
with col2:
    st.markdown(f"### 📅 {datetime.now().strftime('%B %d, %Y')}")
    st.markdown(f"**Total Tweets:** {len(df)}")

st.markdown("---")

# -------------------------------------------------
# Sidebar Filters
# -------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/twitter--v1.png", width=80)
st.sidebar.title("🎯 Filter Options")

selected_sentiment = st.sidebar.multiselect(
    "Select Sentiment Type(s):",
    options=df["sentiment"].unique(),
    default=df["sentiment"].unique()
)

date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(df["date"].min(), df["date"].max()),
    min_value=df["date"].min(),
    max_value=df["date"].max()
)

# Search functionality
search_term = st.sidebar.text_input("🔍 Search in tweets:", "")

# Apply filters
filtered_df = df[df["sentiment"].isin(selected_sentiment)]
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df["date"] >= pd.Timestamp(date_range[0])) & 
                               (filtered_df["date"] <= pd.Timestamp(date_range[1]))]
if search_term:
    filtered_df = filtered_df[filtered_df["tweet"].str.contains(search_term, case=False, na=False)]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Showing **{len(filtered_df)}** of **{len(df)}** tweets")

# -------------------------------------------------
# Summary Metrics with Enhanced Design
# -------------------------------------------------
st.subheader("📊 Key Metrics Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Tweets",
        value=len(filtered_df),
        delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None
    )

with col2:
    positive_count = len(filtered_df[filtered_df["sentiment"] == "Positive"])
    positive_pct = (positive_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric(
        label="😊 Positive",
        value=positive_count,
        delta=f"{positive_pct:.1f}%"
    )

with col3:
    negative_count = len(filtered_df[filtered_df["sentiment"] == "Negative"])
    negative_pct = (negative_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric(
        label="😞 Negative",
        value=negative_count,
        delta=f"-{negative_pct:.1f}%",
        delta_color="inverse"
    )

with col4:
    neutral_count = len(filtered_df[filtered_df["sentiment"] == "Neutral"])
    neutral_pct = (neutral_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric(
        label="😐 Neutral",
        value=neutral_count,
        delta=f"{neutral_pct:.1f}%"
    )

st.markdown("---")

# -------------------------------------------------
# Tabs for organized content
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Analytics", "☁️ Word Cloud", "📊 Timeline", "📋 Data Table"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        if len(filtered_df) > 0:
            sentiment_counts = filtered_df["sentiment"].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={
                    "Positive": "#10b981",
                    "Negative": "#ef4444",
                    "Neutral": "#6b7280"
                },
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to display")
    
    with col2:
        st.subheader("Sentiment Count")
        if len(filtered_df) > 0:
            fig = px.bar(
                filtered_df["sentiment"].value_counts().reset_index(),
                x="sentiment",
                y="count",
                color="sentiment",
                color_discrete_map={
                    "Positive": "#10b981",
                    "Negative": "#ef4444",
                    "Neutral": "#6b7280"
                },
                text="count"
            )
            fig.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Count",
                showlegend=False,
                height=400
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to display")
    
    # NEW FEATURE: Polarity Score Distribution
    st.subheader("📊 Polarity Score Distribution")
    if len(filtered_df) > 0:
        fig = px.histogram(
            filtered_df,
            x="polarity_score",
            color="sentiment",
            nbins=30,
            color_discrete_map={
                "Positive": "#10b981",
                "Negative": "#ef4444",
                "Neutral": "#6b7280"
            },
            labels={"polarity_score": "Polarity Score", "count": "Frequency"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display")

with tab2:
    st.subheader("☁️ Word Cloud of Tweets")
    col1, col2 = st.columns([2, 1])
    
    with col2:
        wordcloud_sentiment = st.selectbox(
            "Select sentiment for word cloud:",
            ["All"] + list(df["sentiment"].unique())
        )
    
    with col1:
        if wordcloud_sentiment == "All":
            wc_df = filtered_df
        else:
            wc_df = filtered_df[filtered_df["sentiment"] == wordcloud_sentiment]
        
        all_words = " ".join(wc_df["clean_tweet"])
        if all_words.strip():
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color="white",
                colormap="viridis",
                max_words=100
            ).generate(all_words)
            
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No words to display. Try adjusting filters.")

with tab3:
    st.subheader("📊 Sentiment Timeline")
    
    # NEW FEATURE: Timeline Analysis
    if len(filtered_df) > 0:
        timeline_df = filtered_df.groupby([pd.Grouper(key="date", freq="W"), "sentiment"]).size().reset_index(name="count")
        
        fig = px.line(
            timeline_df,
            x="date",
            y="count",
            color="sentiment",
            color_discrete_map={
                "Positive": "#10b981",
                "Negative": "#ef4444",
                "Neutral": "#6b7280"
            },
            markers=True
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Tweet Count",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily sentiment trend
        st.subheader("Daily Average Polarity Score")
        daily_polarity = filtered_df.groupby("date")["polarity_score"].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_polarity["date"],
            y=daily_polarity["polarity_score"],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=6)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Polarity Score",
            height=400,
            hovermode='x'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display")

with tab4:
    st.subheader("🧾 Tweet Data")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sort_by = st.selectbox("Sort by:", ["date", "sentiment", "polarity_score"])
    with col2:
        sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
    with col3:
        rows_to_show = st.number_input("Rows to display:", min_value=5, max_value=len(filtered_df), value=min(50, len(filtered_df)))
    
    # Sort data
    ascending = True if sort_order == "Ascending" else False
    display_df = filtered_df.sort_values(by=sort_by, ascending=ascending).head(rows_to_show)
    
    # Color code sentiment
    def highlight_sentiment(row):
        if row["sentiment"] == "Positive":
            return ['background-color: #d1fae5'] * len(row)
        elif row["sentiment"] == "Negative":
            return ['background-color: #fee2e2'] * len(row)
        else:
            return ['background-color: #f3f4f6'] * len(row)
    
    styled_df = display_df[["date", "user", "tweet", "sentiment", "polarity_score"]].style.apply(highlight_sentiment, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download button
    csv = filtered_df[["date", "user", "tweet", "sentiment", "polarity_score"]].to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"twitter_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown(
        "<div style='text-align: center; color: #6b7280;'>Made with ❤️ using Streamlit</div>",
        unsafe_allow_html=True
    )