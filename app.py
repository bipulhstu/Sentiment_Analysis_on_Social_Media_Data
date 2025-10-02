import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set page config
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load the trained model and TF-IDF vectorizer"""
    try:
        model = joblib.load('best_model_nb.joblib')
        tfidf = joblib.load('best_tfidf.joblib')
        return model, tfidf
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'best_model_nb.joblib' and 'best_tfidf.joblib' are in the current directory.")
        return None, None

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('telegram_channels_messages14021213_with_sentiment.csv')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'telegram_channels_messages14021213_with_sentiment.csv' is in the current directory.")
        return None

# Text preprocessing functions
def clean_text(text):
    """Clean the input text"""
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|@\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        return text
    else:
        return ""

def preprocess_text(text):
    """Preprocess text for model prediction"""
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

def predict_sentiment(text, model, tfidf):
    """Predict sentiment for given text"""
    if model is None or tfidf is None:
        return "Model not loaded", 0.0
    
    cleaned_text = clean_text(text)
    processed_text = preprocess_text(cleaned_text)
    
    if not processed_text.strip():
        return "NEUTRAL", 0.5
    
    vectorized_text = tfidf.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text).max()
    
    return prediction, probability

# Main app
def main():
    st.markdown('<h1 class="main-header">📊 Social Media Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load models and data
    model, tfidf = load_models()
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Navigation")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predict Sentiment", 
        "📈 Data Overview", 
        "📊 Visualizations", 
        "🔍 Data Explorer",
        "📋 Model Performance"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🔮 Sentiment Prediction</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Enter text to analyze:")
            user_input = st.text_area(
                "Type your message here...",
                height=150,
                placeholder="Enter social media post, review, or any text to analyze its sentiment..."
            )
            
            if st.button("🚀 Analyze Sentiment", type="primary"):
                if user_input.strip():
                    with st.spinner("Analyzing sentiment..."):
                        prediction, confidence = predict_sentiment(user_input, model, tfidf)
                        
                        # Display results
                        st.markdown("### 📊 Analysis Results:")
                        
                        # Sentiment result with color coding
                        if prediction == "POSITIVE":
                            st.success(f"**Sentiment: {prediction}** ✅")
                        elif prediction == "NEGATIVE":
                            st.error(f"**Sentiment: {prediction}** ❌")
                        else:
                            st.info(f"**Sentiment: {prediction}** ⚪")
                        
                        st.metric("Confidence Score", f"{confidence:.2%}")
                        
                        # Progress bar for confidence
                        st.progress(confidence)
                        
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            st.markdown("### 💡 Tips for better analysis:")
            st.info("""
            - Use complete sentences
            - Include context when possible
            - Avoid excessive special characters
            - The model works best with English text
            """)
            
            # Sample texts
            st.markdown("### 🎯 Try these examples:")
            sample_texts = [
                "I absolutely love this product! It exceeded my expectations.",
                "This is terrible. Worst experience ever.",
                "The weather is okay today, nothing special.",
                "Bitcoin market cap surpasses 1.3 trillion",
                "Crypto market crashed today, losing billions"
            ]
            
            for i, sample in enumerate(sample_texts):
                if st.button(f"Example {i+1}", key=f"sample_{i}"):
                    st.session_state.sample_text = sample
            
            if 'sample_text' in st.session_state:
                st.text_area("Selected example:", value=st.session_state.sample_text, key="example_display")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📈 Data Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Posts", f"{len(df):,}")
        
        with col2:
            positive_pct = (df['sentiment_type'] == 'POSITIVE').mean() * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")
        
        with col3:
            negative_pct = (df['sentiment_type'] == 'NEGATIVE').mean() * 100
            st.metric("Negative %", f"{negative_pct:.1f}%")
        
        with col4:
            neutral_pct = (df['sentiment_type'] == 'NEUTRAL').mean() * 100
            st.metric("Neutral %", f"{neutral_pct:.1f}%")
        
        st.markdown("---")
        
        # Dataset preview
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(
            df[['channel', 'text', 'date', 'views', 'sentiment_type', 'compound']].head(10),
            use_container_width=True
        )
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Sentiment Distribution")
            sentiment_counts = df['sentiment_type'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Distribution of Sentiment Types",
                color_discrete_map={
                    'POSITIVE': '#2ecc71',
                    'NEGATIVE': '#e74c3c',
                    'NEUTRAL': '#95a5a6'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📺 Posts by Channel")
            channel_counts = df['channel'].value_counts()
            fig = px.bar(
                x=channel_counts.values,
                y=channel_counts.index,
                orientation='h',
                title="Number of Posts by Channel"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">📊 Data Visualizations</h2>', unsafe_allow_html=True)
        
        # Visualization options
        viz_option = st.selectbox(
            "Choose a visualization:",
            [
                "Sentiment Trends Over Time",
                "Average Views by Sentiment",
                "Sentiment Distribution by Channel",
                "Word Cloud",
                "Text Length Distribution",
                "Compound Score Distribution"
            ]
        )
        
        if viz_option == "Sentiment Trends Over Time":
            st.markdown("### 📈 Sentiment Trends Over Time")
            
            # Group by date and sentiment
            daily_sentiment = df.groupby([df['date'].dt.date, 'sentiment_type']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            
            colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
            for sentiment in daily_sentiment.columns:
                fig.add_trace(go.Scatter(
                    x=daily_sentiment.index,
                    y=daily_sentiment[sentiment],
                    mode='lines+markers',
                    name=sentiment,
                    line=dict(color=colors.get(sentiment, '#1f77b4'))
                ))
            
            fig.update_layout(
                title="Daily Sentiment Trends",
                xaxis_title="Date",
                yaxis_title="Number of Posts",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Average Views by Sentiment":
            st.markdown("### 👀 Average Views by Sentiment")
            
            avg_views = df.groupby('sentiment_type')['views'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=avg_views.index,
                y=avg_views.values,
                title="Average Views per Sentiment Type",
                color=avg_views.index,
                color_discrete_map={
                    'POSITIVE': '#2ecc71',
                    'NEGATIVE': '#e74c3c',
                    'NEUTRAL': '#95a5a6'
                }
            )
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Sentiment Distribution by Channel":
            st.markdown("### 📺 Sentiment Distribution by Channel")
            
            channel_sentiment = df.groupby(['channel', 'sentiment_type']).size().unstack(fill_value=0)
            
            fig = px.bar(
                channel_sentiment,
                title="Sentiment Distribution by Channel",
                color_discrete_map={
                    'POSITIVE': '#2ecc71',
                    'NEGATIVE': '#e74c3c',
                    'NEUTRAL': '#95a5a6'
                }
            )
            
            fig.update_layout(
                xaxis_title="Channel",
                yaxis_title="Number of Posts",
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Word Cloud":
            st.markdown("### ☁️ Word Cloud")
            
            sentiment_filter = st.selectbox(
                "Select sentiment for word cloud:",
                ["All", "POSITIVE", "NEGATIVE", "NEUTRAL"]
            )
            
            if sentiment_filter == "All":
                text_data = df['text'].fillna('').astype(str)
            else:
                text_data = df[df['sentiment_type'] == sentiment_filter]['text'].fillna('').astype(str)
            
            if len(text_data) > 0:
                # Clean and combine text
                all_text = ' '.join([clean_text(text) for text in text_data])
                
                if all_text.strip():
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='viridis'
                    ).generate(all_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.warning("No text data available for the selected sentiment.")
            else:
                st.warning("No data available for the selected sentiment.")
        
        elif viz_option == "Text Length Distribution":
            st.markdown("### 📏 Text Length Distribution")
            
            df['text_length'] = df['text'].fillna('').astype(str).apply(lambda x: len(x.split()))
            
            fig = px.histogram(
                df,
                x='text_length',
                nbins=50,
                title="Distribution of Text Length (Number of Words)",
                color_discrete_sequence=['#3498db']
            )
            
            fig.update_layout(
                xaxis_title="Text Length (Words)",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Compound Score Distribution":
            st.markdown("### 📊 Compound Score Distribution")
            
            fig = px.histogram(
                df,
                x='compound',
                nbins=50,
                title="Distribution of Sentiment Polarity (Compound Score)",
                color_discrete_sequence=['#9b59b6']
            )
            
            fig.update_layout(
                xaxis_title="Compound Score",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            channel_filter = st.multiselect(
                "Select Channels:",
                options=df['channel'].unique(),
                default=df['channel'].unique()
            )
        
        with col2:
            sentiment_filter = st.multiselect(
                "Select Sentiments:",
                options=df['sentiment_type'].unique(),
                default=df['sentiment_type'].unique()
            )
        
        with col3:
            date_range = st.date_input(
                "Select Date Range:",
                value=(df['date'].min().date(), df['date'].max().date()),
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date()
            )
        
        # Apply filters
        filtered_df = df[
            (df['channel'].isin(channel_filter)) &
            (df['sentiment_type'].isin(sentiment_filter)) &
            (df['date'].dt.date >= date_range[0]) &
            (df['date'].dt.date <= date_range[1])
        ]
        
        st.markdown(f"### 📊 Filtered Results ({len(filtered_df):,} posts)")
        
        # Display filtered data
        st.dataframe(
            filtered_df[['channel', 'text', 'date', 'views', 'sentiment_type', 'compound']],
            use_container_width=True
        )
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_sentiment_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab5:
        st.markdown('<h2 class="sub-header">📋 Model Performance</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🤖 Model Information
        
        **Model Type:** Multinomial Naive Bayes with TF-IDF Vectorization
        
        **Features:**
        - Text preprocessing with stopword removal and lemmatization
        - TF-IDF vectorization with up to 5000 features
        - Hyperparameter tuning for optimal performance
        
        **Performance Metrics:**
        - **Accuracy:** 83.21%
        - **Precision (Macro Avg):** 80%
        - **Recall (Macro Avg):** 78%
        - **F1-Score (Macro Avg):** 79%
        
        ### 📊 Class-wise Performance:
        """)
        
        # Performance metrics table
        performance_data = {
            'Sentiment': ['NEGATIVE', 'NEUTRAL', 'POSITIVE'],
            'Precision': [0.63, 0.90, 0.86],
            'Recall': [0.72, 0.73, 0.90],
            'F1-Score': [0.67, 0.81, 0.88],
            'Support': [408, 771, 1764]
        }
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Model comparison info
        st.markdown("""
        ### 🏆 Model Comparison Results:
        
        During development, multiple models were tested:
        
        1. **Multinomial Naive Bayes:** 83.21% accuracy
        2. **Logistic Regression:** 94.26% accuracy (best performing)
        3. **Support Vector Machine:** 93.54% accuracy
        
        *Note: The current deployed model is Multinomial Naive Bayes for faster inference.*
        """)
        
        # Feature importance (top words)
        if model is not None and tfidf is not None:
            st.markdown("### 🔤 Most Important Features (Words)")
            
            try:
                feature_names = tfidf.get_feature_names_out()
                
                # Get feature log probabilities for each class
                feature_log_prob = model.feature_log_prob_
                
                # Create a dataframe for visualization
                classes = model.classes_
                top_features = {}
                
                for i, class_name in enumerate(classes):
                    top_indices = np.argsort(feature_log_prob[i])[-10:][::-1]
                    top_features[class_name] = [feature_names[idx] for idx in top_indices]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🔴 Negative Features:**")
                    for word in top_features.get('NEGATIVE', []):
                        st.write(f"• {word}")
                
                with col2:
                    st.markdown("**⚪ Neutral Features:**")
                    for word in top_features.get('NEUTRAL', []):
                        st.write(f"• {word}")
                
                with col3:
                    st.markdown("**🟢 Positive Features:**")
                    for word in top_features.get('POSITIVE', []):
                        st.write(f"• {word}")
                        
            except Exception as e:
                st.warning("Could not display feature importance.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Social Media Sentiment Analysis Dashboard | Built with Streamlit</p>
        <p>🔬 Powered by Machine Learning & Natural Language Processing</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
