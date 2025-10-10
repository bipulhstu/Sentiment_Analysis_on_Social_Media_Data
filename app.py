import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Load models and vectorizer
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('best_model_lr.joblib')
        svm_model = joblib.load('best_model_svm.joblib')
        tfidf_vectorizer = joblib.load('best_tfidf.joblib')
        return lr_model, svm_model, tfidf_vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Prediction function
def predict_sentiment(text, model, vectorizer):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    
    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities)
    else:
        confidence = None
    
    return prediction, confidence

# Streamlit app
def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
    
    st.title("💬 Social Media Sentiment Analysis")
    st.markdown("---")
    
    # Load models
    lr_model, svm_model, tfidf_vectorizer = load_models()
    
    if lr_model is None or svm_model is None or tfidf_vectorizer is None:
        st.error("Failed to load models. Please check if model files exist.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction"])
    
    with tab1:
        st.header("Single Text Sentiment Analysis")
        
        # Text input
        user_input = st.text_area("Enter social media text for sentiment analysis:", 
                                  height=150,
                                  placeholder="Type or paste your social media text here...")
        
        # Model selection
        model_choice = st.radio("Select Model:", ("Logistic Regression", "SVM"))
        
        # Prediction button
        if st.button("Analyze Sentiment", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    # Select model based on user choice
                    if model_choice == "Logistic Regression":
                        model = lr_model
                    else:
                        model = svm_model
                    
                    # Make prediction
                    prediction, confidence = predict_sentiment(user_input, model, tfidf_vectorizer)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Map prediction to sentiment label
                    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
                    sentiment_label = sentiment_labels.get(prediction, "Unknown")
                    
                    # Display sentiment with appropriate color
                    if sentiment_label == "Positive":
                        st.success(f"**Sentiment:** {sentiment_label}")
                    elif sentiment_label == "Negative":
                        st.error(f"**Sentiment:** {sentiment_label}")
                    else:
                        st.info(f"**Sentiment:** {sentiment_label}")
                    
                    if confidence is not None:
                        st.write(f"**Confidence:** {confidence:.2%}")
                    
                    # Show processed text
                    with st.expander("See processed text"):
                        processed = preprocess_text(user_input)
                        st.write(processed)
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Sentiment Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file with social media texts", 
                                         type=["csv"],
                                         help="CSV file should have a 'text' column")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column.")
                    return
                
                st.write(f"Uploaded file contains {len(df)} rows.")
                st.dataframe(df.head())
                
                # Model selection for batch processing
                batch_model_choice = st.radio("Select Model for Batch Processing:", 
                                              ("Logistic Regression", "SVM"))
                
                if st.button("Process All Texts", type="primary"):
                    with st.spinner("Processing all texts..."):
                        # Select model
                        if batch_model_choice == "Logistic Regression":
                            model = lr_model
                        else:
                            model = svm_model
                        
                        # Process all texts
                        results = []
                        for text in df['text']:
                            if pd.isna(text):
                                results.append(("Unknown", 0))
                            else:
                                pred, conf = predict_sentiment(str(text), model, tfidf_vectorizer)
                                sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
                                sentiment_label = sentiment_labels.get(pred, "Unknown")
                                results.append((sentiment_label, conf if conf else 0))
                        
                        # Add results to dataframe
                        df['predicted_sentiment'] = [result[0] for result in results]
                        df['confidence'] = [result[1] for result in results]
                        
                        # Display results
                        st.subheader("Batch Processing Results")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                        # Show sentiment distribution
                        st.subheader("Sentiment Distribution")
                        sentiment_counts = df['predicted_sentiment'].value_counts()
                        st.bar_chart(sentiment_counts)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Information section
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses pre-trained machine learning models to analyze the sentiment of social media texts.
    
    **Models Available:**
    - Logistic Regression
    - Support Vector Machine (SVM)
    
    **Sentiment Classes:**
    - Positive
    - Neutral
    - Negative
    """)
    
    st.sidebar.header("How to Use")
    st.sidebar.markdown("""
    1. **Single Prediction**: Enter text manually and get instant sentiment analysis
    2. **Batch Prediction**: Upload a CSV file with multiple texts for bulk analysis
    
    The app will automatically preprocess your text by:
    - Removing URLs and special characters
    - Converting to lowercase
    - Removing stopwords
    - Lemmatizing words
    """)

if __name__ == "__main__":
    main()