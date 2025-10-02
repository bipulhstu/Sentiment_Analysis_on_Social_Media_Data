# 📊 Social Media Sentiment Analysis Streamlit App

This is a comprehensive Streamlit web application for analyzing sentiment in social media posts, built from the Jupyter notebook analysis.

## 🚀 Features

- **🔮 Real-time Sentiment Prediction**: Analyze any text input for sentiment (Positive, Negative, Neutral)
- **📈 Interactive Data Overview**: Explore dataset statistics and key metrics
- **📊 Rich Visualizations**: Multiple chart types including trends, distributions, and word clouds
- **🔍 Data Explorer**: Filter and explore the dataset with interactive controls
- **📋 Model Performance**: View detailed model metrics and feature importance

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the App**:
   Open your browser and go to `http://localhost:8501`

## 📁 Required Files

Make sure these files are in the same directory as `app.py`:
- `best_model_nb.joblib` - Trained Naive Bayes model
- `best_tfidf.joblib` - TF-IDF vectorizer
- `telegram_channels_messages14021213_with_sentiment.csv` - Dataset

## 🎯 How to Use

### 1. Predict Sentiment Tab
- Enter any text in the text area
- Click "Analyze Sentiment" to get predictions
- View confidence scores and try sample texts

### 2. Data Overview Tab
- See key statistics about the dataset
- View sentiment distribution and channel breakdown
- Browse the dataset preview

### 3. Visualizations Tab
- Choose from multiple visualization options:
  - Sentiment trends over time
  - Average views by sentiment
  - Channel-wise sentiment distribution
  - Word clouds for different sentiments
  - Text length and compound score distributions

### 4. Data Explorer Tab
- Filter data by channels, sentiments, and date ranges
- Download filtered results as CSV
- Explore specific subsets of the data

### 5. Model Performance Tab
- View detailed model performance metrics
- See class-wise precision, recall, and F1-scores
- Explore most important features for each sentiment class

## 🤖 Model Information

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF with 5000 features
- **Preprocessing**: Text cleaning, stopword removal, lemmatization
- **Accuracy**: 83.21%
- **Classes**: POSITIVE, NEGATIVE, NEUTRAL

## 📊 Dataset Information

- **Source**: Telegram channels messages
- **Size**: 14,712 posts
- **Channels**: 7 different crypto/trading channels
- **Date Range**: December 2022 - March 2024
- **Features**: Text, channel, date, views, sentiment scores

## 🔧 Technical Details

The app uses:
- **Streamlit** for the web interface
- **Plotly** for interactive visualizations
- **scikit-learn** for machine learning
- **NLTK** for text preprocessing
- **Pandas** for data manipulation

## 🎨 UI Features

- Responsive design with custom CSS styling
- Color-coded sentiment results
- Interactive charts and graphs
- Tabbed navigation for organized content
- Download functionality for filtered data
- Sample text examples for quick testing

## 🚨 Troubleshooting

If you encounter issues:

1. **NLTK Data Error**: The app will automatically download required NLTK data
2. **Model Files Missing**: Ensure `best_model_nb.joblib` and `best_tfidf.joblib` are present
3. **Dataset Missing**: Ensure the CSV file is in the correct location
4. **Dependencies**: Run `pip install -r requirements.txt` to install all required packages

## 📈 Performance Tips

- The app caches models and data for faster loading
- Large datasets may take a moment to load initially
- Word cloud generation may take a few seconds for large text corpora
- Use filters in the Data Explorer to work with smaller subsets for better performance

Enjoy exploring sentiment analysis with this interactive dashboard! 🎉
