# Configuration file for the marketing analytics dashboard

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App Configuration
APP_TITLE = "AI-Powered Marketing Analytics Dashboard"
APP_ICON = "📊"
LAYOUT = "wide"

# Data Configuration
DATA_PATH = "data/marketing_campaign_dataset.csv"
PROCESSED_DATA_PATH = "data/processed_data.pkl"

# Model Configuration
MODEL_CONFIG = {
    'roi_model': {
        'n_estimators': 100,
        'random_state': 42
    },
    'classification_model': {
        'n_estimators': 100,
        'random_state': 42
    },
    'clustering': {
        'n_clusters': 4,
        'random_state': 42
    }
}

# API Keys (stored in .env file)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Dashboard Colors
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17becf'
}

# Page Configuration
PAGES = {
    'overview': '🏠 Overview',
    'campaign_performance': '📈 Campaign Performance', 
    'customer_insights': '👥 Customer Insights',
    'ai_predictions': '🤖 AI Predictions',
    'ai_assistant': '💬 AI Assistant'
}