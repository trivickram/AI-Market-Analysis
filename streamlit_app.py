"""
AI-Powered Marketing Analytics Dashboard - Simplified Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="AI Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data function
@st.cache_data
def load_data(uploaded_file=None):
    """Load and preprocess marketing data"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            return df
        
        # Try sample data
        if os.path.exists("sample_data.csv"):
            df = pd.read_csv("sample_data.csv")
            return df
        
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #1f77b4;">🚀 AI-Powered Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload your marketing campaign dataset (CSV)",
        type=['csv'],
        help="Upload a CSV file with columns like Campaign_Type, ROI, Target_Audience, etc."
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None:
        st.warning("📋 **No data available!** Please upload your marketing campaign dataset.")
        st.info("Your CSV should contain columns like: Campaign_Type, ROI, Target_Audience, Channel_Used, etc.")
        
        # Show sample data format
        st.subheader("📊 Expected Data Format")
        sample_data = {
            'Campaign_Type': ['Email', 'Social Media', 'PPC'],
            'ROI': [3.2, 2.8, 4.1],
            'Target_Audience': ['Young Adults', 'Adults', 'Seniors'],
            'Channel_Used': ['Email', 'Facebook', 'Google Ads'],
            'Conversion_Rate': [5.2, 3.8, 6.1],
            'Clicks': [1500, 2200, 1800],
            'Impressions': [25000, 45000, 32000]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        return
    
    st.success(f"✅ Data loaded successfully! {len(df)} campaigns analyzed.")
    
    # Basic analytics
    st.header("📊 Quick Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'ROI' in df.columns:
            avg_roi = df['ROI'].mean()
            st.metric("Average ROI", f"{avg_roi:.2f}x")
    
    with col2:
        if 'Conversion_Rate' in df.columns:
            avg_conversion = df['Conversion_Rate'].mean()
            st.metric("Avg Conversion Rate", f"{avg_conversion:.1f}%")
    
    with col3:
        if 'Clicks' in df.columns:
            total_clicks = df['Clicks'].sum()
            st.metric("Total Clicks", f"{total_clicks:,}")
    
    with col4:
        st.metric("Total Campaigns", len(df))
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df, use_container_width=True)
    
    # Basic visualization
    if 'ROI' in df.columns and 'Campaign_Type' in df.columns:
        st.subheader("📈 ROI by Campaign Type")
        fig = px.box(df, x='Campaign_Type', y='ROI', title="ROI Distribution by Campaign Type")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    if 'Channel_Used' in df.columns and 'Conversion_Rate' in df.columns:
        st.subheader("🎯 Conversion Rate by Channel")
        fig = px.bar(df.groupby('Channel_Used')['Conversion_Rate'].mean().reset_index(), 
                     x='Channel_Used', y='Conversion_Rate', 
                     title="Average Conversion Rate by Channel")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()