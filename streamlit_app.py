"""
AI-Powered Marketing Analytics Dashboard
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import custom modules
from src.data_preprocessing import DataPreprocessor, preprocess_marketing_data
from src.eda_visualizations import MarketingVisualizer, create_dashboard_plots
from src.ml_models import MarketingMLModels, train_all_models
from src.ai_assistant import MarketingAIAssistant, create_ai_assistant
import config

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load and preprocess marketing data"""
    try:
        # Check if processed data exists
        if os.path.exists(config.PROCESSED_DATA_PATH):
            preprocessor = DataPreprocessor()
            df = preprocessor.load_preprocessed_data(config.PROCESSED_DATA_PATH)
            if df is not None:
                return df, preprocessor
        
        # If not, process the raw data
        if os.path.exists(config.DATA_PATH):
            df, preprocessor = preprocess_marketing_data(config.DATA_PATH, config.PROCESSED_DATA_PATH)
            return df, preprocessor
        else:
            st.error(f"Data file not found: {config.DATA_PATH}")
            return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load and cache ML models
@st.cache_resource
def load_models(df):
    """Load or train ML models"""
    model_path = "models/trained_models.pkl"
    ml_models = MarketingMLModels()
    
    if os.path.exists(model_path):
        if ml_models.load_models(model_path):
            return ml_models
    
    # Train models if not found
    with st.spinner("Training AI models... This may take a few minutes."):
        ml_models = train_all_models(df)
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        ml_models.save_models(model_path)
    
    return ml_models

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI-Powered Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df, preprocessor = load_data()
    if df is None:
        st.error("Unable to load data. Please check your data file.")
        return
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Choose a page", list(config.PAGES.values()))
    
    # Load ML models
    ml_models = load_models(df)
    
    # Page routing
    if page == config.PAGES['overview']:
        show_overview_page(df)
    elif page == config.PAGES['campaign_performance']:
        show_campaign_performance_page(df)
    elif page == config.PAGES['customer_insights']:
        show_customer_insights_page(df)
    elif page == config.PAGES['ai_predictions']:
        show_ai_predictions_page(df, ml_models, preprocessor)
    elif page == config.PAGES['ai_assistant']:
        show_ai_assistant_page(df)

def show_overview_page(df):
    """Display overview page with key metrics and insights"""
    st.header("🏠 Executive Overview")
    
    # Create visualizer
    viz = MarketingVisualizer(df)
    metrics = viz.create_overview_metrics()
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Total Campaigns",
            value=f"{metrics['Total Campaigns']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="💰 Average ROI",
            value=f"{metrics['Avg ROI']:.2f}x",
            delta=f"{(metrics['Avg ROI'] - 3.0):.2f}" if metrics['Avg ROI'] > 3.0 else None
        )
    
    with col3:
        st.metric(
            label="🎯 Avg Conversion Rate",
            value=f"{metrics['Avg Conversion Rate']:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="💸 Total Spend",
            value=f"${metrics['Total Spend']:,.0f}",
            delta=None
        )
    
    with col5:
        st.metric(
            label="📈 Avg Engagement",
            value=f"{metrics['Avg Engagement Score']:.1f}",
            delta=None
        )
    
    # Quick insights
    st.subheader("💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Best performing campaign
        best_campaign = df.loc[df['ROI'].idxmax()]
        st.markdown(f"""
        <div class="insight-box">
            <h4>🏆 Top Performer</h4>
            <p><strong>{best_campaign['Campaign_Type']}</strong> campaign achieved <strong>{best_campaign['ROI']:.2f}x ROI</strong></p>
            <p>Target: {best_campaign['Target_Audience']} | Channel: {best_campaign['Channel_Used']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Channel performance
        channel_perf = df.groupby('Channel_Used')['ROI'].mean().sort_values(ascending=False)
        best_channel = channel_perf.index[0]
        st.markdown(f"""
        <div class="insight-box">
            <h4>🎯 Best Channel</h4>
            <p><strong>{best_channel}</strong> delivers <strong>{channel_perf.iloc[0]:.2f}x</strong> average ROI</p>
            <p>Consistently outperforms other channels</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Overview charts
    st.subheader("📊 Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI distribution
        fig_roi = viz.plot_roi_by_campaign_type()
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col2:
        # Channel effectiveness
        fig_channel = viz.plot_channel_effectiveness()
        st.plotly_chart(fig_channel, use_container_width=True)

def show_campaign_performance_page(df):
    """Display campaign performance analysis"""
    st.header("📈 Campaign Performance Analysis")
    
    viz = MarketingVisualizer(df)
    
    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        campaign_types = ['All'] + list(df['Campaign_Type'].unique())
        selected_campaign = st.selectbox("Campaign Type", campaign_types)
    
    with col2:
        channels = ['All'] + list(df['Channel_Used'].unique())
        selected_channel = st.selectbox("Channel", channels)
    
    with col3:
        segments = ['All'] + list(df['Customer_Segment'].unique())
        selected_segment = st.selectbox("Customer Segment", segments)
    
    # Filter data
    filtered_df = df.copy()
    if selected_campaign != 'All':
        filtered_df = filtered_df[filtered_df['Campaign_Type'] == selected_campaign]
    if selected_channel != 'All':
        filtered_df = filtered_df[filtered_df['Channel_Used'] == selected_channel]
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['Customer_Segment'] == selected_segment]
    
    # Update visualizer with filtered data
    viz_filtered = MarketingVisualizer(filtered_df)
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_roi = filtered_df['ROI'].mean()
        st.metric("Average ROI", f"{avg_roi:.2f}x")
    
    with col2:
        avg_conversion = filtered_df['Conversion_Rate'].mean()
        st.metric("Average Conversion Rate", f"{avg_conversion:.2f}%")
    
    with col3:
        total_clicks = filtered_df['Clicks'].sum()
        st.metric("Total Clicks", f"{total_clicks:,}")
    
    with col4:
        avg_ctr = filtered_df['CTR'].mean() if 'CTR' in filtered_df.columns else 0
        st.metric("Average CTR", f"{avg_ctr:.2f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_conversion = viz_filtered.plot_conversion_rate_distribution()
        st.plotly_chart(fig_conversion, use_container_width=True)
    
    with col2:
        fig_clicks = viz_filtered.plot_clicks_vs_impressions()
        st.plotly_chart(fig_clicks, use_container_width=True)
    
    # Geographic performance
    st.subheader("🌍 Geographic Performance")
    fig_geo = viz_filtered.plot_geographic_performance()
    st.plotly_chart(fig_geo, use_container_width=True)
    
    # Campaign funnel
    fig_funnel = viz_filtered.create_campaign_funnel()
    if fig_funnel:
        st.subheader("📊 Campaign Funnel")
        st.plotly_chart(fig_funnel, use_container_width=True)

def show_customer_insights_page(df):
    """Display customer segmentation and insights"""
    st.header("👥 Customer Insights & Segmentation")
    
    viz = MarketingVisualizer(df)
    
    # Customer segment distribution
    st.subheader("📊 Customer Segment Distribution")
    fig_segments = viz.plot_customer_segment_distribution()
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # Segment performance analysis
    if 'Customer_Cluster' in df.columns:
        st.subheader("🎯 AI-Generated Customer Segments")
        fig_cluster_perf = viz.plot_customer_segment_performance()
        if fig_cluster_perf:
            st.plotly_chart(fig_cluster_perf, use_container_width=True)
        
        # Segment characteristics
        st.subheader("📋 Segment Characteristics")
        segment_analysis = df.groupby('Customer_Cluster').agg({
            'Conversion_Rate': 'mean',
            'ROI': 'mean',
            'Engagement_Score': 'mean',
            'CTR': 'mean',
            'Cost_Per_Click': 'mean'
        }).round(3)
        
        st.dataframe(segment_analysis, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Feature Correlations")
    fig_corr = viz.plot_correlation_heatmap()
    st.plotly_chart(fig_corr, use_container_width=True)

def show_ai_predictions_page(df, ml_models, preprocessor):
    """Display AI prediction interface"""
    st.header("🤖 AI Predictions & Recommendations")
    
    # ROI Prediction
    st.subheader("💰 ROI Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Campaign Configuration**")
        duration = st.slider("Campaign Duration (days)", 7, 90, 30)
        budget = st.number_input("Budget ($)", min_value=1000, max_value=1000000, value=50000, step=1000)
        
        campaign_type = st.selectbox("Campaign Type", df['Campaign_Type'].unique())
        target_audience = st.selectbox("Target Audience", df['Target_Audience'].unique())
        channel = st.selectbox("Channel", df['Channel_Used'].unique())
        location = st.selectbox("Location", df['Location'].unique())
        language = st.selectbox("Language", df['Language'].unique())
        segment = st.selectbox("Customer Segment", df['Customer_Segment'].unique())
    
    with col2:
        if st.button("🎯 Predict ROI"):
            try:
                # Encode categorical inputs using the preprocessor's encoders
                campaign_enc = preprocessor.encoders['Campaign_Type'].transform([campaign_type])[0]
                audience_enc = preprocessor.encoders['Target_Audience'].transform([target_audience])[0]
                channel_enc = preprocessor.encoders['Channel_Used'].transform([channel])[0]
                location_enc = preprocessor.encoders['Location'].transform([location])[0]
                language_enc = preprocessor.encoders['Language'].transform([language])[0]
                segment_enc = preprocessor.encoders['Customer_Segment'].transform([segment])[0]
                
                predicted_roi = ml_models.predict_roi(
                    duration, budget, campaign_enc, audience_enc, 
                    channel_enc, location_enc, language_enc, segment_enc
                )
                
                success_prob = ml_models.predict_campaign_success(
                    duration, budget, campaign_enc, audience_enc, 
                    channel_enc, location_enc, language_enc, segment_enc
                )
                
                # Display predictions
                st.success(f"**Predicted ROI: {predicted_roi:.2f}x**")
                st.info(f"**Success Probability: {success_prob:.1%}**")
                st.write(f"**Expected Return: ${budget * predicted_roi:,.0f}**")
                
                # ROI interpretation
                if predicted_roi > 4:
                    st.balloons()
                    st.success("🚀 Excellent ROI potential!")
                elif predicted_roi > 2:
                    st.success("✅ Good ROI potential")
                elif predicted_roi > 1:
                    st.warning("⚠️ Moderate ROI - consider optimization")
                else:
                    st.error("❌ Low ROI predicted - reconsider strategy")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    # Budget Allocation Recommendations
    st.subheader("💼 Budget Allocation Recommendations")
    
    total_budget = st.number_input("Total Marketing Budget ($)", 
                                  min_value=10000, max_value=10000000, 
                                  value=100000, step=10000)
    
    if st.button("🎯 Get Recommendations"):
        try:
            recommendations = ml_models.generate_recommendations(total_budget, encoders=preprocessor.encoders)
            
            st.write("**Recommended Budget Allocation:**")
            
            for i, rec in enumerate(recommendations[:5], 1):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**{i}. {rec['Campaign_Type']}**")
                with col2:
                    st.write(f"ROI: {rec['Predicted_ROI']:.2f}x")
                with col3:
                    st.write(f"Budget: ${rec['Recommended_Budget']:,.0f}")
                with col4:
                    st.write(f"Return: ${rec['Expected_Return']:,.0f}")
                    
        except Exception as e:
            st.error(f"Recommendation error: {e}")
    
    # Model Performance
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'roi_prediction' in ml_models.model_metrics:
            roi_metrics = ml_models.model_metrics['roi_prediction']
            st.metric("ROI Model R² Score", f"{roi_metrics['r2_score']:.3f}")
            st.metric("ROI Model RMSE", f"{roi_metrics['rmse']:.3f}")
    
    with col2:
        if 'classification' in ml_models.model_metrics:
            class_metrics = ml_models.model_metrics['classification']
            st.metric("Classification Accuracy", f"{class_metrics['accuracy']:.3f}")

def show_ai_assistant_page(df):
    """Display AI assistant chat interface"""
    st.header("💬 AI Marketing Assistant")
    
    # Initialize AI assistant using environment secrets when available
    api_key = config.GEMINI_API_KEY or config.OPENAI_API_KEY
    if api_key:
        assistant = create_ai_assistant(df, api_key)
    else:
        assistant = create_ai_assistant(df)
        st.info("💡 For enhanced AI responses, set GEMINI_API_KEY or OPENAI_API_KEY in your environment or Streamlit settings.")
    
    # Quick insights
    st.subheader("⚡ Quick Insights")
    if st.button("Get Quick Insights"):
        insights = assistant.get_quick_insights()
        for insight in insights:
            st.write(f"• {insight}")
    
    # Chat interface
    st.subheader("💬 Ask Questions About Your Marketing Data")
    
    # Sample questions
    st.write("**Sample Questions:**")
    sample_questions = [
        "Which campaign type has the highest ROI?",
        "Compare performance across different channels",
        "What customer segment should we focus on?",
        "Recommend budget allocation strategy",
        "What are the key performance trends?"
    ]
    
    for question in sample_questions:
        if st.button(f"❓ {question}"):
            st.session_state.user_query = question
    
    # User input
    user_query = st.text_input("Your question:", 
                              value=st.session_state.get('user_query', ''),
                              placeholder="Ask anything about your marketing campaigns...")
    
    if st.button("🚀 Get Answer") and user_query:
        with st.spinner("🤖 Analyzing your data..."):
            response = assistant.generate_response(user_query)
            st.write("**AI Assistant:**")
            st.write(response)

if __name__ == "__main__":
    main()