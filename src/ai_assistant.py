"""
AI Assistant module for natural language queries about marketing data
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json

# Optional imports for different LLM providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class MarketingAIAssistant:
    def __init__(self, df: pd.DataFrame, api_key: str = None, provider: str = "gemini"):
        self.df = df
        self.provider = provider
        self.api_key = api_key
        self.setup_llm()
        
    def setup_llm(self):
        """Setup the selected LLM provider"""
        if self.provider == "gemini" and GEMINI_AVAILABLE:
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                print("✅ Gemini AI Assistant initialized")
            else:
                print("❌ Gemini API key not provided")
                self.model = None
        elif self.provider == "openai" and OPENAI_AVAILABLE:
            if self.api_key:
                openai.api_key = self.api_key
                print("✅ OpenAI Assistant initialized")
            else:
                print("❌ OpenAI API key not provided")
        else:
            print(f"❌ {self.provider} not available or not supported")
            self.model = None
    
    def get_data_summary(self) -> str:
        """Generate a summary of the dataset for context"""
        summary = f"""
        Dataset Summary:
        - Total Campaigns: {len(self.df)}
        - Date Range: {self.df['Date'].min()} to {self.df['Date'].max()} (if Date column exists)
        - Average ROI: {self.df['ROI'].mean():.2f}
        - Average Conversion Rate: {self.df['Conversion_Rate'].mean():.2f}%
        - Total Marketing Spend: ${self.df['Acquisition_Cost'].sum():,.0f}
        
        Campaign Types: {', '.join(self.df['Campaign_Type'].unique())}
        Channels Used: {', '.join(self.df['Channel_Used'].unique())}
        Target Audiences: {', '.join(self.df['Target_Audience'].unique())}
        Locations: {', '.join(self.df['Location'].unique())}
        Customer Segments: {', '.join(self.df['Customer_Segment'].unique())}
        
        Key Metrics Available:
        - ROI, Conversion Rate, Engagement Score
        - Clicks, Impressions, CTR, Cost per Click
        - Duration, Acquisition Cost
        """
        return summary
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query and extract relevant data"""
        
        # Keywords for different types of analysis
        analysis_keywords = {
            'performance': ['roi', 'conversion', 'performance', 'best', 'worst', 'effective'],
            'comparison': ['compare', 'vs', 'versus', 'difference', 'better', 'higher', 'lower'],
            'trend': ['trend', 'over time', 'timeline', 'change', 'growth', 'decline'],
            'segmentation': ['segment', 'audience', 'group', 'demographic', 'cluster'],
            'prediction': ['predict', 'forecast', 'future', 'estimate', 'expect', 'will'],
            'recommendation': ['recommend', 'suggest', 'advice', 'optimize', 'improve']
        }
        
        query_lower = user_query.lower()
        analysis_type = 'general'
        
        for analysis, keywords in analysis_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis_type = analysis
                break
        
        return {
            'query': user_query,
            'analysis_type': analysis_type,
            'contains_metrics': any(metric in query_lower for metric in ['roi', 'conversion', 'engagement', 'clicks', 'impressions'])
        }
    
    def get_relevant_data(self, query_analysis: Dict[str, Any]) -> str:
        """Extract relevant data based on query analysis"""
        analysis_type = query_analysis['analysis_type']
        
        if analysis_type == 'performance':
            # Get top and bottom performers
            top_campaigns = self.df.nlargest(3, 'ROI')[['Campaign_Type', 'ROI', 'Conversion_Rate']].to_string()
            bottom_campaigns = self.df.nsmallest(3, 'ROI')[['Campaign_Type', 'ROI', 'Conversion_Rate']].to_string()
            return f"Top Performing Campaigns:\n{top_campaigns}\n\nBottom Performing Campaigns:\n{bottom_campaigns}"
        
        elif analysis_type == 'comparison':
            # Get performance by campaign type
            comparison_data = self.df.groupby('Campaign_Type').agg({
                'ROI': 'mean',
                'Conversion_Rate': 'mean',
                'Engagement_Score': 'mean'
            }).round(2).to_string()
            return f"Performance Comparison by Campaign Type:\n{comparison_data}"
        
        elif analysis_type == 'segmentation':
            # Get segment performance
            segment_data = self.df.groupby('Customer_Segment').agg({
                'ROI': 'mean',
                'Conversion_Rate': 'mean',
                'Engagement_Score': 'mean'
            }).round(2).to_string()
            return f"Performance by Customer Segment:\n{segment_data}"
        
        elif analysis_type == 'trend':
            if 'Date' in self.df.columns:
                # Get monthly trends
                monthly_trends = self.df.set_index('Date').resample('M').agg({
                    'ROI': 'mean',
                    'Conversion_Rate': 'mean',
                    'Engagement_Score': 'mean'
                }).round(2).to_string()
                return f"Monthly Trends:\n{monthly_trends}"
            else:
                return "Date information not available for trend analysis."
        
        else:
            # General statistics
            general_stats = self.df[['ROI', 'Conversion_Rate', 'Engagement_Score', 'CTR']].describe().round(2).to_string()
            return f"General Statistics:\n{general_stats}"
    
    def generate_response(self, user_query: str) -> str:
        """Generate AI response to user query"""
        if not self.model and self.provider == "gemini":
            return "❌ AI Assistant not available. Please provide a valid API key."
        
        # Analyze the query
        query_analysis = self.analyze_query(user_query)
        
        # Get relevant data
        relevant_data = self.get_relevant_data(query_analysis)
        
        # Get data summary for context
        data_summary = self.get_data_summary()
        
        # Create prompt for the LLM
        prompt = f"""
        You are a marketing analytics expert AI assistant. You have access to a marketing campaign dataset.
        
        Dataset Context:
        {data_summary}
        
        User Question: {user_query}
        
        Relevant Data:
        {relevant_data}
        
        Please provide a comprehensive, insightful answer to the user's question based on the data provided. 
        Include specific numbers, insights, and actionable recommendations where appropriate.
        Keep the response clear, concise, and professional.
        """
        
        try:
            if self.provider == "gemini" and self.model:
                response = self.model.generate_content(prompt)
                return response.text
            elif self.provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a marketing analytics expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            else:
                return self._fallback_response(user_query, relevant_data)
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return self._fallback_response(user_query, relevant_data)
    
    def _fallback_response(self, user_query: str, relevant_data: str) -> str:
        """Fallback response when AI services are not available"""
        query_analysis = self.analyze_query(user_query)
        analysis_type = query_analysis['analysis_type']
        
        fallback_responses = {
            'performance': f"Based on the data analysis:\n\n{relevant_data}\n\nThe campaigns with highest ROI show strong performance, while those with lower ROI may need optimization.",
            
            'comparison': f"Here's the performance comparison:\n\n{relevant_data}\n\nLook for campaigns with higher ROI and conversion rates for best performance.",
            
            'segmentation': f"Customer segment analysis:\n\n{relevant_data}\n\nFocus on segments with higher engagement and conversion rates.",
            
            'trend': f"Trend analysis:\n\n{relevant_data}\n\nMonitor these trends to identify patterns and optimize future campaigns.",
            
            'prediction': f"For predictions, consider these current performance metrics:\n\n{relevant_data}\n\nUse historical performance to estimate future results.",
            
            'recommendation': f"Based on current performance:\n\n{relevant_data}\n\nRecommendation: Focus budget on high-performing campaign types and segments."
        }
        
        return fallback_responses.get(analysis_type, f"Here's the relevant data for your query:\n\n{relevant_data}")
    
    def get_quick_insights(self) -> List[str]:
        """Generate quick insights about the marketing data"""
        insights = []
        
        # Best performing campaign type
        best_campaign = self.df.loc[self.df['ROI'].idxmax()]
        insights.append(f"🏆 Best ROI: {best_campaign['Campaign_Type']} campaign achieved {best_campaign['ROI']:.2f} ROI")
        
        # Average metrics
        avg_roi = self.df['ROI'].mean()
        avg_conversion = self.df['Conversion_Rate'].mean()
        insights.append(f"📊 Average ROI: {avg_roi:.2f}, Average Conversion Rate: {avg_conversion:.2f}%")
        
        # Channel effectiveness
        channel_performance = self.df.groupby('Channel_Used')['ROI'].mean().sort_values(ascending=False)
        best_channel = channel_performance.index[0]
        insights.append(f"🎯 Most effective channel: {best_channel} with {channel_performance.iloc[0]:.2f} average ROI")
        
        # Customer segment insights
        segment_performance = self.df.groupby('Customer_Segment')['Conversion_Rate'].mean().sort_values(ascending=False)
        best_segment = segment_performance.index[0]
        insights.append(f"👥 Best converting segment: {best_segment} with {segment_performance.iloc[0]:.2f}% conversion rate")
        
        # Budget efficiency
        total_spend = self.df['Acquisition_Cost'].sum()
        total_return = (self.df['Acquisition_Cost'] * self.df['ROI']).sum()
        overall_roi = total_return / total_spend
        insights.append(f"💰 Overall portfolio ROI: {overall_roi:.2f} on ${total_spend:,.0f} total spend")
        
        return insights

# Example usage functions
def create_ai_assistant(df: pd.DataFrame, api_key: str = None, provider: str = "gemini"):
    """Create and return an AI assistant instance"""
    return MarketingAIAssistant(df, api_key, provider)

def demo_ai_assistant(df: pd.DataFrame):
    """Demo the AI assistant with sample queries"""
    assistant = MarketingAIAssistant(df)
    
    sample_queries = [
        "Which campaign type has the highest ROI?",
        "Compare the performance of different channels",
        "What are the trends in engagement over time?",
        "Which customer segment should we target?",
        "Recommend budget allocation for next quarter"
    ]
    
    print("🤖 AI Assistant Demo")
    print("=" * 50)
    
    # Show quick insights
    insights = assistant.get_quick_insights()
    print("📊 Quick Insights:")
    for insight in insights:
        print(f"  {insight}")
    
    print("\n💬 Sample Queries & Responses:")
    for query in sample_queries:
        print(f"\n❓ {query}")
        response = assistant.generate_response(query)
        print(f"🤖 {response[:200]}..." if len(response) > 200 else f"🤖 {response}")
        print("-" * 50)