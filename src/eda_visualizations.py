"""
EDA and visualization module for marketing analytics dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class MarketingVisualizer:
    def __init__(self, df):
        self.df = df
        # High-contrast color palette for better visibility
        self.colors = ['#0066cc', '#28a745', '#dc3545', '#ffc107', '#6f42c1', '#fd7e14', '#20c997', '#e83e8c', '#6c757d', '#17a2b8']
    
    def create_overview_metrics(self):
        """Create overview KPI metrics"""
        metrics = {
            'Total Campaigns': len(self.df),
            'Avg ROI': self.df['ROI'].mean(),
            'Avg Conversion Rate': self.df['Conversion_Rate'].mean(),
            'Total Spend': self.df['Acquisition_Cost'].sum(),
            'Avg Engagement Score': self.df['Engagement_Score'].mean()
        }
        return metrics
    
    def plot_conversion_rate_distribution(self):
        """Create histogram of conversion rates"""
        fig = px.histogram(
            self.df, 
            x='Conversion_Rate', 
            nbins=20,
            title='Distribution of Conversion Rates',
            labels={'count': 'Number of Campaigns', 'Conversion_Rate': 'Conversion Rate (%)'},
            color_discrete_sequence=[self.colors[0]]
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='#2c3e50')
        )
        return fig
    
    def plot_roi_by_campaign_type(self):
        """Create box plot of ROI by campaign type"""
        fig = px.box(
            self.df, 
            x='Campaign_Type', 
            y='ROI',
            title='ROI Distribution by Campaign Type',
            color='Campaign_Type',
            color_discrete_sequence=self.colors
        )
        fig.update_layout(
            xaxis_tickangle=45,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='#2c3e50'),
            showlegend=True
        )
        return fig
    
    def plot_clicks_vs_impressions(self):
        """Create scatter plot of clicks vs impressions"""
        fig = px.scatter(
            self.df, 
            x='Impressions', 
            y='Clicks',
            color='Campaign_Type',
            color_discrete_sequence=self.colors,
            size='ROI',
            hover_data=['Conversion_Rate', 'Engagement_Score'],
            title='Clicks vs Impressions by Campaign Type'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='#2c3e50'),
            showlegend=True
        )
        return fig
    
    def plot_customer_segment_distribution(self):
        """Create count plot of customer segments"""
        segment_counts = self.df['Customer_Segment'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segment Distribution',
            color_discrete_sequence=self.colors
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='#2c3e50')
        )
        return fig
    
    def plot_engagement_over_time(self):
        """Create line plot of engagement score over time"""
        if 'Date' in self.df.columns:
            df_sorted = self.df.sort_values('Date')
            
            fig = px.line(
                df_sorted, 
                x='Date', 
                y='Engagement_Score',
                color='Campaign_Type',
                title='Engagement Score Trends Over Time',
                color_discrete_sequence=self.colors
            )
            fig.update_layout(
                xaxis_tickangle=45,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12, color='#2c3e50'),
                showlegend=True
            )
            return fig
        else:
            return None
    
    def plot_geographic_performance(self):
        """Create bar plot of performance by location with distinct colors"""
        location_performance = self.df.groupby('Location').agg({
            'Conversion_Rate': 'mean',
            'ROI': 'mean',
            'Engagement_Score': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Avg Conversion Rate (%)', 'Avg ROI (x)', 'Avg Engagement Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Define distinct colors for each metric
        colors = {
            'conversion': '#28a745',  # Green for conversion rate
            'roi': '#ffc107',         # Gold for ROI 
            'engagement': '#dc3545'   # Red for engagement
        }
        
        # Conversion Rate
        fig.add_trace(
            go.Bar(x=location_performance['Location'], 
                   y=location_performance['Conversion_Rate'],
                   name='Conversion Rate',
                   marker_color=colors['conversion'],
                   text=[f"{val:.1f}%" for val in location_performance['Conversion_Rate']],
                   textposition='outside'),
            row=1, col=1
        )
        
        # ROI
        fig.add_trace(
            go.Bar(x=location_performance['Location'], 
                   y=location_performance['ROI'],
                   name='ROI',
                   marker_color=colors['roi'],
                   text=[f"{val:.2f}x" for val in location_performance['ROI']],
                   textposition='outside'),
            row=1, col=2
        )
        
        # Engagement
        fig.add_trace(
            go.Bar(x=location_performance['Location'], 
                   y=location_performance['Engagement_Score'],
                   name='Engagement',
                   marker_color=colors['engagement'],
                   text=[f"{val:.1f}" for val in location_performance['Engagement_Score']],
                   textposition='outside'),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text="Performance Metrics by Location",
            showlegend=False,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='#2c3e50')
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Location", row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="Location", row=1, col=2, tickangle=45)
        fig.update_xaxes(title_text="Location", row=1, col=3, tickangle=45)
        
        # Update y-axes with appropriate labels
        fig.update_yaxes(title_text="Conversion Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="ROI (x)", row=1, col=2)
        fig.update_yaxes(title_text="Engagement Score", row=1, col=3)
        
        return fig
    
    def plot_channel_effectiveness(self):
        """Create channel performance analysis"""
        channel_stats = self.df.groupby('Channel_Used').agg({
            'ROI': ['mean', 'count'],
            'Conversion_Rate': 'mean',
            'Cost_Per_Click': 'mean',
            'CTR': 'mean'
        }).round(3)
        
        # Flatten column names
        channel_stats.columns = ['_'.join(col).strip() for col in channel_stats.columns]
        channel_stats = channel_stats.reset_index()
        
        fig = px.scatter(
            channel_stats,
            x='Conversion_Rate_mean',
            y='ROI_mean',
            size='ROI_count',
            color='Channel_Used',
            title='Channel Effectiveness: ROI vs Conversion Rate',
            labels={
                'Conversion_Rate_mean': 'Average Conversion Rate (%)',
                'ROI_mean': 'Average ROI',
                'ROI_count': 'Number of Campaigns'
            }
        )
        
        return fig
    
    def plot_correlation_heatmap(self):
        """Create correlation heatmap of numerical features"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title='Feature Correlation Heatmap',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, color='#2c3e50')
        )
        
        return fig
    
    def plot_customer_segment_performance(self):
        """Create customer segment performance analysis"""
        if 'Customer_Cluster' in self.df.columns:
            cluster_performance = self.df.groupby('Customer_Cluster').agg({
                'Conversion_Rate': 'mean',
                'ROI': 'mean',
                'Engagement_Score': 'mean',
                'CTR': 'mean',
                'Cost_Per_Click': 'mean'
            }).reset_index()
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Conversion Rate', 'ROI', 'Engagement Score', 
                               'CTR', 'Cost Per Click', ''),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            
            metrics = ['Conversion_Rate', 'ROI', 'Engagement_Score', 'CTR', 'Cost_Per_Click']
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
            
            for i, (metric, pos) in enumerate(zip(metrics, positions)):
                fig.add_trace(
                    go.Bar(
                        x=cluster_performance['Customer_Cluster'],
                        y=cluster_performance[metric],
                        name=metric,
                        showlegend=False
                    ),
                    row=pos[0], col=pos[1]
                )
            
            fig.update_layout(
                title_text="Customer Segment Performance Analysis",
                height=600
            )
            
            return fig
        else:
            return None
    
    def create_campaign_funnel(self):
        """Create campaign funnel visualization"""
        if all(col in self.df.columns for col in ['Impressions', 'Clicks', 'Conversion_Rate']):
            total_impressions = self.df['Impressions'].sum()
            total_clicks = self.df['Clicks'].sum()
            # Estimate conversions based on conversion rate and clicks
            estimated_conversions = (self.df['Clicks'] * self.df['Conversion_Rate'] / 100).sum()
            
            funnel_data = {
                'Stage': ['Impressions', 'Clicks', 'Conversions'],
                'Count': [total_impressions, total_clicks, estimated_conversions],
                'Percentage': [100, (total_clicks/total_impressions)*100, 
                              (estimated_conversions/total_impressions)*100]
            }
            
            fig = go.Figure(go.Funnel(
                y = funnel_data['Stage'],
                x = funnel_data['Count'],
                texttemplate = "%{label}: %{value:,.0f}<br>(%{percentTotal})",
                textposition = "inside",
                textfont = {"size": 14, "color": "white"}
            ))
            
            fig.update_layout(
                title="Marketing Campaign Funnel",
                font=dict(size=12)
            )
            
            return fig
        else:
            return None

def create_dashboard_plots(df):
    """Create all dashboard visualizations"""
    viz = MarketingVisualizer(df)
    
    plots = {
        'overview_metrics': viz.create_overview_metrics(),
        'conversion_dist': viz.plot_conversion_rate_distribution(),
        'roi_by_campaign': viz.plot_roi_by_campaign_type(),
        'clicks_impressions': viz.plot_clicks_vs_impressions(),
        'segment_distribution': viz.plot_customer_segment_distribution(),
        'engagement_time': viz.plot_engagement_over_time(),
        'geographic_performance': viz.plot_geographic_performance(),
        'channel_effectiveness': viz.plot_channel_effectiveness(),
        'correlation_heatmap': viz.plot_correlation_heatmap(),
        'segment_performance': viz.plot_customer_segment_performance(),
        'campaign_funnel': viz.create_campaign_funnel()
    }
    
    return plots