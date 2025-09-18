"""
Machine Learning models module for marketing analytics dashboard
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class MarketingMLModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metrics = {}
    
    def prepare_roi_prediction_data(self, df):
        """Prepare data for ROI prediction model"""
        # Select features for ROI prediction
        roi_features = ['Duration', 'Acquisition_Cost', 'Clicks', 'Impressions', 'Engagement_Score']
        
        # Get encoded categorical features
        categorical_features = ['Campaign_Type_encoded', 'Target_Audience_encoded', 
                               'Channel_Used_encoded', 'Location_encoded', 
                               'Language_encoded', 'Customer_Segment_encoded']
        
        # Final feature set
        roi_feature_columns = roi_features + categorical_features
        
        X = df[roi_feature_columns]
        y = df['ROI']
        
        return X, y, roi_feature_columns
    
    def train_roi_prediction_model(self, df):
        """Train ROI prediction model"""
        X, y, feature_columns = self.prepare_roi_prediction_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Store model and artifacts
        self.models['roi_prediction'] = model
        self.scalers['roi_prediction'] = scaler
        self.model_metrics['roi_prediction'] = {
            'r2_score': r2,
            'rmse': rmse,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }
        
        print(f"✅ ROI Prediction Model trained - R² Score: {r2:.4f}")
        return model, scaler, self.model_metrics['roi_prediction']
    
    def prepare_classification_data(self, df):
        """Prepare data for campaign success classification"""
        classification_features = ['Duration', 'Acquisition_Cost', 'Campaign_Type_encoded', 
                                  'Target_Audience_encoded', 'Channel_Used_encoded', 
                                  'Location_encoded', 'Language_encoded', 'Customer_Segment_encoded']
        
        X = df[classification_features]
        y = df['High_Engagement']  # Binary target
        
        return X, y, classification_features
    
    def train_classification_model(self, df):
        """Train campaign success classification model"""
        X, y, feature_columns = self.prepare_classification_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = model.score(X_test_scaled, y_test)
        
        # Store model and artifacts
        self.models['classification'] = model
        self.scalers['classification'] = scaler
        self.model_metrics['classification'] = {
            'accuracy': accuracy,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }
        
        print(f"✅ Classification Model trained - Accuracy: {accuracy:.4f}")
        return model, scaler, self.model_metrics['classification']
    
    def train_clustering_model(self, df):
        """Train customer segmentation clustering model"""
        segment_features = ['Conversion_Rate', 'ROI', 'Engagement_Score', 'CTR', 'Cost_Per_Click']
        X = df[segment_features].fillna(df[segment_features].median())
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train K-means clustering
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Store model and artifacts
        self.models['clustering'] = kmeans
        self.scalers['clustering'] = scaler
        self.model_metrics['clustering'] = {
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'feature_columns': segment_features
        }
        
        print(f"✅ Clustering Model trained - {n_clusters} clusters")
        return kmeans, scaler, cluster_labels
    
    def predict_roi(self, duration, budget, campaign_type_enc, target_audience_enc, 
                   channel_enc, location_enc, language_enc, segment_enc,
                   estimated_clicks=None, estimated_impressions=None, estimated_engagement=None):
        """Predict ROI for a new campaign"""
        if 'roi_prediction' not in self.models:
            return "Model not trained yet"
        
        # Estimate missing values if not provided
        if estimated_clicks is None:
            estimated_impressions = budget * 10 if estimated_impressions is None else estimated_impressions
            estimated_clicks = estimated_impressions * 0.02
        if estimated_engagement is None:
            estimated_engagement = 50 + (budget / 1000) * 5
        
        # Create feature array
        features = np.array([[duration, budget, estimated_clicks, estimated_impressions, 
                             estimated_engagement, campaign_type_enc, target_audience_enc, 
                             channel_enc, location_enc, language_enc, segment_enc]])
        
        # Scale and predict
        features_scaled = self.scalers['roi_prediction'].transform(features)
        predicted_roi = self.models['roi_prediction'].predict(features_scaled)[0]
        
        return predicted_roi
    
    def predict_campaign_success(self, duration, budget, campaign_type_enc, target_audience_enc, 
                                channel_enc, location_enc, language_enc, segment_enc):
        """Predict campaign success probability"""
        if 'classification' not in self.models:
            return "Model not trained yet"
        
        # Create feature array
        features = np.array([[duration, budget, campaign_type_enc, target_audience_enc, 
                             channel_enc, location_enc, language_enc, segment_enc]])
        
        # Scale and predict
        features_scaled = self.scalers['classification'].transform(features)
        success_probability = self.models['classification'].predict_proba(features_scaled)[0][1]
        
        return success_probability
    
    def predict_customer_segment(self, conversion_rate, roi, engagement_score, ctr, cost_per_click):
        """Predict customer segment for given characteristics"""
        if 'clustering' not in self.models:
            return "Model not trained yet"
        
        # Create feature array
        features = np.array([[conversion_rate, roi, engagement_score, ctr, cost_per_click]])
        
        # Scale and predict
        features_scaled = self.scalers['clustering'].transform(features)
        cluster = self.models['clustering'].predict(features_scaled)[0]
        
        return cluster
    
    def generate_recommendations(self, total_budget, target_roi=3.0, encoders=None):
        """Generate budget allocation recommendations"""
        if encoders is None or 'roi_prediction' not in self.models:
            return "Models or encoders not available"
        
        recommendations = []
        
        # Test different campaign configurations
        campaign_configs = [
            ("Email", "Young Adults", "Email", "USA", "English", "Tech Enthusiasts"),
            ("Social Media", "Adults", "Social Media", "USA", "English", "Fashionistas"),
            ("Influencer", "Young Adults", "Social Media", "USA", "English", "Health and Wellness"),
            ("Display", "Adults", "Website", "USA", "English", "Tech Enthusiasts"),
            ("Search", "Adults", "Google Ads", "USA", "English", "Outdoor Adventurers")
        ]
        
        for config in campaign_configs:
            campaign_type, audience, channel, location, language, segment = config
            try:
                # Patch: Add missing labels to encoders before transforming
                def safe_transform(encoder, label):
                    if label not in encoder.classes_:
                        encoder.classes_ = np.append(encoder.classes_, label)
                    return encoder.transform([label])[0]
                campaign_enc = safe_transform(encoders['Campaign_Type'], campaign_type)
                audience_enc = safe_transform(encoders['Target_Audience'], audience)
                channel_enc = safe_transform(encoders['Channel_Used'], channel)
                location_enc = safe_transform(encoders['Location'], location)
                language_enc = safe_transform(encoders['Language'], language)
                segment_enc = safe_transform(encoders['Customer_Segment'], segment)
                # Test with 30-day duration and 20% of total budget
                test_budget = total_budget * 0.2
                predicted_roi = self.predict_roi(30, test_budget, campaign_enc, audience_enc, 
                                               channel_enc, location_enc, language_enc, segment_enc)
                success_prob = self.predict_campaign_success(30, test_budget, campaign_enc, audience_enc, 
                                                           channel_enc, location_enc, language_enc, segment_enc)
                recommendations.append({
                    'Campaign_Type': campaign_type,
                    'Target_Audience': audience,
                    'Channel': channel,
                    'Predicted_ROI': predicted_roi,
                    'Success_Probability': success_prob,
                    'Recommended_Budget': test_budget,
                    'Expected_Return': test_budget * predicted_roi
                })
            except Exception as e:
                print(f"Error processing {campaign_type}: {e}")
                continue
        
        # Sort by predicted ROI
        recommendations = sorted(recommendations, key=lambda x: x['Predicted_ROI'], reverse=True)
        return recommendations
    
    def save_models(self, file_path):
        """Save all trained models and artifacts"""
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'metrics': self.model_metrics
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Models saved to {file_path}")
    
    def load_models(self, file_path):
        """Load all trained models and artifacts"""
        try:
            with open(file_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.models = model_package['models']
            self.scalers = model_package['scalers']
            self.encoders = model_package.get('encoders', {})
            self.model_metrics = model_package['metrics']
            
            print(f"✅ Models loaded from {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def train_all_models(df):
    """Train all ML models for the marketing analytics dashboard"""
    ml_models = MarketingMLModels()
    
    # Train ROI prediction model
    ml_models.train_roi_prediction_model(df)
    
    # Train classification model
    ml_models.train_classification_model(df)
    
    # Train clustering model
    ml_models.train_clustering_model(df)
    
    return ml_models