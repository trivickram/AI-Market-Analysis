"""
FIXED Machine Learning models module for marketing analytics dashboard
This version addresses data leakage, feature selection, and model performance issues.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import pickle
import warnings
warnings.filterwarnings('ignore')

class MarketingMLModelsFixed:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.model_metrics = {}
    
    def prepare_roi_prediction_data(self, df):
        """
        FIXED: Prepare data for ROI prediction WITHOUT data leakage
        Only use features available BEFORE campaign execution
        """
        print("🔧 Preparing ROI prediction data (NO LEAKAGE)...")
        
        # PRE-CAMPAIGN features only (no results-based features)
        base_features = [
            'Duration',           # Campaign duration - known beforehand  
            'Acquisition_Cost'    # Budget allocated - known beforehand
        ]
        
        # Categorical features (encoded)
        categorical_features = [
            'Campaign_Type_encoded', 'Target_Audience_encoded', 
            'Channel_Used_encoded', 'Location_encoded', 
            'Language_encoded', 'Customer_Segment_encoded'
        ]
        
        # Create domain-specific engineered features
        df_copy = df.copy()
        
        # Cost efficiency features (pre-campaign estimates)
        df_copy['Budget_Per_Day'] = df_copy['Acquisition_Cost'] / df_copy['Duration']
        df_copy['Log_Budget'] = np.log1p(df_copy['Acquisition_Cost'])
        
        # Temporal features
        df_copy['Month'] = pd.to_datetime(df_copy['Date']).dt.month
        df_copy['Quarter'] = pd.to_datetime(df_copy['Date']).dt.quarter
        df_copy['Is_Weekend_Start'] = pd.to_datetime(df_copy['Date']).dt.dayofweek >= 5
        
        # Channel-Duration interaction
        df_copy['Channel_Duration_Interaction'] = (
            df_copy['Channel_Used_encoded'] * df_copy['Duration']
        )
        
        # All pre-campaign features
        engineered_features = [
            'Budget_Per_Day', 'Log_Budget', 'Month', 'Quarter', 
            'Is_Weekend_Start', 'Channel_Duration_Interaction'
        ]
        
        roi_feature_columns = base_features + categorical_features + engineered_features
        
        # Remove any rows with missing target
        clean_data = df_copy.dropna(subset=['ROI'])
        
        X = clean_data[roi_feature_columns]
        y = clean_data['ROI']
        
        # Remove outliers using IQR method
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        X = X[mask]
        y = y[mask]
        
        print(f"✅ ROI Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Target range: {y.min():.2f} to {y.max():.2f}, mean: {y.mean():.2f}")
        
        return X, y, roi_feature_columns
    
    def train_roi_prediction_model(self, df):
        """FIXED: Train ROI prediction model with proper validation"""
        X, y, feature_columns = self.prepare_roi_prediction_data(df)
        
        # Split with stratification on ROI quartiles
        y_quartiles = pd.qcut(y, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y_quartiles
        )
        
        # Use RobustScaler for better outlier handling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(12, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_selected, y_train)
        
        model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = model.predict(X_test_selected)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
        
        # Store model and artifacts
        self.models['roi_prediction'] = model
        self.scalers['roi_prediction'] = scaler
        self.feature_selectors['roi_prediction'] = selector
        
        # Get selected feature names
        selected_features = np.array(feature_columns)[selector.get_support()]
        
        self.model_metrics['roi_prediction'] = {
            'r2_score': r2,
            'rmse': rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'feature_importance': dict(zip(selected_features, model.feature_importances_)),
            'selected_features': selected_features.tolist()
        }
        
        print(f"✅ ROI Model - R² Score: {r2:.4f}, CV Score: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        return model, scaler, self.model_metrics['roi_prediction']
    
    def prepare_classification_data(self, df):
        """FIXED: Prepare data for campaign success classification"""
        print("🔧 Preparing classification data...")
        
        # Create meaningful binary target based on ROI performance
        df_copy = df.copy()
        roi_median = df_copy['ROI'].median()
        df_copy['High_ROI'] = (df_copy['ROI'] > roi_median).astype(int)
        
        # Use same pre-campaign features as ROI prediction
        base_features = ['Duration', 'Acquisition_Cost']
        categorical_features = [
            'Campaign_Type_encoded', 'Target_Audience_encoded', 
            'Channel_Used_encoded', 'Location_encoded', 
            'Language_encoded', 'Customer_Segment_encoded'
        ]
        
        # Add same engineered features
        df_copy['Budget_Per_Day'] = df_copy['Acquisition_Cost'] / df_copy['Duration']
        df_copy['Log_Budget'] = np.log1p(df_copy['Acquisition_Cost'])
        df_copy['Month'] = pd.to_datetime(df_copy['Date']).dt.month
        df_copy['Channel_Duration_Interaction'] = (
            df_copy['Channel_Used_encoded'] * df_copy['Duration']
        )
        
        engineered_features = [
            'Budget_Per_Day', 'Log_Budget', 'Month', 'Channel_Duration_Interaction'
        ]
        
        classification_features = base_features + categorical_features + engineered_features
        
        X = df_copy[classification_features].dropna()
        y = df_copy.loc[X.index, 'High_ROI']
        
        print(f"✅ Classification data: {X.shape[0]} samples, Class distribution: {y.value_counts().to_dict()}")
        return X, y, classification_features
    
    def train_classification_model(self, df):
        """FIXED: Train campaign success classification model"""
        X, y, feature_columns = self.prepare_classification_data(df)
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [8, 12, 15],
            'min_samples_split': [5, 10],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_selected, y_train)
        
        model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = model.predict(X_test_selected)
        
        # Calculate metrics
        accuracy = model.score(X_test_selected, y_test)
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
        
        # Store model and artifacts
        self.models['classification'] = model
        self.scalers['classification'] = scaler
        self.feature_selectors['classification'] = selector
        
        selected_features = np.array(feature_columns)[selector.get_support()]
        
        self.model_metrics['classification'] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'feature_importance': dict(zip(selected_features, model.feature_importances_)),
            'selected_features': selected_features.tolist()
        }
        
        print(f"✅ Classification Model - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        return model, scaler, self.model_metrics['classification']
    
    def train_clustering_model(self, df):
        """FIXED: Train customer segmentation clustering model"""
        print("🔧 Training clustering model...")
        
        # Use engagement and performance metrics for clustering
        segment_features = ['Conversion_Rate', 'ROI', 'Engagement_Score']
        
        # Add derived metrics
        df_copy = df.copy()
        df_copy['Cost_Per_Click'] = df_copy['Acquisition_Cost'] / df_copy['Clicks'].replace(0, 1)
        df_copy['CTR'] = df_copy['Clicks'] / df_copy['Impressions'].replace(0, 1)
        
        extended_features = segment_features + ['Cost_Per_Click', 'CTR']
        X = df_copy[extended_features].fillna(df_copy[extended_features].median())
        
        # Remove outliers
        for col in extended_features:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            X = X[(X[col] >= Q1 - 1.5*IQR) & (X[col] <= Q3 + 1.5*IQR)]
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Choose optimal k (you can implement elbow detection)
        optimal_k = 4  # Can be made more sophisticated
        
        # Train final model
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Store model and artifacts
        self.models['clustering'] = kmeans
        self.scalers['clustering'] = scaler
        self.model_metrics['clustering'] = {
            'n_clusters': optimal_k,
            'inertia': kmeans.inertia_,
            'feature_columns': extended_features,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        print(f"✅ Clustering Model - {optimal_k} clusters, Inertia: {kmeans.inertia_:.2f}")
        return kmeans, scaler, cluster_labels
    
    def predict_roi(self, duration, budget, campaign_type_enc, target_audience_enc, 
                   channel_enc, location_enc, language_enc, segment_enc, month=6):
        """FIXED: Predict ROI using only pre-campaign features"""
        if 'roi_prediction' not in self.models:
            return "Model not trained yet"
        
        # Create features matching training data
        budget_per_day = budget / duration
        log_budget = np.log1p(budget)
        quarter = (month - 1) // 3 + 1
        is_weekend_start = 0  # Default assumption
        channel_duration_interaction = channel_enc * duration
        
        # Create feature array (match training order)
        features = np.array([[
            duration, budget,  # base features
            campaign_type_enc, target_audience_enc, channel_enc, 
            location_enc, language_enc, segment_enc,  # categorical
            budget_per_day, log_budget, month, quarter, 
            is_weekend_start, channel_duration_interaction  # engineered
        ]])
        
        # Scale and select features
        features_scaled = self.scalers['roi_prediction'].transform(features)
        features_selected = self.feature_selectors['roi_prediction'].transform(features_scaled)
        
        # Predict
        predicted_roi = self.models['roi_prediction'].predict(features_selected)[0]
        
        return max(0, predicted_roi)  # Ensure non-negative ROI
    
    def predict_campaign_success(self, duration, budget, campaign_type_enc, target_audience_enc, 
                                channel_enc, location_enc, language_enc, segment_enc, month=6):
        """FIXED: Predict campaign success probability"""
        if 'classification' not in self.models:
            return "Model not trained yet"
        
        # Create same features as ROI prediction
        budget_per_day = budget / duration
        log_budget = np.log1p(budget)
        channel_duration_interaction = channel_enc * duration
        
        features = np.array([[
            duration, budget,
            campaign_type_enc, target_audience_enc, channel_enc, 
            location_enc, language_enc, segment_enc,
            budget_per_day, log_budget, month, channel_duration_interaction
        ]])
        
        # Scale and select features
        features_scaled = self.scalers['classification'].transform(features)
        features_selected = self.feature_selectors['classification'].transform(features_scaled)
        
        # Predict probability
        success_probability = self.models['classification'].predict_proba(features_selected)[0][1]
        
        return success_probability
    
    def generate_recommendations(self, total_budget, target_roi=3.0, encoders=None):
        """FIXED: Generate recommendations with proper error handling"""
        if encoders is None or 'roi_prediction' not in self.models:
            return "Models or encoders not available"
        
        recommendations = []
        
        # Test campaign configurations
        campaign_configs = [
            ("Email", "Young Adults", "Email", "USA", "English", "Tech Enthusiasts"),
            ("Social Media", "Adults", "Social Media", "USA", "English", "Fashionistas"),
            ("Influencer", "Young Adults", "Social Media", "USA", "English", "Health & Wellness"),
            ("Display", "Adults", "Website", "USA", "English", "Tech Enthusiasts"),
            ("Search", "Adults", "Google Ads", "USA", "English", "Outdoor Adventurers")
        ]
        
        for config in campaign_configs:
            campaign_type, audience, channel, location, language, segment = config
            try:
                # Safe encoding with fallback
                def safe_encode(encoder, value):
                    if value in encoder.classes_:
                        return encoder.transform([value])[0]
                    else:
                        return 0  # Use first class as fallback
                
                campaign_enc = safe_encode(encoders['Campaign_Type'], campaign_type)
                audience_enc = safe_encode(encoders['Target_Audience'], audience)
                channel_enc = safe_encode(encoders['Channel_Used'], channel)
                location_enc = safe_encode(encoders['Location'], location)
                language_enc = safe_encode(encoders['Language'], language)
                segment_enc = safe_encode(encoders['Customer_Segment'], segment)
                
                # Test configuration
                test_budget = total_budget * 0.2
                predicted_roi = self.predict_roi(30, test_budget, campaign_enc, audience_enc, 
                                               channel_enc, location_enc, language_enc, segment_enc)
                success_prob = self.predict_campaign_success(30, test_budget, campaign_enc, audience_enc, 
                                                           channel_enc, location_enc, language_enc, segment_enc)
                
                if isinstance(predicted_roi, (int, float)) and isinstance(success_prob, (int, float)):
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
                print(f"⚠️ Skipping {campaign_type}: {e}")
                continue
        
        # Sort by predicted ROI
        recommendations = sorted(recommendations, key=lambda x: x['Predicted_ROI'], reverse=True)
        return recommendations[:3]  # Return top 3 recommendations
    
    def save_models(self, file_path):
        """Save all trained models and artifacts"""
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_selectors': self.feature_selectors,
            'metrics': self.model_metrics
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Fixed models saved to {file_path}")
    
    def load_models(self, file_path):
        """Load all trained models and artifacts"""
        try:
            with open(file_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.models = model_package['models']
            self.scalers = model_package['scalers']
            self.encoders = model_package.get('encoders', {})
            self.feature_selectors = model_package.get('feature_selectors', {})
            self.model_metrics = model_package['metrics']
            
            print(f"✅ Fixed models loaded from {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def train_all_models_fixed(df):
    """Train all FIXED ML models"""
    print("🚀 Training FIXED ML models...")
    ml_models = MarketingMLModelsFixed()
    
    # Train all models
    ml_models.train_roi_prediction_model(df)
    ml_models.train_classification_model(df)
    ml_models.train_clustering_model(df)
    
    return ml_models