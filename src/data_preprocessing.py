"""
Data preprocessing module for marketing analytics dashboard
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def load_data(self, file_path):
        """Load marketing campaign data"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and prepare the data"""
        df_clean = df.copy()
        
        # Remove dollar signs and convert to float
        if 'Acquisition_Cost' in df_clean.columns:
            if df_clean['Acquisition_Cost'].dtype == 'object':
                df_clean['Acquisition_Cost'] = df_clean['Acquisition_Cost'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Convert duration to numeric
        if 'Duration' in df_clean.columns:
            if df_clean['Duration'].dtype == 'object':
                df_clean['Duration'] = df_clean['Duration'].str.extract(r'(\d+)').astype(int)
            else:
                df_clean['Duration'] = df_clean['Duration'].astype(int)
        
        # Convert date column
        if 'Date' in df_clean.columns:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        
        # Convert categorical columns
        categorical_columns = ['Campaign_Type', 'Target_Audience', 'Channel_Used', 
                             'Location', 'Language', 'Customer_Segment']
        
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
        
        print(f"✅ Data cleaned successfully")
        return df_clean
    
    def engineer_features(self, df):
        """Create additional features for better AI predictions"""
        df_features = df.copy()
        
        # Feature: Cost per Click (CPC)
        df_features['Cost_Per_Click'] = df_features['Acquisition_Cost'] / df_features['Clicks'].replace(0, 1)
        
        # Feature: Click-Through Rate (CTR)
        df_features['CTR'] = df_features['Clicks'] / df_features['Impressions'].replace(0, 1)
        
        # Feature: Cost per Impression (CPM)
        df_features['CPM'] = (df_features['Acquisition_Cost'] / df_features['Impressions'].replace(0, 1)) * 1000
        
        # Feature: ROI Category
        df_features['ROI_Category'] = pd.cut(df_features['ROI'], 
                                           bins=[-float('inf'), 0, 2, 5, float('inf')], 
                                           labels=['Poor', 'Average', 'Good', 'Excellent'])
        
        # Feature: High Engagement
        df_features['High_Engagement'] = (df_features['Engagement_Score'] > df_features['Engagement_Score'].median()).astype(int)
        
        print(f"✅ Features engineered successfully")
        return df_features
    
    def encode_categorical_features(self, df):
        """Encode categorical variables for ML models"""
        df_encoded = df.copy()
        
        categorical_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used', 
                           'Location', 'Language', 'Customer_Segment']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                
                df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(df_encoded[col])
        
        print(f"✅ Categorical features encoded")
        return df_encoded
    
    def scale_features(self, df, feature_columns, scaler_name='default'):
        """Scale numerical features"""
        if scaler_name not in self.scalers:
            self.scalers[scaler_name] = StandardScaler()
        
        scaled_features = self.scalers[scaler_name].fit_transform(df[feature_columns])
        
        return scaled_features
    
    def save_preprocessed_data(self, df, file_path):
        """Save preprocessed data and scalers/encoders"""
        # Save data
        df.to_pickle(file_path)
        
        # Save scalers and encoders
        model_artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders
        }
        
        artifacts_path = file_path.replace('.pkl', '_artifacts.pkl')
        with open(artifacts_path, 'wb') as f:
            pickle.dump(model_artifacts, f)
        
        print(f"✅ Preprocessed data saved to {file_path}")
        print(f"✅ Model artifacts saved to {artifacts_path}")
    
    def load_preprocessed_data(self, file_path):
        """Load preprocessed data and artifacts"""
        try:
            # Load data
            df = pd.read_pickle(file_path)
            
            # Load artifacts
            artifacts_path = file_path.replace('.pkl', '_artifacts.pkl')
            if os.path.exists(artifacts_path):
                with open(artifacts_path, 'rb') as f:
                    artifacts = pickle.load(f)
                    self.scalers = artifacts['scalers']
                    self.encoders = artifacts['encoders']
            
            print(f"✅ Preprocessed data loaded successfully")
            return df
        except Exception as e:
            print(f"❌ Error loading preprocessed data: {e}")
            return None

def preprocess_marketing_data(input_file, output_file):
    """Main preprocessing function"""
    preprocessor = DataPreprocessor()
    
    # Load raw data
    df = preprocessor.load_data(input_file)
    if df is None:
        return None
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Engineer features
    df_features = preprocessor.engineer_features(df_clean)
    
    # Encode categorical variables
    df_final = preprocessor.encode_categorical_features(df_features)
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(df_final, output_file)
    
    return df_final, preprocessor

if __name__ == "__main__":
    # Example usage
    input_path = "data/marketing_campaign_dataset.csv"
    output_path = "data/processed_data.pkl"
    
    df, preprocessor = preprocess_marketing_data(input_path, output_path)
    print(f"Preprocessing complete! Shape: {df.shape}")