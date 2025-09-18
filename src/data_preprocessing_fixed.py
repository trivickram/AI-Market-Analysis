"""
FIXED Data preprocessing module for marketing analytics dashboard
This version addresses data quality, validation, and feature engineering issues.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessorFixed:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.data_stats = {}
        
    def load_data(self, file_path):
        """Load marketing campaign data with validation"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {df.shape}")
            
            # Basic data validation
            self._validate_data(df)
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _validate_data(self, df):
        """Validate data quality and structure"""
        print("🔍 Validating data quality...")
        
        # Check required columns
        required_cols = ['Campaign_ID', 'Campaign_Type', 'Target_Audience', 'Duration', 
                        'Channel_Used', 'Conversion_Rate', 'Acquisition_Cost', 'ROI', 
                        'Location', 'Language', 'Clicks', 'Impressions', 'Engagement_Score', 
                        'Customer_Segment', 'Date']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
        
        # Check data types and ranges
        print(f"📊 Data shape: {df.shape}")
        print(f"📊 Missing values: {df.isnull().sum().sum()}")
        print(f"📊 Duplicate rows: {df.duplicated().sum()}")
        
        # Check ROI distribution
        if 'ROI' in df.columns:
            roi_stats = df['ROI'].describe()
            print(f"📊 ROI distribution: min={roi_stats['min']:.2f}, max={roi_stats['max']:.2f}, mean={roi_stats['mean']:.2f}")
            
            # Flag potential data quality issues
            if roi_stats['min'] < -10:
                print("⚠️ Warning: Very negative ROI values detected")
            if roi_stats['max'] > 50:
                print("⚠️ Warning: Very high ROI values detected - possible outliers")
        
        print("✅ Data validation completed")
    
    def clean_data(self, df):
        """Clean and prepare the data with robust preprocessing"""
        print("🧹 Cleaning data...")
        df_clean = df.copy()
        
        # Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            print(f"🗑️ Removed {removed_duplicates} duplicate rows")
        
        # Clean Acquisition_Cost column
        if 'Acquisition_Cost' in df_clean.columns:
            df_clean['Acquisition_Cost'] = (df_clean['Acquisition_Cost']
                                          .astype(str)
                                          .str.replace('$', '', regex=False)
                                          .str.replace(',', '', regex=False)
                                          .astype(float))
        
        # Clean Duration column
        if 'Duration' in df_clean.columns:
            df_clean['Duration'] = (df_clean['Duration']
                                  .astype(str)
                                  .str.extract('(\\d+)')
                                  .astype(int))
        
        # Convert and validate date column
        if 'Date' in df_clean.columns:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            # Remove rows with invalid dates
            date_na_count = df_clean['Date'].isna().sum()
            if date_na_count > 0:
                print(f"🗑️ Removing {date_na_count} rows with invalid dates")
                df_clean = df_clean.dropna(subset=['Date'])
        
        # Handle missing values in numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['ROI', 'Conversion_Rate', 'Engagement_Score']:
                # For target and key metrics, remove missing values
                missing_count = df_clean[col].isna().sum()
                if missing_count > 0:
                    print(f"🗑️ Removing {missing_count} rows with missing {col}")
                    df_clean = df_clean.dropna(subset=[col])
            else:
                # For other numeric columns, impute with median
                if df_clean[col].isna().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"🔧 Imputed {col} missing values with median: {median_val:.2f}")
        
        # Handle categorical columns
        categorical_columns = ['Campaign_Type', 'Target_Audience', 'Channel_Used', 
                             'Location', 'Language', 'Customer_Segment']
        
        for col in categorical_columns:
            if col in df_clean.columns:
                # Fill missing categorical values with mode
                if df_clean[col].isna().sum() > 0:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_val, inplace=True)
                    print(f"🔧 Imputed {col} missing values with mode: {mode_val}")
                
                # Convert to category type
                df_clean[col] = df_clean[col].astype('category')
        
        # Remove extreme outliers using IQR method for key metrics
        for col in ['ROI', 'Acquisition_Cost', 'Conversion_Rate']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.01)  # Use 1st and 99th percentile for less aggressive filtering
                Q3 = df_clean[col].quantile(0.99)
                
                outlier_mask = (df_clean[col] < Q1) | (df_clean[col] > Q3)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    print(f"🗑️ Removing {outlier_count} extreme outliers from {col}")
                    df_clean = df_clean[~outlier_mask]
        
        print(f"✅ Data cleaned: {df_clean.shape[0]} rows remaining")
        return df_clean
    
    def engineer_features(self, df):
        """Create additional features for better AI predictions"""
        print("⚙️ Engineering features...")
        df_features = df.copy()
        
        # Validate required columns exist
        required_for_engineering = ['Acquisition_Cost', 'Clicks', 'Impressions', 'ROI', 'Engagement_Score']
        missing_required = [col for col in required_for_engineering if col not in df_features.columns]
        if missing_required:
            print(f"⚠️ Cannot engineer features: missing {missing_required}")
            return df_features
        
        # Cost efficiency features
        df_features['Cost_Per_Click'] = df_features['Acquisition_Cost'] / df_features['Clicks'].replace(0, 1)
        df_features['Cost_Per_Impression'] = df_features['Acquisition_Cost'] / df_features['Impressions'].replace(0, 1)
        
        # Performance ratios
        df_features['CTR'] = df_features['Clicks'] / df_features['Impressions'].replace(0, 1)
        df_features['CPM'] = (df_features['Acquisition_Cost'] / df_features['Impressions'].replace(0, 1)) * 1000
        
        # Budget efficiency
        df_features['Budget_Per_Day'] = df_features['Acquisition_Cost'] / df_features['Duration']
        df_features['Log_Budget'] = np.log1p(df_features['Acquisition_Cost'])
        
        # Temporal features
        if 'Date' in df_features.columns:
            df_features['Month'] = df_features['Date'].dt.month
            df_features['Quarter'] = df_features['Date'].dt.quarter
            df_features['Day_of_Week'] = df_features['Date'].dt.dayofweek
            df_features['Is_Weekend'] = (df_features['Day_of_Week'] >= 5).astype(int)
        
        # Performance categories (for classification targets)
        roi_median = df_features['ROI'].median()
        df_features['High_ROI'] = (df_features['ROI'] > roi_median).astype(int)
        
        engagement_median = df_features['Engagement_Score'].median()
        df_features['High_Engagement'] = (df_features['Engagement_Score'] > engagement_median).astype(int)
        
        # Interaction features
        if 'Campaign_Type_encoded' in df_features.columns:
            df_features['Campaign_Duration_Interaction'] = (df_features['Campaign_Type_encoded'] * 
                                                           df_features['Duration'])
        
        # Remove infinite or NaN values created by feature engineering
        numeric_features = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan)
            if df_features[col].isna().sum() > 0:
                df_features[col].fillna(df_features[col].median(), inplace=True)
        
        print(f"✅ Features engineered: {len([col for col in df_features.columns if col not in df.columns])} new features")
        return df_features\n    \n    def encode_categorical_features(self, df):\n        \"\"\"Encode categorical variables for ML models with validation\"\"\"\n        print(\"🔤 Encoding categorical features...\")\n        df_encoded = df.copy()\n        \n        categorical_cols = ['Campaign_Type', 'Target_Audience', 'Channel_Used', \n                           'Location', 'Language', 'Customer_Segment']\n        \n        for col in categorical_cols:\n            if col in df_encoded.columns:\n                # Initialize encoder if not exists\n                if col not in self.encoders:\n                    self.encoders[col] = LabelEncoder()\n                \n                # Fit and transform\n                try:\n                    df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(df_encoded[col].astype(str))\n                    \n                    # Store encoding information\n                    self.data_stats[f'{col}_classes'] = list(self.encoders[col].classes_)\n                    \n                    print(f\"✅ Encoded {col}: {len(self.encoders[col].classes_)} unique values\")\n                except Exception as e:\n                    print(f\"❌ Error encoding {col}: {e}\")\n                    continue\n        \n        print(\"✅ Categorical features encoded\")\n        return df_encoded\n    \n    def validate_and_split_data(self, df, target_col='ROI', test_size=0.2):\n        \"\"\"Validate and split data for training with proper stratification\"\"\"\n        print(f\"🎯 Preparing data for modeling (target: {target_col})...\")\n        \n        # Remove rows with missing target\n        initial_rows = len(df)\n        df_clean = df.dropna(subset=[target_col])\n        removed_rows = initial_rows - len(df_clean)\n        \n        if removed_rows > 0:\n            print(f\"🗑️ Removed {removed_rows} rows with missing target values\")\n        \n        # For continuous targets, create stratification bins\n        if target_col in ['ROI', 'Conversion_Rate']:\n            try:\n                stratify_bins = pd.qcut(df_clean[target_col], q=5, labels=False, duplicates='drop')\n                stratify = stratify_bins\n            except:\n                stratify = None\n                print(\"⚠️ Could not create stratification bins - using random split\")\n        else:\n            stratify = df_clean[target_col]\n        \n        # Split data\n        if len(df_clean) > 100:  # Only split if we have enough data\n            if stratify is not None:\n                train_df, test_df = train_test_split(\n                    df_clean, test_size=test_size, random_state=42, stratify=stratify\n                )\n            else:\n                train_df, test_df = train_test_split(\n                    df_clean, test_size=test_size, random_state=42\n                )\n            \n            print(f\"✅ Data split: {len(train_df)} train, {len(test_df)} test samples\")\n            return train_df, test_df\n        else:\n            print(\"⚠️ Dataset too small for splitting - using full dataset\")\n            return df_clean, df_clean\n    \n    def save_preprocessed_data(self, df, file_path):\n        \"\"\"Save preprocessed data and all artifacts\"\"\"\n        try:\n            # Save data\n            df.to_pickle(file_path)\n            \n            # Save all preprocessing artifacts\n            artifacts = {\n                'scalers': self.scalers,\n                'encoders': self.encoders,\n                'data_stats': self.data_stats\n            }\n            \n            artifacts_path = file_path.replace('.pkl', '_artifacts.pkl')\n            with open(artifacts_path, 'wb') as f:\n                pickle.dump(artifacts, f)\n            \n            print(f\"✅ Preprocessed data saved to {file_path}\")\n            print(f\"✅ Preprocessing artifacts saved to {artifacts_path}\")\n            return True\n        except Exception as e:\n            print(f\"❌ Error saving preprocessed data: {e}\")\n            return False\n    \n    def load_preprocessed_data(self, file_path):\n        \"\"\"Load preprocessed data and artifacts\"\"\"\n        try:\n            # Load data\n            df = pd.read_pickle(file_path)\n            \n            # Load artifacts\n            artifacts_path = file_path.replace('.pkl', '_artifacts.pkl')\n            if os.path.exists(artifacts_path):\n                with open(artifacts_path, 'rb') as f:\n                    artifacts = pickle.load(f)\n                    self.scalers = artifacts.get('scalers', {})\n                    self.encoders = artifacts.get('encoders', {})\n                    self.data_stats = artifacts.get('data_stats', {})\n            \n            print(f\"✅ Preprocessed data loaded: {df.shape}\")\n            return df\n        except Exception as e:\n            print(f\"❌ Error loading preprocessed data: {e}\")\n            return None\n\ndef preprocess_marketing_data_fixed(input_file, output_file=None):\n    \"\"\"Main preprocessing function with comprehensive data preparation\"\"\"\n    print(\"🚀 Starting FIXED data preprocessing pipeline...\")\n    \n    preprocessor = DataPreprocessorFixed()\n    \n    # Load and validate data\n    df = preprocessor.load_data(input_file)\n    if df is None:\n        return None, None\n    \n    # Clean data\n    df_clean = preprocessor.clean_data(df)\n    \n    # Encode categorical features first\n    df_encoded = preprocessor.encode_categorical_features(df_clean)\n    \n    # Engineer features\n    df_final = preprocessor.engineer_features(df_encoded)\n    \n    # Save if output path provided\n    if output_file:\n        preprocessor.save_preprocessed_data(df_final, output_file)\n    \n    print(f\"✅ FIXED preprocessing completed: {df_final.shape}\")\n    print(f\"📊 Final columns: {list(df_final.columns)}\")\n    \n    return df_final, preprocessor\n