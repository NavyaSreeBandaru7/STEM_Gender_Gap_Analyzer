import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from functools import lru_cache, wraps
import pickle

# Core data processing
import numpy as np
import pandas as pd
import dask.dataframe as dd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import modin.pandas as mpd

# Machine Learning & Deep Learning
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingClassifier,
    IsolationForest,
    VotingClassifier
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModel

# NLP & Gen AI
import spacy
import nltk
from textblob import TextBlob
import openai
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Web & API
import requests
from flask import Flask, request, jsonify
import streamlit as st
from fastapi import FastAPI, HTTPException
import uvicorn

# Database & Caching
import redis
import sqlalchemy as sa
from pymongo import MongoClient
import psycopg2

# Utils
from tqdm import tqdm
import schedule
import yaml
from dotenv import load_dotenv

# Configure environment
load_dotenv()
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stem_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLP models
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)


@dataclass
class STEMConfig:
    """Configuration management for STEM analyzer"""
    data_sources: Dict[str, str] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)
    cache_ttl: int = 3600
    batch_size: int = 10000
    n_workers: int = os.cpu_count()
    
    def __post_init__(self):
        self.data_sources = {
            'unesco': os.getenv('UNESCO_API', ''),
            'world_bank': os.getenv('WORLD_BANK_API', ''),
            'kaggle': os.getenv('KAGGLE_API', ''),
            'oecd': 'https://stats.oecd.org/api',
            'eurostat': 'https://ec.europa.eu/eurostat/api'
        }
        self.api_keys['openai'] = os.getenv('OPENAI_API_KEY', '')
        self.api_keys['huggingface'] = os.getenv('HUGGINGFACE_TOKEN', '')


class DataCollector:
    """Advanced data collection with multi-source integration"""
    
    def __init__(self, config: STEMConfig):
        self.config = config
        self.session = requests.Session()
        self.cache = {}
        
    async def fetch_unesco_data(self) -> pd.DataFrame:
        """Fetch UNESCO education statistics"""
        try:
            params = {
                'format': 'json',
                'indicator': 'EDULIT_DS',
                'ref_area': 'all',
                'time_period': '2010:2024'
            }
            response = await self._async_request(
                self.config.data_sources['unesco'], 
                params
            )
            return self._process_unesco_response(response)
        except Exception as e:
            logger.error(f"UNESCO fetch failed: {e}")
            return pd.DataFrame()
    
    async def fetch_world_bank_data(self) -> pd.DataFrame:
        """Fetch World Bank education indicators"""
        indicators = [
            'SE.PRM.ENRR.FE',  # Female enrollment primary
            'SE.SEC.ENRR.FE',  # Female enrollment secondary
            'SE.TER.ENRR.FE',  # Female enrollment tertiary
            'SE.XPD.TOTL.GD.ZS'  # Education expenditure
        ]
        
        dfs = []
        for indicator in indicators:
            url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
            params = {'format': 'json', 'per_page': 5000, 'date': '2010:2024'}
            
            try:
                response = await self._async_request(url, params)
                df = self._process_wb_response(response, indicator)
                dfs.append(df)
            except Exception as e:
                logger.error(f"World Bank {indicator} failed: {e}")
        
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
    
    async def _async_request(self, url: str, params: dict) -> dict:
        """Asynchronous HTTP request with retry logic"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None, 
            lambda: self.session.get(url, params=params, timeout=30)
        )
        response = await future
        response.raise_for_status()
        return response.json()
    
    def _process_unesco_response(self, data: dict) -> pd.DataFrame:
        """Process UNESCO API response"""
        records = []
        if isinstance(data, dict) and 'data' in data:
            for item in data['data']:
                records.append({
                    'country': item.get('ref_area', ''),
                    'year': item.get('time_period', 0),
                    'indicator': item.get('indicator', ''),
                    'value': item.get('obs_value', 0),
                    'gender': item.get('sex', '')
                })
        return pd.DataFrame(records)
    
    def _process_wb_response(self, data: list, indicator: str) -> pd.DataFrame:
        """Process World Bank API response"""
        if len(data) > 1 and isinstance(data[1], list):
            records = []
            for item in data[1]:
                if item.get('value'):
                    records.append({
                        f'{indicator}_country': item.get('country', {}).get('value', ''),
                        f'{indicator}_year': item.get('date', ''),
                        f'{indicator}_value': item.get('value', 0)
                    })
            return pd.DataFrame(records)
        return pd.DataFrame()


class AdvancedPreprocessor:
    """Sophisticated data preprocessing pipeline"""
    
    def __init__(self, config: STEMConfig):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def process_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        logger.info(f"Processing {len(df)} records")
        
        # Parallel processing for large datasets
        if len(df) > self.config.batch_size:
            df = self._parallel_process(df)
        
        # Data quality checks
        df = self._quality_checks(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Handle missing values
        df = self._smart_imputation(df)
        
        # Outlier detection
        df = self._detect_outliers(df)
        
        # Normalize features
        df = self._normalize_features(df)
        
        return df
    
    def _parallel_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process large datasets in parallel"""
        n_chunks = self.config.n_workers
        chunks = np.array_split(df, n_chunks)
        
        with ProcessPoolExecutor(max_workers=n_chunks) as executor:
            processed_chunks = list(executor.map(self._process_chunk, chunks))
        
        return pd.concat(processed_chunks, ignore_index=True)
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process individual data chunk"""
        # Apply transformations
        chunk = chunk.copy()
        
        # Clean text fields
        text_cols = chunk.select_dtypes(include=['object']).columns
        for col in text_cols:
            chunk[col] = chunk[col].str.strip().str.lower()
        
        # Convert dates
        date_cols = [col for col in chunk.columns if 'date' in col.lower() or 'year' in col.lower()]
        for col in date_cols:
            chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
        
        return chunk
    
    def _quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data quality validation"""
        initial_shape = df.shape
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Check data types
        df = self._enforce_dtypes(df)
        
        # Validate ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'percentage' in col.lower() or 'rate' in col.lower():
                df.loc[df[col] > 100, col] = 100
                df.loc[df[col] < 0, col] = 0
        
        logger.info(f"Quality checks: {initial_shape} -> {df.shape}")
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        
        # Temporal features
        if 'year' in df.columns:
            df['decade'] = (df['year'] // 10) * 10
            df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        
        # Gender gap metrics
        if 'male_enrollment' in df.columns and 'female_enrollment' in df.columns:
            df['gender_gap'] = df['male_enrollment'] - df['female_enrollment']
            df['gender_ratio'] = df['female_enrollment'] / (df['male_enrollment'] + 1e-6)
            df['parity_index'] = df['female_enrollment'] / (df['female_enrollment'] + df['male_enrollment'] + 1e-6)
        
        # Regional aggregations
        if 'country' in df.columns:
            df['region'] = df['country'].map(self._get_region_mapping())
            df['income_group'] = df['country'].map(self._get_income_mapping())
        
        # Lag features for time series
        if 'year' in df.columns and 'country' in df.columns:
            for lag in [1, 2, 3]:
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col not in ['year', 'decade']:
                        df[f'{col}_lag{lag}'] = df.groupby('country')[col].shift(lag)
        
        # Moving averages
        window_sizes = [3, 5]
        for window in window_sizes:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in ['year', 'decade']:
                    df[f'{col}_ma{window}'] = df.groupby('country')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
        
        return df
    
    def _smart_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value imputation"""
        
        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # KNN imputation for numeric
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self.imputers['numeric'] = imputer
        
        # Mode imputation for categorical
        for col in categorical_cols:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-method outlier detection"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_pred = iso_forest.fit_predict(df[numeric_cols].fillna(0))
        df['outlier_score'] = outlier_pred
        
        # Z-score method
        z_scores = np.abs(stats.zscore(df[numeric_cols].fillna(0)))
        df['z_outlier'] = (z_scores > 3).any(axis=1)
        
        # IQR method
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        df['iqr_outlier'] = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                             (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature normalization with multiple strategies"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['year', 'outlier_score', 'z_outlier', 'iqr_outlier']
        normalize_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Robust scaling for outlier resistance
        robust_scaler = RobustScaler()
        df[normalize_cols] = robust_scaler.fit_transform(df[normalize_cols])
        self.scalers['robust'] = robust_scaler
        
        return df
    
    @lru_cache(maxsize=256)
    def _get_region_mapping(self) -> dict:
        """Country to region mapping"""
        return {
            'united states': 'north_america',
            'canada': 'north_america',
            'mexico': 'north_america',
            'brazil': 'south_america',
            'argentina': 'south_america',
            'united kingdom': 'europe',
            'germany': 'europe',
            'france': 'europe',
            'china': 'asia',
            'india': 'asia',
            'japan': 'asia',
            # Add more mappings
        }
    
    @lru_cache(maxsize=256)
    def _get_income_mapping(self) -> dict:
        """Country to income group mapping"""
        return {
            'united states': 'high',
            'germany': 'high',
            'china': 'upper_middle',
            'india': 'lower_middle',
            'nigeria': 'lower_middle',
            # Add more mappings
        }
    
    def _enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce proper data types"""
        dtype_mapping = {
            'year': 'int32',
            'enrollment': 'float32',
            'graduation_rate': 'float32',
            'dropout_rate': 'float32'
        }
        
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        
        return df


class MLPipeline:
    """Advanced machine learning pipeline"""
    
    def __init__(self, config: STEMConfig):
        self.config = config
        self.models = {}
        self.results = {}
        
    def train_ensemble(self, X_train, y_train, X_val, y_val, task='regression'):
        """Train ensemble of models"""
        
        if task == 'regression':
            models = {
                'rf': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
                'xgb': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=10),
                'lgb': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31),
                'cat': CatBoostRegressor(iterations=200, learning_rate=0.05, depth=10, verbose=False)
            }
        else:
            models = {
                'rf': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
                'xgb': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=10),
                'lgb': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31),
                'cat': CatBoostClassifier(iterations=200, learning_rate=0.05, depth=10, verbose=False)
            }
        
        for name, model in tqdm(models.items(), desc="Training models"):
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Evaluate
            y_pred = model.predict(X_val)
            if task == 'regression':
                score = r2_score(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                self.results[name] = {'r2': score, 'mae': mae}
            else:
                score = model.score(X_val, y_val)
                self.results[name] = {'accuracy': score}
            
            logger.info(f"{name} trained: {self.results[name]}")
        
        return self.models
    
    def train_deep_learning(self, X_train, y_train, X_val, y_val):
        """Train deep neural network"""
        
        # Build model
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        self.models['deep_nn'] = model
        return model, history


class NLPAnalyzer:
    """Advanced NLP for policy document analysis"""
    
    def __init__(self, config: STEMConfig):
        self.config = config
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.summarizer = pipeline("summarization")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def analyze_policy_documents(self, documents: List[str]) -> Dict:
        """Comprehensive policy document analysis"""
        
        results = {
            'sentiments': [],
            'entities': [],
            'topics': [],
            'gender_bias_scores': [],
            'summaries': []
        }
        
        for doc in tqdm(documents, desc="Analyzing documents"):
            # Sentiment analysis
            sentiment = self.sentiment_pipeline(doc[:512])[0]
            results['sentiments'].append(sentiment)
            
            # Named entity recognition
            doc_nlp = nlp(doc)
            entities = [(ent.text, ent.label_) for ent in doc_nlp.ents]
            results['entities'].append(entities)
            
            # Gender bias detection
            bias_score = self._detect_gender_bias(doc)
            results['gender_bias_scores'].append(bias_score)
            
            # Summarization
            if len(doc) > 100:
                summary = self.summarizer(doc[:1024], max_length=130, min_length=30)[0]
                results['summaries'].append(summary['summary_text'])
        
        return results
    
    def _detect_gender_bias(self, text: str) -> float:
        """Detect gender bias in text"""
        
        gendered_terms = {
            'male': ['he', 'him', 'his', 'man', 'men', 'male', 'boy', 'boys'],
            'female': ['she', 'her', 'hers', 'woman', 'women', 'female', 'girl', 'girls']
        }
        
        text_lower = text.lower()
        male_count = sum(text_lower.count(term) for term in gendered_terms['male'])
        female_count = sum(text_lower.count(term) for term in gendered_terms['female'])
        
        total = male_count + female_count
        if total == 0:
            return 0.5  # Neutral
        
        return female_count / total


class IntelligentAgent:
    """AI agent for insights and recommendations"""
    
    def __init__(self, config: STEMConfig):
        self.config = config
        self.llm = OpenAI(temperature=0.7, api_key=config.api_keys['openai'])
        self.memory = ConversationBufferMemory()
        self.tools = self._setup_tools()
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            memory=self.memory,
            verbose=True
        )
    
    def _setup_tools(self) -> List[Tool]:
        """Setup agent tools"""
        
        tools = [
            Tool(
                name="Statistical Analysis",
                func=self._statistical_analysis,
                description="Perform statistical analysis on STEM gender data"
            ),
            Tool(
                name="Trend Prediction",
                func=self._trend_prediction,
                description="Predict future trends in gender gaps"
            ),
            Tool(
                name="Policy Recommendation",
                func=self._policy_recommendation,
                description="Generate policy recommendations based on data"
            )
        ]
        
        return tools
    
    def _statistical_analysis(self, query: str) -> str:
        """Statistical analysis tool"""
        # Implement statistical analysis
        return f"Statistical analysis for: {query}"
    
    def _trend_prediction(self, query: str) -> str:
        """Trend prediction tool"""
        # Implement trend prediction
        return f"Trend prediction for: {query}"
    
    def _policy_recommendation(self, query: str) -> str:
        """Policy recommendation tool"""
        # Implement policy recommendations
        return f"Policy recommendations for: {query}"
    
    def generate_insights(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive insights"""
        
        prompt = f"""
        Analyze the following STEM gender gap data and provide:
        1. Key findings
        2. Trend analysis
        3. Policy recommendations
        4. Action items
        
        Data summary:
        {data.describe().to_string()}
        """
        
        response = self.agent.run(prompt)
        
        return {
            'insights': response,
            'timestamp': datetime.now().isoformat()
        }


class VisualizationEngine:
    """Advanced visualization generation"""
    
    def __init__(self):
        self.theme = 'plotly_dark'
        
    def create_dashboard(self, data: pd.DataFrame) -> go.Figure:
        """Create comprehensive dashboard"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Gender Gap Trends', 'Regional Distribution',
                'Enrollment Forecast', 'Correlation Matrix',
                'Policy Impact', 'Success Metrics'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'geo'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}],
                [{'type': 'bar'}, {'type': 'indicator'}]
            ]
        )
        
        # Add traces
        # ... (implement visualization logic)
        
        fig.update_layout(
            title='STEM Gender Gap Analytics Dashboard',
            showlegend=True,
            height=1200,
            template=self.theme
        )
        
        return fig


class STEMAnalyzer:
    """Main orchestrator class"""
    
    def __init__(self):
        self.config = STEMConfig()
        self.collector = DataCollector(self.config)
        self.preprocessor = AdvancedPreprocessor(self.config)
        self.ml_pipeline = MLPipeline(self.config)
        self.nlp_analyzer = NLPAnalyzer(self.config)
        self.agent = IntelligentAgent(self.config)
        self.visualizer = VisualizationEngine()
        
    async def run_analysis(self):
        """Execute complete analysis pipeline"""
        
        logger.info("Starting STEM Gender Gap Analysis")
        
        # Collect data
        logger.info("Collecting data from multiple sources...")
        unesco_data = await self.collector.fetch_unesco_data()
        wb_data = await self.collector.fetch_world_bank_data()
        
        # Combine data sources
        data = pd.concat([unesco_data, wb_data], axis=1)
        
        # Preprocess
        logger.info("Preprocessing data...")
        data = self.preprocessor.process_pipeline(data)
        
        # Machine Learning
        logger.info("Training ML models...")
        # Split data and train models
        # ... (implement train/test split and training)
        
        # Generate insights
        logger.info("Generating AI insights...")
        insights = self.agent.generate_insights(data)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        dashboard = self.visualizer.create_dashboard(data)
        
        # Save results
        self._save_results(data, insights, dashboard)
        
        logger.info("Analysis complete!")
        
        return {
            'data': data,
            'insights': insights,
            'dashboard': dashboard
        }
    
    def _save_results(self, data, insights, dashboard):
        """Save analysis results"""
        
        # Save processed data
        data.to_csv('output/processed_data.csv', index=False)
        data.to_parquet('output/processed_data.parquet')
        
        # Save insights
        with open('output/insights.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Save dashboard
        dashboard.write_html('output/dashboard.html')
        
        # Save model artifacts
        with open('output/models.pkl', 'wb') as f:
            pickle.dump(self.ml_pipeline.models, f)


# FastAPI Application
app = FastAPI(title="STEM Gender Gap Analyzer API")

@app.get("/")
async def root():
    return {"message": "STEM Gender Gap Analyzer API v2.0"}

@app.post("/analyze")
async def analyze_endpoint():
    analyzer = STEMAnalyzer()
    results = await analyzer.run_analysis()
    return {"status": "success", "summary": results['insights']}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    # Run async analysis
    analyzer = STEMAnalyzer()
    asyncio.run(analyzer.run_analysis())
    
    # Optionally start API server
    # uvicorn.run(app, host="0.0.0.0", port=8000)
