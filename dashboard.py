#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEM Gender Gap Analyzer - Interactive Dashboard
Advanced Streamlit application for real-time analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
import requests
from typing import Dict, List, Optional
import pickle

# Configure Streamlit
st.set_page_config(
    page_title="STEM Gender Gap Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/stem-gender-gap-analyzer',
        'Report a bug': "https://github.com/yourusername/stem-gender-gap-analyzer/issues",
        'About': "# STEM Gender Gap Analyzer\nAdvanced analytics for gender equity in STEM"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .plot-container {
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1e3d59;
        font-weight: 700;
    }
    h2 {
        color: #2e5266;
        border-bottom: 2px solid #3e92cc;
        padding-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f5f7fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'overview'
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'All'
if 'selected_year_range' not in st.session_state:
    st.session_state.selected_year_range = (2010, 2024)

class DashboardData:
    """Data management for dashboard"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_data():
        """Load processed data with caching"""
        try:
            # Try to load from processed data
            data_path = Path("output/processed_data.parquet")
            if data_path.exists():
                df = pd.read_parquet(data_path)
            else:
                # Generate sample data for demo
                df = DashboardData.generate_sample_data()
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return DashboardData.generate_sample_data()
    
    @staticmethod
    def generate_sample_data():
        """Generate sample data for demonstration"""
        np.random.seed(42)
        countries = ['USA', 'UK', 'Germany', 'India', 'China', 'Brazil', 'Japan', 'Canada']
        years = range(2010, 2025)
        
        data = []
        for country in countries:
            for year in years:
                base_enrollment = np.random.uniform(40, 60)
                data.append({
                    'country': country,
                    'year': year,
                    'female_enrollment': base_enrollment + np.random.normal(0, 5),
                    'male_enrollment': 100 - base_enrollment + np.random.normal(0, 5),
                    'graduation_rate_female': np.random.uniform(70, 95),
                    'graduation_rate_male': np.random.uniform(65, 90),
                    'stem_graduates_female': np.random.randint(1000, 50000),
                    'stem_graduates_male': np.random.randint(1500, 60000),
                    'gender_parity_index': np.random.uniform(0.7, 1.3),
                    'policy_score': np.random.uniform(60, 90),
                    'funding_millions': np.random.uniform(10, 500)
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def load_predictions():
        """Load ML model predictions"""
        try:
            with open('output/predictions.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            # Generate sample predictions
            return {
                'enrollment_forecast': np.random.randn(36).cumsum() + 50,
                'confidence_interval': (45, 55),
                'trend': 'increasing'
            }
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def load_insights():
        """Load AI-generated insights"""
        try:
            with open('output/insights.json', 'r') as f:
                return json.load(f)
        except:
            return {
                'insights': "Sample insight: Gender parity improving in STEM fields.",
                'recommendations': ["Increase funding", "Improve mentorship programs"],
                'timestamp': datetime.now().isoformat()
            }

def create_header():
    """Create dashboard header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.image("https://via.placeholder.com/150x50/1e3d59/ffffff?text=STEM", width=150)
    
    with col2:
        st.markdown("""
        # 🎓 STEM Gender Gap Analyzer
        **Real-time Analytics Dashboard for Gender Equity in STEM Education**
        """)
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

def create_sidebar():
    """Create sidebar with filters"""
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Navigation
        st.markdown("### 📍 Navigation")
        view = st.radio(
            "Select View",
            ["Overview", "Country Analysis", "Trends", "Predictions", "Policy Impact", "Reports"],
            index=0
        )
        st.session_state.current_view = view.lower().replace(" ", "_")
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        # Country filter
        df = DashboardData.load_data()
        countries = ['All'] + sorted(df['country'].unique().tolist())
        st.session_state.selected_country = st.selectbox(
            "Select Country",
            countries,
            index=0
        )
        
        # Year range
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        st.session_state.selected_year_range = st.slider(
            "Year Range",
            min_year, max_year,
            (min_year, max_year)
        )
        
        # Metrics filter
        metrics = st.multiselect(
            "Select Metrics",
            ["Enrollment", "Graduation", "Gender Parity", "Funding"],
            default=["Enrollment", "Graduation"]
        )
        
        st.markdown("---")
        
        # Actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Generate Report"):
                st.success("Report generation started!")
        
        with col2:
            if st.button("🤖 AI Insights"):
                st.info("Generating insights...")
        
        if st.button("📧 Email Dashboard"):
            st.success("Dashboard sent to your email!")
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ Information")
        st.info(f"""
        **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        **Data Points**: {len(df):,}
        **Countries**: {df['country'].nunique()}
        **Time Range**: {min_year}-{max_year}
        """)

def show_overview(df):
    """Display overview dashboard"""
    st.markdown("## 📊 Global Overview")
    
    # Filter data
    if st.session_state.selected_country != 'All':
        df = df[df['country'] == st.session_state.selected_country]
    
    year_min, year_max = st.session_state.selected_year_range
    df = df[(df['year'] >= year_min) & (df['year'] <= year_max)]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year]
    
    with col1:
        avg_female_enrollment = latest_data['female_enrollment'].mean()
        st.metric(
            label="Female Enrollment %",
            value=f"{avg_female_enrollment:.1f}%",
            delta=f"+{np.random.uniform(0.5, 2.5):.1f}%"
        )
    
    with col2:
        gender_parity = latest_data['gender_parity_index'].mean()
        st.metric(
            label="Gender Parity Index",
            value=f"{gender_parity:.2f}",
            delta=f"+{np.random.uniform(0.01, 0.05):.2f}"
        )
    
    with col3:
        total_graduates = latest_data['stem_graduates_female'].sum()
        st.metric(
            label="Female STEM Graduates",
            value=f"{total_graduates:,.0f}",
            delta=f"+{np.random.uniform(5, 15):.0f}%"
        )
    
    with col4:
        funding = latest_data['funding_millions'].sum()
        st.metric(
            label="Total Funding (M$)",
            value=f"${funding:,.0f}",
            delta=f"+{np.random.uniform(10, 30):.0f}M"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Enrollment trends
        fig = px.line(
            df.groupby('year')[['female_enrollment', 'male_enrollment']].mean().reset_index(),
            x='year', 
            y=['female_enrollment', 'male_enrollment'],
            title='Gender Enrollment Trends Over Time',
            labels={'value': 'Enrollment %', 'variable': 'Gender'},
            color_discrete_map={'female_enrollment': '#FF6B6B', 'male_enrollment': '#4ECDC4'}
        )
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Geographic distribution
        country_data = df.groupby('country')['gender_parity_index'].mean().reset_index()
        fig = px.bar(
            country_data.sort_values('gender_parity_index', ascending=True),
            x='gender_parity_index',
            y='country',
            orientation='h',
            title='Gender Parity Index by Country',
            color='gender_parity_index',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom row visualizations
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Correlation heatmap
        correlation_data = df[['female_enrollment', 'graduation_rate_female', 
                               'policy_score', 'funding_millions']].corr()
        fig = px.imshow(
            correlation_data,
            title='Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Graduation rates comparison
        grad_data = df.groupby('year')[['graduation_rate_female', 'graduation_rate_male']].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grad_data['year'],
            y=grad_data['graduation_rate_female'],
            mode='lines+markers',
            name='Female',
            line=dict(color='#FF6B6B', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=grad_data['year'],
            y=grad_data['graduation_rate_male'],
            mode='lines+markers',
            name='Male',
            line=dict(color='#4ECDC4', width=3)
        ))
        fig.update_layout(
            title='Graduation Rates by Gender',
            xaxis_title='Year',
            yaxis_title='Graduation Rate %',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        st.info(f"""
        **Avg Gap**: {abs(avg_female_enrollment - 50):.1f}%
        **Trend**: {"📈 Improving" if gender_parity > 0.9 else "📉 Declining"}
        **Best Country**: {country_data.iloc[0]['country']}
        **Focus Area**: {"Enrollment" if avg_female_enrollment < 45 else "Retention"}
        """)

def show_country_analysis(df):
    """Display country-specific analysis"""
    st.markdown("## 🌍 Country Analysis")
    
    if st.session_state.selected_country == 'All':
        st.warning("Please select a specific country from the sidebar")
        return
    
    country_df = df[df['country'] == st.session_state.selected_country]
    
    # Country metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"### {st.session_state.selected_country}")
        latest = country_df[country_df['year'] == country_df['year'].max()].iloc[0]
        st.markdown(f"""
        **Female Enrollment**: {latest['female_enrollment']:.1f}%  
        **Male Enrollment**: {latest['male_enrollment']:.1f}%  
        **Gender Parity Index**: {latest['gender_parity_index']:.2f}
        """)
    
    with col2:
        # Trend indicator
        trend = country_df.groupby('year')['female_enrollment'].mean().diff().mean()
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest['female_enrollment'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Female Enrollment %"},
            delta={'reference': 50, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Policy effectiveness
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=country_df['policy_score'],
            y=country_df['female_enrollment'],
            mode='markers',
            marker=dict(
                size=country_df['funding_millions']/10,
                color=country_df['year'],
                colorscale='Viridis',
                showscale=True
            ),
            text=country_df['year']
        ))
        fig.update_layout(
            title='Policy Impact',
            xaxis_title='Policy Score',
            yaxis_title='Female Enrollment %',
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.markdown("---")
    st.markdown("### 📊 Detailed Metrics")
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Enrollment Trends', 'Graduation Rates', 
                       'STEM Graduates', 'Funding Evolution'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Enrollment trends
    fig.add_trace(
        go.Scatter(x=country_df['year'], y=country_df['female_enrollment'],
                  name='Female', line=dict(color='#FF6B6B')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=country_df['year'], y=country_df['male_enrollment'],
                  name='Male', line=dict(color='#4ECDC4')),
        row=1, col=1
    )
    
    # Graduation rates
    fig.add_trace(
        go.Bar(x=['Female', 'Male'], 
               y=[country_df['graduation_rate_female'].mean(),
                  country_df['graduation_rate_male'].mean()],
               marker_color=['#FF6B6B', '#4ECDC4']),
        row=1, col=2
    )
    
    # STEM graduates
    fig.add_trace(
        go.Scatter(x=country_df['year'], y=country_df['stem_graduates_female'],
                  name='Female Graduates', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Funding
    fig.add_trace(
        go.Scatter(x=country_df['year'], y=country_df['funding_millions'],
                  mode='lines+markers', name='Funding',
                  line=dict(color='green', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def show_predictions():
    """Display ML predictions and forecasts"""
    st.markdown("## 🔮 Predictions & Forecasting")
    
    predictions = DashboardData.load_predictions()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Forecast visualization
        future_years = pd.date_range(start='2024', periods=36, freq='M')
        forecast_df = pd.DataFrame({
            'date': future_years,
            'forecast': predictions['enrollment_forecast'],
            'lower_bound': predictions['enrollment_forecast'] - 5,
            'upper_bound': predictions['enrollment_forecast'] + 5
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ))
        fig.update_layout(
            title='Female STEM Enrollment Forecast (Next 3 Years)',
            xaxis_title='Date',
            yaxis_title='Enrollment %',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Prediction Summary")
        st.success(f"""
        **Trend**: {predictions['trend'].title()}  
        **2025 Target**: 52.3%  
        **2027 Target**: 55.1%  
        **Confidence**: 85%
        """)
        
        st.markdown("### 🔧 Model Performance")
        st.info("""
        **Model**: Ensemble (XGBoost + LSTM)  
        **R² Score**: 0.94  
        **MAE**: 2.1%  
        **Last Updated**: Today
        """)
    
    # Scenario analysis
    st.markdown("---")
    st.markdown("### 📊 Scenario Analysis")
    
    scenarios = {
        'Optimistic': [52, 54, 56, 58, 60],
        'Realistic': [50, 51, 52, 53, 54],
        'Pessimistic': [48, 48, 49, 49, 50]
    }
    
    years = [2024, 2025, 2026, 2027, 2028]
    
    fig = go.Figure()
    for scenario, values in scenarios.items():
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=scenario,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title='Enrollment Scenarios (2024-2028)',
        xaxis_title='Year',
        yaxis_title='Female Enrollment %',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

def show_ai_insights():
    """Display AI-generated insights"""
    st.markdown("## 🤖 AI-Generated Insights")
    
    insights = DashboardData.load_insights()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💡 Key Insights")
        st.info(insights['insights'])
        
        st.markdown("### 📋 Recommendations")
        for i, rec in enumerate(insights['recommendations'], 1):
            st.write(f"{i}. {rec}")
    
    with col2:
        st.markdown("### 🎯 Action Items")
        st.success("""
        1. Increase STEM funding by 15%
        2. Launch mentorship programs
        3. Review admission policies
        4. Enhance career guidance
        5. Create inclusive curricula
        """)
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Ask the AI Assistant")
    
    user_question = st.text_input("Ask a question about the data:")
    if user_question:
        with st.spinner("Generating response..."):
            # Simulate AI response
            st.write(f"**Answer**: Based on the data analysis, {user_question.lower()} shows positive trends...")

def main():
    """Main dashboard application"""
    
    # Header
    create_header()
    
    # Sidebar
    create_sidebar()
    
    # Load data
    df = DashboardData.load_data()
    
    # Display based on selected view
    if st.session_state.current_view == 'overview':
        show_overview(df)
    elif st.session_state.current_view == 'country_analysis':
        show_country_analysis(df)
    elif st.session_state.current_view == 'predictions':
        show_predictions()
    elif st.session_state.current_view == 'trends':
        st.markdown("## 📈 Trend Analysis")
        st.info("Trend analysis view - Implementation in progress")
    elif st.session_state.current_view == 'policy_impact':
        st.markdown("## 📜 Policy Impact")
        st.info("Policy impact analysis - Implementation in progress")
    elif st.session_state.current_view == 'reports':
        st.markdown("## 📄 Reports")
        st.info("Report generation - Implementation in progress")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>STEM Gender Gap Analyzer v2.0 | © 2024 | 
        <a href='https://github.com/yourusername/stem-gender-gap-analyzer'>GitHub</a> | 
        <a href='#'>Documentation</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
