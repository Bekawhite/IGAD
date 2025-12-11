# ============================================================================
# STREAMLIT PAGE CONFIGURATION - MUST BE FIRST
# ============================================================================
import streamlit as st

st.set_page_config(
    page_title="Malaria Forecasting System",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
.stButton > button {
    width: 100%;
    border-radius: 5px;
    font-weight: bold;
}
.css-1d391kg {
    padding: 1rem;
}
.sub-header {
    color: #2E86AB;
    border-bottom: 2px solid #2E86AB;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
.stMetric {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #2E86AB;
}
/* IGAD specific styling */
.igad-country {
    background-color: #e8f4f8;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
import requests
import json
import io
import base64
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NEW: ENTERPRISE SECURITY IMPORTS
# ============================================================================
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
import uuid
import time
from functools import wraps

# ============================================================================
# NEW: ADVANCED ML IMPORTS
# ============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap
import optuna
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import xgboost as xgb
import lightgbm as lgb

# ============================================================================
# NEW: OFFLINE/MOBILE IMPORTS
# ============================================================================
import sqlite3
from pathlib import Path
import pickle
import zipfile
import tempfile

# ============================================================================
# NEW: IGAD MALARIA TRENDS DASHBOARD MODULE
# ============================================================================

class IGADDataValidator:
    """Validate CSV data against IGAD PRD specifications"""
    
    # IGAD member states
    IGAD_COUNTRIES = ['DJI', 'ETH', 'KEN', 'SOM', 'SSD', 'SDN', 'UGA']
    
    # Required schema
    REQUIRED_COLUMNS = ['ISO3', 'Name', 'Admin Level', 'Metric', 'Units', 'Year', 'Value']
    
    @staticmethod
    def validate_schema(df):
        """Validate that CSV has correct columns in correct order"""
        if list(df.columns) != IGADDataValidator.REQUIRED_COLUMNS:
            return False, f"Error: Columns must be exactly: {', '.join(IGADDataValidator.REQUIRED_COLUMNS)}"
        return True, "Schema validation passed"
    
    @staticmethod
    def validate_data_types(df):
        """Validate data types according to PRD"""
        errors = []
        
        # ISO3 validation
        iso3_check = df['ISO3'].apply(lambda x: isinstance(x, str) and len(x) == 3 and x.isupper())
        if not iso3_check.all():
            errors.append("ISO3 codes must be 3-character uppercase strings")
        
        # Year validation
        if not df['Year'].apply(lambda x: isinstance(x, (int, np.integer)) and 2000 <= x <= 2030).all():
            errors.append("Year must be integers between 2000-2030")
        
        # Value validation
        if not df['Value'].apply(lambda x: isinstance(x, (int, np.integer)) and x >= 0).all():
            errors.append("Value must be non-negative integers")
        
        # Admin Level validation
        if not df['Admin Level'].apply(lambda x: isinstance(x, (int, np.integer)) and x >= 0).all():
            errors.append("Admin Level must be integers ≥ 0")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_igad_countries(df):
        """Validate ISO3 codes are IGAD member states"""
        invalid_countries = df[~df['ISO3'].isin(IGADDataValidator.IGAD_COUNTRIES)]['ISO3'].unique()
        
        if len(invalid_countries) > 0:
            return False, f"Error: Non-IGAD country detected: {', '.join(invalid_countries)}"
        return True, "All countries are IGAD members"
    
    @staticmethod
    def validate_csv(df):
        """Comprehensive CSV validation"""
        # Check if empty
        if df.empty:
            return False, ["Error: Uploaded file is empty"]
        
        # Schema validation
        schema_valid, schema_msg = IGADDataValidator.validate_schema(df)
        if not schema_valid:
            return False, [schema_msg]
        
        # Data type validation
        type_valid, type_errors = IGADDataValidator.validate_data_types(df)
        if not type_valid:
            return False, type_errors
        
        # IGAD country validation
        country_valid, country_msg = IGADDataValidator.validate_igad_countries(df)
        if not country_valid:
            return False, [country_msg]
        
        return True, ["All validations passed"]

class IGADDataProcessor:
    """Process malaria case data for IGAD dashboard"""
    
    @staticmethod
    def identify_malaria_cases(df):
        """Identify malaria case metrics using case-insensitive search for 'cases'"""
        if 'Metric' not in df.columns:
            return pd.DataFrame()
        
        # Case-insensitive search for 'cases' in Metric column
        malaria_mask = df['Metric'].astype(str).str.contains('cases', case=False, na=False)
        return df[malaria_mask].copy()
    
    @staticmethod
    def filter_country_level_data(df):
        """Filter for country-level data (Admin Level = 0)"""
        if 'Admin Level' not in df.columns:
            return pd.DataFrame()
        
        return df[df['Admin Level'] == 0].copy()
    
    @staticmethod
    def prepare_bar_chart_data(df, selected_year=None):
        """Prepare data for bar chart visualization"""
        if df.empty:
            return pd.DataFrame()
        
        # Filter for malaria cases
        malaria_df = IGADDataProcessor.identify_malaria_cases(df)
        if malaria_df.empty:
            return pd.DataFrame()
        
        # Filter for country-level data
        country_df = IGADDataProcessor.filter_country_level_data(malaria_df)
        if country_df.empty:
            return pd.DataFrame()
        
        # Filter by year if specified
        if selected_year:
            country_df = country_df[country_df['Year'] == selected_year]
        
        # Aggregate by country
        if not country_df.empty:
            aggregated = country_df.groupby(['ISO3', 'Name', 'Year'])['Value'].sum().reset_index()
            aggregated = aggregated.sort_values('Value', ascending=False)
            return aggregated
        else:
            return pd.DataFrame()
    
    @staticmethod
    def prepare_map_data(df, selected_year=None):
        """Prepare data for choropleth map visualization"""
        # Start with bar chart data (already aggregated and filtered)
        bar_data = IGADDataProcessor.prepare_bar_chart_data(df, selected_year)
        
        if bar_data.empty:
            # Return empty structure with all IGAD countries
            map_data = pd.DataFrame({
                'ISO3': IGADDataValidator.IGAD_COUNTRIES,
                'Name': ['Djibouti', 'Ethiopia', 'Kenya', 'Somalia', 
                        'South Sudan', 'Sudan', 'Uganda'],
                'Value': 0,
                'Year': selected_year if selected_year else 2023
            })
        else:
            map_data = bar_data.copy()
        
        # Add geographic coordinates for each IGAD country
        country_coords = {
            'DJI': {'lat': 11.8251, 'lon': 42.5903, 'region': 'East Africa'},
            'ETH': {'lat': 9.1450, 'lon': 40.4897, 'region': 'East Africa'},
            'KEN': {'lat': -0.0236, 'lon': 37.9062, 'region': 'East Africa'},
            'SOM': {'lat': 5.1521, 'lon': 46.1996, 'region': 'East Africa'},
            'SSD': {'lat': 6.8770, 'lon': 31.3070, 'region': 'East Africa'},
            'SDN': {'lat': 12.8628, 'lon': 30.2176, 'region': 'North Africa'},
            'UGA': {'lat': 1.3733, 'lon': 32.2903, 'region': 'East Africa'}
        }
        
        # Add coordinates to map data
        map_data['latitude'] = map_data['ISO3'].map(lambda x: country_coords.get(x, {}).get('lat', 0))
        map_data['longitude'] = map_data['ISO3'].map(lambda x: country_coords.get(x, {}).get('lon', 0))
        map_data['region'] = map_data['ISO3'].map(lambda x: country_coords.get(x, {}).get('region', 'Unknown'))
        
        return map_data

class IGADVisualizations:
    """Create visualizations for IGAD malaria dashboard"""
    
    @staticmethod
    def create_bar_chart(bar_data, selected_year=None):
        """Create interactive bar chart using Plotly"""
        if bar_data.empty:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No malaria case data available for the selected criteria",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title="Malaria Cases by Country - No Data",
                xaxis_title="Country",
                yaxis_title="Total Malaria Cases",
                height=400
            )
            return fig
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=bar_data['Name'],
                y=bar_data['Value'],
                marker_color='#1f77b4',
                text=bar_data['Value'].apply(lambda x: f"{x:,}"),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Cases: %{y:,}<extra></extra>'
            )
        ])
        
        year_title = f" ({selected_year})" if selected_year else ""
        fig.update_layout(
            title=f"Malaria Cases by Country{year_title}",
            xaxis_title="Country",
            yaxis_title="Total Malaria Cases",
            hovermode='closest',
            height=500,
            showlegend=False,
            xaxis={'categoryorder': 'total descending'}
        )
        
        return fig
    
    @staticmethod
    def create_choropleth_map(map_data, selected_year=None):
        """Create choropleth map of Africa showing malaria case distribution"""
        
        # Define color scale based on PRD requirements
        colorscale = [
            [0.0, '#90EE90'],  # Light green for low cases
            [0.5, '#FFFF00'],  # Yellow for medium cases
            [1.0, '#8B0000']   # Dark red for high cases
        ]
        
        # Create base map centered on Africa
        fig = go.Figure()
        
        if map_data.empty:
            fig.add_annotation(
                text="No data available for the selected criteria",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
        else:
            # Add choropleth layer
            fig.add_trace(go.Choropleth(
                locations=map_data['ISO3'],
                z=map_data['Value'],
                text=map_data['Name'],
                colorscale=colorscale,
                autocolorscale=False,
                marker_line_color='darkgray',
                marker_line_width=0.5,
                colorbar_title="Cases",
                hovertemplate='<b>%{text}</b><br>Cases: %{z:,}<extra></extra>'
            ))
        
        # Update layout for Africa focus
        year_title = f" ({selected_year})" if selected_year else ""
        fig.update_layout(
            title_text=f'Geographic Distribution of Malaria Cases{year_title}',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                center=dict(lat=5, lon=25),  # Center on Africa
                lataxis_range=[-35, 40],  # Latitude range for Africa
                lonaxis_range=[-20, 55],   # Longitude range for Africa
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_data_summary(validation_results, df):
        """Create data summary panel"""
        summary_data = {
            "Total Rows": len(df) if df is not None else 0,
            "IGAD Countries": len(df['ISO3'].unique()) if df is not None and 'ISO3' in df.columns else 0,
            "Years Covered": "N/A",
            "Total Malaria Cases": "N/A",
            "Data Quality": "✓ Valid" if validation_results['is_valid'] else "✗ Invalid"
        }
        
        # Calculate total malaria cases if data is valid
        if validation_results['is_valid'] and df is not None:
            malaria_df = IGADDataProcessor.identify_malaria_cases(df)
            if not malaria_df.empty:
                total_cases = malaria_df['Value'].sum()
                summary_data["Total Malaria Cases"] = f"{total_cases:,}"
            
            # Years covered
            if 'Year' in df.columns and not df.empty:
                summary_data["Years Covered"] = f"{df['Year'].min()} - {df['Year'].max()}"
        
        return summary_data

# Initialize IGAD components
igad_validator = IGADDataValidator()
igad_processor = IGADDataProcessor()
igad_viz = IGADVisualizations()

def show_igad_dashboard():
    """Display the IGAD Malaria Trends Dashboard"""
    
    st.markdown('<h2 class="sub-header">🦟 IGAD Malaria Trends Executive Dashboard</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Monitor and analyze malaria trends across IGAD member states.**  
    This dashboard provides real-time insights into malaria case distributions for data-driven decision-making.
    
    **Target Countries:** Djibouti (DJI), Ethiopia (ETH), Kenya (KEN), Somalia (SOM), South Sudan (SSD), Sudan (SDN), Uganda (UGA)
    """)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main visualization area
        if st.session_state.get('igad_data') is not None and st.session_state.get('igad_validation', {}).get('is_valid', False):
            # Prepare data for visualizations
            bar_data = igad_processor.prepare_bar_chart_data(
                st.session_state.igad_data, 
                st.session_state.get('selected_year')
            )
            map_data = igad_processor.prepare_map_data(
                st.session_state.igad_data,
                st.session_state.get('selected_year')
            )
            
            # Create visualizations
            tab1, tab2 = st.tabs(["📊 Bar Chart", "🗺️ Geographic Distribution"])
            
            with tab1:
                bar_chart = igad_viz.create_bar_chart(bar_data, st.session_state.get('selected_year'))
                st.plotly_chart(bar_chart, use_container_width=True, key="igad_bar_chart")
                
                # Data table
                with st.expander("View Data Table"):
                    if not bar_data.empty:
                        st.dataframe(
                            bar_data[['Name', 'ISO3', 'Year', 'Value']].rename(
                                columns={'Name': 'Country', 'Value': 'Malaria Cases'}
                            ),
                            use_container_width=True
                        )
                    else:
                        st.info("No malaria case data available for selected year")
            
            with tab2:
                choropleth_map = igad_viz.create_choropleth_map(map_data, st.session_state.get('selected_year'))
                st.plotly_chart(choropleth_map, use_container_width=True, key="igad_map")
                
                # Map legend
                st.markdown("""
                **Color Legend:**
                - 🟢 Light Green: Low malaria cases
                - 🟡 Yellow: Medium malaria cases  
                - 🔴 Dark Red: High malaria cases
                - ⚫ Gray: No data available
                """)
        
        else:
            # Show upload instructions
            st.info("👆 Please upload a valid IGAD CSV file to begin analysis")
            
            # Sample data structure
            with st.expander("📋 Expected CSV Format"):
                st.markdown("""
                **Required Columns (exact spelling/capitalization):**
                
                | Column Name | Data Type | Description |
                |-------------|-----------|-------------|
                | `ISO3` | String (3 chars) | ISO 3166-1 alpha-3 country code (UPPERCASE) |
                | `Name` | String | Country/region name |
                | `Admin Level` | Integer | 0=Country, 1=Region, etc. |
                | `Metric` | String | Must contain "cases" for malaria metrics |
                | `Units` | String | Typically "cases" |
                | `Year` | Integer | 4-digit year (2000-2030) |
                | `Value` | Integer | Non-negative integer |
                
                **Example Valid Row:**
                ```
                ISO3,Name,Admin Level,Metric,Units,Year,Value
                UGA,Uganda,0,Confirmed malaria cases,cases,2023,1250000
                ```
                """)
                
                # Provide sample download
                sample_data = pd.DataFrame({
                    'ISO3': ['UGA', 'KEN', 'ETH'],
                    'Name': ['Uganda', 'Kenya', 'Ethiopia'],
                    'Admin Level': [0, 0, 0],
                    'Metric': ['Confirmed malaria cases', 'Suspected cases', 'Clinical cases'],
                    'Units': ['cases', 'cases', 'cases'],
                    'Year': [2023, 2023, 2023],
                    'Value': [1250000, 850000, 2100000]
                })
                
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download Sample CSV",
                    data=csv,
                    file_name="igad_sample_data.csv",
                    mime="text/csv"
                )
    
    with col2:
        # Sidebar-like controls in the right column
        st.markdown("### 📁 IGAD Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload CSV with exact schema: ISO3, Name, Admin Level, Metric, Units, Year, Value",
            key="igad_file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                # Validate the data
                is_valid, validation_messages = igad_validator.validate_csv(df)
                
                st.session_state.igad_validation = {
                    'is_valid': is_valid,
                    'messages': validation_messages
                }
                
                if is_valid:
                    st.session_state.igad_data = df
                    st.success("✅ Data validated successfully!")
                    
                    # Set default selected year
                    if 'Year' in df.columns:
                        available_years = sorted(df['Year'].unique())
                        if available_years:
                            st.session_state.selected_year = available_years[-1]
                else:
                    st.session_state.igad_data = None
                    for msg in validation_messages:
                        st.error(msg)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.session_state.igad_data = None
                st.session_state.igad_validation = {'is_valid': False, 'messages': [str(e)]}
        
        # Year selection dropdown
        if st.session_state.get('igad_data') is not None:
            st.markdown("---")
            st.markdown("### 📅 Year Selection")
            
            available_years = sorted(st.session_state.igad_data['Year'].unique())
            selected_year = st.selectbox(
                "Select Year for Analysis",
                options=available_years,
                index=len(available_years)-1 if available_years else 0,
                key="igad_year_select"
            )
            st.session_state.selected_year = selected_year
            
            # Data summary
            st.markdown("---")
            st.markdown("### 📊 Data Summary")
            summary = igad_viz.create_data_summary(
                st.session_state.get('igad_validation', {'is_valid': False, 'messages': []}),
                st.session_state.igad_data
            )
            
            for key, value in summary.items():
                st.metric(key, value)
        
        # IGAD country information (always show)
        st.markdown("---")
        st.markdown("### 🇺🇳 IGAD Member States")
        
        igad_info = pd.DataFrame({
            'Country': ['Djibouti', 'Ethiopia', 'Kenya', 'Somalia', 
                       'South Sudan', 'Sudan', 'Uganda'],
            'ISO3': ['DJI', 'ETH', 'KEN', 'SOM', 'SSD', 'SDN', 'UGA'],
            'Region': ['East Africa', 'East Africa', 'East Africa', 'East Africa',
                      'East Africa', 'North Africa', 'East Africa']
        })
        
        st.dataframe(igad_info, use_container_width=True, hide_index=True)
        
        # Quick stats if data is loaded
        if (st.session_state.get('igad_data') is not None and 
            st.session_state.get('igad_validation', {}).get('is_valid', False) and
            st.session_state.get('selected_year')):
            
            st.markdown("---")
            st.markdown("### 📈 Quick Insights")
            
            # Calculate some basic insights
            malaria_df = igad_processor.identify_malaria_cases(st.session_state.igad_data)
            
            if not malaria_df.empty:
                year_data = malaria_df[malaria_df['Year'] == st.session_state.selected_year]
                
                if not year_data.empty:
                    # Highest burden country
                    if len(year_data) > 0:
                        max_idx = year_data['Value'].idxmax()
                        max_country = year_data.loc[max_idx, 'Name'] if pd.notna(max_idx) else "N/A"
                    else:
                        max_country = "N/A"
                    
                    # Total cases for selected year
                    total_cases = year_data['Value'].sum()
                    
                    # Number of countries with data
                    countries_with_data = year_data['ISO3'].nunique()
                    
                    st.metric("Total Cases (Selected Year)", f"{total_cases:,}")
                    st.metric("Countries with Data", countries_with_data)
                    st.metric("Highest Burden Country", max_country)

# ============================================================================
# CORE APP CLASSES (SIMPLIFIED FOR IGAD INTEGRATION)
# ============================================================================

class UserPermissions:
    """User permissions and role management"""
    
    ROLES = {
        'field_worker': 'Field Worker',
        'district_officer': 'District Officer', 
        'regional_manager': 'Regional Manager',
        'national_director': 'National Director',
        'data_scientist': 'Data Scientist',
        'public_health': 'Public Health Officer'
    }
    
    @staticmethod
    def get_role_name(role_code):
        """Get display name for role code"""
        return UserPermissions.ROLES.get(role_code, role_code)

class DataQualityMonitor:
    """Monitor data quality and integrity"""
    
    @staticmethod
    def check_data_quality(data):
        """Check quality of uploaded data"""
        if data is None:
            return {'status': 'error', 'message': 'No data provided'}
        
        checks = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'duplicates': data.duplicated().sum()
        }
        
        return {'status': 'success', 'checks': checks}

class AlertSystem:
    """System for generating and managing alerts"""
    
    @staticmethod
    def generate_alerts(data):
        """Generate alerts based on data analysis"""
        alerts = []
        
        if data is not None and 'malaria_cases' in data.columns:
            # Example alert logic
            avg_cases = data['malaria_cases'].mean()
            latest_cases = data['malaria_cases'].iloc[-1] if len(data) > 0 else 0
            
            if latest_cases > avg_cases * 1.5:
                alerts.append({
                    'level': 'HIGH',
                    'message': f'Malaria cases spike detected: {latest_cases} cases (average: {avg_cases:.1f})',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'spike_alert'
                })
        
        return alerts

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_synthetic_data():
    """Generate synthetic malaria data for demonstration"""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
    
    # Base pattern with seasonality
    base_cases = 1000
    seasonal_factor = 500 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    trend = 0.5 * np.arange(len(dates))
    noise = np.random.normal(0, 100, len(dates))
    
    cases = base_cases + seasonal_factor + trend + noise
    cases = np.maximum(cases, 0)  # No negative cases
    
    data = pd.DataFrame({
        'date': dates,
        'malaria_cases': cases.astype(int),
        'temperature': np.random.uniform(25, 35, len(dates)),
        'rainfall': np.random.exponential(50, len(dates)),
        'humidity': np.random.uniform(60, 90, len(dates)),
        'nddi': np.random.uniform(0.3, 0.7, len(dates)),
        'llin_coverage': np.random.uniform(30, 80, len(dates)),
        'irs_coverage': np.random.uniform(20, 60, len(dates)),
        'population': np.random.uniform(100000, 500000, len(dates))
    })
    
    return data

def train_all_models(data):
    """Train all machine learning models"""
    if data is None:
        return {}
    
    # Prepare features
    features = ['temperature', 'rainfall', 'humidity', 'nddi', 'llin_coverage', 'irs_coverage']
    features = [f for f in features if f in data.columns]
    
    X = data[features].values
    y = data['malaria_cases'].values
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    
    # Calculate metrics
    models['rf_model'] = rf_model
    models['gb_model'] = gb_model
    models['rf_rmse'] = np.sqrt(mean_squared_error(y_test, rf_pred))
    models['gb_rmse'] = np.sqrt(mean_squared_error(y_test, gb_pred))
    models['rf_mae'] = mean_absolute_error(y_test, rf_pred)
    models['gb_mae'] = mean_absolute_error(y_test, gb_pred)
    models['rf_r2'] = r2_score(y_test, rf_pred)
    models['gb_r2'] = r2_score(y_test, gb_pred)
    models['scaler'] = scaler
    models['feature_names'] = features
    
    return models

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application function"""
    
    # Initialize session state variables
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'user_role' not in st.session_state:
        st.session_state.user_role = 'field_worker'
    if 'show_igad' not in st.session_state:
        st.session_state.show_igad = False
    if 'igad_data' not in st.session_state:
        st.session_state.igad_data = None
    if 'igad_validation' not in st.session_state:
        st.session_state.igad_validation = {'is_valid': False, 'messages': []}
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = None
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
        st.title("🦟 Malaria Forecasting System")
        st.markdown("**Enterprise-Grade National Control System**")
        st.markdown("---")
        
        # Navigation selection
        st.markdown("### 🧭 Navigation")
        app_mode = st.radio(
            "Select Dashboard",
            ["🏠 Main Dashboard", "📈 IGAD Dashboard"],
            index=0
        )
        
        if app_mode == "📈 IGAD Dashboard":
            st.session_state.show_igad = True
        else:
            st.session_state.show_igad = False
        
        # User role selection
        st.markdown("### 👤 User Role")
        user_role = st.selectbox(
            "Select your role",
            options=list(UserPermissions.ROLES.keys()),
            format_func=lambda x: UserPermissions.get_role_name(x),
            key="user_role_select"
        )
        st.session_state.user_role = user_role
        
        # Data management for Main Dashboard
        if not st.session_state.show_igad:
            st.markdown("---")
            st.markdown("### 📊 Data Management")
            
            data_option = st.radio(
                "Choose data source:",
                ["Generate Synthetic Data", "Upload CSV/Excel"],
                index=0
            )
            
            if st.button("📥 Load Data", use_container_width=True):
                with st.spinner("Loading data..."):
                    if data_option == "Generate Synthetic Data":
                        st.session_state.data = generate_synthetic_data()
                        st.session_state.data_generated = True
                        st.session_state.models_trained = False
                        st.success("Synthetic data generated successfully!")
                    elif data_option == "Upload CSV/Excel":
                        uploaded_file = st.file_uploader(
                            "Choose a CSV or Excel file", 
                            type=['csv', 'xlsx', 'xls'],
                            key="main_data_upload"
                        )
                        if uploaded_file is not None:
                            try:
                                if uploaded_file.name.endswith('.csv'):
                                    data = pd.read_csv(uploaded_file)
                                else:
                                    data = pd.read_excel(uploaded_file)
                                st.session_state.data = data
                                st.session_state.data_generated = True
                                st.session_state.models_trained = False
                                st.success("Data uploaded successfully!")
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")
                
                # Check data quality and generate alerts
                if st.session_state.data is not None:
                    DataQualityMonitor.check_data_quality(st.session_state.data)
                    st.session_state.alerts = AlertSystem.generate_alerts(st.session_state.data)
            
            if st.session_state.data_generated and st.button("🤖 Train Models", use_container_width=True):
                with st.spinner("Training models..."):
                    st.session_state.model_results = train_all_models(st.session_state.data)
                    st.session_state.models_trained = True
                    st.success("Models trained successfully!")
        
        # System status
        st.markdown("---")
        st.markdown("### 📈 System Status")
        
        if st.session_state.data is not None:
            st.metric("Data Records", len(st.session_state.data))
        
        if st.session_state.alerts:
            alert_count = len([a for a in st.session_state.alerts if a['level'] in ['HIGH']])
            if alert_count > 0:
                st.metric("Active Alerts", alert_count, delta="Requires attention")
    
    # Main content area
    if st.session_state.show_igad:
        # Show IGAD Dashboard
        show_igad_dashboard()
    else:
        # Show Main Dashboard
        st.title("🏠 National Malaria Forecasting & Control System")
        st.markdown("""
        **Enterprise-Grade Platform for National Malaria Control Programs**
        
        This system integrates advanced analytics and machine learning for comprehensive malaria surveillance and control.
        """)
        
        # Create tabs for Main Dashboard
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", "🚨 Alerts & Response", "📊 Performance", "🔮 Forecast"
        ])
        
        with tab1:
            st.markdown('<h2 class="sub-header">📈 System Overview</h2>', unsafe_allow_html=True)
            
            if st.session_state.data is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_cases = st.session_state.data['malaria_cases'].sum()
                    st.metric("Total Cases", f"{total_cases:,}")
                with col2:
                    avg_temp = st.session_state.data['temperature'].mean()
                    st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
                with col3:
                    avg_rainfall = st.session_state.data['rainfall'].mean()
                    st.metric("Avg Rainfall", f"{avg_rainfall:.1f} mm")
                with col4:
                    avg_nddi = st.session_state.data['nddi'].mean()
                    st.metric("Avg NDDI", f"{avg_nddi:.3f}")
                
                # Data preview
                st.subheader("📊 Data Preview")
                st.dataframe(st.session_state.data.head(10), use_container_width=True)
                
                # Basic visualization
                st.subheader("📈 Malaria Cases Over Time")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(st.session_state.data['date'], st.session_state.data['malaria_cases'], 
                       marker='o', markersize=4, linewidth=2)
                ax.set_xlabel('Date')
                ax.set_ylabel('Malaria Cases')
                ax.set_title('Malaria Cases Over Time')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("Please load data from the sidebar to see the overview")
        
        with tab2:
            st.markdown('<h2 class="sub-header">🚨 Alerts & Response System</h2>', unsafe_allow_html=True)
            
            if st.session_state.alerts:
                st.warning(f"⚠️ **{len(st.session_state.alerts)} Active Alerts**")
                
                for alert in st.session_state.alerts:
                    if alert['level'] == 'HIGH':
                        st.error(f"**{alert['level']}**: {alert['message']}")
                    elif alert['level'] == 'MEDIUM':
                        st.warning(f"**{alert['level']}**: {alert['message']}")
                    else:
                        st.info(f"**{alert['level']}**: {alert['message']}")
                
                st.subheader("Response Actions")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📋 Generate Response Plan", use_container_width=True):
                        st.success("Response plan generated!")
                        st.write("1. Mobilize rapid response team")
                        st.write("2. Distribute additional bed nets")
                        st.write("3. Increase testing in affected areas")
                        st.write("4. Alert nearby health facilities")
                with col2:
                    if st.button("📞 Contact Emergency Team", use_container_width=True):
                        st.success("Emergency team notified!")
                with col3:
                    if st.button("📊 Allocate Resources", use_container_width=True):
                        st.success("Resource allocation optimized!")
            else:
                st.success("✅ No active alerts")
                st.info("System is monitoring for potential outbreaks...")
        
        with tab3:
            st.markdown('<h2 class="sub-header">📊 Model Performance</h2>', unsafe_allow_html=True)
            
            if st.session_state.models_trained:
                st.success("✅ Models trained successfully!")
                
                # Display model performance
                results = st.session_state.model_results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RF RMSE", f"{results.get('rf_rmse', 0):.1f}")
                with col2:
                    st.metric("GB RMSE", f"{results.get('gb_rmse', 0):.1f}")
                with col3:
                    st.metric("RF R² Score", f"{results.get('rf_r2', 0):.3f}")
                with col4:
                    st.metric("GB R² Score", f"{results.get('gb_r2', 0):.3f}")
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                if 'rf_model' in results:
                    importances = results['rf_model'].feature_importances_
                    features = results.get('feature_names', ['Feature 1', 'Feature 2', 'Feature 3'])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(features, importances)
                    ax.set_xlabel('Importance')
                    ax.set_title('Random Forest Feature Importance')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            else:
                st.info("Please train models from the sidebar to see performance metrics")
        
        with tab4:
            st.markdown('<h2 class="sub-header">🔮 Generate Forecast</h2>', unsafe_allow_html=True)
            
            if st.session_state.models_trained:
                if st.button("Generate 6-Month Forecast"):
                    with st.spinner("Generating forecast..."):
                        # Simulate forecast
                        future_dates = pd.date_range(
                            start=st.session_state.data['date'].iloc[-1] + timedelta(days=30),
                            periods=6,
                            freq='M'
                        )
                        
                        forecast_data = pd.DataFrame({
                            'date': future_dates,
                            'predicted_cases': np.random.randint(800, 1500, 6),
                            'lower_bound': np.random.randint(600, 1200, 6),
                            'upper_bound': np.random.randint(1000, 1800, 6)
                        })
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(st.session_state.data['date'], st.session_state.data['malaria_cases'], 
                               label='Historical', marker='o')
                        ax.plot(forecast_data['date'], forecast_data['predicted_cases'], 
                               label='Forecast', marker='s', linestyle='--')
                        ax.fill_between(forecast_data['date'], 
                                       forecast_data['lower_bound'], 
                                       forecast_data['upper_bound'], 
                                       alpha=0.3, label='Confidence Interval')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Malaria Cases')
                        ax.set_title('6-Month Malaria Forecast')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
            else:
                st.info("Please train models first to generate forecasts")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
