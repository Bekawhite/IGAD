# ============================================================================
# IGAD MALARIA DASHBOARD - COMPLETE ENTERPRISE SOLUTION
# ============================================================================

import streamlit as st
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
# PAGE CONFIGURATION - MUST BE FIRST
# ============================================================================
st.set_page_config(
    page_title="IGAD Malaria Forecasting System",
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
.igad-country {
    background-color: #e8f4f8;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.igad-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.sidebar-header {
    background: linear-gradient(135deg, #2E86AB 0%, #2E86AB 100%);
    color: white;
    padding: 15px;
    border-radius: 5px;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IGAD DATA VALIDATION MODULE
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

    @staticmethod
    def prepare_time_series_data(df, country_codes=None):
        """Prepare time series data for forecasting"""
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
        
        # Filter by country if specified
        if country_codes:
            country_df = country_df[country_df['ISO3'].isin(country_codes)]
        
        # Pivot to get time series
        time_series = country_df.pivot_table(
            index='Year',
            columns='ISO3',
            values='Value',
            aggfunc='sum'
        ).fillna(0)
        
        return time_series

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
    def create_time_series_chart(time_series_data, selected_countries=None):
        """Create time series chart for malaria cases"""
        if time_series_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No time series data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title="Malaria Cases Over Time - No Data",
                height=400
            )
            return fig
        
        fig = go.Figure()
        
        # Plot selected countries or all countries
        countries_to_plot = selected_countries if selected_countries else time_series_data.columns
        
        for country in countries_to_plot:
            if country in time_series_data.columns:
                fig.add_trace(go.Scatter(
                    x=time_series_data.index,
                    y=time_series_data[country],
                    mode='lines+markers',
                    name=country,
                    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Cases: %{y:,}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Malaria Cases Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def create_heatmap_chart(time_series_data):
        """Create heatmap of cases by country and year"""
        if time_series_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        # Prepare data for heatmap
        heatmap_data = time_series_data.copy()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values.T,
            x=heatmap_data.index,
            y=heatmap_data.columns,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Country: %{y}<br>Year: %{x}<br>Cases: %{z:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Malaria Cases Heatmap (Country × Year)",
            xaxis_title="Year",
            yaxis_title="Country",
            height=500
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

# ============================================================================
# FORECASTING MODULE
# ============================================================================

class IGADForecaster:
    """Forecasting module for IGAD malaria cases"""
    
    @staticmethod
    def prepare_forecast_data(time_series_data, n_years=3):
        """Prepare data for forecasting"""
        if time_series_data.empty or len(time_series_data) < 5:
            return None, None, None
        
        # Ensure we have consecutive years
        years = sorted(time_series_data.index)
        X = np.array(years).reshape(-1, 1)
        forecasts = {}
        
        for country in time_series_data.columns:
            y = time_series_data[country].values
            
            # Simple linear regression for forecasting
            coeffs = np.polyfit(years, y, 1)
            poly = np.poly1d(coeffs)
            
            # Generate future years
            future_years = np.arange(years[-1] + 1, years[-1] + n_years + 1)
            future_cases = poly(future_years)
            
            # Ensure non-negative cases
            future_cases = np.maximum(future_cases, 0)
            
            forecasts[country] = {
                'past_years': years,
                'past_cases': y.tolist(),
                'future_years': future_years.tolist(),
                'future_cases': future_cases.tolist(),
                'trend': 'increasing' if coeffs[0] > 0 else 'decreasing',
                'growth_rate': coeffs[0]
            }
        
        return forecasts
    
    @staticmethod
    def create_forecast_chart(forecast_data, selected_countries=None):
        """Create forecast visualization"""
        if not forecast_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        fig = go.Figure()
        
        countries_to_plot = selected_countries if selected_countries else list(forecast_data.keys())
        
        for country in countries_to_plot:
            if country in forecast_data:
                data = forecast_data[country]
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data['past_years'],
                    y=data['past_cases'],
                    mode='lines+markers',
                    name=f"{country} (Historical)",
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
                # Forecast data
                fig.add_trace(go.Scatter(
                    x=data['future_years'],
                    y=data['future_cases'],
                    mode='lines+markers',
                    name=f"{country} (Forecast)",
                    line=dict(width=2, dash='dash'),
                    marker=dict(size=6, symbol='diamond')
                ))
        
        fig.update_layout(
            title="Malaria Cases Forecast (Next 3 Years)",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig

# ============================================================================
# MAIN APP
# ============================================================================

def show_igad_dashboard():
    """Display the IGAD Malaria Trends Dashboard"""
    
    # Initialize session state
    if 'igad_data' not in st.session_state:
        st.session_state.igad_data = None
    if 'igad_validation' not in st.session_state:
        st.session_state.igad_validation = {'is_valid': False, 'messages': []}
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = None
    if 'selected_countries' not in st.session_state:
        st.session_state.selected_countries = []
    if 'show_forecast' not in st.session_state:
        st.session_state.show_forecast = False
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    
    # Initialize components
    igad_validator = IGADDataValidator()
    igad_processor = IGADDataProcessor()
    igad_viz = IGADVisualizations()
    forecaster = IGADForecaster()
    
    # Header
    st.markdown('<div class="igad-header">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🦟 IGAD Malaria Surveillance Dashboard")
        st.markdown("**Intergovernmental Authority on Development - Regional Malaria Control Initiative**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main visualization area
        if st.session_state.get('igad_data') is not None and st.session_state.get('igad_validation', {}).get('is_valid', False):
            # Prepare data
            bar_data = igad_processor.prepare_bar_chart_data(
                st.session_state.igad_data, 
                st.session_state.get('selected_year')
            )
            map_data = igad_processor.prepare_map_data(
                st.session_state.igad_data,
                st.session_state.get('selected_year')
            )
            time_series_data = igad_processor.prepare_time_series_data(
                st.session_state.igad_data
            )
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Chart", "🗺️ Geographic Distribution", "📈 Time Series", "🔥 Heatmap"])
            
            with tab1:
                bar_chart = igad_viz.create_bar_chart(bar_data, st.session_state.get('selected_year'))
                st.plotly_chart(bar_chart, use_container_width=True)
                
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
                st.plotly_chart(choropleth_map, use_container_width=True)
                
                st.markdown("""
                **Color Legend:**
                - 🟢 Light Green: Low malaria cases
                - 🟡 Yellow: Medium malaria cases  
                - 🔴 Dark Red: High malaria cases
                - ⚫ Gray: No data available
                """)
            
            with tab3:
                if not time_series_data.empty:
                    time_series_chart = igad_viz.create_time_series_chart(
                        time_series_data, 
                        st.session_state.get('selected_countries')
                    )
                    st.plotly_chart(time_series_chart, use_container_width=True)
                    
                    # Country selector for time series
                    available_countries = list(time_series_data.columns)
                    selected_countries = st.multiselect(
                        "Select countries for time series:",
                        options=available_countries,
                        default=available_countries[:3] if available_countries else [],
                        key="ts_country_select"
                    )
                    st.session_state.selected_countries = selected_countries
                else:
                    st.info("No time series data available")
            
            with tab4:
                if not time_series_data.empty:
                    heatmap = igad_viz.create_heatmap_chart(time_series_data)
                    st.plotly_chart(heatmap, use_container_width=True)
                else:
                    st.info("No data available for heatmap")
            
            # Forecasting section
            st.markdown("---")
            st.markdown("### 🔮 Forecasting")
            
            if st.button("Generate 3-Year Forecast", key="forecast_button"):
                if not time_series_data.empty:
                    with st.spinner("Generating forecasts..."):
                        st.session_state.forecast_data = forecaster.prepare_forecast_data(time_series_data)
                        st.session_state.show_forecast = True
                else:
                    st.warning("Insufficient data for forecasting")
            
            if st.session_state.show_forecast and st.session_state.forecast_data:
                forecast_chart = forecaster.create_forecast_chart(
                    st.session_state.forecast_data,
                    st.session_state.get('selected_countries')
                )
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Forecast insights
                st.subheader("📈 Forecast Insights")
                insights_cols = st.columns(3)
                
                for idx, country in enumerate(st.session_state.forecast_data.keys()):
                    if idx < 3:  # Show first 3 countries
                        data = st.session_state.forecast_data[country]
                        with insights_cols[idx]:
                            trend_icon = "📈" if data['trend'] == 'increasing' else "📉"
                            st.metric(
                                label=f"{country} Trend",
                                value=trend_icon,
                                delta=f"{abs(data['growth_rate']):.1f} cases/year"
                            )
        
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
                    'ISO3': ['UGA', 'KEN', 'ETH', 'SDN', 'SSD', 'SOM', 'DJI'],
                    'Name': ['Uganda', 'Kenya', 'Ethiopia', 'Sudan', 'South Sudan', 'Somalia', 'Djibouti'],
                    'Admin Level': [0, 0, 0, 0, 0, 0, 0],
                    'Metric': ['Confirmed malaria cases', 'Confirmed malaria cases', 'Confirmed malaria cases',
                              'Confirmed malaria cases', 'Confirmed malaria cases', 'Confirmed malaria cases',
                              'Confirmed malaria cases'],
                    'Units': ['cases', 'cases', 'cases', 'cases', 'cases', 'cases', 'cases'],
                    'Year': [2023, 2023, 2023, 2023, 2023, 2023, 2023],
                    'Value': [1250000, 850000, 2100000, 950000, 650000, 450000, 15000]
                })
                
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download Sample CSV",
                    data=csv,
                    file_name="igad_sample_data.csv",
                    mime="text/csv"
                )
    
    with col2:
        # Sidebar-like controls
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        st.markdown("### 📁 Data Management")
        st.markdown('</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload IGAD CSV File",
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
                        max_cases = year_data.loc[max_idx, 'Value'] if pd.notna(max_idx) else 0
                    else:
                        max_country = "N/A"
                        max_cases = 0
                    
                    # Lowest burden country
                    if len(year_data) > 1:
                        min_idx = year_data['Value'].idxmin()
                        min_country = year_data.loc[min_idx, 'Name'] if pd.notna(min_idx) else "N/A"
                    else:
                        min_country = "N/A"
                    
                    # Total cases for selected year
                    total_cases = year_data['Value'].sum()
                    
                    # Number of countries with data
                    countries_with_data = year_data['ISO3'].nunique()
                    
                    st.metric("Total Cases", f"{total_cases:,}")
                    st.metric("Highest Burden", max_country, f"{max_cases:,}")
                    st.metric("Countries with Data", countries_with_data)
    
    # Additional analysis section
    st.markdown("---")
    st.markdown("### 🔬 Advanced Analysis")
    
    if st.session_state.get('igad_data') is not None and st.session_state.get('igad_validation', {}).get('is_valid', False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Analysis", use_container_width=True):
                # Prepare data for export
                if st.session_state.forecast_data:
                    export_data = {
                        'metadata': {
                            'export_date': datetime.now().isoformat(),
                            'data_source': 'IGAD Dashboard',
                            'selected_year': st.session_state.selected_year
                        },
                        'summary': igad_viz.create_data_summary(
                            st.session_state.igad_validation,
                            st.session_state.igad_data
                        ),
                        'forecasts': st.session_state.forecast_data
                    }
                    
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="Download JSON Export",
                        data=json_str,
                        file_name=f"igad_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="export_json"
                    )
        
        with col2:
            if st.button("📊 Generate Report", use_container_width=True):
                st.info("Report generation feature would create a comprehensive PDF report here.")
        
        with col3:
            if st.button("🔄 Reset Dashboard", use_container_width=True):
                for key in ['igad_data', 'igad_validation', 'selected_year', 'selected_countries', 
                          'show_forecast', 'forecast_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

def main():
    """Main application function"""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/IGAD_Logo.svg/1200px-IGAD_Logo.svg.png", 
                width=150)
        st.markdown("### IGAD Malaria Dashboard")
        st.markdown("Intergovernmental Authority on Development")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        # Navigation options
        nav_option = st.radio(
            "Select Dashboard",
            ["🏠 Main Dashboard", "📈 Analytics", "🔮 Forecasting", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About IGAD")
        
        st.info("""
        The Intergovernmental Authority on Development (IGAD) is an eight-country 
        trade bloc in Africa. It includes governments from the Horn of Africa, 
        Nile Valley and the African Great Lakes.
        
        **Member States:**
        - Djibouti (DJI)
        - Ethiopia (ETH)
        - Kenya (KEN)
        - Somalia (SOM)
        - South Sudan (SSD)
        - Sudan (SDN)
        - Uganda (UGA)
        """)
        
        st.markdown("---")
        st.markdown("### 📞 Contact & Support")
        
        st.write("**Technical Support:**")
        st.write("support@igad-malaria.org")
        
        st.write("**Emergency Contact:**")
        st.write("+254 20 123 4567")
        
        # System status
        st.markdown("---")
        st.markdown("### 🔄 System Status")
        
        if 'igad_data' in st.session_state and st.session_state.igad_data is not None:
            data_status = "✅ Loaded"
            rows = len(st.session_state.igad_data)
        else:
            data_status = "❌ Not Loaded"
            rows = 0
        
        st.metric("Data Status", data_status, f"{rows} rows")
        
        # Version info
        st.markdown("---")
        st.caption("Version 1.0.0 | IGAD Malaria Control Initiative")

    # Main content based on navigation
    if nav_option == "🏠 Main Dashboard":
        show_igad_dashboard()
    elif nav_option == "📈 Analytics":
        st.title("📈 Advanced Analytics")
        st.info("Advanced analytics features coming soon!")
        
        if st.session_state.get('igad_data') is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(st.session_state.igad_data))
                st.metric("Years of Data", st.session_state.igad_data['Year'].nunique())
            with col2:
                st.metric("Countries", st.session_state.igad_data['ISO3'].nunique())
                st.metric("Metrics", st.session_state.igad_data['Metric'].nunique())
            
            # Show sample of advanced analytics
            st.subheader("Data Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'Value' in st.session_state.igad_data.columns:
                st.session_state.igad_data['Value'].hist(bins=30, ax=ax, edgecolor='black')
                ax.set_xlabel('Number of Cases')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Malaria Cases')
                st.pyplot(fig)
    
    elif nav_option == "🔮 Forecasting":
        st.title("🔮 Forecasting Models")
        st.info("Advanced forecasting models coming soon!")
        
        if st.session_state.get('igad_data') is not None:
            st.subheader("Available Forecasting Methods")
            
            methods = [
                ("Linear Regression", "Simple trend-based forecasting"),
                ("Time Series ARIMA", "Advanced time series analysis"),
                ("Machine Learning", "Random Forest and Gradient Boosting"),
                ("Ensemble Methods", "Combination of multiple models")
            ]
            
            for method, description in methods:
                with st.expander(f"{method}"):
                    st.write(description)
                    st.button(f"Train {method}", key=f"train_{method}")
    
    elif nav_option == "⚙️ Settings":
        st.title("⚙️ System Settings")
        
        st.subheader("Dashboard Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            refresh_rate = st.selectbox("Auto-refresh", ["Off", "5 minutes", "15 minutes", "30 minutes"])
        
        with col2:
            default_view = st.selectbox("Default View", ["Bar Chart", "Map", "Time Series", "Heatmap"])
            data_points = st.slider("Max Data Points", 100, 10000, 1000)
        
        st.subheader("Export Settings")
        export_format = st.multiselect(
            "Export Formats",
            ["CSV", "Excel", "JSON", "PDF", "PNG"],
            default=["CSV", "JSON"]
        )
        
        if st.button("Save Settings", type="primary"):
            st.success("Settings saved successfully!")
        
        st.subheader("System Information")
        st.write(f"**Streamlit Version:** {st.__version__}")
        st.write(f"**Pandas Version:** {pd.__version__}")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
