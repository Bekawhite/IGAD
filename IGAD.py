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
        """Create interactive bar chart using Plotly[citation:4]"""
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
            "Total Rows": len(df),
            "IGAD Countries": len(df['ISO3'].unique()),
            "Years Covered": f"{df['Year'].min()} - {df['Year'].max()}",
            "Total Malaria Cases": "N/A",
            "Data Quality": "✓ Valid" if validation_results['is_valid'] else "✗ Invalid"
        }
        
        # Calculate total malaria cases if data is valid
        if validation_results['is_valid']:
            malaria_df = IGADDataProcessor.identify_malaria_cases(df)
            if not malaria_df.empty:
                total_cases = malaria_df['Value'].sum()
                summary_data["Total Malaria Cases"] = f"{total_cases:,}"
        
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
    
    # Initialize session state for IGAD data
    if 'igad_data' not in st.session_state:
        st.session_state.igad_data = None
    if 'igad_validation' not in st.session_state:
        st.session_state.igad_validation = {'is_valid': False, 'messages': []}
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = None
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📁 IGAD Data Upload")
        
        # File uploader[citation:10]
        uploaded_file = st.file_uploader(
            "Upload IGAD CSV file",
            type=['csv'],
            help="Upload CSV with exact schema: ISO3, Name, Admin Level, Metric, Units, Year, Value"
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
                else:
                    st.session_state.igad_data = None
                    for msg in validation_messages:
                        st.error(msg)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.session_state.igad_data = None
                st.session_state.igad_validation = {'is_valid': False, 'messages': [str(e)]}
        
        # Year selection dropdown
        if st.session_state.igad_data is not None:
            st.markdown("---")
            st.markdown("### 📅 Year Selection")
            
            available_years = sorted(st.session_state.igad_data['Year'].unique())
            selected_year = st.selectbox(
                "Select Year for Analysis",
                options=available_years,
                index=len(available_years)-1 if available_years else 0
            )
            st.session_state.selected_year = selected_year
            
            # Data summary
            st.markdown("---")
            st.markdown("### 📊 Data Summary")
            summary = igad_viz.create_data_summary(
                st.session_state.igad_validation,
                st.session_state.igad_data
            )
            
            for key, value in summary.items():
                st.metric(key, value)
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.igad_data is not None and st.session_state.igad_validation['is_valid']:
            # Prepare data for visualizations
            bar_data = igad_processor.prepare_bar_chart_data(
                st.session_state.igad_data, 
                st.session_state.selected_year
            )
            map_data = igad_processor.prepare_map_data(
                st.session_state.igad_data,
                st.session_state.selected_year
            )
            
            # Create visualizations
            tab1, tab2 = st.tabs(["📊 Bar Chart", "🗺️ Geographic Distribution"])
            
            with tab1:
                bar_chart = igad_viz.create_bar_chart(bar_data, st.session_state.selected_year)
                st.plotly_chart(bar_chart, use_container_width=True, key="igad_bar_chart")[citation:4]
                
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
                choropleth_map = igad_viz.create_choropleth_map(map_data, st.session_state.selected_year)
                st.plotly_chart(choropleth_map, use_container_width=True, key="igad_map")[citation:4]
                
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
        # Validation results panel
        st.markdown("### ✅ Validation Results")
        
        if st.session_state.igad_validation['messages']:
            for msg in st.session_state.igad_validation['messages']:
                if "Error" in msg:
                    st.error(msg)
                elif "passed" in msg.lower():
                    st.success(msg)
                else:
                    st.info(msg)
        else:
            st.info("No file uploaded yet")
        
        # IGAD country information
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
        if st.session_state.igad_data is not None and st.session_state.igad_validation['is_valid']:
            st.markdown("---")
            st.markdown("### 📈 Quick Insights")
            
            # Calculate some basic insights
            malaria_df = igad_processor.identify_malaria_cases(st.session_state.igad_data)
            
            if not malaria_df.empty and st.session_state.selected_year:
                year_data = malaria_df[malaria_df['Year'] == st.session_state.selected_year]
                
                if not year_data.empty:
                    # Highest burden country
                    max_country = year_data.loc[year_data['Value'].idxmax(), 'Name'] if len(year_data) > 0 else "N/A"
                    
                    # Total cases for selected year
                    total_cases = year_data['Value'].sum()
                    
                    # Number of countries with data
                    countries_with_data = year_data['ISO3'].nunique()
                    
                    st.metric("Total Cases (Selected Year)", f"{total_cases:,}")
                    st.metric("Countries with Data", countries_with_data)
                    st.metric("Highest Burden Country", max_country)

# ============================================================================
# MODIFIED MAIN APP WITH IGAD DASHBOARD INTEGRATION
# ============================================================================

def main():
    # Initialize IGAD session state variables
    if 'igad_data' not in st.session_state:
        st.session_state.igad_data = None
    if 'igad_validation' not in st.session_state:
        st.session_state.igad_validation = {'is_valid': False, 'messages': []}
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = None
    
    # (Keep all existing session state initializations from your original code)
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
    if 'mobile_view' not in st.session_state:
        st.session_state.mobile_view = False
    if 'show_alerts' not in st.session_state:
        st.session_state.show_alerts = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = 'field_worker'
    
    # (Keep all existing feature flag initializations)
    if 'show_live_weather' not in st.session_state:
        st.session_state.show_live_weather = False
    if 'train_deep_learning' not in st.session_state:
        st.session_state.train_deep_learning = False
    if 'generate_fhir' not in st.session_state:
        st.session_state.generate_fhir = False
    if 'run_security_audit' not in st.session_state:
        st.session_state.run_security_audit = False
    if 'run_shap' not in st.session_state:
        st.session_state.run_shap = False
    if 'generate_who_report' not in st.session_state:
        st.session_state.generate_who_report = False
    if 'show_compliance' not in st.session_state:
        st.session_state.show_compliance = False
    
    # Enhanced sidebar with IGAD option
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
        st.title("🦟 Malaria Forecasting System")
        st.markdown("**Enterprise-Grade National Control System**")
        st.markdown("---")
        
        # Navigation options
        st.markdown("### 🧭 Navigation")
        
        nav_options = ["🏠 Main Dashboard", "📈 IGAD Dashboard"] + (
            ["🌐 Real-Time Data", "🤖 Advanced AI", "🏥 Interoperability", 
             "🔐 Security", "💎 Premium"] if st.session_state.data is not None else []
        )
        
        selected_nav = st.selectbox("Go to", nav_options)
        
        # Set navigation state
        if selected_nav == "📈 IGAD Dashboard":
            st.session_state.show_igad_dashboard = True
        else:
            st.session_state.show_igad_dashboard = False
        
        # Rest of your existing sidebar code remains the same...
        # (Keep all your existing sidebar code for user roles, data loading, etc.)
        
        # Only show the IGAD-specific uploader in the IGAD dashboard
        if st.session_state.get('show_igad_dashboard', False):
            st.markdown("---")
            st.markdown("### 📁 IGAD Data Upload")
            
            uploaded_file = st.file_uploader(
                "Upload IGAD CSV file",
                type=['csv'],
                help="Upload CSV with exact schema: ISO3, Name, Admin Level, Metric, Units, Year, Value",
                key="igad_file_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    is_valid, validation_messages = igad_validator.validate_csv(df)
                    
                    st.session_state.igad_validation = {
                        'is_valid': is_valid,
                        'messages': validation_messages
                    }
                    
                    if is_valid:
                        st.session_state.igad_data = df
                        st.success("✅ IGAD data validated successfully!")
                        
                        # Set default selected year
                        if 'Year' in df.columns:
                            available_years = sorted(df['Year'].unique())
                            st.session_state.selected_year = available_years[-1] if available_years else None
                    else:
                        for msg in validation_messages:
                            st.error(msg)
                            
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Year selection for IGAD dashboard
        if st.session_state.get('show_igad_dashboard', False) and st.session_state.igad_data is not None:
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
        
        # (Keep all your existing sidebar widgets and controls below...)
        # [Your existing sidebar code continues here...]
        
        # For brevity in this example, I'm showing where your existing code continues
        # In your actual implementation, all your existing sidebar code from the original
        # app should remain here, just add the IGAD-specific sections above
        
        # Example of how your existing sidebar structure continues:
        st.markdown("---")
        st.markdown("### 👤 User Role")
        user_role = st.selectbox(
            "Select your role",
            options=list(UserPermissions.ROLES.keys()),
            format_func=lambda x: UserPermissions.get_role_name(x),
            key="user_role_select"
        )
        st.session_state.user_role = user_role
        
        # [Continue with all your existing sidebar code...]

    # Main content routing
    if st.session_state.get('show_igad_dashboard', False):
        show_igad_dashboard()
        return
    
    # Check for other feature activations (keep existing)
    if st.session_state.get('show_live_weather', False):
        show_live_weather_dashboard()
        return
    
    if st.session_state.get('train_deep_learning', False):
        show_deep_learning_dashboard()
        return
    
    if st.session_state.get('generate_fhir', False):
        show_fhir_dashboard()
        return
    
    if st.session_state.get('run_security_audit', False):
        show_security_dashboard()
        return
    
    if st.session_state.get('run_shap', False):
        show_shap_analysis()
        return
    
    if st.session_state.get('mobile_view', False):
        MobileInterface.mobile_view()
        return

    # Main dashboard tabs - ADD IGAD TAB HERE
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Overview", "🚨 Alerts & Response", "💰 Resources", 
        "📊 Performance", "🤖 Advanced AI", "🏥 Interoperability",
        "🔐 Security", "💎 Premium", "🌍 IGAD Dashboard"
    ])
    
    # Your existing tab content (tab1 through tab8) remains exactly the same
    with tab1:
        # [Your existing tab1 content...]
        st.markdown('<h2 class="sub-header">📈 System Overview</h2>', unsafe_allow_html=True)
        # ... rest of your tab1 code
    
    with tab2:
        # [Your existing tab2 content...]
        pass
    
    # ... tabs 3-8 remain the same
    
    with tab9:  # NEW: IGAD Dashboard Tab
        show_igad_dashboard()

# ============================================================================
# RUN THE ENHANCED APP
# ============================================================================

if __name__ == "__main__":
    # Set page configuration
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
    
    # Run the app
    main()