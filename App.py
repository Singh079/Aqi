import streamlit as st

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="AQI MONITOR APP",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
import threading

import data_handler as dh
import visualization as viz
import prediction_model as pm
import utils
import data_updater
import api_client

# Initialize database on startup
from database import init_db
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization error: {str(e)}")

# Initialize session state variables
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "Delhi"
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = None  # No state filter initially
if 'alert_threshold' not in st.session_state:
    st.session_state.alert_threshold = 200
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = True
if 'comparison_city' not in st.session_state:
    st.session_state.comparison_city = None
if 'date_range' not in st.session_state:
    st.session_state.date_range = (datetime.now() - timedelta(days=30)).date(), datetime.now().date()
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Title and description
st.title("🍃 AQI MONITOR APP")
st.markdown("An AI-powered Air Quality Index monitoring application for Indian cities")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # State filter
    all_states = ["All States"] + dh.get_available_states()
    state_index = 0
    if st.session_state.selected_state in all_states:
        state_index = all_states.index(st.session_state.selected_state)
    
    selected_state_filter = st.selectbox(
        "Filter by State",
        all_states,
        index=state_index
    )
    
    # Update selected state in session state
    st.session_state.selected_state = None if selected_state_filter == "All States" else selected_state_filter
    
    # City selection based on state filter
    if st.session_state.selected_state:
        filtered_cities = dh.get_available_cities(state=st.session_state.selected_state)
    else:
        filtered_cities = dh.get_available_cities()
    
    # Make sure there's at least one city available
    if not filtered_cities:
        st.warning(f"No cities available for state: {st.session_state.selected_state}. Showing all cities.")
        filtered_cities = dh.get_available_cities()
        st.session_state.selected_state = None
        
    # If the currently selected city is not in the filtered list, select the first city
    if st.session_state.selected_city not in filtered_cities and filtered_cities:
        st.session_state.selected_city = filtered_cities[0]
        
    # City selection
    city_index = 0
    if st.session_state.selected_city in filtered_cities:
        city_index = filtered_cities.index(st.session_state.selected_city)
        
    st.session_state.selected_city = st.selectbox(
        "Select City",
        filtered_cities,
        index=city_index
    )
    
    # Date range for historical data
    st.subheader("Historical Data Range")
    date_range = st.date_input(
        "Select date range",
        value=st.session_state.date_range,
        min_value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
    if len(date_range) == 2:
        st.session_state.date_range = date_range
    
    # Alert settings
    st.subheader("Alert Settings")
    st.session_state.alert_threshold = st.slider(
        "Alert Threshold AQI",
        min_value=100,
        max_value=500,
        value=st.session_state.alert_threshold,
        step=10,
        help="You'll receive alerts when AQI exceeds this value"
    )
    
    # Prediction toggle
    st.session_state.show_prediction = st.checkbox(
        "Show AI Predictions",
        value=st.session_state.show_prediction,
        help="Display AI-based AQI predictions"
    )
    
    # City comparison
    st.subheader("Compare Cities")
    # Get all cities for comparison (not just filtered ones)
    all_cities_for_comparison = dh.get_available_cities()
    comparison_options = ["None"] + [city for city in all_cities_for_comparison if city != st.session_state.selected_city]
    comparison_selection = st.selectbox(
        "Compare with",
        comparison_options
    )
    st.session_state.comparison_city = None if comparison_selection == "None" else comparison_selection
    
    # API Settings
    st.markdown("---")
    st.markdown("### API Settings")
    api_key = os.environ.get('AQI_API_KEY', '')
    
    # Check if API key exists
    if not api_key:
        st.warning("No API key found. Using simulated data.")
        api_key = st.text_input(
            "Enter your AQI API Key to get real-time data",
            value="",
            type="password"
        )
        if api_key:
            if st.button("Save API Key"):
                os.environ['AQI_API_KEY'] = api_key
                st.success("API key saved! Refreshing data...")
                # Trigger an update of the data
                try:
                    data_updater.update_aqi_data()
                    time.sleep(1)  # Short pause to allow update to complete
                    st.refresh()
                except Exception as e:
                    st.error(f"Error updating data: {str(e)}")
    else:
        st.success("API key is configured")
        if st.button("Clear API Key"):
            os.environ['AQI_API_KEY'] = ''
            st.warning("API key removed. Using simulated data.")
            time.sleep(1)
            st.refresh()
            
    # Data Update Button
    if st.button("Update AQI Data Now"):
        with st.spinner("Updating AQI data..."):
            try:
                success = data_updater.update_aqi_data()
                if success:
                    st.success("Data updated successfully!")
                else:
                    st.warning("Data update completed with some errors.")
                time.sleep(1)
                st.refresh()
            except Exception as e:
                st.error(f"Error updating data: {str(e)}")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This application uses data from the Central Pollution Control Board (CPCB) and AI models to provide accurate AQI monitoring and forecasting for Indian cities.")

# Main content
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Current AQI card
    try:
        current_aqi_data = dh.get_current_aqi(st.session_state.selected_city)
        aqi_value = current_aqi_data['aqi']
        aqi_category, aqi_color, aqi_description = utils.get_aqi_category(aqi_value)
        
        st.markdown(f"## Current AQI: {st.session_state.selected_city}, {current_aqi_data['state']}")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("AQI Value", f"{aqi_value}", 
                      delta=f"{current_aqi_data['change_24h']:.1f} from yesterday")
        with metric_col2:
            st.markdown(f"""
            <div style='background-color:{aqi_color}; padding:10px; border-radius:5px; text-align:center;'>
                <h3 style='color:white; margin:0;'>{aqi_category}</h3>
            </div>
            """, unsafe_allow_html=True)
        with metric_col3:
            st.markdown(f"**Last Updated:** {current_aqi_data['timestamp']}")
        
        st.markdown(f"**Health Advisory:** {aqi_description}")
        
        # Alert notification if AQI exceeds threshold
        if aqi_value > st.session_state.alert_threshold:
            st.warning(f"⚠️ AQI level in {st.session_state.selected_city} has exceeded your alert threshold of {st.session_state.alert_threshold}!")
    
    except Exception as e:
        st.error(f"Unable to fetch current AQI data: {str(e)}")
        st.markdown("Please check your connection or try again later.")

with main_col2:
    # Pollutant breakdown
    st.markdown(f"## Pollutant Breakdown")
    try:
        pollutant_data = dh.get_pollutant_breakdown(st.session_state.selected_city)
        fig_pollutants = viz.create_pollutant_chart(pollutant_data)
        st.plotly_chart(fig_pollutants, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to fetch pollutant data: {str(e)}")

# Map section
st.markdown("## AQI Map of India")
try:
    map_data = dh.get_all_cities_current_aqi()
    map_fig = viz.create_aqi_map(map_data)
    st.plotly_chart(map_fig, use_container_width=True)
except Exception as e:
    st.error(f"Unable to display AQI map: {str(e)}")
    
# Historical data section
st.markdown("## Historical AQI Trends")
tab1, tab2 = st.tabs(["Time Series", "Monthly Averages"])

with tab1:
    try:
        # Date range validation
        start_date, end_date = st.session_state.date_range
        if start_date >= end_date:
            st.warning("Start date must be before end date. Adjusting to last 30 days.")
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
        
        historical_data = dh.get_historical_aqi(
            st.session_state.selected_city, 
            start_date, 
            end_date
        )
        
        comparison_data = None
        if st.session_state.comparison_city:
            comparison_data = dh.get_historical_aqi(
                st.session_state.comparison_city, 
                start_date, 
                end_date
            )
            
        fig_historical = viz.create_historical_chart(
            historical_data, 
            st.session_state.selected_city,
            comparison_data,
            st.session_state.comparison_city
        )
        st.plotly_chart(fig_historical, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to display historical trends: {str(e)}")

with tab2:
    try:
        monthly_data = dh.get_monthly_avg_aqi(st.session_state.selected_city)
        fig_monthly = viz.create_monthly_chart(monthly_data, st.session_state.selected_city)
        st.plotly_chart(fig_monthly, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to display monthly averages: {str(e)}")

# AI Prediction section
if st.session_state.show_prediction:
    st.markdown("## AI-Powered AQI Prediction")
    
    # Create tabs for short-term and long-term predictions
    short_term_tab, long_term_tab = st.tabs(["Short-Term Forecast (7 Days)", "Long-Term Forecast (Up to 90 Days)"])
    
    with short_term_tab:
        pred_col1, pred_col2 = st.columns([1, 2])
        
        with pred_col1:
            st.markdown("### Next 7 Days Forecast")
            try:
                prediction_data = pm.predict_aqi(st.session_state.selected_city)
                fig_prediction = viz.create_prediction_chart(prediction_data)
                st.plotly_chart(fig_prediction, use_container_width=True)
            except Exception as e:
                st.error(f"Unable to generate prediction: {str(e)}")
        
        with pred_col2:
            st.markdown("### Model Insights")
            st.markdown("""
            Our AI model analyzes historical AQI patterns along with weather data and seasonal trends to predict future AQI values.
            Key factors affecting predictions:
            - Weather conditions (temperature, humidity, wind)
            - Seasonal patterns
            - Recent pollution trends
            - Local industrial and traffic activity
            """)
            
            explanation = pm.get_prediction_explanation(st.session_state.selected_city)
            st.info(explanation)
    
    with long_term_tab:
        st.markdown("### Advanced Long-Term AQI Prediction")
        st.markdown("""
        This advanced forecast uses our ensemble ML models to predict AQI trends over extended periods.
        Perfect for planning outdoor activities and travel weeks or months in advance.
        """)
        
        # Long-term prediction settings
        settings_col1, settings_col2, settings_col3 = st.columns(3)
        
        with settings_col1:
            # Prediction timeframe
            prediction_days = st.slider(
                "Prediction Horizon (Days)",
                min_value=7,
                max_value=90,
                value=30,
                step=1,
                key="long_term_days"
            )
        
        with settings_col2:
            # Model type
            model_type = st.selectbox(
                "Model Type",
                ["ensemble", "xgboost", "gbm"],
                index=0,
                help="Ensemble combines multiple models for better accuracy",
                key="long_term_model"
            )
        
        with settings_col3:
            # Include uncertainty
            show_uncertainty = st.checkbox(
                "Show Prediction Uncertainty",
                value=True,
                help="Shows the confidence range of the prediction",
                key="long_term_uncertainty"
            )
        
        # Generate button
        generate_long_term = st.button("Generate Long-Term Prediction", 
                                        type="primary",
                                        key="generate_long_term")
        
        # Session state for long-term prediction
        if "run_long_term_prediction" not in st.session_state:
            st.session_state.run_long_term_prediction = False
            
        if generate_long_term:
            st.session_state.run_long_term_prediction = True
            st.session_state.prediction_params = {
                "city": st.session_state.selected_city,
                "days": prediction_days,
                "model_type": model_type,
                "show_uncertainty": show_uncertainty
            }
        
        # Main predictions area
        if not st.session_state.get("run_long_term_prediction", False):
            # Show initial information when no prediction has been run
            st.info("Select settings above and click 'Generate Long-Term Prediction' to see forecasts.")
            
            # Add a visual explanation
            st.image("assets/aqi_scale.svg", caption="The predictions will show AQI levels across the selected range", width=400)
            
        else:
            # Get prediction parameters
            params = st.session_state.prediction_params
            city = params["city"]
            days = params["days"]
            model_type = params["model_type"]
            show_uncertainty = params["show_uncertainty"]
            
            # Show a spinner while generating predictions
            with st.spinner(f"Generating {days}-day prediction for {city}..."):
                try:
                    # Generate long-term prediction
                    prediction_data = pm.predict_aqi_long_term(city, days=days, model_type=model_type)
                    
                    # Get current AQI for reference
                    current_aqi_data = dh.get_current_aqi(city)
                    current_aqi = current_aqi_data["aqi"]
                    
                    # Get historical data for context
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=30)  # Show last 30 days
                    historical_data = dh.get_historical_aqi(city, start_date, end_date)
                    
                    # Key metrics
                    avg_predicted = prediction_data["predicted_aqi"].mean()
                    max_predicted = prediction_data["predicted_aqi"].max()
                    max_date = prediction_data.loc[prediction_data["predicted_aqi"].idxmax(), "date"].strftime("%b %d, %Y")
                    min_predicted = prediction_data["predicted_aqi"].min()
                    min_date = prediction_data.loc[prediction_data["predicted_aqi"].idxmin(), "date"].strftime("%b %d, %Y")
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current AQI", f"{current_aqi:.1f}")
                        category, color, _ = utils.get_aqi_category(current_aqi)
                        st.markdown(f"<div style='color:{color};font-weight:bold;'>{category}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Average Predicted AQI", f"{avg_predicted:.1f}")
                        category, color, _ = utils.get_aqi_category(avg_predicted)
                        st.markdown(f"<div style='color:{color};font-weight:bold;'>{category}</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("Highest Predicted", f"{max_predicted:.1f}")
                        st.caption(f"On {max_date}")
                    
                    with col4:
                        st.metric("Lowest Predicted", f"{min_predicted:.1f}")
                        st.caption(f"On {min_date}")
                    
                    import plotly.graph_objects as go
                    
                    # Combine historical and prediction data for visualization
                    historical_plot_data = historical_data[["date", "aqi"]].copy()
                    historical_plot_data["type"] = "Historical"
                    historical_plot_data = historical_plot_data.rename(columns={"aqi": "value"})
                    
                    prediction_plot_data = prediction_data[["date", "predicted_aqi"]].copy()
                    prediction_plot_data["type"] = "Predicted"
                    prediction_plot_data = prediction_plot_data.rename(columns={"predicted_aqi": "value"})
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical_plot_data["date"],
                        y=historical_plot_data["value"],
                        mode="lines+markers",
                        name="Historical AQI",
                        line=dict(color="#1F77B4", width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Add prediction data
                    fig.add_trace(go.Scatter(
                        x=prediction_plot_data["date"],
                        y=prediction_plot_data["value"],
                        mode="lines",
                        name="Predicted AQI",
                        line=dict(color="#FF7F0E", width=2)
                    ))
                    
                    # Add uncertainty bands if selected
                    if show_uncertainty:
                        # Generate uncertainty bands that increase with time
                        dates = prediction_plot_data["date"]
                        values = prediction_plot_data["value"]
                        lower_bound = []
                        upper_bound = []
                        
                        for i, val in enumerate(values):
                            # Uncertainty increases with prediction horizon
                            uncertainty = 0.05 + (i / len(values)) * 0.15  # 5% to 20%
                            lower_bound.append(val * (1 - uncertainty))
                            upper_bound.append(val * (1 + uncertainty))
                        
                        # Add lower and upper uncertainty bands
                        fig.add_trace(go.Scatter(
                            x=list(dates) + list(dates)[::-1],
                            y=list(upper_bound) + list(lower_bound)[::-1],
                            fill="toself",
                            fillcolor="rgba(255, 127, 14, 0.2)",
                            line=dict(color="rgba(255, 127, 14, 0)"),
                            hoverinfo="skip",
                            showlegend=False
                        ))
                    
                    # Add a vertical line for today
                    fig.add_vline(
                        x=datetime.now().date(),
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Today",
                        annotation_position="top right"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{days}-Day AQI Forecast for {city}",
                        xaxis_title="Date",
                        yaxis_title="Air Quality Index (AQI)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        hovermode="x unified",
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    
                    # Add colored background zones for AQI categories
                    aqi_categories = [
                        {"name": "Good", "color": "#55A84F", "range": (0, 50)},
                        {"name": "Satisfactory", "color": "#A3C853", "range": (51, 100)},
                        {"name": "Moderate", "color": "#FFF833", "range": (101, 200)},
                        {"name": "Poor", "color": "#F29C33", "range": (201, 300)},
                        {"name": "Very Poor", "color": "#E93F33", "range": (301, 400)},
                        {"name": "Severe", "color": "#AF2D24", "range": (401, 500)}
                    ]
                    
                    for category in aqi_categories:
                        fig.add_hrect(
                            y0=category["range"][0],
                            y1=category["range"][1],
                            fillcolor=category["color"],
                            opacity=0.15,
                            line_width=0,
                            annotation_text=category["name"],
                            annotation_position="right",
                            annotation=dict(font_size=10)
                        )
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed prediction data
                    with st.expander("View Detailed Forecast Data"):
                        # Format date for display
                        prediction_data["formatted_date"] = prediction_data["date"].dt.strftime("%b %d, %Y")
                        
                        # Add day of week
                        prediction_data["day_of_week"] = prediction_data["date"].dt.strftime("%A")
                        
                        # Add AQI category
                        prediction_data["category"] = prediction_data["predicted_aqi"].apply(
                            lambda x: utils.get_aqi_category(x)[0]
                        )
                        
                        # Display data table
                        st.dataframe(
                            prediction_data[["formatted_date", "day_of_week", "predicted_aqi", "category"]].rename(
                                columns={
                                    "formatted_date": "Date",
                                    "day_of_week": "Day",
                                    "predicted_aqi": "Predicted AQI",
                                    "category": "AQI Category"
                                }
                            ),
                            use_container_width=True,
                            height=300
                        )
                        
                except Exception as e:
                    st.error(f"Error generating long-term prediction: {str(e)}")
                    st.warning("Please try a different city or prediction timeframe.")

# AQI Scale Explanation
st.markdown("## Understanding the AQI Scale")
scale_col1, scale_col2 = st.columns([1, 1])

with scale_col1:
    st.markdown("""
    The Air Quality Index (AQI) is divided into six categories:
    - **Good (0-50)**: Air quality is satisfactory, poses little or no risk
    - **Satisfactory (51-100)**: Acceptable air quality for most people
    - **Moderately Polluted (101-200)**: May cause breathing discomfort to sensitive people
    - **Poor (201-300)**: May cause breathing discomfort to many people
    - **Very Poor (301-400)**: May cause respiratory illness on prolonged exposure
    - **Severe (401-500)**: May cause serious respiratory effects even to healthy people
    """)

with scale_col2:
    st.image("assets/aqi_scale.svg", caption="AQI Scale Visualization")

# Health recommendations
st.markdown("## Health Recommendations")
try:
    # Always fetch the current AQI value for safety
    current_data = dh.get_current_aqi(st.session_state.selected_city)
    current_aqi = current_data['aqi']
    health_recommendations = utils.get_health_recommendations(current_aqi)
    st.markdown(health_recommendations)
except Exception as e:
    st.error(f"Unable to fetch health recommendations: {str(e)}")
    st.markdown("Health recommendations are not available at the moment.")

# Footer section
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("Data Source: Central Pollution Control Board (CPCB)")
with col2:
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"Last Updated: {last_update}")
with col3:
    if st.button("Refresh Data"):
        st.session_state.last_update = datetime.now()
        st.refresh()
