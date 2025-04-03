"""
Long-Term Prediction Page for AQI Monitoring Application
Using advanced ML models for forecasting up to 90 days

Note: This page should appear in the sidebar when Streamlit is running correctly.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_handler as dh
import prediction_model as pm
import visualization as viz
import utils

# Set page config
st.set_page_config(
    page_title="Long-Term AQI Prediction",
    page_icon="🔮",
    layout="wide"
)

# Title and description
st.title("🔮 Long-Term AQI Prediction")
st.markdown("""
### Advanced AI forecasting for extended air quality planning
This page uses sophisticated machine learning models optimized for forecasting AQI trends over extended periods.
These models combine multiple prediction techniques including LSTM neural networks, ensemble methods, and seasonality analysis.

Use this page to:
- Plan outdoor activities and travel weeks or months in advance
- Understand potential seasonal trends affecting your city
- Identify potential high-risk periods for air quality
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # State filter
    all_states = ["All States"] + dh.get_available_states()
    selected_state = st.selectbox(
        "Filter by State",
        all_states,
        index=0
    )
    
    # Cities filtered by state
    if selected_state == "All States":
        filtered_cities = dh.get_available_cities()
    else:
        filtered_cities = dh.get_available_cities(state=selected_state)
    
    # Make sure there's at least one city available
    if not filtered_cities:
        st.warning(f"No cities available for state: {selected_state}. Showing all cities.")
        filtered_cities = dh.get_available_cities()
    
    # City selection
    selected_city = st.selectbox(
        "Select City",
        filtered_cities
    )
    
    # Prediction settings
    st.subheader("Prediction Settings")
    
    # Prediction timeframe
    prediction_days = st.slider(
        "Prediction Horizon (Days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )
    
    # Model type
    model_type = st.selectbox(
        "Model Type",
        ["ensemble", "lstm", "xgboost"],
        index=0,
        help="Ensemble combines multiple models, LSTM uses neural networks, XGBoost is a powerful gradient boosting model"
    )
    
    # Include uncertainty
    show_uncertainty = st.checkbox(
        "Show Prediction Uncertainty",
        value=True,
        help="Shows the confidence range of the prediction"
    )
    
    # Apply predictions
    if st.button("Generate Long-Term Prediction", type="primary"):
        st.session_state.run_long_term_prediction = True
        st.session_state.prediction_params = {
            "city": selected_city,
            "days": prediction_days,
            "model_type": model_type,
            "show_uncertainty": show_uncertainty
        }
    else:
        if "run_long_term_prediction" not in st.session_state:
            st.session_state.run_long_term_prediction = False

# Main area
main_container = st.container()

with main_container:
    if not st.session_state.get("run_long_term_prediction", False):
        # Show initial information when no prediction has been run
        st.info("Select a city and prediction settings, then click 'Generate Long-Term Prediction' to see forecasts.")
        
        # Show some sample visualizations or explanations
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("How Long-Term Prediction Works")
            st.markdown("""
            Our advanced long-term prediction system uses multiple AI techniques:
            
            1. **Historical Pattern Analysis**: Identifies seasonal and cyclical patterns
            2. **LSTM Neural Networks**: Specialized for time series forecasting
            3. **Ensemble Methods**: Combines multiple models for better accuracy
            4. **Uncertainty Quantification**: Provides confidence ranges for predictions
            """)
        
        with col2:
            st.subheader("When to Use Long-Term Predictions")
            st.markdown("""
            Long-term predictions are most useful for:
            
            - **Seasonal Planning**: Understanding air quality across different seasons
            - **Health Management**: Preparation for sensitive individuals
            - **Travel Planning**: Identifying optimal periods for outdoor activities
            - **Policy Development**: Supporting air quality improvement initiatives
            """)
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
                
                # Display prediction summary
                st.subheader(f"Long-Term AQI Forecast for {city}")
                
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
                
                # Plot the prediction data with historical context
                st.subheader("Historical & Predicted AQI Trends")
                
                # Combine historical and prediction data for visualization
                historical_plot_data = historical_data[["date", "aqi"]].copy()
                historical_plot_data["type"] = "Historical"
                historical_plot_data = historical_plot_data.rename(columns={"aqi": "value"})
                
                prediction_plot_data = prediction_data[["date", "predicted_aqi"]].copy()
                prediction_plot_data["type"] = "Predicted"
                prediction_plot_data = prediction_plot_data.rename(columns={"predicted_aqi": "value"})
                
                # Combine the datasets
                combined_data = pd.concat([historical_plot_data, prediction_plot_data], ignore_index=True)
                
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
                
                # Monthly average prediction
                st.subheader("Monthly Average Predictions")
                
                # Group predictions by month
                prediction_data["month"] = prediction_data["date"].dt.strftime("%b %Y")
                monthly_avg = prediction_data.groupby("month")["predicted_aqi"].mean().reset_index()
                monthly_avg["predicted_aqi"] = monthly_avg["predicted_aqi"].round(1)
                
                # Create a bar chart for monthly averages
                fig_monthly = px.bar(
                    monthly_avg,
                    x="month",
                    y="predicted_aqi",
                    title=f"Predicted Monthly Average AQI for {city}",
                    labels={"month": "Month", "predicted_aqi": "Average AQI"},
                    color="predicted_aqi",
                    color_continuous_scale=[
                        "#55A84F", "#A3C853", "#FFF833", "#F29C33", "#E93F33", "#AF2D24"
                    ],
                    range_color=[0, 400]
                )
                
                fig_monthly.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Average AQI",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Detailed prediction data
                st.subheader("Detailed Forecast Data")
                
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
                    height=400
                )
                
                # Health recommendations based on average prediction
                st.subheader("Health Recommendations")
                avg_category, avg_color, _ = utils.get_aqi_category(avg_predicted)
                health_rec = utils.get_health_recommendations(avg_predicted)
                
                st.markdown(f"""
                <div style="background-color:{avg_color}20; padding:15px; border-radius:5px; border-left:5px solid {avg_color};">
                <h4 style="color:{avg_color};">Based on {days}-day forecast ({avg_category})</h4>
                <p>{health_rec}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Warning about long-term prediction uncertainty
                if days > 14:
                    st.caption("""
                    **Note**: Long-term predictions become less certain as we forecast further into the future.
                    Predictions beyond 14 days should be used as general trend indicators rather than precise forecasts.
                    """)
                
            except Exception as e:
                st.error(f"Error generating long-term prediction: {str(e)}")
                st.warning("Please try a different city or prediction timeframe.")