"""
City Comparison Page for AQI Monitoring Application
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
import visualization as viz
import utils

# Set page config
st.set_page_config(
    page_title="City Comparison",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 City Comparison")
st.markdown("Compare Air Quality Index across multiple Indian cities")

# Sidebar
with st.sidebar:
    st.header("Comparison Settings")
    
    # State filter
    all_states = ["All States"] + dh.get_available_states()
    selected_state_filter = st.selectbox(
        "Filter by State",
        all_states,
        index=0
    )
    
    # Cities filtered by state
    if selected_state_filter == "All States":
        filtered_cities = dh.get_available_cities()
    else:
        filtered_cities = dh.get_available_cities(state=selected_state_filter)
    
    # Make sure there's at least one city available
    if not filtered_cities:
        st.warning(f"No cities available for state: {selected_state_filter}. Showing all cities.")
        filtered_cities = dh.get_available_cities()
    
    # Multi-select for cities
    selected_cities = st.multiselect(
        "Select Cities to Compare",
        filtered_cities,
        default=filtered_cities[:3] if len(filtered_cities) >= 3 else filtered_cities
    )
    
    if not selected_cities:
        st.warning("Please select at least one city")
        selected_cities = filtered_cities[:1] if filtered_cities else []
    
    # Date range for comparison
    st.subheader("Date Range")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    date_range = st.date_input(
        "Select date range",
        value=(start_date, end_date),
        min_value=end_date - timedelta(days=365),
        max_value=end_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    
    # Comparison metrics
    st.subheader("Comparison Metrics")
    metrics = st.multiselect(
        "Select Metrics",
        ["Current AQI", "Average AQI", "Maximum AQI", "Minimum AQI", "Days Above Threshold"],
        default=["Current AQI", "Average AQI"]
    )
    
    # AQI threshold for "Days Above Threshold" metric
    threshold = st.slider(
        "AQI Threshold",
        min_value=50,
        max_value=400,
        value=200,
        step=10,
        help="For counting days above threshold"
    )
    
    # Chart type
    chart_type = st.radio(
        "Chart Type",
        ["Bar Chart", "Line Chart", "Radar Chart"],
        index=0
    )

# Main content
try:
    # Date range validation
    if start_date >= end_date:
        st.warning("Start date must be before end date. Adjusting to last 30 days.")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
    
    # Current AQI comparison
    st.markdown("## Current AQI Comparison")
    
    # Fetch current AQI data for all selected cities
    current_data = []
    for city in selected_cities:
        try:
            aqi_data = dh.get_current_aqi(city)
            aqi_value = aqi_data['aqi']
            category, color, _ = utils.get_aqi_category(aqi_value)
            current_data.append({
                "city": city,
                "aqi": aqi_value,
                "category": category,
                "color": color,
                "timestamp": aqi_data['timestamp']
            })
        except Exception as e:
            st.warning(f"Could not fetch data for {city}: {str(e)}")
    
    current_df = pd.DataFrame(current_data)
    
    if not current_df.empty:
        # Display as metrics in columns
        cols = st.columns(len(current_df))
        for i, (_, row) in enumerate(current_df.iterrows()):
            with cols[i]:
                st.metric(row['city'], f"{row['aqi']:.1f}")
                st.markdown(f"""
                <div style='background-color:{row['color']}; padding:10px; border-radius:5px; text-align:center;'>
                    <p style='color:white; margin:0;'>{row['category']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Bar chart for current AQI
        fig_current = px.bar(
            current_df,
            x='city',
            y='aqi',
            title="Current AQI Comparison",
            labels={'city': 'City', 'aqi': 'AQI Value'},
            color='aqi',
            color_continuous_scale=px.colors.sequential.Plasma_r,
            text='aqi'
        )
        fig_current.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        
        st.plotly_chart(fig_current, use_container_width=True)
    else:
        st.warning("No current AQI data available for the selected cities.")
    
    # Historical comparison
    st.markdown("## Historical AQI Comparison")
    
    # Fetch historical data for all selected cities
    historical_data = {}
    for city in selected_cities:
        try:
            historical_df = dh.get_historical_aqi(
                city, 
                start_date, 
                end_date
            )
            historical_data[city] = historical_df
        except Exception as e:
            st.warning(f"Could not fetch historical data for {city}: {str(e)}")
    
    if historical_data:
        # Create comparison chart based on chart type
        if chart_type == "Line Chart":
            # Line chart for historical comparison
            fig = go.Figure()
            
            for city, data in historical_data.items():
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data['aqi'],
                    mode='lines',
                    name=city
                ))
            
            fig.update_layout(
                title="Historical AQI Comparison",
                xaxis_title="Date",
                yaxis_title="AQI Value",
                legend_title="City",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Bar Chart":
            # Calculate average AQI for each city
            avg_data = []
            for city, data in historical_data.items():
                avg_aqi = data['aqi'].mean()
                avg_data.append({
                    "city": city,
                    "avg_aqi": avg_aqi
                })
            
            avg_df = pd.DataFrame(avg_data)
            
            fig_avg = px.bar(
                avg_df,
                x='city',
                y='avg_aqi',
                title="Average AQI Comparison",
                labels={'city': 'City', 'avg_aqi': 'Average AQI'},
                color='avg_aqi',
                color_continuous_scale=px.colors.sequential.Plasma_r,
                text='avg_aqi'
            )
            fig_avg.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            
            st.plotly_chart(fig_avg, use_container_width=True)
            
        elif chart_type == "Radar Chart":
            # Radar chart for multi-metric comparison
            categories = ["Current AQI", "Average AQI", "Maximum AQI", "Minimum AQI"]
            fig = go.Figure()
            
            for city, data in historical_data.items():
                current_aqi = next((item['aqi'] for item in current_data if item['city'] == city), 0)
                avg_aqi = data['aqi'].mean()
                max_aqi = data['aqi'].max()
                min_aqi = data['aqi'].min()
                
                fig.add_trace(go.Scatterpolar(
                    r=[current_aqi, avg_aqi, max_aqi, min_aqi],
                    theta=categories,
                    fill='toself',
                    name=city
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max([data['aqi'].max() for data in historical_data.values()]) * 1.1]
                    )
                ),
                title="Multi-Metric AQI Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
        # Statistics comparison
        st.markdown("## Statistical Comparison")
        
        # Calculate statistics for each city
        stats_data = []
        for city, data in historical_data.items():
            avg_aqi = data['aqi'].mean()
            max_aqi = data['aqi'].max()
            min_aqi = data['aqi'].min()
            days_above = (data['aqi'] > threshold).sum()
            
            stats_data.append({
                "city": city,
                "avg_aqi": avg_aqi,
                "max_aqi": max_aqi,
                "min_aqi": min_aqi,
                "days_above_threshold": days_above
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Display statistics
        st.dataframe(stats_df, use_container_width=True)
        
        # Visualize selected metrics
        if "Average AQI" in metrics:
            fig_avg = px.bar(
                stats_df,
                x='city',
                y='avg_aqi',
                title=f"Average AQI ({start_date} to {end_date})",
                labels={'city': 'City', 'avg_aqi': 'Average AQI'},
                color='avg_aqi',
                color_continuous_scale=px.colors.sequential.Plasma_r,
                text='avg_aqi'
            )
            fig_avg.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_avg, use_container_width=True)
        
        if "Maximum AQI" in metrics:
            fig_max = px.bar(
                stats_df,
                x='city',
                y='max_aqi',
                title=f"Maximum AQI ({start_date} to {end_date})",
                labels={'city': 'City', 'max_aqi': 'Maximum AQI'},
                color='max_aqi',
                color_continuous_scale=px.colors.sequential.Plasma_r,
                text='max_aqi'
            )
            fig_max.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_max, use_container_width=True)
        
        if "Minimum AQI" in metrics:
            fig_min = px.bar(
                stats_df,
                x='city',
                y='min_aqi',
                title=f"Minimum AQI ({start_date} to {end_date})",
                labels={'city': 'City', 'min_aqi': 'Minimum AQI'},
                color='min_aqi',
                color_continuous_scale=px.colors.sequential.Plasma_r,
                text='min_aqi'
            )
            fig_min.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_min, use_container_width=True)
        
        if "Days Above Threshold" in metrics:
            fig_days = px.bar(
                stats_df,
                x='city',
                y='days_above_threshold',
                title=f"Days with AQI Above {threshold} ({start_date} to {end_date})",
                labels={'city': 'City', 'days_above_threshold': 'Number of Days'},
                color='days_above_threshold',
                color_continuous_scale=px.colors.sequential.Plasma_r,
                text='days_above_threshold'
            )
            fig_days.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_days, use_container_width=True)
        
        # Pollution category distribution
        st.markdown("## Pollution Category Distribution")
        
        # Create stacked bar chart for category distribution
        category_data = []
        for city, data in historical_data.items():
            # Get category for each AQI value
            categories = []
            for aqi in data['aqi']:
                category, _, _ = utils.get_aqi_category(aqi)
                categories.append(category)
            
            # Count categories
            category_counts = pd.Series(categories).value_counts()
            
            # Create row for each category
            for category, count in category_counts.items():
                category_data.append({
                    "city": city,
                    "category": category,
                    "count": count
                })
        
        if category_data:
            category_df = pd.DataFrame(category_data)
            
            # Create stacked bar chart
            fig_categories = px.bar(
                category_df,
                x='city',
                y='count',
                color='category',
                title="AQI Category Distribution by City",
                labels={'city': 'City', 'count': 'Number of Days', 'category': 'AQI Category'},
                color_discrete_map={
                    'Good': '#3BB143',
                    'Satisfactory': '#AFE1AF',
                    'Moderately Polluted': '#FFF700',
                    'Poor': '#FF7F00',
                    'Very Poor': '#FF0000',
                    'Severe': '#800000'
                }
            )
            
            st.plotly_chart(fig_categories, use_container_width=True)
    else:
        st.warning("No historical AQI data available for the selected cities and date range.")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.markdown("Please try again later or select different cities/date range.")

# Footer
st.markdown("---")
st.markdown("Data Source: AQI Monitoring Database")