"""
Historical Data Page for AQI Monitoring Application
"""
import streamlit as st
import pandas as pd
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
    page_title="Historical AQI Data",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Historical AQI Data")
st.markdown("View and analyze historical AQI trends for Indian cities")

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
    
    # Date range for historical data
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
    
    # City comparison
    st.subheader("Compare Cities")
    # Either use filtered cities or cities from the same state for more relevant comparison
    if selected_state != "All States":
        # If state is filtered, use other cities from same state
        comparison_options = ["None"] + [city for city in filtered_cities if city != selected_city]
    else:
        # Otherwise use all cities
        comparison_options = ["None"] + [city for city in dh.get_available_cities() if city != selected_city]
        
    comparison_selection = st.selectbox(
        "Compare with",
        comparison_options
    )
    comparison_city = None if comparison_selection == "None" else comparison_selection
    
    # Chart type
    chart_type = st.radio(
        "Chart Type",
        ["Line Chart", "Bar Chart", "Area Chart"],
        index=0
    )
    
    # Aggregation
    aggregation = st.radio(
        "Data Aggregation",
        ["Daily", "Weekly", "Monthly"],
        index=0
    )
    
    # Download option
    st.subheader("Export Data")
    if st.button("Download Historical Data"):
        st.markdown("Preparing download...")
        # This will be handled in the main content

# Main content
try:
    # Date range validation
    if start_date >= end_date:
        st.warning("Start date must be before end date. Adjusting to last 30 days.")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
    
    st.markdown(f"## Historical AQI Data for {selected_city}")
    
    # Fetch historical data
    with st.spinner("Fetching historical data..."):
        historical_data = dh.get_historical_aqi(
            selected_city, 
            start_date, 
            end_date
        )
        
        comparison_data = None
        if comparison_city:
            comparison_data = dh.get_historical_aqi(
                comparison_city, 
                start_date, 
                end_date
            )
    
    # Data analysis section
    st.markdown("### Data Analysis")
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    
    # Calculate statistics
    max_aqi = historical_data['aqi'].max()
    min_aqi = historical_data['aqi'].min()
    avg_aqi = historical_data['aqi'].mean()
    
    with analysis_col1:
        st.metric("Maximum AQI", f"{max_aqi:.1f}")
    with analysis_col2:
        st.metric("Minimum AQI", f"{min_aqi:.1f}")
    with analysis_col3:
        st.metric("Average AQI", f"{avg_aqi:.1f}")
    
    # Create historical trend chart
    st.markdown("### AQI Trend")
    fig_historical = viz.create_historical_chart(
        historical_data, 
        selected_city,
        comparison_data,
        comparison_city
    )
    st.plotly_chart(fig_historical, use_container_width=True)
    
    # Monthly average section
    st.markdown("### Monthly Average AQI")
    current_year = datetime.now().year
    monthly_data = dh.get_monthly_avg_aqi(selected_city, current_year)
    fig_monthly = viz.create_monthly_chart(monthly_data, selected_city)
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Day of week analysis
    if 'date' in historical_data.columns:
        st.markdown("### Day of Week Analysis")
        historical_data['day_of_week'] = pd.to_datetime(historical_data['date']).dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        day_of_week_data = historical_data.groupby('day_of_week')['aqi'].mean().reset_index()
        day_of_week_data['day_of_week'] = pd.Categorical(day_of_week_data['day_of_week'], categories=day_order, ordered=True)
        day_of_week_data = day_of_week_data.sort_values('day_of_week')
        
        fig_day_of_week = px.bar(
            day_of_week_data, 
            x='day_of_week', 
            y='aqi',
            title=f"Average AQI by Day of Week in {selected_city}",
            labels={'day_of_week': 'Day of Week', 'aqi': 'Average AQI'},
            color='aqi',
            color_continuous_scale=px.colors.sequential.Plasma_r
        )
        
        st.plotly_chart(fig_day_of_week, use_container_width=True)
    
    # AQI threshold breakdown
    st.markdown("### AQI Category Breakdown")
    if 'aqi' in historical_data.columns:
        categories = []
        for aqi in historical_data['aqi']:
            category, _, _ = utils.get_aqi_category(aqi)
            categories.append(category)
        
        historical_data['category'] = categories
        category_counts = historical_data['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        
        # Create pie chart
        fig_categories = px.pie(
            category_counts, 
            values='count', 
            names='category', 
            title=f"AQI Category Distribution in {selected_city}",
            color='category',
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
    
    # Data table
    st.markdown("### Data Table")
    st.dataframe(historical_data, use_container_width=True)
    
    # Download section
    csv_data = historical_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download as CSV",
        csv_data,
        f"{selected_city}_aqi_data_{start_date}_to_{end_date}.csv",
        "text/csv",
        key="download-csv"
    )

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.markdown("Please try again later or select a different city/date range.")

# Footer
st.markdown("---")
st.markdown("Data Source: AQI Monitoring Database")