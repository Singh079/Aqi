import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import utils

def create_pollutant_chart(pollutant_data):
    """
    Create a bar chart for pollutant breakdown
    """
    # Create a DataFrame from the pollutant data
    df = pd.DataFrame({
        'Pollutant': list(pollutant_data.keys()),
        'Value': list(pollutant_data.values())
    })
    
    # Create color scale based on value
    colors = df['Value'].apply(lambda x: utils.get_color_for_value(x))
    
    # Create the bar chart
    fig = px.bar(
        df, 
        x='Pollutant', 
        y='Value',
        title='Pollutant Levels',
        color='Value',
        color_continuous_scale=['green', 'yellow', 'orange', 'red', 'purple'],
        labels={'Value': 'Concentration'}
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    
    return fig

def create_aqi_map(map_data):
    """
    Create an interactive map of India with AQI levels
    """
    # Create the map
    fig = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        hover_name="city",
        hover_data={"lat": False, "lon": False, "aqi": True, "state": True, "timestamp": True},
        color="aqi",
        size="aqi",
        size_max=25,
        zoom=4,
        center={"lat": 23, "lon": 80},  # Center of India
        color_continuous_scale=['green', 'yellow', 'orange', 'red', 'purple'],
        range_color=[0, 500],
        labels={"aqi": "AQI Value", "state": "State"}
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def create_historical_chart(data, city_name, comparison_data=None, comparison_city=None):
    """
    Create a time series chart for historical AQI data
    """
    fig = go.Figure()
    
    # Add the primary city data
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['aqi'],
        mode='lines+markers',
        name=city_name,
        line=dict(width=3),
        marker=dict(size=6)
    ))
    
    # Add comparison city data if available
    if comparison_data is not None and comparison_city is not None:
        fig.add_trace(go.Scatter(
            x=comparison_data['date'],
            y=comparison_data['aqi'],
            mode='lines+markers',
            name=comparison_city,
            line=dict(width=3, dash='dash'),
            marker=dict(size=6)
        ))
    
    # Add colored background regions for AQI categories
    fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="green", opacity=0.1)
    fig.add_hrect(y0=50, y1=100, line_width=0, fillcolor="yellow", opacity=0.1)
    fig.add_hrect(y0=100, y1=200, line_width=0, fillcolor="orange", opacity=0.1)
    fig.add_hrect(y0=200, y1=300, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=300, y1=400, line_width=0, fillcolor="purple", opacity=0.1)
    fig.add_hrect(y0=400, y1=500, line_width=0, fillcolor="maroon", opacity=0.1)
    
    # Update layout
    fig.update_layout(
        title=f"Historical AQI Trends for {city_name}" + (f" vs {comparison_city}" if comparison_city else ""),
        xaxis_title="Date",
        yaxis_title="AQI Value",
        yaxis_range=[0, 500],
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add annotations for AQI categories
    categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    y_positions = [25, 75, 150, 250, 350, 450]
    
    for cat, y_pos in zip(categories, y_positions):
        fig.add_annotation(
            x=data['date'].iloc[0],
            y=y_pos,
            text=cat,
            showarrow=False,
            font=dict(size=10),
            xanchor="left"
        )
    
    return fig

def create_monthly_chart(data, city_name):
    """
    Create a bar chart for monthly average AQI
    """
    # Color the bars based on AQI value
    colors = data['avg_aqi'].apply(lambda x: utils.get_color_for_value(x))
    
    fig = px.bar(
        data,
        x='month_name',
        y='avg_aqi',
        title=f"Monthly Average AQI for {city_name}",
        labels={"month_name": "Month", "avg_aqi": "Average AQI"},
        color='avg_aqi',
        color_continuous_scale=['green', 'yellow', 'orange', 'red', 'purple'],
        range_color=[0, 400]
    )
    
    # Add horizontal lines for AQI category thresholds
    fig.add_shape(type="line", x0=-0.5, y0=50, x1=11.5, y1=50, line=dict(color="green", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=100, x1=11.5, y1=100, line=dict(color="yellow", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=200, x1=11.5, y1=200, line=dict(color="orange", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=300, x1=11.5, y1=300, line=dict(color="red", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=400, x1=11.5, y1=400, line=dict(color="purple", width=1, dash="dash"))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    
    return fig

def create_prediction_chart(data):
    """
    Create a line chart for AQI predictions
    """
    # Get colors for each prediction based on the AQI value
    colors = data['predicted_aqi'].apply(lambda x: utils.get_color_for_value(x))
    
    fig = go.Figure()
    
    # Add predicted AQI line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['predicted_aqi'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(width=3, color='rgba(0, 0, 255, 0.7)'),
        marker=dict(
            size=10,
            color=colors,
            line=dict(width=2, color='DarkSlateGrey')
        )
    ))
    
    # Add confidence interval (just for visualization purposes)
    upper_bound = data['predicted_aqi'] * 1.15
    lower_bound = data['predicted_aqi'] * 0.85
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=upper_bound,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=lower_bound,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.1)',
        showlegend=False
    ))
    
    # Add colored background regions for AQI categories
    fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="green", opacity=0.1)
    fig.add_hrect(y0=50, y1=100, line_width=0, fillcolor="yellow", opacity=0.1)
    fig.add_hrect(y0=100, y1=200, line_width=0, fillcolor="orange", opacity=0.1)
    fig.add_hrect(y0=200, y1=300, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=300, y1=400, line_width=0, fillcolor="purple", opacity=0.1)
    fig.add_hrect(y0=400, y1=500, line_width=0, fillcolor="maroon", opacity=0.1)
    
    # Update layout
    fig.update_layout(
        title="AI-Powered AQI Prediction",
        xaxis_title="Date",
        yaxis_title="Predicted AQI",
        yaxis_range=[0, 500],
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
