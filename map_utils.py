"""
Map utilities for AQI visualization
"""

import streamlit as st
import folium
from folium.plugins import HeatMap
import numpy as np
from streamlit_folium import folium_static
from constants import MAJOR_CITIES, DEFAULT_MAP_ZOOM, AQI_LEVELS

class AQIMapVisualizer:
    def __init__(self):
        pass
    
    def create_aqi_map(self, aqi_data):
        """
        Create an interactive map of India showing AQI levels
        """
        if aqi_data is None or aqi_data.empty:
            st.warning("No AQI data available for map visualization.")
            return
        
        # Create base map centered on India
        m = folium.Map(
            location=[20.5937, 78.9629],  # Center of India
            zoom_start=DEFAULT_MAP_ZOOM,
            tiles="CartoDB positron"
        )
        
        # Add AQI markers for each city
        for _, row in aqi_data.iterrows():
            # Create a circular marker with color based on AQI
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=max(5, min(15, row['aqi'] / 30)),  # Size based on AQI (capped)
                color=row['color'],
                fill=True,
                fill_color=row['color'],
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['city']}, {row['state']}</b><br>"
                    f"AQI: {row['aqi']}<br>"
                    f"Category: {row['category']}<br>"
                    f"PM2.5: {row['pollutants']['PM2.5']}<br>"
                    f"PM10: {row['pollutants']['PM10']}<br>"
                    f"NO2: {row['pollutants']['NO2']}<br>"
                    f"SO2: {row['pollutants']['SO2']}<br>"
                    f"CO: {row['pollutants']['CO']}<br>"
                    f"O3: {row['pollutants']['O3']}",
                    max_width=300
                )
            ).add_to(m)
        
        # Add a heat map layer
        heat_data = [
            [row['lat'], row['lon'], row['aqi'] / 100]  # Normalize AQI for heatmap intensity
            for _, row in aqi_data.iterrows()
        ]
        
        HeatMap(
            heat_data,
            radius=25,
            blur=15,
            gradient={
                0.0: AQI_LEVELS["Good"]["color"],
                0.2: AQI_LEVELS["Satisfactory"]["color"],
                0.4: AQI_LEVELS["Moderate"]["color"],
                0.6: AQI_LEVELS["Poor"]["color"],
                0.8: AQI_LEVELS["Very Poor"]["color"],
                1.0: AQI_LEVELS["Severe"]["color"]
            },
            overlay=True,
            control=True,
            show=False  # Hidden by default
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add title
        title_html = '''
            <div style="position: fixed; 
                        top: 10px; left: 50px; width: 200px; height: 35px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; font-weight: bold; text-align: center;
                        padding: 5px;">AQI Levels Across India</div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add legend
        legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px;
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:12px; padding: 10px;">
                <div style="margin-bottom: 5px;"><b>AQI Categories</b></div>
        '''
        
        for category, details in AQI_LEVELS.items():
            legend_html += f'''
                <div style="display: flex; align-items: center; margin-bottom: 2px;">
                    <div style="width: 15px; height: 15px; background-color: {details['color']}; margin-right: 5px;"></div>
                    <div>{category} ({details['range'][0]}-{details['range'][1]})</div>
                </div>
            '''
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def display_map(self, m):
        """
        Display the folium map in Streamlit
        """
        if m:
            folium_static(m)
