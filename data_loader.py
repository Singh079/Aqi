"""
Data Loader for AQI Monitoring Application
Handles fetching and processing of AQI data from CPCB API
"""

import requests
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import time
from constants import CPCB_API_ENDPOINT, CPCB_HISTORICAL_ENDPOINT, MAJOR_CITIES, AQI_LEVELS

class AQIDataLoader:
    def __init__(self):
        self.api_endpoint = CPCB_API_ENDPOINT
        self.historical_endpoint = CPCB_HISTORICAL_ENDPOINT
        
    @st.cache_data(ttl=3600)  # Cache data for 1 hour
    def get_current_aqi_data(_self, api_key=None):
        """
        Fetch current AQI data for Indian cities from CPCB API
        """
        try:
            # In a real scenario, we would use the actual CPCB API
            # For now, we'll create realistic data based on major cities
            
            # This should be replaced with actual API call once API key is available
            # headers = {'api-key': api_key} if api_key else {}
            # response = requests.get(self.api_endpoint, headers=headers)
            # if response.status_code == 200:
            #     data = response.json()
            #     # Process the data
            # else:
            #     st.error(f"Failed to fetch data: {response.status_code}")
            #     return None
            
            # Simulate AQI data for demonstration
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = []
            
            for city, location in MAJOR_CITIES.items():
                # Generate realistic AQI values (normally higher in Delhi, Mumbai etc.)
                base_aqi = np.random.normal(
                    loc=200 if city in ["Delhi", "Kanpur", "Lucknow"] else 
                        150 if city in ["Mumbai", "Kolkata", "Ahmedabad"] else 
                        100, 
                    scale=30
                )
                aqi = max(20, min(500, int(base_aqi)))  # Ensure it's within 20-500 range
                
                # Determine AQI category and color
                category = next((level for level, details in AQI_LEVELS.items() 
                               if details["range"][0] <= aqi <= details["range"][1]), "Severe")
                
                # Generate realistic pollutant values
                pm25 = max(10, min(300, int(aqi * 0.7 + np.random.normal(0, 10))))
                pm10 = max(20, min(450, int(aqi * 1.2 + np.random.normal(0, 15))))
                no2 = max(5, min(200, int(aqi * 0.3 + np.random.normal(0, 8))))
                so2 = max(2, min(100, int(aqi * 0.2 + np.random.normal(0, 5))))
                co = max(0.3, min(30, aqi * 0.05 + np.random.normal(0, 1)))
                o3 = max(10, min(180, int(aqi * 0.25 + np.random.normal(0, 7))))
                
                data.append({
                    "city": city,
                    "state": _self._get_state_for_city(city),
                    "aqi": aqi,
                    "category": category,
                    "color": AQI_LEVELS[category]["color"],
                    "lat": location["lat"],
                    "lon": location["lon"],
                    "timestamp": current_time,
                    "pollutants": {
                        "PM2.5": pm25,
                        "PM10": pm10,
                        "NO2": no2,
                        "SO2": so2,
                        "CO": round(co, 2),
                        "O3": o3
                    }
                })
            
            return pd.DataFrame(data)
        
        except Exception as e:
            st.error(f"Error fetching current AQI data: {e}")
            return None
    
    @st.cache_data(ttl=86400)  # Cache historical data for a day
    def get_historical_aqi_data(_self, city, days=30, api_key=None):
        """
        Fetch historical AQI data for a specific city
        """
        try:
            # In a real scenario, we would fetch historical data from the CPCB API
            # For now, generate realistic historical data
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate base AQI trend with seasonal and weekly patterns
            base_pattern = []
            for date in date_range:
                # Higher AQI in winter months (Oct-Feb)
                seasonal_factor = 1.3 if date.month in [10, 11, 12, 1, 2] else 1.0
                # Higher AQI on weekdays
                weekday_factor = 1.1 if date.weekday() < 5 else 0.9
                
                # Base value depends on city
                if city in ["Delhi", "Kanpur", "Lucknow"]:
                    base_value = 220 * seasonal_factor * weekday_factor
                elif city in ["Mumbai", "Kolkata", "Ahmedabad"]:
                    base_value = 160 * seasonal_factor * weekday_factor
                else:
                    base_value = 110 * seasonal_factor * weekday_factor
                
                # Add some random variation
                aqi_value = max(30, min(500, int(base_value + np.random.normal(0, 20))))
                base_pattern.append(aqi_value)
            
            # Create dataframe
            df = pd.DataFrame({
                'date': date_range,
                'aqi': base_pattern
            })
            
            # Add categories and colors
            df['category'] = df['aqi'].apply(lambda x: next((level for level, details in AQI_LEVELS.items() 
                               if details["range"][0] <= x <= details["range"][1]), "Severe"))
            df['color'] = df['category'].apply(lambda x: AQI_LEVELS[x]["color"])
            
            # Generate pollutant data
            for date, aqi in zip(date_range, base_pattern):
                pm25 = max(10, min(300, int(aqi * 0.7 + np.random.normal(0, 10))))
                pm10 = max(20, min(450, int(aqi * 1.2 + np.random.normal(0, 15))))
                no2 = max(5, min(200, int(aqi * 0.3 + np.random.normal(0, 8))))
                so2 = max(2, min(100, int(aqi * 0.2 + np.random.normal(0, 5))))
                co = max(0.3, min(30, aqi * 0.05 + np.random.normal(0, 1)))
                o3 = max(10, min(180, int(aqi * 0.25 + np.random.normal(0, 7))))
                
                idx = df[df['date'] == date].index[0]
                df.at[idx, 'PM2.5'] = pm25
                df.at[idx, 'PM10'] = pm10
                df.at[idx, 'NO2'] = no2
                df.at[idx, 'SO2'] = so2
                df.at[idx, 'CO'] = round(co, 2)
                df.at[idx, 'O3'] = o3
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching historical AQI data for {city}: {e}")
            return None
    
    def _get_state_for_city(self, city):
        """Helper method to return state for a given city"""
        states = {
            "Delhi": "Delhi",
            "Mumbai": "Maharashtra",
            "Kolkata": "West Bengal",
            "Chennai": "Tamil Nadu",
            "Bangalore": "Karnataka",
            "Hyderabad": "Telangana",
            "Ahmedabad": "Gujarat",
            "Pune": "Maharashtra",
            "Jaipur": "Rajasthan",
            "Lucknow": "Uttar Pradesh",
            "Kanpur": "Uttar Pradesh",
            "Nagpur": "Maharashtra",
            "Indore": "Madhya Pradesh",
            "Thane": "Maharashtra",
            "Bhopal": "Madhya Pradesh",
            "Visakhapatnam": "Andhra Pradesh",
            "Patna": "Bihar",
            "Vadodara": "Gujarat",
            "Ghaziabad": "Uttar Pradesh",
            "Varanasi": "Uttar Pradesh"
        }
        return states.get(city, "Unknown")
    
    @st.cache_data(ttl=86400)  # Cache for a day
    def get_multiple_cities_data(_self, cities, days=30):
        """
        Get historical data for multiple cities for comparison
        """
        all_data = {}
        for city in cities:
            city_data = _self.get_historical_aqi_data(city, days)
            if city_data is not None:
                all_data[city] = city_data
        
        return all_data

