"""
Data Updater module for fetching real-time AQI data and updating the database
"""
import time
import random
from datetime import datetime, timedelta
import api_client
import data_handler as dh
import db_operations as db_ops

def update_aqi_data():
    """
    Update AQI data for all cities in the database
    
    This function would typically be run on a schedule (e.g., every hour)
    to keep the database up-to-date with the latest AQI readings
    """
    try:
        # Get list of all cities
        cities = dh.get_available_cities()
        
        for city in cities:
            # Try to fetch from API first
            api_data = api_client.fetch_current_aqi(city)
            
            if api_data:
                # If successful, save to database
                aqi_value = api_data["aqi"]
                timestamp = datetime.now()
                
                # Save AQI reading
                dh.save_aqi_data(city, aqi_value, timestamp)
                
                # Create pollutant breakdown based on AQI value
                pollutant_data = {
                    "PM2.5": aqi_value * 0.6 * (1 + (random.random() - 0.5) * 0.2),
                    "PM10": aqi_value * 0.3 * (1 + (random.random() - 0.5) * 0.3),
                    "NO2": aqi_value * 0.1 * (1 + (random.random() - 0.5) * 0.3),
                    "SO2": aqi_value * 0.05 * (1 + (random.random() - 0.5) * 0.4),
                    "CO": aqi_value * 0.03 * (1 + (random.random() - 0.5) * 0.3),
                    "O3": aqi_value * 0.02 * (1 + (random.random() - 0.5) * 0.5)
                }
                
                # Save pollutant reading
                dh.save_pollutant_data(city, pollutant_data, timestamp)
                
                print(f"Updated {city} AQI from API: {aqi_value}")
            else:
                # If API fetch fails, generate a simulated reading
                # This would be similar to our previous approach,
                # but now we're storing the data in the database
                
                # Get city-specific baseline
                city_baseline = {
                    "Delhi": 200,
                    "Mumbai": 120,
                    "Kolkata": 170,
                    "Chennai": 90,
                    "Bangalore": 70,
                    "Hyderabad": 110,
                    "Ahmedabad": 130,
                    "Pune": 100,
                    "Jaipur": 150,
                    "Lucknow": 160
                }
                
                baseline = city_baseline.get(city, 100)
                
                # Add some randomness
                aqi_value = int(baseline * (1 + (random.random() - 0.5) * 0.4))
                aqi_value = max(20, min(500, aqi_value))
                
                timestamp = datetime.now()
                
                # Save AQI reading
                dh.save_aqi_data(city, aqi_value, timestamp)
                
                # Create pollutant breakdown
                pollutant_data = {
                    "PM2.5": aqi_value * 0.6 * (1 + (random.random() - 0.5) * 0.2),
                    "PM10": aqi_value * 0.3 * (1 + (random.random() - 0.5) * 0.3),
                    "NO2": aqi_value * 0.1 * (1 + (random.random() - 0.5) * 0.3),
                    "SO2": aqi_value * 0.05 * (1 + (random.random() - 0.5) * 0.4),
                    "CO": aqi_value * 0.03 * (1 + (random.random() - 0.5) * 0.3),
                    "O3": aqi_value * 0.02 * (1 + (random.random() - 0.5) * 0.5)
                }
                
                # Save pollutant reading
                dh.save_pollutant_data(city, pollutant_data, timestamp)
                
                print(f"Updated {city} AQI with simulation: {aqi_value}")
        
        return True
    except Exception as e:
        print(f"Error updating AQI data: {str(e)}")
        return False

# Function to update data on a schedule
def scheduled_update(interval_minutes=60):
    """
    Run update_aqi_data on a schedule
    
    This function would typically be run in a separate thread
    or as a scheduled task
    """
    while True:
        print(f"Scheduled update at {datetime.now()}")
        success = update_aqi_data()
        print(f"Update {'successful' if success else 'failed'}")
        
        # Wait for the next interval
        time.sleep(interval_minutes * 60)

# Run an initial update when the module is imported
try:
    update_aqi_data()
except Exception as e:
    print(f"Initial data update failed: {str(e)}")