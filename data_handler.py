import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import random
import os
from constants import MAJOR_CITIES, CPCB_API_ENDPOINT, INDIAN_STATES
import db_operations as db_ops

# Initialize the database on first import
from database import init_db, Base, engine

# Check if tables exist and initialize if needed
try:
    init_db()
    db_ops.seed_database_with_test_data()
except Exception as e:
    print(f"Database initialization error: {str(e)}")

def get_available_cities(state=None):
    """
    Returns the list of available cities for monitoring
    If state is provided, returns only cities in that state
    """
    # Always use the constants to ensure consistency with our comprehensive city list
    if state:
        cities = sorted([city for city, data in MAJOR_CITIES.items() if data.get('state') == state])
        print(f"Returning {len(cities)} cities for state: {state}")
        return cities
    else:
        all_cities = sorted(list(MAJOR_CITIES.keys()))
        print(f"Returning all {len(all_cities)} cities")
        return all_cities

def get_available_states():
    """Returns the list of available states with cities"""
    # Always use the comprehensive list of Indian states
    # This ensures all states are always available in the UI
    print(f"Returning {len(INDIAN_STATES)} Indian states")
    return INDIAN_STATES

def get_current_aqi(city):
    """
    Get current AQI data for a specific city
    Returns a dictionary with AQI value, timestamp, and change from yesterday
    """
    try:
        # Get data from database
        return db_ops.get_current_aqi(city)
    except Exception as e:
        raise Exception(f"Error fetching current AQI data: {str(e)}")

def get_pollutant_breakdown(city):
    """
    Get detailed pollutant breakdown for a specific city
    Returns a dictionary with pollutant values
    """
    try:
        # Get data from database
        return db_ops.get_pollutant_breakdown(city)
    except Exception as e:
        raise Exception(f"Error fetching pollutant breakdown: {str(e)}")

def get_all_cities_current_aqi():
    """
    Get current AQI data for all available cities
    Returns a DataFrame with city, AQI value, and coordinates
    """
    try:
        # Get data from database
        return db_ops.get_all_cities_current_aqi()
    except Exception as e:
        raise Exception(f"Error fetching all cities AQI data: {str(e)}")

def get_historical_aqi(city, start_date, end_date):
    """
    Get historical AQI data for a specific city in a date range
    Returns a DataFrame with date and AQI values
    """
    try:
        # Get data from database
        return db_ops.get_historical_aqi(city, start_date, end_date)
    except Exception as e:
        raise Exception(f"Error fetching historical AQI data: {str(e)}")

def get_monthly_avg_aqi(city, year=None):
    """
    Get monthly average AQI for a specific city
    Returns a DataFrame with month and average AQI values
    """
    try:
        # Get data from database
        return db_ops.get_monthly_avg_aqi(city, year)
    except Exception as e:
        raise Exception(f"Error fetching monthly average AQI data: {str(e)}")

# Add a function to save AQI data to the database
def save_aqi_data(city, aqi_value, timestamp=None):
    """Save AQI data to the database"""
    try:
        return db_ops.add_aqi_reading(city, aqi_value, timestamp)
    except Exception as e:
        raise Exception(f"Error saving AQI data: {str(e)}")

# Add a function to save pollutant data to the database
def save_pollutant_data(city, pollutant_data, timestamp=None):
    """Save pollutant data to the database"""
    try:
        return db_ops.add_pollutant_reading(city, pollutant_data, timestamp)
    except Exception as e:
        raise Exception(f"Error saving pollutant data: {str(e)}")
