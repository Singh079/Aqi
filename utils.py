import pandas as pd
import numpy as np

def get_aqi_category(aqi_value):
    """
    Get the AQI category, color, and health advisory based on the AQI value
    """
    if aqi_value <= 50:
        return "Good", "#3BB143", "Air quality is satisfactory, and poses little or no risk."
    elif aqi_value <= 100:
        return "Satisfactory", "#B2D732", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi_value <= 200:
        return "Moderately Polluted", "#FFC30B", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi_value <= 300:
        return "Poor", "#FF5733", "Health alert: The risk of health effects is increased for everyone. Reduce outdoor activities."
    elif aqi_value <= 400:
        return "Very Poor", "#C70039", "Health warning of emergency conditions: everyone is more likely to be affected. Avoid outdoor activities."
    else:
        return "Severe", "#900C3F", "Health alert: everyone may experience more serious health effects. Avoid all outdoor activities."

def get_color_for_value(value):
    """
    Get the color for a value based on the AQI scale
    """
    if value <= 50:
        return "#3BB143"  # Green
    elif value <= 100:
        return "#B2D732"  # Light Green-Yellow
    elif value <= 200:
        return "#FFC30B"  # Yellow-Orange
    elif value <= 300:
        return "#FF5733"  # Orange-Red
    elif value <= 400:
        return "#C70039"  # Red-Purple
    else:
        return "#900C3F"  # Dark Purple-Maroon

def get_health_recommendations(aqi_value):
    """
    Get health recommendations based on the AQI value
    """
    if aqi_value <= 50:
        return """
        ### Recommendations for Good Air Quality (0-50)
        
        - Enjoy outdoor activities as usual
        - Great time for outdoor exercises and sports
        - No special precautions needed
        - Ideal conditions for sensitive groups
        """
    elif aqi_value <= 100:
        return """
        ### Recommendations for Satisfactory Air Quality (51-100)
        
        - Acceptable time for outdoor activities
        - Those unusually sensitive to air pollution may consider reducing prolonged outdoor exertion
        - No special precautions needed for the general public
        - Keep windows open for good ventilation
        """
    elif aqi_value <= 200:
        return """
        ### Recommendations for Moderately Polluted Air (101-200)
        
        - Sensitive groups (children, elderly, and those with respiratory or heart conditions) should limit prolonged outdoor exertion
        - Everyone else can continue outdoor activities but consider taking more breaks
        - Consider wearing masks during extended outdoor activities
        - Close windows during peak traffic hours
        - Stay hydrated and maintain good indoor air quality
        """
    elif aqi_value <= 300:
        return """
        ### Recommendations for Poor Air Quality (201-300)
        
        - Sensitive groups should avoid all outdoor activities
        - Everyone else should limit outdoor exertion
        - Wear N95 masks when outdoors
        - Keep windows closed and use air purifiers if available
        - Consider rescheduling outdoor events
        - Stay hydrated and watch for symptoms like coughing or shortness of breath
        """
    elif aqi_value <= 400:
        return """
        ### Recommendations for Very Poor Air Quality (301-400)
        
        - Everyone should avoid all outdoor activities
        - Wear N95 masks when outdoors (mandatory for sensitive groups)
        - Keep windows and doors closed at all times
        - Use air purifiers indoors
        - Avoid exercise even indoors if air filtration is not available
        - Monitor for respiratory symptoms and seek medical attention if needed
        - Consider temporarily relocating sensitive individuals if possible
        """
    else:
        return """
        ### Recommendations for Severe Air Quality (401-500)
        
        - HEALTH EMERGENCY: Everyone should stay indoors
        - All outdoor activities should be avoided completely
        - Keep all windows and doors sealed
        - Use multiple air purifiers if available
        - Wear N95 masks even indoors if air purification is insufficient
        - Do not vacuum as it stirs up particles
        - Contact health services immediately if experiencing difficulty breathing
        - Evacuation to areas with better air quality may be considered for vulnerable individuals
        """
