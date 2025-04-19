"""
Weather Data Service
Handles acquisition of weather data from Open-Meteo API
"""

import logging
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from dt_project.config import ConfigManager

logger = logging.getLogger(__name__)

class WeatherService:
    """Service for retrieving weather data from Open-Meteo API."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the weather service.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        
        # Get base URL from config or use default
        self.api_endpoint = self.config.get('WEATHER_API_ENDPOINT', 
                                            'https://api.open-meteo.com/v1/forecast')
        
        # Set up request caching to avoid excessive API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = {
            'current': 1800,  # 30 minutes for current weather
            'forecast': 3600,  # 1 hour for forecast
            'historical': 86400  # 24 hours for historical data
        }
    
    def get_current_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Get current weather data for a location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary with current weather data
        """
        cache_key = f"current_{latitude}_{longitude}"
        
        # Check cache first
        if self._is_cache_valid(cache_key, 'current'):
            logger.info(f"Using cached current weather data for {latitude}, {longitude}")
            return self.cache[cache_key]
        
        # Prepare request parameters
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'current': ['temperature_2m', 'relative_humidity_2m', 'precipitation', 
                      'wind_speed_10m', 'wind_direction_10m'],
            'timezone': 'auto',
            'models': 'best_match'
        }
        
        # Make API request
        try:
            response = requests.get(self.api_endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Format the response data
            current_weather = {
                'temperature': data['current']['temperature_2m'],
                'humidity': data['current']['relative_humidity_2m'],
                'precipitation': data['current']['precipitation'],
                'wind_speed': data['current']['wind_speed_10m'],
                'wind_direction': data['current']['wind_direction_10m'],
                'timestamp': data['current']['time'],
                'units': {
                    'temperature': data['current_units']['temperature_2m'],
                    'humidity': data['current_units']['relative_humidity_2m'],
                    'precipitation': data['current_units']['precipitation'],
                    'wind_speed': data['current_units']['wind_speed_10m'],
                    'wind_direction': data['current_units']['wind_direction_10m']
                }
            }
            
            # Cache the results
            self._cache_result(cache_key, current_weather, 'current')
            
            return current_weather
            
        except Exception as e:
            logger.error(f"Error fetching current weather data: {str(e)}")
            
            # Return fallback data if we had an error
            return self._get_fallback_weather()
    
    def get_forecast(self, latitude: float, longitude: float, days: int = 3) -> List[Dict[str, Any]]:
        """
        Get weather forecast data for a location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            days: Number of days to forecast (max 16)
            
        Returns:
            List of daily weather forecasts
        """
        days = min(16, max(1, days))  # Ensure days is between 1 and 16
        cache_key = f"forecast_{latitude}_{longitude}_{days}"
        
        # Check cache first
        if self._is_cache_valid(cache_key, 'forecast'):
            logger.info(f"Using cached forecast data for {latitude}, {longitude}")
            return self.cache[cache_key]
        
        # Prepare request parameters
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'daily': ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
                    'wind_speed_10m_max', 'wind_direction_10m_dominant'],
            'timezone': 'auto',
            'forecast_days': days,
            'models': 'best_match'
        }
        
        # Make API request
        try:
            response = requests.get(self.api_endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Format the response data
            daily_forecasts = []
            
            for i in range(len(data['daily']['time'])):
                forecast = {
                    'date': data['daily']['time'][i],
                    'temp_max': data['daily']['temperature_2m_max'][i],
                    'temp_min': data['daily']['temperature_2m_min'][i],
                    'precipitation': data['daily']['precipitation_sum'][i],
                    'wind_speed': data['daily']['wind_speed_10m_max'][i],
                    'wind_direction': data['daily']['wind_direction_10m_dominant'][i],
                    'units': {
                        'temp_max': data['daily_units']['temperature_2m_max'],
                        'temp_min': data['daily_units']['temperature_2m_min'],
                        'precipitation': data['daily_units']['precipitation_sum'],
                        'wind_speed': data['daily_units']['wind_speed_10m_max'],
                        'wind_direction': data['daily_units']['wind_direction_10m_dominant']
                    }
                }
                daily_forecasts.append(forecast)
            
            # Cache the results
            self._cache_result(cache_key, daily_forecasts, 'forecast')
            
            return daily_forecasts
            
        except Exception as e:
            logger.error(f"Error fetching forecast data: {str(e)}")
            
            # Return fallback forecast if we had an error
            return self._get_fallback_forecast(days)
    
    def get_historical_weather(self, latitude: float, longitude: float, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get historical weather data for a location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            days: Number of past days to retrieve (max 92)
            
        Returns:
            List of daily historical weather data
        """
        days = min(92, max(1, days))  # Ensure days is between 1 and 92
        cache_key = f"historical_{latitude}_{longitude}_{days}"
        
        # Check cache first
        if self._is_cache_valid(cache_key, 'historical'):
            logger.info(f"Using cached historical data for {latitude}, {longitude}")
            return self.cache[cache_key]
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Prepare request parameters
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'daily': ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
                    'wind_speed_10m_max', 'wind_direction_10m_dominant'],
            'timezone': 'auto',
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'models': 'best_match'
        }
        
        # Make API request - use the historical endpoint
        historical_endpoint = self.api_endpoint.replace('forecast', 'archive')
        
        try:
            response = requests.get(historical_endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Format the response data
            historical_data = []
            
            for i in range(len(data['daily']['time'])):
                day_data = {
                    'date': data['daily']['time'][i],
                    'temp_max': data['daily']['temperature_2m_max'][i],
                    'temp_min': data['daily']['temperature_2m_min'][i],
                    'precipitation': data['daily']['precipitation_sum'][i],
                    'wind_speed': data['daily']['wind_speed_10m_max'][i],
                    'wind_direction': data['daily']['wind_direction_10m_dominant'][i],
                    'units': {
                        'temp_max': data['daily_units']['temperature_2m_max'],
                        'temp_min': data['daily_units']['temperature_2m_min'],
                        'precipitation': data['daily_units']['precipitation_sum'],
                        'wind_speed': data['daily_units']['wind_speed_10m_max'],
                        'wind_direction': data['daily_units']['wind_direction_10m_dominant']
                    }
                }
                historical_data.append(day_data)
            
            # Cache the results
            self._cache_result(cache_key, historical_data, 'historical')
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            
            # Return fallback historical data if we had an error
            return self._get_fallback_historical(days)
    
    def _is_cache_valid(self, key: str, data_type: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache or key not in self.cache_expiry:
            return False
        
        expiry_time = self.cache_expiry[key]
        current_time = time.time()
        
        return current_time < expiry_time
    
    def _cache_result(self, key: str, data: Any, data_type: str) -> None:
        """Cache API result with appropriate expiry time."""
        self.cache[key] = data
        expiry_time = time.time() + self.cache_duration[data_type]
        self.cache_expiry[key] = expiry_time
    
    def _get_fallback_weather(self) -> Dict[str, Any]:
        """Get fallback weather data when API requests fail."""
        return {
            'temperature': 20.0,
            'humidity': 50.0,
            'precipitation': 0.0,
            'wind_speed': 5.0,
            'wind_direction': 0,
            'timestamp': datetime.now().isoformat(),
            'units': {
                'temperature': '°C',
                'humidity': '%',
                'precipitation': 'mm',
                'wind_speed': 'km/h',
                'wind_direction': '°'
            },
            'is_fallback': True
        }
    
    def _get_fallback_forecast(self, days: int) -> List[Dict[str, Any]]:
        """Get fallback forecast data when API requests fail."""
        forecasts = []
        base_date = datetime.now().date()
        
        for i in range(days):
            forecast_date = base_date + timedelta(days=i)
            forecasts.append({
                'date': forecast_date.isoformat(),
                'temp_max': 25.0,
                'temp_min': 15.0,
                'precipitation': 0.0,
                'wind_speed': 5.0,
                'wind_direction': 0,
                'units': {
                    'temp_max': '°C',
                    'temp_min': '°C',
                    'precipitation': 'mm',
                    'wind_speed': 'km/h',
                    'wind_direction': '°'
                },
                'is_fallback': True
            })
        
        return forecasts
    
    def _get_fallback_historical(self, days: int) -> List[Dict[str, Any]]:
        """Get fallback historical data when API requests fail."""
        historical = []
        end_date = datetime.now().date()
        
        for i in range(days):
            hist_date = end_date - timedelta(days=i)
            historical.append({
                'date': hist_date.isoformat(),
                'temp_max': 22.0,
                'temp_min': 12.0,
                'precipitation': 0.0,
                'wind_speed': 5.0,
                'wind_direction': 0,
                'units': {
                    'temp_max': '°C',
                    'temp_min': '°C',
                    'precipitation': 'mm',
                    'wind_speed': 'km/h',
                    'wind_direction': '°'
                },
                'is_fallback': True
            })
        
        return historical 