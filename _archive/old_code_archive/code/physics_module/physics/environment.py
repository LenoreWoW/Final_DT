"""
Environmental Simulation Module
Handles simulation of environmental conditions including temperature, humidity, and wind.
"""

import math
import random
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from dt_project.config import ConfigManager

logger = logging.getLogger(__name__)

class EnvironmentalSimulation:
    """Simulates environmental conditions with controlled randomness."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the environmental simulation engine.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        
        # Load configuration parameters
        self._load_config()
        
        # Set initial values
        self.current_time = datetime.now()
        self.base_temperature = 20.0  # Celsius
        self.base_humidity = 50.0     # Percentage
        self.base_wind_speed = 5.0    # km/h
        self.base_wind_direction = 0  # Degrees (0 = North, 90 = East, etc.)
        
        # Random seeds for each variable to ensure consistent patterns
        self.temp_seed = random.randint(0, 1000)
        self.humidity_seed = random.randint(0, 1000)
        self.wind_speed_seed = random.randint(0, 1000)
        self.wind_dir_seed = random.randint(0, 1000)
        
        # Noise generators
        self.noise_generators = {
            'temperature': self._perlin_noise,
            'humidity': self._perlin_noise,
            'wind_speed': self._perlin_noise,
            'wind_direction': self._perlin_noise
        }
        
    def _load_config(self) -> None:
        """Load configuration parameters for the simulation."""
        env_config = self.config.get("simulation.environment", {})
        
        # Temperature range
        self.temp_range = env_config.get("temp_range", [-20, 45])
        
        # Humidity range
        self.humidity_range = env_config.get("humidity_range", [0, 100])
        
        # Wind speed range
        self.wind_speed_range = env_config.get("wind_speed_range", [0, 30])
        
        # Time step in seconds
        self.time_step = env_config.get("time_step", 60)
        
        # Frequencies for sinusoidal patterns
        self.temp_daily_freq = 1.0 / (24 * 60 * 60)  # Daily temperature cycle
        self.temp_yearly_freq = 1.0 / (365.25 * 24 * 60 * 60)  # Yearly temperature cycle
        self.humidity_daily_freq = 1.0 / (24 * 60 * 60)  # Daily humidity cycle
        self.wind_daily_freq = 1.0 / (12 * 60 * 60)  # 12-hour wind cycle
        
        # Amplitudes for variations
        self.temp_daily_amplitude = 5.0  # Celsius
        self.temp_yearly_amplitude = 15.0  # Celsius
        self.humidity_daily_amplitude = 20.0  # Percentage
        self.wind_speed_amplitude = 5.0  # km/h
        
        # Noise levels
        self.noise_level = env_config.get("noise_level", 0.2)
        
    def set_base_conditions(self, temperature: float, humidity: float, 
                          wind_speed: float, wind_direction: float) -> None:
        """
        Set base environmental conditions.
        
        Args:
            temperature: Base temperature in Celsius
            humidity: Base humidity percentage
            wind_speed: Base wind speed in km/h
            wind_direction: Base wind direction in degrees
        """
        self.base_temperature = temperature
        self.base_humidity = humidity
        self.base_wind_speed = wind_speed
        self.base_wind_direction = wind_direction
        
    def set_time(self, simulation_time: datetime) -> None:
        """
        Set the simulation time.
        
        Args:
            simulation_time: Datetime object representing the simulation time
        """
        self.current_time = simulation_time
        
    def advance_time(self, seconds: int = None) -> datetime:
        """
        Advance the simulation time by the specified number of seconds.
        If seconds is None, uses the configured time_step.
        
        Args:
            seconds: Number of seconds to advance
            
        Returns:
            New simulation time
        """
        if seconds is None:
            seconds = self.time_step
            
        self.current_time += timedelta(seconds=seconds)
        return self.current_time
        
    def get_conditions(self, time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get environmental conditions for the specified time.
        If time is None, uses the current simulation time.
        
        Args:
            time: Datetime object for which to calculate conditions
            
        Returns:
            Dictionary with environmental conditions
        """
        if time is None:
            time = self.current_time
            
        # Calculate conditions
        temperature = self._calculate_temperature(time)
        humidity = self._calculate_humidity(time)
        wind_speed = self._calculate_wind_speed(time)
        wind_direction = self._calculate_wind_direction(time)
        
        # Apply range constraints
        temperature = max(min(temperature, self.temp_range[1]), self.temp_range[0])
        humidity = max(min(humidity, self.humidity_range[1]), self.humidity_range[0])
        wind_speed = max(min(wind_speed, self.wind_speed_range[1]), self.wind_speed_range[0])
        wind_direction = wind_direction % 360
        
        # Calculate "feels like" temperature using heat index and wind chill
        feels_like = self._calculate_feels_like(temperature, humidity, wind_speed)
        
        return {
            'time': time.isoformat(),
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'wind_speed': round(wind_speed, 1),
            'wind_direction': round(wind_direction),
            'feels_like': round(feels_like, 1),
            'units': {
                'temperature': '°C',
                'humidity': '%',
                'wind_speed': 'km/h',
                'wind_direction': '°'
            }
        }
        
    def get_conditions_range(self, start_time: datetime, end_time: datetime, 
                           interval_seconds: int = 3600) -> List[Dict[str, Any]]:
        """
        Get environmental conditions over a time range.
        
        Args:
            start_time: Start datetime
            end_time: End datetime
            interval_seconds: Interval between data points in seconds
            
        Returns:
            List of condition dictionaries
        """
        conditions = []
        current = start_time
        
        while current <= end_time:
            conditions.append(self.get_conditions(current))
            current += timedelta(seconds=interval_seconds)
            
        return conditions
        
    def _calculate_temperature(self, time: datetime) -> float:
        """
        Calculate temperature for the given time using sinusoidal patterns with noise.
        
        Args:
            time: Datetime for which to calculate temperature
            
        Returns:
            Temperature in Celsius
        """
        # Convert time to seconds for sinusoidal calculations
        epoch = datetime(2000, 1, 1)
        time_seconds = (time - epoch).total_seconds()
        
        # Daily cycle (coolest at ~5am, warmest at ~3pm)
        day_of_year = time.timetuple().tm_yday
        time_of_day = time.hour + time.minute / 60
        
        # Daily variation
        daily_offset = 5  # Hours (coolest at 5am)
        daily_phase = 2 * math.pi * ((time_of_day - daily_offset) % 24) / 24
        daily_variation = self.temp_daily_amplitude * math.sin(daily_phase)
        
        # Yearly variation
        yearly_offset = 15  # Days (coldest is day 15, January 15th)
        yearly_phase = 2 * math.pi * ((day_of_year - yearly_offset) % 365) / 365
        yearly_variation = self.temp_yearly_amplitude * math.sin(yearly_phase)
        
        # Add noise
        noise_value = self._get_noise('temperature', time_seconds) * self.noise_level * 8.0
        
        # Combine base temperature with variations
        temperature = self.base_temperature + daily_variation + yearly_variation + noise_value
        
        return temperature
        
    def _calculate_humidity(self, time: datetime) -> float:
        """
        Calculate humidity for the given time.
        
        Args:
            time: Datetime for which to calculate humidity
            
        Returns:
            Humidity percentage
        """
        # Convert time to seconds for sinusoidal calculations
        epoch = datetime(2000, 1, 1)
        time_seconds = (time - epoch).total_seconds()
        
        # Daily cycle (highest at dawn, lowest in afternoon)
        time_of_day = time.hour + time.minute / 60
        
        # Daily variation (inverse of temperature cycle)
        daily_offset = 15  # Hours (highest at 3am)
        daily_phase = 2 * math.pi * ((time_of_day - daily_offset) % 24) / 24
        daily_variation = self.humidity_daily_amplitude * math.sin(daily_phase)
        
        # Temperature affects humidity (inverse relationship)
        temperature = self._calculate_temperature(time)
        temperature_factor = max(0, (30 - temperature) / 30)  # Higher temps = lower humidity
        temperature_effect = 15.0 * temperature_factor
        
        # Add noise
        noise_value = self._get_noise('humidity', time_seconds) * self.noise_level * 15.0
        
        # Combine base humidity with variations
        humidity = self.base_humidity + daily_variation + temperature_effect + noise_value
        
        return humidity
        
    def _calculate_wind_speed(self, time: datetime) -> float:
        """
        Calculate wind speed for the given time.
        
        Args:
            time: Datetime for which to calculate wind speed
            
        Returns:
            Wind speed in km/h
        """
        # Convert time to seconds for sinusoidal calculations
        epoch = datetime(2000, 1, 1)
        time_seconds = (time - epoch).total_seconds()
        
        # Daily cycle
        time_of_day = time.hour + time.minute / 60
        
        # Wind typically picks up during the day and drops at night
        daily_offset = 4  # Hours (lowest at 4am)
        daily_phase = 2 * math.pi * ((time_of_day - daily_offset) % 24) / 24
        daily_variation = self.wind_speed_amplitude * math.sin(daily_phase)
        
        # Add noise (wind is quite variable)
        noise_value = self._get_noise('wind_speed', time_seconds) * self.noise_level * 10.0
        
        # Occasional gusts
        gust_probability = 0.05  # 5% chance of a gust
        gust = 0
        if random.random() < gust_probability:
            gust = random.uniform(3, 10)
        
        # Combine base wind speed with variations
        wind_speed = max(0, self.base_wind_speed + daily_variation + noise_value + gust)
        
        return wind_speed
        
    def _calculate_wind_direction(self, time: datetime) -> float:
        """
        Calculate wind direction for the given time.
        
        Args:
            time: Datetime for which to calculate wind direction
            
        Returns:
            Wind direction in degrees
        """
        # Convert time to seconds for sinusoidal calculations
        epoch = datetime(2000, 1, 1)
        time_seconds = (time - epoch).total_seconds()
        
        # Wind direction changes more slowly
        slow_variation_freq = 1.0 / (4 * 60 * 60)  # 4-hour cycle
        slow_phase = 2 * math.pi * time_seconds * slow_variation_freq
        slow_variation = 45.0 * math.sin(slow_phase)
        
        # Add noise for small fluctuations
        noise_value = self._get_noise('wind_direction', time_seconds) * self.noise_level * 30.0
        
        # Combine base direction with variations
        wind_direction = (self.base_wind_direction + slow_variation + noise_value) % 360
        
        return wind_direction
        
    def _calculate_feels_like(self, temperature: float, humidity: float, wind_speed: float) -> float:
        """
        Calculate "feels like" temperature using heat index and wind chill.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity percentage
            wind_speed: Wind speed in km/h
            
        Returns:
            "Feels like" temperature in Celsius
        """
        # Convert to Fahrenheit for standard formulas
        temp_f = temperature * 9/5 + 32
        
        if temp_f <= 50:
            # Wind chill applies when it's cold and windy
            # Convert km/h to mph for the formula
            wind_mph = wind_speed * 0.621371
            
            if wind_mph >= 3:
                # Wind chill formula
                wind_chill_f = 35.74 + 0.6215 * temp_f - 35.75 * wind_mph**0.16 + 0.4275 * temp_f * wind_mph**0.16
                # Convert back to Celsius
                feels_like = (wind_chill_f - 32) * 5/9
            else:
                feels_like = temperature
        
        elif temp_f >= 80 and humidity >= 40:
            # Heat index applies when it's hot and humid
            # Heat index formula
            hi_f = -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
            hi_f -= 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2
            hi_f -= 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity
            hi_f += 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2
            
            # Convert back to Celsius
            feels_like = (hi_f - 32) * 5/9
        
        else:
            # Neither wind chill nor heat index applies
            feels_like = temperature
        
        return feels_like
    
    def _get_noise(self, variable: str, time_seconds: float) -> float:
        """
        Get noise value for the given variable and time.
        
        Args:
            variable: Environmental variable name
            time_seconds: Time in seconds
            
        Returns:
            Noise value between -1 and 1
        """
        # Use the appropriate noise generator for the variable
        generator = self.noise_generators.get(variable, self._perlin_noise)
        
        # Get seed for the variable
        seed = getattr(self, f"{variable.split('_')[0]}_seed", 0)
        
        # Calculate noise
        return generator(time_seconds, seed)
    
    def _perlin_noise(self, x: float, seed: int = 0) -> float:
        """
        Simple Perlin-like noise function.
        
        Args:
            x: Input value
            seed: Seed for the noise
            
        Returns:
            Noise value between -1 and 1
        """
        # Add seed to x
        x = x + seed
        
        # Multiple frequencies for more natural noise
        high_freq = math.sin(x * 0.01) * 0.5
        med_freq = math.sin(x * 0.05 + 100) * 0.3
        low_freq = math.sin(x * 0.2 + 300) * 0.2
        
        # Combine frequencies
        return high_freq + med_freq + low_freq
        
    def get_seasonal_adjustment(self, time: datetime) -> Dict[str, float]:
        """
        Get seasonal adjustment factors for the given time.
        
        Args:
            time: Datetime for which to calculate adjustments
            
        Returns:
            Dictionary with seasonal adjustment factors
        """
        # Day of year from 0 to 365
        day_of_year = time.timetuple().tm_yday - 1
        
        # Northern hemisphere seasons
        # (0 = winter, 0.25 = spring, 0.5 = summer, 0.75 = fall)
        season_phase = ((day_of_year / 365) - 0.5) % 1
        
        # Calculate seasonal factors
        temperature_factor = math.sin(2 * math.pi * season_phase)
        humidity_factor = math.sin(2 * math.pi * (season_phase + 0.125))
        precipitation_factor = math.sin(2 * math.pi * (season_phase + 0.25))
        
        # Season names
        season_names = ["Winter", "Spring", "Summer", "Fall"]
        season_idx = int((season_phase * 4) % 4)
        
        return {
            'season': season_names[season_idx],
            'temperature_factor': round(temperature_factor, 2),
            'humidity_factor': round(humidity_factor, 2),
            'precipitation_factor': round(precipitation_factor, 2)
        }
        
    def get_diurnal_adjustment(self, time: datetime) -> Dict[str, float]:
        """
        Get diurnal (daily) adjustment factors for the given time.
        
        Args:
            time: Datetime for which to calculate adjustments
            
        Returns:
            Dictionary with diurnal adjustment factors
        """
        # Hour of day from 0 to 24
        hour_of_day = time.hour + time.minute / 60
        
        # Diurnal phase (0 = midnight, 0.25 = 6am, 0.5 = noon, 0.75 = 6pm)
        diurnal_phase = hour_of_day / 24
        
        # Calculate diurnal factors
        temperature_factor = math.sin(2 * math.pi * (diurnal_phase - 0.25))
        humidity_factor = -math.sin(2 * math.pi * (diurnal_phase - 0.25))
        
        # Time of day description
        if 5 <= hour_of_day < 12:
            time_of_day = "Morning"
        elif 12 <= hour_of_day < 18:
            time_of_day = "Afternoon"
        elif 18 <= hour_of_day < 22:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"
        
        return {
            'time_of_day': time_of_day,
            'temperature_factor': round(temperature_factor, 2),
            'humidity_factor': round(humidity_factor, 2)
        } 