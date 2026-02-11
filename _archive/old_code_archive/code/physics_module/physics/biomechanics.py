"""
Biomechanical Simulation Module
Handles the simulation of biomechanical aspects of athletic performance.
"""

import math
import random
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from dt_project.config import ConfigManager
from dt_project.physics.terrain import TerrainType

logger = logging.getLogger(__name__)

class BiomechanicalModel:
    """Simulates biomechanical aspects of athletic performance."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the biomechanical model.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        
        # Load configuration
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration parameters for the simulation."""
        biomech_config = self.config.get("simulation.biomechanics", {})
        
        # Energy model type (linear, exponential, etc.)
        self.energy_model = biomech_config.get("energy_model", "linear")
        
        # Fatigue parameters
        self.fatigue_factor = biomech_config.get("fatigue_factor", 0.05)
        self.recovery_rate = biomech_config.get("recovery_rate", 0.02)
        
        # Speed limits
        self.max_speed = biomech_config.get("max_speed", 10.0)  # m/s
        
    def calculate_speed(self, athlete_profile: Dict[str, Any], 
                     terrain_point: Dict[str, Any],
                     environmental_conditions: Dict[str, Any],
                     current_fatigue: float) -> Dict[str, Any]:
        """
        Calculate achievable speed based on athlete profile, terrain, environment, and fatigue.
        
        Args:
            athlete_profile: Athlete profile with biomechanical parameters
            terrain_point: Terrain information including gradient and type
            environmental_conditions: Environmental conditions (temp, humidity, etc.)
            current_fatigue: Current fatigue level (0 to 1)
            
        Returns:
            Dictionary with speed and related metrics
        """
        # Extract relevant parameters
        max_athlete_speed = athlete_profile.get("max_speed", 5.0)  # m/s
        athlete_type = athlete_profile.get("athlete_type", "recreational")
        endurance = athlete_profile.get("endurance", 60.0) / 100.0  # Convert to 0-1 scale
        strength = athlete_profile.get("strength", 60.0) / 100.0  # Convert to 0-1 scale
        fatigue_resistance = athlete_profile.get("fatigue_resistance", 60.0) / 100.0  # Convert to 0-1 scale
        
        # Extract biomechanical parameters
        biomechanics = athlete_profile.get("biomechanics", {})
        stride_length = biomechanics.get("stride_length", 1.5)  # meters
        ground_contact_time = biomechanics.get("ground_contact_time", 0.25)  # seconds
        
        # Extract terrain parameters
        gradient_percent = terrain_point.get("gradient", 0.0)
        terrain_type = terrain_point.get("terrain_type", TerrainType.ROAD)
        
        # Extract environmental parameters
        temperature = environmental_conditions.get("temperature", 20.0)
        humidity = environmental_conditions.get("humidity", 50.0)
        wind_speed = environmental_conditions.get("wind_speed", 0.0)
        wind_direction = environmental_conditions.get("wind_direction", 0.0)
        
        # Base speed calculation
        base_speed = max_athlete_speed * (1.0 - current_fatigue * (1.0 - fatigue_resistance))
        
        # Adjust for gradient
        gradient_adjustment = self._calculate_gradient_adjustment(gradient_percent, strength)
        
        # Adjust for terrain type
        terrain_adjustment = self._calculate_terrain_adjustment(terrain_type)
        
        # Adjust for environmental conditions
        env_adjustment = self._calculate_environmental_adjustment(
            temperature, humidity, wind_speed, wind_direction)
        
        # Calculate stride rate based on speed and stride length
        # At max speed, use the athlete's biomechanics
        # At lower speeds, adjust stride length and rate
        adjusted_stride_length = stride_length * (0.7 + 0.3 * (base_speed / max_athlete_speed))
        stride_rate = base_speed / adjusted_stride_length  # strides per second
        
        # Calculate final speed
        speed = base_speed * gradient_adjustment * terrain_adjustment * env_adjustment
        
        # Cap at maximum possible
        speed = min(speed, self.max_speed)
        
        # Convert to km/h for display
        speed_kmh = speed * 3.6
        
        # Calculate energy expenditure rate
        energy_rate = self._calculate_energy_rate(
            speed, gradient_percent, terrain_type, 
            athlete_profile.get("weight", 70.0))
        
        # Calculate fatigue increase rate
        fatigue_increase = self._calculate_fatigue_increase(
            speed, max_athlete_speed, gradient_percent, endurance, fatigue_resistance)
        
        # Calculate recovery rate if at rest (speed very low)
        recovery_rate = 0.0
        if speed < 0.5:  # If essentially at rest
            recovery_rate = self.recovery_rate * (1.0 + fatigue_resistance)
        
        return {
            "speed": round(speed, 2),  # m/s
            "speed_kmh": round(speed_kmh, 2),  # km/h
            "stride_length": round(adjusted_stride_length, 2),  # meters
            "stride_rate": round(stride_rate, 2),  # strides/second
            "energy_rate": round(energy_rate, 2),  # joules/second
            "fatigue_increase": round(fatigue_increase, 4),  # fatigue units/second
            "recovery_rate": round(recovery_rate, 4),  # fatigue units/second
            "adjustments": {
                "gradient": round(gradient_adjustment, 2),
                "terrain": round(terrain_adjustment, 2),
                "environmental": round(env_adjustment, 2)
            }
        }
    
    def simulate_performance(self, athlete_profile: Dict[str, Any],
                          terrain_profile: List[Dict[str, Any]],
                          environmental_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simulate athletic performance over a terrain profile.
        
        Args:
            athlete_profile: Athlete profile with biomechanical parameters
            terrain_profile: List of terrain points
            environmental_conditions: Environmental conditions
            
        Returns:
            List of performance data points corresponding to terrain points
        """
        performance_data = []
        current_fatigue = 0.0
        total_distance = 0.0
        total_time = 0.0
        total_energy = 0.0
        
        # Get weight for energy calculations
        weight_kg = athlete_profile.get("weight", 70.0)
        
        for i, terrain_point in enumerate(terrain_profile):
            # Calculate speed at this point
            speed_data = self.calculate_speed(
                athlete_profile, terrain_point, environmental_conditions, current_fatigue)
            
            # Extract values
            speed = speed_data["speed"]  # m/s
            energy_rate = speed_data["energy_rate"]  # joules/second
            fatigue_increase = speed_data["fatigue_increase"]
            recovery_rate = speed_data["recovery_rate"]
            
            # Calculate segment metrics
            if i > 0:
                # Distance between points
                prev_point = terrain_profile[i-1]
                segment_distance = terrain_point["distance"] - prev_point["distance"]
                
                # Time to cover segment
                segment_time = segment_distance / speed if speed > 0 else 0
                
                # Update cumulative values
                total_distance += segment_distance
                total_time += segment_time
                
                # Energy expenditure for this segment
                segment_energy = energy_rate * segment_time
                total_energy += segment_energy
                
                # Update fatigue
                fatigue_change = fatigue_increase * segment_time - recovery_rate * segment_time
                current_fatigue = min(1.0, max(0.0, current_fatigue + fatigue_change))
            
            # Create performance data point
            performance_point = {
                "index": i,
                "distance": round(total_distance, 2),  # meters
                "time": round(total_time, 2),  # seconds
                "speed": round(speed, 2),  # m/s
                "speed_kmh": round(speed * 3.6, 2),  # km/h
                "energy": round(total_energy, 2),  # joules
                "fatigue": round(current_fatigue, 2),  # 0-1 scale
                "altitude": terrain_point["altitude"],
                "gradient": terrain_point["gradient"],
                "terrain_type": terrain_point["terrain_type"]
            }
            
            performance_data.append(performance_point)
        
        # Add some derived metrics
        if performance_data:
            last_point = performance_data[-1]
            total_distance = last_point["distance"]
            total_time = last_point["time"]
            
            # Add summary data to each point
            for point in performance_data:
                point["percent_complete"] = round(100 * point["distance"] / total_distance, 1)
                point["avg_speed"] = round(point["distance"] / point["time"], 2) if point["time"] > 0 else 0
                point["avg_speed_kmh"] = round(point["avg_speed"] * 3.6, 2)
                
                # Pace in min/km
                pace_mins_per_km = (1000 / point["avg_speed"]) / 60 if point["avg_speed"] > 0 else 0
                point["pace_mins_per_km"] = round(pace_mins_per_km, 2)
        
        return performance_data
    
    def calculate_fatigue_recovery(self, athlete_profile: Dict[str, Any],
                               activity_intensity: float,
                               duration_minutes: float,
                               current_fatigue: float) -> Dict[str, float]:
        """
        Calculate fatigue development and recovery for an activity.
        
        Args:
            athlete_profile: Athlete profile with relevant parameters
            activity_intensity: Intensity of the activity (0-1)
            duration_minutes: Duration of the activity in minutes
            current_fatigue: Current fatigue level (0-1)
            
        Returns:
            Dictionary with new fatigue level and recovery metrics
        """
        # Extract relevant parameters
        fatigue_resistance = athlete_profile.get("fatigue_resistance", 60.0) / 100.0
        recovery_rate = athlete_profile.get("recovery_rate", 70.0) / 100.0
        endurance = athlete_profile.get("endurance", 60.0) / 100.0
        
        # Calculate fatigue increase rate based on intensity and athlete parameters
        fatigue_increase_rate = self.fatigue_factor * activity_intensity * (1 - fatigue_resistance)
        
        # Calculate fatigue increase
        fatigue_increase = fatigue_increase_rate * duration_minutes
        
        # Calculate new fatigue level
        new_fatigue = min(1.0, current_fatigue + fatigue_increase)
        
        # Calculate recovery time
        if new_fatigue > 0:
            # Time to recover to different levels
            recovery_to_75_percent = self._calculate_recovery_time(new_fatigue, 0.25, recovery_rate)
            recovery_to_50_percent = self._calculate_recovery_time(new_fatigue, 0.5, recovery_rate)
            recovery_to_25_percent = self._calculate_recovery_time(new_fatigue, 0.75, recovery_rate)
            recovery_to_full = self._calculate_recovery_time(new_fatigue, 0.95, recovery_rate)
            
            recovery_metrics = {
                "new_fatigue": round(new_fatigue, 2),
                "fatigue_increase": round(fatigue_increase, 2),
                "recovery_to_75_percent_mins": round(recovery_to_75_percent, 1),
                "recovery_to_50_percent_mins": round(recovery_to_50_percent, 1),
                "recovery_to_25_percent_mins": round(recovery_to_25_percent, 1),
                "recovery_to_full_mins": round(recovery_to_full, 1)
            }
        else:
            recovery_metrics = {
                "new_fatigue": 0.0,
                "fatigue_increase": 0.0,
                "recovery_to_75_percent_mins": 0.0,
                "recovery_to_50_percent_mins": 0.0,
                "recovery_to_25_percent_mins": 0.0,
                "recovery_to_full_mins": 0.0
            }
        
        return recovery_metrics
    
    def _calculate_gradient_adjustment(self, gradient_percent: float, strength: float) -> float:
        """
        Calculate speed adjustment factor for gradient.
        
        Args:
            gradient_percent: Gradient percentage (-ve for downhill, +ve for uphill)
            strength: Athlete strength parameter (0-1)
            
        Returns:
            Gradient adjustment factor
        """
        # For uphill (positive gradient)
        if gradient_percent > 0:
            # Steeper gradients require more strength
            # Formula derived from empirical data on incline running
            # Higher strength reduces the impact of the gradient
            adjustment = 1.0 - (gradient_percent / 100) * (2.0 - strength)
            
            # Ensure adjustment doesn't go below a minimum value
            return max(0.2, adjustment)
        
        # For downhill (negative gradient)
        elif gradient_percent < 0:
            # Slight downhill can increase speed, but too steep will reduce speed
            # Strength has less impact on downhill compared to uphill
            adjustment = 1.0 + min(4.0, abs(gradient_percent)) / 100 * 0.5
            
            # Very steep downhill reduces speed dramatically
            if gradient_percent < -15:
                adjustment *= max(0.5, 1.0 - (abs(gradient_percent) - 15) / 30)
                
            return adjustment
        
        # Flat terrain
        else:
            return 1.0
    
    def _calculate_terrain_adjustment(self, terrain_type: str) -> float:
        """
        Calculate speed adjustment factor for terrain type.
        
        Args:
            terrain_type: Type of terrain
            
        Returns:
            Terrain adjustment factor
        """
        # Terrain coefficients (inverse of energy factors)
        terrain_coefficients = {
            TerrainType.ROAD: 1.0,
            TerrainType.TRAIL: 0.85,
            TerrainType.GRASS: 0.8,
            TerrainType.SAND: 0.55,
            TerrainType.GRAVEL: 0.75,
            TerrainType.CONCRETE: 1.05,
            TerrainType.ASPHALT: 1.1
        }
        
        return terrain_coefficients.get(terrain_type, 1.0)
    
    def _calculate_environmental_adjustment(self, temperature: float, humidity: float,
                                        wind_speed: float, wind_direction: float) -> float:
        """
        Calculate speed adjustment factor for environmental conditions.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            wind_speed: Wind speed in km/h
            wind_direction: Wind direction in degrees
            
        Returns:
            Environmental adjustment factor
        """
        # Temperature effect (performance peaks around 10-15°C)
        temp_adjustment = 1.0
        if temperature < 10:
            # Cold affects performance
            temp_adjustment = 1.0 - max(0, (10 - temperature) / 30)
        elif temperature > 25:
            # Heat affects performance more significantly
            temp_adjustment = 1.0 - min(0.3, (temperature - 25) / 40)
        
        # Humidity effect (high humidity reduces performance, especially in heat)
        humidity_adjustment = 1.0
        if humidity > 60 and temperature > 20:
            # High humidity in heat reduces performance
            humidity_factor = (humidity - 60) / 100
            temp_factor = (temperature - 20) / 20
            humidity_adjustment = 1.0 - min(0.2, humidity_factor * temp_factor * 0.4)
        
        # Wind effect (headwind reduces speed, tailwind increases it slightly)
        # We assume athlete is moving in 0 degrees direction for simplicity
        # In a real app, we'd use the route bearing
        wind_adjustment = 1.0
        
        # Convert wind speed from km/h to m/s
        wind_speed_ms = wind_speed / 3.6
        
        # Simplified wind adjustment
        # Assuming 0 degrees is a headwind, 180 is a tailwind
        if 315 <= wind_direction or wind_direction <= 45:
            # Headwind
            wind_adjustment = 1.0 - min(0.2, wind_speed_ms / 20)
        elif 135 <= wind_direction <= 225:
            # Tailwind (less benefit than the negative of headwind)
            wind_adjustment = 1.0 + min(0.1, wind_speed_ms / 30)
        else:
            # Crosswind (slight negative effect)
            wind_adjustment = 1.0 - min(0.05, wind_speed_ms / 40)
        
        # Combine all adjustments
        return temp_adjustment * humidity_adjustment * wind_adjustment
    
    def _calculate_energy_rate(self, speed: float, gradient_percent: float, 
                          terrain_type: str, weight_kg: float) -> float:
        """
        Calculate energy expenditure rate.
        
        Args:
            speed: Speed in m/s
            gradient_percent: Gradient percentage
            terrain_type: Type of terrain
            weight_kg: Athlete weight in kg
            
        Returns:
            Energy expenditure rate in joules/second (watts)
        """
        # Base metabolic rate (BMR) contribution
        # Simplified formula for average human: ~1.2 watts/kg at rest
        bmr_watts = 1.2 * weight_kg
        
        # Energy cost of horizontal motion
        # ~4.0 watts per kg per m/s on flat ground
        horizontal_watts = 4.0 * weight_kg * speed
        
        # Energy cost of vertical motion (gradient)
        # Positive gradient requires work against gravity
        vertical_speed = speed * gradient_percent / 100  # m/s in vertical direction
        if vertical_speed > 0:
            # Full gravity cost for uphill
            vertical_watts = 9.81 * weight_kg * vertical_speed
        else:
            # Reduced cost for downhill (eccentric muscle action)
            vertical_watts = 0.4 * 9.81 * weight_kg * vertical_speed
        
        # Terrain effect on energy expenditure
        terrain_factor = TerrainType.get_energy_factor(terrain_type)
        terrain_watts = horizontal_watts * (terrain_factor - 1.0)
        
        # Efficiency factor (humans are ~25% efficient at converting
        # chemical energy to mechanical work)
        efficiency = 0.25
        
        # Total energy expenditure rate (joules/second = watts)
        total_watts = (bmr_watts + horizontal_watts + vertical_watts + terrain_watts) / efficiency
        
        return total_watts
    
    def _calculate_fatigue_increase(self, speed: float, max_speed: float,
                              gradient_percent: float, endurance: float,
                              fatigue_resistance: float) -> float:
        """
        Calculate fatigue increase rate.
        
        Args:
            speed: Current speed in m/s
            max_speed: Maximum speed in m/s
            gradient_percent: Gradient percentage
            endurance: Endurance parameter (0-1)
            fatigue_resistance: Fatigue resistance parameter (0-1)
            
        Returns:
            Fatigue increase rate in fatigue units/second
        """
        # Relative intensity (fraction of max speed)
        relative_intensity = speed / max_speed if max_speed > 0 else 0
        
        # Base fatigue rate increases with intensity
        base_fatigue_rate = self.fatigue_factor * (relative_intensity ** 2)
        
        # Gradient effect (uphill causes more fatigue)
        gradient_factor = 1.0
        if gradient_percent > 0:
            gradient_factor = 1.0 + min(3.0, gradient_percent / 10)
        
        # Endurance and fatigue resistance reduce fatigue rate
        athlete_factor = 1.0 - 0.7 * endurance - 0.3 * fatigue_resistance
        
        # Combine factors
        fatigue_rate = base_fatigue_rate * gradient_factor * max(0.2, athlete_factor)
        
        return fatigue_rate
    
    def _calculate_recovery_time(self, fatigue_level: float, recovery_percent: float,
                             recovery_rate: float) -> float:
        """
        Calculate time to recover to a specific percentage of fatigue.
        
        Args:
            fatigue_level: Current fatigue level (0-1)
            recovery_percent: Target recovery percentage (0-1)
            recovery_rate: Recovery rate parameter (0-1)
            
        Returns:
            Recovery time in minutes
        """
        # Target fatigue level
        target_fatigue = fatigue_level * (1.0 - recovery_percent)
        
        # Calculate recovery time (simple linear model)
        recovery_time = (fatigue_level - target_fatigue) / (self.recovery_rate * recovery_rate)
        
        # Convert to minutes
        recovery_time_minutes = recovery_time * 60
        
        return recovery_time_minutes 