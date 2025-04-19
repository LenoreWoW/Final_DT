"""
Terrain Simulation Module
Handles the generation and simulation of terrain features including altitude and incline.
"""

import math
import random
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass

from dt_project.config import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class Point:
    """Represents a point in 3D space with latitude, longitude, and altitude."""
    lat: float
    lon: float
    alt: float = 0.0
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate the great-circle distance to another point in meters."""
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.lon)
        lat2 = math.radians(other.lat)
        lon2 = math.radians(other.lon)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance

class TerrainType:
    """Enum-like class for terrain types."""
    ROAD = "road"
    TRAIL = "trail"
    GRASS = "grass"
    SAND = "sand"
    GRAVEL = "gravel"
    CONCRETE = "concrete"
    ASPHALT = "asphalt"
    
    @staticmethod
    def get_friction_coefficient(terrain_type: str) -> float:
        """Get the friction coefficient for a terrain type."""
        coefficients = {
            TerrainType.ROAD: 0.7,
            TerrainType.TRAIL: 0.8,
            TerrainType.GRASS: 0.9,
            TerrainType.SAND: 1.5,
            TerrainType.GRAVEL: 1.2,
            TerrainType.CONCRETE: 0.6,
            TerrainType.ASPHALT: 0.5
        }
        return coefficients.get(terrain_type, 0.7)
    
    @staticmethod
    def get_energy_factor(terrain_type: str) -> float:
        """Get the energy expenditure factor for a terrain type."""
        factors = {
            TerrainType.ROAD: 1.0,
            TerrainType.TRAIL: 1.2,
            TerrainType.GRASS: 1.3,
            TerrainType.SAND: 1.8,
            TerrainType.GRAVEL: 1.4,
            TerrainType.CONCRETE: 0.95,
            TerrainType.ASPHALT: 0.9
        }
        return factors.get(terrain_type, 1.0)

class TerrainSimulation:
    """Simulates terrain features with sinusoidal patterns and controlled randomness."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the terrain simulation.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        
        # Load configuration
        self._load_config()
        
        # Random seeds for consistent patterns
        self.altitude_seed = random.randint(0, 1000)
        self.roughness_seed = random.randint(0, 1000)
        
    def _load_config(self) -> None:
        """Load configuration parameters for the simulation."""
        terrain_config = self.config.get("simulation.terrain", {})
        
        # Resolution in meters (distance between sample points)
        self.resolution = terrain_config.get("resolution", 50)
        
        # Altitude parameters
        self.max_altitude = terrain_config.get("max_altitude", 5000)
        self.min_altitude = terrain_config.get("min_altitude", 0)
        
        # Frequency for sinusoidal patterns (cycles per meter)
        self.frequency = terrain_config.get("frequency", 0.01)
        
        # Noise level for randomness (0.0 to 1.0)
        self.noise_level = terrain_config.get("noise_level", 0.2)
        
    def generate_terrain_profile(self, start_point: Point, end_point: Point, 
                              num_points: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate a terrain profile between two points.
        
        Args:
            start_point: Starting point
            end_point: Ending point
            num_points: Number of points to generate. If None, calculated based on resolution.
            
        Returns:
            List of terrain points with altitude and gradient information
        """
        # Calculate distance between points
        distance = start_point.distance_to(end_point)
        
        # Determine number of sample points if not provided
        if num_points is None:
            num_points = max(10, int(distance / self.resolution))
        
        # Create interpolated points
        profile = []
        bearing = self._calculate_bearing(start_point, end_point)
        
        for i in range(num_points):
            # Interpolate position
            progress = i / (num_points - 1)
            point = self._interpolate_position(start_point, end_point, progress)
            
            # Generate altitude for this point
            altitude = self._generate_altitude(point, distance * progress)
            point.alt = altitude
            
            # Terrain type varies along the route
            terrain_type = self._determine_terrain_type(point, progress)
            
            # Add to profile
            profile_point = {
                'index': i,
                'distance': distance * progress,
                'latitude': point.lat,
                'longitude': point.lon,
                'altitude': altitude,
                'terrain_type': terrain_type
            }
            
            profile.append(profile_point)
        
        # Calculate gradients and additional metrics
        self._add_gradients(profile, distance / (num_points - 1))
        
        return profile
    
    def generate_terrain_grid(self, center_point: Point, 
                           width_meters: float, height_meters: float,
                           resolution_meters: Optional[float] = None) -> np.ndarray:
        """
        Generate a 2D terrain grid centered on a point.
        
        Args:
            center_point: Center point of the grid
            width_meters: Width of the grid in meters
            height_meters: Height of the grid in meters
            resolution_meters: Resolution in meters. If None, uses config resolution.
            
        Returns:
            2D numpy array of altitudes
        """
        # Use configured resolution if not specified
        if resolution_meters is None:
            resolution_meters = self.resolution
        
        # Calculate grid dimensions
        cols = max(2, int(width_meters / resolution_meters))
        rows = max(2, int(height_meters / resolution_meters))
        
        # Calculate grid bounds
        # 111,111 meters is approximately 1 degree latitude
        # Longitude degrees vary with latitude: 111,111 * cos(latitude)
        lat_offset = height_meters / (2 * 111111)
        lon_offset = width_meters / (2 * 111111 * math.cos(math.radians(center_point.lat)))
        
        min_lat = center_point.lat - lat_offset
        max_lat = center_point.lat + lat_offset
        min_lon = center_point.lon - lon_offset
        max_lon = center_point.lon + lon_offset
        
        # Generate altitude grid
        grid = np.zeros((rows, cols))
        
        for row in range(rows):
            for col in range(cols):
                # Interpolate position
                lat = min_lat + (max_lat - min_lat) * row / (rows - 1)
                lon = min_lon + (max_lon - min_lon) * col / (cols - 1)
                point = Point(lat, lon)
                
                # Distance from center
                dist_x = (col / (cols - 1) - 0.5) * width_meters
                dist_y = (row / (rows - 1) - 0.5) * height_meters
                distance = math.sqrt(dist_x**2 + dist_y**2)
                
                # Generate altitude
                grid[row, col] = self._generate_altitude(point, distance)
        
        return grid
        
    def generate_route(self, start_point: Point, length_meters: float, 
                     route_type: str = "loop", num_points: int = 100) -> List[Dict[str, Any]]:
        """
        Generate a route with the specified characteristics.
        
        Args:
            start_point: Starting point
            length_meters: Approximate length of the route in meters
            route_type: Type of route ("loop", "out_and_back", "random")
            num_points: Number of points to generate
            
        Returns:
            List of terrain points forming a route
        """
        if route_type == "loop":
            return self._generate_loop_route(start_point, length_meters, num_points)
        elif route_type == "out_and_back":
            return self._generate_out_and_back_route(start_point, length_meters, num_points)
        else:  # "random" or any other value
            return self._generate_random_route(start_point, length_meters, num_points)
    
    def calculate_energy_expenditure(self, profile: List[Dict[str, Any]], 
                                  weight_kg: float) -> List[float]:
        """
        Calculate energy expenditure for a terrain profile.
        
        Args:
            profile: Terrain profile
            weight_kg: Weight of the person/vehicle in kg
            
        Returns:
            List of energy expenditure values in joules for each segment
        """
        # Energy expenditure for each segment
        energy_values = []
        
        for i in range(1, len(profile)):
            prev = profile[i-1]
            curr = profile[i]
            
            # Distance in meters
            distance = curr['distance'] - prev['distance']
            
            # Altitude change in meters
            altitude_change = curr['altitude'] - prev['altitude']
            
            # Calculate gradient energy component
            # Positive values require more energy (going uphill)
            if altitude_change > 0:
                # Energy = mass * g * height
                gradient_energy = weight_kg * 9.81 * altitude_change
            else:
                # Downhill requires less energy but still some
                gradient_energy = weight_kg * 9.81 * altitude_change * 0.3
            
            # Calculate horizontal energy component
            # Basic work formula considering terrain type
            terrain_factor = TerrainType.get_energy_factor(curr['terrain_type'])
            horizontal_energy = weight_kg * distance * terrain_factor * 0.5
            
            # Total energy (joules)
            total_energy = gradient_energy + horizontal_energy
            
            # Ensure always positive minimum energy expenditure
            energy_values.append(max(100, total_energy))
        
        return energy_values
    
    def _generate_altitude(self, point: Point, distance: float) -> float:
        """
        Generate altitude for a point using sinusoidal patterns with noise.
        
        Args:
            point: Point for which to generate altitude
            distance: Distance along the route in meters
            
        Returns:
            Altitude in meters
        """
        # Create multiple frequency components
        base_frequency = self.frequency
        
        # Base sinusoidal pattern with multiple components
        component1 = math.sin(2 * math.pi * base_frequency * distance)
        component2 = 0.5 * math.sin(2 * math.pi * base_frequency * 2 * distance + 1.3)
        component3 = 0.25 * math.sin(2 * math.pi * base_frequency * 4 * distance + 2.6)
        
        # Position-dependent component (varies with lat/lon)
        position_hash = math.sin(point.lat * 0.1) + math.cos(point.lon * 0.1)
        position_component = position_hash * 0.2
        
        # Combine components
        combined = component1 + component2 + component3 + position_component
        
        # Scale to altitude range
        altitude_range = self.max_altitude - self.min_altitude
        normalized = (combined + 2) / 4  # Convert from [-2,2] to [0,1]
        altitude = self.min_altitude + normalized * altitude_range
        
        # Add noise
        noise = self._perlin_noise(distance, self.altitude_seed) * self.noise_level * altitude_range * 0.2
        altitude += noise
        
        return altitude
    
    def _calculate_bearing(self, p1: Point, p2: Point) -> float:
        """
        Calculate the initial bearing from point 1 to point 2.
        
        Args:
            p1: Starting point
            p2: Ending point
            
        Returns:
            Bearing in degrees (0-360)
        """
        # Convert to radians
        lat1 = math.radians(p1.lat)
        lon1 = math.radians(p1.lon)
        lat2 = math.radians(p2.lat)
        lon2 = math.radians(p2.lon)
        
        # Calculate bearing
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.atan2(y, x)
        
        # Convert to degrees
        bearing = math.degrees(bearing)
        
        # Normalize to 0-360
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def _interpolate_position(self, p1: Point, p2: Point, t: float) -> Point:
        """
        Interpolate between two points using great-circle path.
        
        Args:
            p1: Starting point
            p2: Ending point
            t: Interpolation parameter (0 to 1)
            
        Returns:
            Interpolated point
        """
        # Simple linear interpolation for small distances
        # For longer distances, spherical interpolation would be more accurate
        lat = p1.lat + t * (p2.lat - p1.lat)
        lon = p1.lon + t * (p2.lon - p1.lon)
        return Point(lat, lon)
    
    def _determine_terrain_type(self, point: Point, progress: float) -> str:
        """
        Determine terrain type based on position and route progress.
        
        Args:
            point: Position
            progress: Progress along the route (0 to 1)
            
        Returns:
            Terrain type string
        """
        # Use a combination of position, progress, and randomness
        # Position hash to get some spatial consistency
        position_hash = (math.sin(point.lat * 100) + math.cos(point.lon * 100)) * 0.5
        
        # Progress-based variation
        progress_variation = math.sin(progress * 4 * math.pi)
        
        # Combine factors
        combined = position_hash + 0.3 * progress_variation
        
        # Map to terrain types with higher probability for roads
        terrain_types = [
            TerrainType.ROAD,
            TerrainType.ROAD,
            TerrainType.TRAIL,
            TerrainType.GRASS,
            TerrainType.ASPHALT,
            TerrainType.CONCRETE,
            TerrainType.GRAVEL
        ]
        
        index = min(len(terrain_types) - 1, max(0, int((combined + 1) * len(terrain_types) / 2)))
        return terrain_types[index]
    
    def _add_gradients(self, profile: List[Dict[str, Any]], segment_length: float) -> None:
        """
        Add gradient information to the profile.
        
        Args:
            profile: Terrain profile to update
            segment_length: Length of each segment in meters
        """
        for i in range(len(profile)):
            # Gradient requires looking at adjacent points
            if i == 0:
                # First point - forward gradient
                rise = profile[i+1]['altitude'] - profile[i]['altitude']
                gradient = rise / segment_length
            elif i == len(profile) - 1:
                # Last point - backward gradient
                rise = profile[i]['altitude'] - profile[i-1]['altitude']
                gradient = rise / segment_length
            else:
                # Interior point - average of backward and forward gradients
                rise_back = profile[i]['altitude'] - profile[i-1]['altitude']
                rise_forward = profile[i+1]['altitude'] - profile[i]['altitude']
                gradient = (rise_back + rise_forward) / (2 * segment_length)
            
            # Convert to percentage
            gradient_percent = gradient * 100.0
            
            # Calculate difficulty rating based on gradient
            difficulty = self._calculate_difficulty(gradient_percent)
            
            # Add to profile
            profile[i]['gradient'] = round(gradient_percent, 1)
            profile[i]['difficulty'] = difficulty
    
    def _calculate_difficulty(self, gradient_percent: float) -> str:
        """
        Calculate difficulty rating based on gradient.
        
        Args:
            gradient_percent: Gradient in percent
            
        Returns:
            Difficulty rating
        """
        abs_gradient = abs(gradient_percent)
        
        if abs_gradient < 2:
            return "easy"
        elif abs_gradient < 5:
            return "moderate"
        elif abs_gradient < 10:
            return "challenging"
        elif abs_gradient < 15:
            return "difficult"
        else:
            return "extreme"
    
    def _generate_loop_route(self, start_point: Point, length_meters: float, 
                          num_points: int) -> List[Dict[str, Any]]:
        """
        Generate a loop route that starts and ends at the same point.
        
        Args:
            start_point: Starting point
            length_meters: Approximate length of the route in meters
            num_points: Number of points to generate
            
        Returns:
            List of terrain points forming a loop
        """
        # Calculate radius for a circular route
        radius_meters = length_meters / (2 * math.pi)
        
        # Create points along a circular path
        route = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # Calculate offset from center (start_point)
            # 111,111 meters is approximately 1 degree latitude
            # Longitude degrees vary with latitude: 111,111 * cos(latitude)
            lat_offset = (radius_meters * math.sin(angle)) / 111111
            lon_offset = (radius_meters * math.cos(angle)) / (111111 * math.cos(math.radians(start_point.lat)))
            
            # Create point
            lat = start_point.lat + lat_offset
            lon = start_point.lon + lon_offset
            point = Point(lat, lon)
            
            # Progress along the route (0 to 1)
            progress = i / num_points
            
            # Generate altitude
            distance = progress * length_meters
            altitude = self._generate_altitude(point, distance)
            
            # Terrain type
            terrain_type = self._determine_terrain_type(point, progress)
            
            # Add to route
            route_point = {
                'index': i,
                'distance': distance,
                'latitude': point.lat,
                'longitude': point.lon,
                'altitude': altitude,
                'terrain_type': terrain_type
            }
            
            route.append(route_point)
        
        # Ensure it's a perfect loop
        route[-1]['latitude'] = route[0]['latitude']
        route[-1]['longitude'] = route[0]['longitude']
        
        # Add gradients
        segment_length = length_meters / num_points
        self._add_gradients(route, segment_length)
        
        return route
    
    def _generate_out_and_back_route(self, start_point: Point, length_meters: float,
                                  num_points: int) -> List[Dict[str, Any]]:
        """
        Generate an out and back route (goes out and returns on the same path).
        
        Args:
            start_point: Starting point
            length_meters: Approximate length of the route in meters
            num_points: Number of points to generate
            
        Returns:
            List of terrain points forming an out and back route
        """
        # Half the points for the outbound journey
        half_points = num_points // 2
        outbound_length = length_meters / 2
        
        # Generate a random direction
        random_angle = random.uniform(0, 2 * math.pi)
        
        # Calculate end point
        # 111,111 meters is approximately 1 degree latitude
        # Longitude degrees vary with latitude: 111,111 * cos(latitude)
        lat_offset = (outbound_length * math.sin(random_angle)) / 111111
        lon_offset = (outbound_length * math.cos(random_angle)) / (111111 * math.cos(math.radians(start_point.lat)))
        
        end_lat = start_point.lat + lat_offset
        end_lon = start_point.lon + lon_offset
        end_point = Point(end_lat, end_lon)
        
        # Generate outbound journey
        outbound_profile = self.generate_terrain_profile(start_point, end_point, half_points)
        
        # Create return journey (same points in reverse but with updated indices and distances)
        inbound_profile = []
        
        for i, point in enumerate(reversed(outbound_profile)):
            # Deep copy to avoid modifying the original
            new_point = point.copy()
            
            # Update index and distance
            new_point['index'] = half_points + i
            new_point['distance'] = outbound_length + (outbound_length - point['distance'])
            
            inbound_profile.append(new_point)
        
        # Combine outbound and inbound (skip the duplicate at the turnaround)
        full_profile = outbound_profile + inbound_profile[1:]
        
        return full_profile
    
    def _generate_random_route(self, start_point: Point, length_meters: float,
                            num_points: int) -> List[Dict[str, Any]]:
        """
        Generate a random route with the given length.
        
        Args:
            start_point: Starting point
            length_meters: Approximate length of the route in meters
            num_points: Number of points to generate
            
        Returns:
            List of terrain points forming a random route
        """
        # Create a random walk
        route = []
        current_point = start_point
        segment_length = length_meters / (num_points - 1)
        
        for i in range(num_points):
            # Progress along the route
            progress = i / (num_points - 1)
            distance = progress * length_meters
            
            # Add point to route
            altitude = self._generate_altitude(current_point, distance)
            terrain_type = self._determine_terrain_type(current_point, progress)
            
            route_point = {
                'index': i,
                'distance': distance,
                'latitude': current_point.lat,
                'longitude': current_point.lon,
                'altitude': altitude,
                'terrain_type': terrain_type
            }
            
            route.append(route_point)
            
            # Generate next point (random direction with slight preference to return to start)
            if i < num_points - 1:
                # Random angle with bias toward start point as we get further
                random_angle = random.uniform(0, 2 * math.pi)
                
                # As progress increases, increase bias to return to start
                return_bias = 0.5 * progress
                
                # Calculate bearing to start
                bearing_to_start = self._calculate_bearing(current_point, start_point)
                bearing_to_start_rad = math.radians(bearing_to_start)
                
                # Combine random angle with bearing to start
                combined_angle = (1 - return_bias) * random_angle + return_bias * bearing_to_start_rad
                
                # Move in the combined direction
                lat_offset = (segment_length * math.sin(combined_angle)) / 111111
                lon_offset = (segment_length * math.cos(combined_angle)) / (111111 * math.cos(math.radians(current_point.lat)))
                
                next_lat = current_point.lat + lat_offset
                next_lon = current_point.lon + lon_offset
                current_point = Point(next_lat, next_lon)
        
        # Add gradients
        self._add_gradients(route, segment_length)
        
        return route
    
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