"""
Terrain Data Service Module
Handles elevation data acquisition from Open-Elevation API.
Provides terrain profile generation for routes between coordinates.
"""

import logging
import requests
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
from dt_project.config import ConfigManager

logger = logging.getLogger(__name__)

class TerrainService:
    """Service for acquiring terrain and elevation data."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the terrain service.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        
        # Configure API endpoints
        self.api_endpoint = self.config.get('ELEVATION_API_ENDPOINT', 
                                           'https://api.open-elevation.com/api/v1/lookup')
        self.batch_api_endpoint = self.config.get('ELEVATION_BATCH_API_ENDPOINT', 
                                                 'https://api.open-elevation.com/api/v1/lookup')
        
        # Set up request caching to avoid excessive API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 604800  # 7 days for elevation data (rarely changes)
        
        # Track API calls to handle rate limiting
        self.last_api_call = 0
        self.min_call_interval = 1  # minimum seconds between API calls
        
        # Max locations per batch request
        self.max_batch_size = 100
    
    def get_elevation(self, latitude: float, longitude: float) -> float:
        """
        Get the elevation for a single location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Elevation in meters above sea level
        """
        # Check cache first
        cache_key = f"elev_{latitude:.6f}_{longitude:.6f}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached elevation data for {latitude}, {longitude}")
            return self.cache[cache_key]
            
        # Prepare request parameters
        params = {
            'locations': f"{latitude},{longitude}"
        }
        
        # Respect rate limits
        self._respect_rate_limit()
            
        # Make API request
        try:
            response = requests.get(self.api_endpoint, params=params)
            self.last_api_call = time.time()
            response.raise_for_status()
            
            data = response.json()
            
            # Extract elevation value
            results = data.get('results', [])
            if not results:
                logger.warning(f"No elevation data found for {latitude}, {longitude}")
                return 0.0
                
            elevation = float(results[0].get('elevation', 0.0))
            
            # Cache the result
            self._cache_result(cache_key, elevation)
            
            return elevation
            
        except Exception as e:
            logger.error(f"Error getting elevation for {latitude}, {longitude}: {str(e)}")
            return 0.0
    
    def get_batch_elevations(self, coordinates: List[Tuple[float, float]]) -> List[float]:
        """
        Get elevations for multiple locations in one batch request.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            
        Returns:
            List of elevation values in meters
        """
        if not coordinates:
            return []
            
        # Check if all coordinates are in cache
        elevations = []
        uncached_coords = []
        uncached_indices = []
        
        for i, (lat, lon) in enumerate(coordinates):
            cache_key = f"elev_{lat:.6f}_{lon:.6f}"
            if self._is_cache_valid(cache_key):
                elevations.append(self.cache[cache_key])
            else:
                elevations.append(None)  # Placeholder for uncached values
                uncached_coords.append((lat, lon))
                uncached_indices.append(i)
        
        # If all coordinates were cached, return immediately
        if not uncached_coords:
            return elevations
            
        # Process uncached coordinates in batches
        for start_idx in range(0, len(uncached_coords), self.max_batch_size):
            batch = uncached_coords[start_idx:start_idx + self.max_batch_size]
            batch_indices = uncached_indices[start_idx:start_idx + self.max_batch_size]
            
            # Format locations for the API
            locations = []
            for lat, lon in batch:
                locations.append({"latitude": lat, "longitude": lon})
            
            # Respect rate limits
            self._respect_rate_limit()
                
            # Make API request
            try:
                response = requests.post(
                    self.batch_api_endpoint,
                    json={"locations": locations}
                )
                self.last_api_call = time.time()
                response.raise_for_status()
                
                data = response.json()
                
                # Extract elevation values
                results = data.get('results', [])
                
                for i, result in enumerate(results):
                    lat = batch[i][0]
                    lon = batch[i][1]
                    elevation = float(result.get('elevation', 0.0))
                    
                    # Update result and cache
                    idx = batch_indices[i]
                    elevations[idx] = elevation
                    
                    cache_key = f"elev_{lat:.6f}_{lon:.6f}"
                    self._cache_result(cache_key, elevation)
                
            except Exception as e:
                logger.error(f"Error getting batch elevations: {str(e)}")
                # Fill missing values with fallbacks (0.0)
                for idx in batch_indices:
                    if elevations[idx] is None:
                        elevations[idx] = 0.0
        
        return elevations
    
    def generate_terrain_profile(self, 
                                start_coord: Tuple[float, float], 
                                end_coord: Tuple[float, float], 
                                num_points: int = 50) -> Dict[str, Any]:
        """
        Generate a terrain profile (elevations) along a straight line path.
        
        Args:
            start_coord: Starting (latitude, longitude) tuple
            end_coord: Ending (latitude, longitude) tuple
            num_points: Number of points to sample along the path
            
        Returns:
            Dictionary with profile data including distances, elevations, and stats
        """
        start_lat, start_lon = start_coord
        end_lat, end_lon = end_coord
        
        # Generate evenly spaced points along the path
        path_coords = self._generate_path_points(start_coord, end_coord, num_points)
        
        # Get elevations for all points
        elevations = self.get_batch_elevations(path_coords)
        
        # Calculate distances from start point
        distances = [self._haversine_distance(start_coord, coord) for coord in path_coords]
        
        # Calculate slope between adjacent points
        slopes = []
        for i in range(1, len(distances)):
            if distances[i] - distances[i-1] > 0:
                # Convert to percent grade: rise/run * 100
                rise = elevations[i] - elevations[i-1]
                run = (distances[i] - distances[i-1]) * 1000  # convert km to m
                slope_percent = (rise / run) * 100
                slopes.append(slope_percent)
            else:
                slopes.append(0)
                
        # Add a 0 at the beginning to match the length of other arrays
        slopes.insert(0, 0)
        
        # Calculate statistics
        elevation_gain = sum(max(0, elevations[i] - elevations[i-1]) for i in range(1, len(elevations)))
        elevation_loss = sum(max(0, elevations[i-1] - elevations[i]) for i in range(1, len(elevations)))
        avg_elevation = sum(elevations) / len(elevations) if elevations else 0
        max_elevation = max(elevations) if elevations else 0
        min_elevation = min(elevations) if elevations else 0
        steepest_uphill = max(slopes) if slopes else 0
        steepest_downhill = min(slopes) if slopes else 0
        total_distance = distances[-1] if distances else 0
        
        # Create result dictionary
        profile = {
            "path_coordinates": path_coords,
            "distances": distances,  # in km
            "elevations": elevations,  # in meters
            "slopes": slopes,  # in percent
            "stats": {
                "total_distance": total_distance,  # km
                "elevation_gain": elevation_gain,  # meters
                "elevation_loss": elevation_loss,  # meters
                "avg_elevation": avg_elevation,  # meters
                "max_elevation": max_elevation,  # meters
                "min_elevation": min_elevation,  # meters
                "steepest_uphill": steepest_uphill,  # percent
                "steepest_downhill": steepest_downhill,  # percent
            }
        }
        
        return profile
    
    def generate_terrain_matrix(self, 
                               center_coord: Tuple[float, float], 
                               radius_km: float, 
                               resolution: int = 25) -> Dict[str, Any]:
        """
        Generate a square terrain matrix centered on the provided coordinates.
        
        Args:
            center_coord: Center (latitude, longitude) tuple
            radius_km: Radius from center in kilometers
            resolution: Number of points on each side of the square grid
            
        Returns:
            Dictionary with terrain matrix data
        """
        center_lat, center_lon = center_coord
        
        # Generate grid coordinates
        lat_km = 111.32  # approximate km per degree latitude
        lon_km = 111.32 * math.cos(math.radians(center_lat))  # km per degree longitude
        
        lat_radius = radius_km / lat_km  # convert km to degrees
        lon_radius = radius_km / lon_km
        
        lat_min = center_lat - lat_radius
        lat_max = center_lat + lat_radius
        lon_min = center_lon - lon_radius
        lon_max = center_lon + lon_radius
        
        lat_vals = np.linspace(lat_min, lat_max, resolution)
        lon_vals = np.linspace(lon_min, lon_max, resolution)
        
        # Create coordinate grid
        coordinates = []
        for lat in lat_vals:
            for lon in lon_vals:
                coordinates.append((lat, lon))
        
        # Get elevations for all grid points
        elevations = self.get_batch_elevations(coordinates)
        
        # Reshape into grid
        elevation_grid = np.array(elevations).reshape((resolution, resolution))
        
        # Calculate terrain properties
        # Calculate slope gradients in x and y directions
        dx = (lon_max - lon_min) / (resolution - 1) * lon_km * 1000  # m
        dy = (lat_max - lat_min) / (resolution - 1) * lat_km * 1000  # m
        
        # Compute terrain derivatives
        slope_x = np.zeros_like(elevation_grid)
        slope_y = np.zeros_like(elevation_grid)
        roughness = np.zeros_like(elevation_grid)
        
        for i in range(1, resolution-1):
            for j in range(1, resolution-1):
                # Compute x and y gradients (central difference)
                slope_x[i, j] = (elevation_grid[i, j+1] - elevation_grid[i, j-1]) / (2 * dx)
                slope_y[i, j] = (elevation_grid[i+1, j] - elevation_grid[i-1, j]) / (2 * dy)
                
                # Compute local roughness (variation in 3x3 neighborhood)
                neighborhood = elevation_grid[i-1:i+2, j-1:j+2]
                roughness[i, j] = np.std(neighborhood)
        
        # Calculate slope magnitude and direction
        slope_magnitude = np.sqrt(slope_x**2 + slope_y**2) * 100  # percent
        slope_direction = np.degrees(np.arctan2(slope_y, slope_x))
        
        # Calculate statistics
        avg_elevation = np.mean(elevation_grid)
        max_elevation = np.max(elevation_grid)
        min_elevation = np.min(elevation_grid)
        avg_slope = np.mean(slope_magnitude[1:-1, 1:-1])  # exclude edges
        max_slope = np.max(slope_magnitude[1:-1, 1:-1])
        avg_roughness = np.mean(roughness[1:-1, 1:-1])
        
        # Create result dictionary
        terrain_data = {
            "center": center_coord,
            "radius_km": radius_km,
            "resolution": resolution,
            "lat_bounds": (lat_min, lat_max),
            "lon_bounds": (lon_min, lon_max),
            "elevation_grid": elevation_grid.tolist(),
            "slope_magnitude": slope_magnitude.tolist(),
            "slope_direction": slope_direction.tolist(),
            "roughness": roughness.tolist(),
            "stats": {
                "avg_elevation": float(avg_elevation),
                "max_elevation": float(max_elevation),
                "min_elevation": float(min_elevation),
                "elevation_range": float(max_elevation - min_elevation),
                "avg_slope": float(avg_slope),
                "max_slope": float(max_slope),
                "avg_roughness": float(avg_roughness),
            }
        }
        
        return terrain_data
    
    def classify_terrain(self, terrain_data: Dict[str, Any]) -> str:
        """
        Classify terrain based on elevation, slope and roughness data.
        
        Args:
            terrain_data: Terrain data from generate_terrain_matrix
            
        Returns:
            Terrain classification string
        """
        stats = terrain_data.get("stats", {})
        
        avg_elevation = stats.get("avg_elevation", 0)
        elevation_range = stats.get("elevation_range", 0)
        avg_slope = stats.get("avg_slope", 0)
        max_slope = stats.get("max_slope", 0)
        avg_roughness = stats.get("avg_roughness", 0)
        
        # Apply classification rules
        if max_slope > 50:
            return "EXTREME"
        elif max_slope > 35:
            return "MOUNTAINOUS"
        elif max_slope > 25:
            return "HILLY"
        elif max_slope > 15:
            return "ROLLING"
        elif avg_roughness > 5:
            return "ROUGH"
        elif elevation_range < 10 and avg_slope < 5:
            return "FLAT"
        else:
            return "MODERATE"
    
    def _generate_path_points(self, 
                             start_coord: Tuple[float, float], 
                             end_coord: Tuple[float, float], 
                             num_points: int) -> List[Tuple[float, float]]:
        """
        Generate evenly spaced points along a great circle path.
        
        Args:
            start_coord: Starting (latitude, longitude) tuple
            end_coord: Ending (latitude, longitude) tuple
            num_points: Number of points to generate
            
        Returns:
            List of (latitude, longitude) tuples
        """
        # Ensure we have at least 2 points
        num_points = max(2, num_points)
        
        # Convert to radians
        start_lat, start_lon = math.radians(start_coord[0]), math.radians(start_coord[1])
        end_lat, end_lon = math.radians(end_coord[0]), math.radians(end_coord[1])
        
        # Generate fractions of the path
        fractions = np.linspace(0, 1, num_points)
        
        # Compute intermediate points
        path_coords = []
        for f in fractions:
            # Formula for interpolating along a great circle
            a = math.sin((1 - f) * math.acos(math.sin(start_lat) * math.sin(end_lat) + 
                    math.cos(start_lat) * math.cos(end_lat) * math.cos(start_lon - end_lon)))
            b = math.sin(f * math.acos(math.sin(start_lat) * math.sin(end_lat) + 
                    math.cos(start_lat) * math.cos(end_lat) * math.cos(start_lon - end_lon)))
            
            x = (math.sin(start_lat) * b + math.sin(end_lat) * a) / (a + b)
            y = (math.cos(start_lat) * math.cos(start_lon) * b + 
                math.cos(end_lat) * math.cos(end_lon) * a) / (a + b)
            z = (math.cos(start_lat) * math.sin(start_lon) * b + 
                math.cos(end_lat) * math.sin(end_lon) * a) / (a + b)
            
            lat = math.asin(x)
            lon = math.atan2(z, y)
            
            path_coords.append((math.degrees(lat), math.degrees(lon)))
        
        return path_coords
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate the great circle distance between two points in kilometers.
        
        Args:
            coord1: First (latitude, longitude) tuple
            coord2: Second (latitude, longitude) tuple
            
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache or key not in self.cache_expiry:
            return False
            
        expiry_time = self.cache_expiry[key]
        current_time = time.time()
        
        return current_time < expiry_time
    
    def _cache_result(self, key: str, data: Any) -> None:
        """Cache API result with appropriate expiry time."""
        self.cache[key] = data
        expiry_time = time.time() + self.cache_duration
        self.cache_expiry[key] = expiry_time
    
    def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_call_interval:
            sleep_time = self.min_call_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time) 