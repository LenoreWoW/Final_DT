"""
Location Service Module
Handles geocoding, reverse geocoding, and location-based services using Maps.co API.
"""

import logging
import requests
import json
import time
from typing import Dict, Any, List, Optional, Tuple

from dt_project.config import ConfigManager

logger = logging.getLogger(__name__)

class LocationService:
    """Service for geocoding and location-based services."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the location service.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        
        # Get API keys and endpoints from config
        self.api_key = self.config.get('GEOCODING_API_KEY', '')
        self.forward_endpoint = self.config.get('GEOCODING_FORWARD_ENDPOINT', 
                                               'https://geocode.maps.co/search')
        self.reverse_endpoint = self.config.get('GEOCODING_REVERSE_ENDPOINT', 
                                               'https://geocode.maps.co/reverse')
        
        # Set up request caching to avoid excessive API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 86400  # 24 hours for location data
        
        # Track API calls to handle rate limiting
        self.last_api_call = 0
        self.min_call_interval = 1  # minimum seconds between API calls
        
        # Initialize major cities database
        self._init_major_cities()
    
    def search_places(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for locations matching the query.
        
        Args:
            query: Search query (e.g. city name, address)
            limit: Maximum number of results to return
            
        Returns:
            List of matching location dictionaries
        """
        if not query or len(query.strip()) < 2:
            return []
            
        # Normalize and trim query
        query = query.strip().lower()
        
        # Check cache first
        cache_key = f"search_{query}_{limit}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached location search results for '{query}'")
            return self.cache[cache_key]
        
        # Check if query matches any major city
        city_matches = self._search_major_cities(query, limit)
        if city_matches:
            # If we have exact matches from our database, use them
            # This also saves API calls for common queries
            self._cache_result(cache_key, city_matches)
            return city_matches
            
        # Prepare request parameters
        params = {
            'q': query,
            'limit': limit,
            'format': 'json'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
            
        # Respect rate limits
        self._respect_rate_limit()
            
        # Make API request
        try:
            response = requests.get(self.forward_endpoint, params=params)
            self.last_api_call = time.time()
            response.raise_for_status()
            
            data = response.json()
            
            # Process results
            places = []
            for place in data:
                processed_place = {
                    'display_name': place.get('display_name', ''),
                    'latitude': float(place.get('lat', 0)),
                    'longitude': float(place.get('lon', 0)),
                    'type': place.get('type', 'unknown'),
                    'importance': place.get('importance', 0),
                    'country': place.get('address', {}).get('country', ''),
                    'country_code': place.get('address', {}).get('country_code', '').upper(),
                    'state': place.get('address', {}).get('state', ''),
                    'city': place.get('address', {}).get('city', '') or 
                            place.get('address', {}).get('town', '') or 
                            place.get('address', {}).get('village', ''),
                }
                places.append(processed_place)
                
            # Cache the results
            self._cache_result(cache_key, places)
            
            return places
            
        except Exception as e:
            logger.error(f"Error searching for location '{query}': {str(e)}")
            return []
    
    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Convert coordinates to location information.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Location information dictionary or None if not found
        """
        # Check cache first
        cache_key = f"reverse_{latitude:.6f}_{longitude:.6f}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached reverse geocoding results for {latitude}, {longitude}")
            return self.cache[cache_key]
            
        # Check if coordinates match any major city
        city_match = self._match_coordinates_to_city(latitude, longitude)
        if city_match:
            # This also saves API calls for common locations
            self._cache_result(cache_key, city_match)
            return city_match
            
        # Prepare request parameters
        params = {
            'lat': latitude,
            'lon': longitude,
            'format': 'json'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
            
        # Respect rate limits
        self._respect_rate_limit()
            
        # Make API request
        try:
            response = requests.get(self.reverse_endpoint, params=params)
            self.last_api_call = time.time()
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'error' in data:
                logger.warning(f"No results found for coordinates {latitude}, {longitude}")
                return None
                
            # Process the result
            address = data.get('address', {})
            location = {
                'display_name': data.get('display_name', ''),
                'latitude': float(data.get('lat', latitude)),
                'longitude': float(data.get('lon', longitude)),
                'country': address.get('country', ''),
                'country_code': address.get('country_code', '').upper(),
                'state': address.get('state', ''),
                'city': address.get('city', '') or address.get('town', '') or address.get('village', ''),
                'postcode': address.get('postcode', ''),
                'road': address.get('road', ''),
                'neighbourhood': address.get('neighbourhood', ''),
            }
            
            # Cache the result
            self._cache_result(cache_key, location)
            
            return location
            
        except Exception as e:
            logger.error(f"Error reverse geocoding coordinates {latitude}, {longitude}: {str(e)}")
            return None
    
    def get_major_cities(self) -> List[Dict[str, Any]]:
        """
        Get a list of major world cities for UI dropdowns.
        
        Returns:
            List of city dictionaries with name, country and coordinates
        """
        return self.major_cities
    
    def _init_major_cities(self) -> None:
        """Initialize the list of major cities."""
        # This is a subset of major world cities with their coordinates
        # In a real implementation, this would be stored in a database
        self.major_cities = [
            {
                'city': 'New York',
                'country': 'United States',
                'country_code': 'US',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'display_name': 'New York, United States'
            },
            {
                'city': 'London',
                'country': 'United Kingdom',
                'country_code': 'GB',
                'latitude': 51.5074,
                'longitude': -0.1278,
                'display_name': 'London, United Kingdom'
            },
            {
                'city': 'Paris',
                'country': 'France',
                'country_code': 'FR',
                'latitude': 48.8566,
                'longitude': 2.3522,
                'display_name': 'Paris, France'
            },
            {
                'city': 'Tokyo',
                'country': 'Japan',
                'country_code': 'JP',
                'latitude': 35.6762,
                'longitude': 139.6503,
                'display_name': 'Tokyo, Japan'
            },
            {
                'city': 'Sydney',
                'country': 'Australia',
                'country_code': 'AU',
                'latitude': -33.8688,
                'longitude': 151.2093,
                'display_name': 'Sydney, Australia'
            },
            {
                'city': 'Moscow',
                'country': 'Russia',
                'country_code': 'RU',
                'latitude': 55.7558,
                'longitude': 37.6173,
                'display_name': 'Moscow, Russia'
            },
            {
                'city': 'Beijing',
                'country': 'China',
                'country_code': 'CN',
                'latitude': 39.9042,
                'longitude': 116.4074,
                'display_name': 'Beijing, China'
            },
            {
                'city': 'Cairo',
                'country': 'Egypt',
                'country_code': 'EG',
                'latitude': 30.0444,
                'longitude': 31.2357,
                'display_name': 'Cairo, Egypt'
            },
            {
                'city': 'Rio de Janeiro',
                'country': 'Brazil',
                'country_code': 'BR',
                'latitude': -22.9068,
                'longitude': -43.1729,
                'display_name': 'Rio de Janeiro, Brazil'
            },
            {
                'city': 'Doha',
                'country': 'Qatar',
                'country_code': 'QA',
                'latitude': 25.2854,
                'longitude': 51.5310,
                'display_name': 'Doha, Qatar'
            },
            {
                'city': 'Washington D.C.',
                'country': 'United States',
                'country_code': 'US',
                'latitude': 38.9072,
                'longitude': -77.0369,
                'display_name': 'Washington D.C., United States'
            },
            {
                'city': 'Berlin',
                'country': 'Germany',
                'country_code': 'DE',
                'latitude': 52.5200,
                'longitude': 13.4050,
                'display_name': 'Berlin, Germany'
            },
            {
                'city': 'Toronto',
                'country': 'Canada',
                'country_code': 'CA',
                'latitude': 43.6532,
                'longitude': -79.3832,
                'display_name': 'Toronto, Canada'
            },
            {
                'city': 'Singapore',
                'country': 'Singapore',
                'country_code': 'SG',
                'latitude': 1.3521,
                'longitude': 103.8198,
                'display_name': 'Singapore, Singapore'
            },
            {
                'city': 'Dubai',
                'country': 'United Arab Emirates',
                'country_code': 'AE',
                'latitude': 25.2048,
                'longitude': 55.2708,
                'display_name': 'Dubai, United Arab Emirates'
            }
        ]
    
    def _search_major_cities(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Search for cities matching the query in our major cities database.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching cities
        """
        matches = []
        query = query.lower()
        
        for city in self.major_cities:
            # Check if query matches city name or country
            if query in city['city'].lower() or query in city['country'].lower():
                matches.append(city)
                
            # Check exact matches for country code
            if query.upper() == city['country_code']:
                matches.append(city)
                
            if len(matches) >= limit:
                break
                
        return matches
    
    def _match_coordinates_to_city(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Check if coordinates closely match any major city.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Matching city or None
        """
        # Define a small radius (~5km at equator)
        radius = 0.045  
        
        for city in self.major_cities:
            if (abs(city['latitude'] - latitude) < radius and 
                abs(city['longitude'] - longitude) < radius):
                return city
                
        return None
    
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