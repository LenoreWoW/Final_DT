"""
Athlete Profile Module
Handles the creation, management, and retrieval of athlete profiles for digital twin simulations.
"""

import os
import json
import uuid
import random
import logging
import math
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field

from dt_project.config import ConfigManager

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AthleteProfile:
    """
    Represents an athlete's profile with physical and performance attributes.
    """
    id: str
    name: str
    athlete_type: str  # e.g., runner, cyclist, military, etc.
    age: int
    gender: str
    height: float  # in cm
    weight: float  # in kg
    
    # Physical attributes (normalized 0-1 scale)
    strength: float = 0.5
    endurance: float = 0.5
    speed: float = 0.5
    agility: float = 0.5
    
    # Performance metrics
    vo2max: float = field(default=40.0)  # mL/(kg·min)
    lactate_threshold: float = field(default=0.7)  # % of VO2max
    running_economy: float = field(default=200.0)  # mL/kg/km
    critical_power: float = field(default=250.0)  # watts
    anaerobic_capacity: float = field(default=20.0)  # kJ
    
    # Military-specific attributes (if applicable)
    military_rank: Optional[str] = None
    years_of_service: Optional[int] = None
    specialty: Optional[str] = None
    
    # Biomechanical parameters
    stride_length: float = field(default=0.0)  # meters
    cadence: float = field(default=0.0)  # steps per minute
    ground_contact_time: float = field(default=0.0)  # milliseconds
    vertical_oscillation: float = field(default=0.0)  # cm
    
    # Fatigue parameters
    fatigue_resistance: float = field(default=0.5)  # 0-1 scale
    recovery_rate: float = field(default=0.5)  # 0-1 scale
    
    # Training status
    training_status: str = field(default="Moderate")  # Untrained, Moderate, Trained, Elite
    
    # Historical data
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize derived attributes based on primary attributes."""
        if self.stride_length == 0.0:
            # Estimate stride length based on height
            # Average stride length is approximately 42% of height
            self.stride_length = self.height * 0.0042
        
        if self.cadence == 0.0:
            # Estimate cadence based on height and speed attribute
            # Base cadence is 170 steps/min, adjusted by height and speed preference
            height_factor = 180.0 / self.height  # Taller people typically have lower cadence
            self.cadence = 170.0 * height_factor * (0.9 + 0.2 * self.speed)
        
        if self.ground_contact_time == 0.0:
            # Estimate ground contact time based on speed attribute
            # Higher speed generally means less ground contact time
            # Range from ~300ms (slow) to ~200ms (fast)
            self.ground_contact_time = 300 - (self.speed * 100)
        
        if self.vertical_oscillation == 0.0:
            # Estimate vertical oscillation based on height and efficiency
            # Range from 6-12cm, with more efficient runners having less oscillation
            efficiency = (self.endurance + self.speed) / 2.0
            self.vertical_oscillation = 12 - (efficiency * 6)
    
    def get_profile(self) -> Dict[str, Any]:
        """Return the athlete profile as a dictionary."""
        return asdict(self)
    
    def update_profile(self, attributes: Dict[str, Any]) -> None:
        """
        Update profile attributes with new values.
        
        Args:
            attributes: Dictionary containing attribute names and new values
        """
        for key, value in attributes.items():
            if hasattr(self, key) and key not in ['id', 'created_at']:
                setattr(self, key, value)
        
        # Update the updated_at timestamp
        self.updated_at = datetime.now().isoformat()
        
        # Recalculate derived attributes
        self.__post_init__()
    
    def calculate_power(self, speed: float, gradient: float, 
                        terrain_type: str = 'asphalt') -> float:
        """
        Calculate power output based on speed, gradient, and terrain type.
        
        Args:
            speed: Speed in m/s
            gradient: Gradient as a decimal (0.01 = 1%)
            terrain_type: Type of terrain (asphalt, trail, etc.)
            
        Returns:
            Power output in watts
        """
        # Basic physics-based power calculation
        gravity = 9.81  # m/s^2
        rolling_resistance = {
            'asphalt': 0.005,
            'concrete': 0.007,
            'trail': 0.010,
            'grass': 0.025,
            'sand': 0.060,
            'snow': 0.050
        }.get(terrain_type, 0.010)
        
        # Air density and drag coefficient
        air_density = 1.225  # kg/m^3
        drag_area = 0.5  # m^2, depends on position
        
        # Calculate rolling resistance force
        roll_force = rolling_resistance * self.weight * gravity * math.cos(gradient)
        
        # Calculate gravitational force (for climbing)
        gravity_force = self.weight * gravity * math.sin(gradient)
        
        # Calculate air resistance force
        air_force = 0.5 * air_density * drag_area * speed**2
        
        # Calculate total force
        total_force = roll_force + gravity_force + air_force
        
        # Calculate power
        power = total_force * speed
        
        # Adjust based on efficiency (skill-dependent)
        efficiency_factor = 0.8 + (0.15 * self.endurance)
        
        return power / efficiency_factor


class AthleteManager:
    """
    Manages athlete profiles including creation, storage, retrieval, and deletion.
    """
    
    def __init__(self, data_dir: str = "data/profiles"):
        """
        Initialize the AthleteManager.
        
        Args:
            data_dir: Directory to store athlete profiles
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load athlete types
        self.athlete_types = [
            'runner', 'cyclist', 'triathlete', 'military'
        ]
        
        logger.info(f"AthleteManager initialized with data directory: {self.data_dir}")
    
    def _get_profile_path(self, athlete_id: str) -> str:
        """
        Get the file path for an athlete profile.
        
        Args:
            athlete_id: ID of the athlete
            
        Returns:
            Path to the athlete profile JSON file
        """
        return os.path.join(self.data_dir, f"{athlete_id}.json")
    
    def create_profile(self, name: str, athlete_type: str, age: int, 
                       gender: str, height: float, weight: float, 
                       attributes: Dict[str, Any] = None) -> AthleteProfile:
        """
        Create a new athlete profile.
        
        Args:
            name: Athlete's name
            athlete_type: Type of athlete (runner, cyclist, etc.)
            age: Age in years
            gender: Gender identity
            height: Height in cm
            weight: Weight in kg
            attributes: Additional attributes to set
            
        Returns:
            New AthleteProfile instance
        """
        # Generate a unique ID
        profile_id = str(uuid.uuid4())
        
        # Create base profile
        profile = AthleteProfile(
            id=profile_id,
            name=name,
            athlete_type=athlete_type,
            age=age,
            gender=gender,
            height=height,
            weight=weight
        )
        
        # Apply additional attributes if provided
        if attributes:
            profile.update_profile(attributes)
        
        # Set appropriate values based on athlete type
        self._customize_for_athlete_type(profile, athlete_type)
        
        # Save the profile
        self._save_profile(profile)
        
        logger.info(f"Created new athlete profile: {name} (ID: {profile_id})")
        return profile
    
    def _customize_for_athlete_type(self, profile: AthleteProfile, athlete_type: str) -> None:
        """
        Customize profile attributes based on athlete type.
        
        Args:
            profile: AthleteProfile instance to customize
            athlete_type: Type of athlete
        """
        if athlete_type == 'runner':
            # Runners typically have higher endurance and speed
            profile.endurance = min(profile.endurance * 1.2, 1.0)
            profile.speed = min(profile.speed * 1.1, 1.0)
            profile.vo2max += 5.0  # Higher VO2max
            profile.running_economy -= 10.0  # Better running economy (lower is better)
            
        elif athlete_type == 'cyclist':
            # Cyclists typically have higher endurance and strength
            profile.endurance = min(profile.endurance * 1.2, 1.0)
            profile.strength = min(profile.strength * 1.15, 1.0)
            profile.critical_power += 30.0  # Higher power output
            
        elif athlete_type == 'triathlete':
            # Triathletes are balanced
            profile.endurance = min(profile.endurance * 1.25, 1.0)
            profile.lactate_threshold += 0.05  # Higher lactate threshold
            
        elif athlete_type == 'military':
            # Military personnel typically have balanced attributes
            profile.strength = min(profile.strength * 1.1, 1.0)
            profile.endurance = min(profile.endurance * 1.1, 1.0)
            profile.fatigue_resistance = min(profile.fatigue_resistance * 1.2, 1.0)
            
            # Set military-specific attributes
            if not profile.military_rank:
                profile.military_rank = random.choice([
                    'Private', 'Corporal', 'Sergeant', 'Lieutenant', 'Captain'
                ])
            
            if not profile.years_of_service:
                profile.years_of_service = random.randint(1, 15)
                
            if not profile.specialty:
                profile.specialty = random.choice([
                    'Infantry', 'Artillery', 'Special Forces', 'Communications', 
                    'Engineering', 'Medical', 'Intelligence'
                ])
    
    def _save_profile(self, profile: AthleteProfile) -> bool:
        """
        Save an athlete profile to file.
        
        Args:
            profile: AthleteProfile instance to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            profile_path = self._get_profile_path(profile.id)
            
            with open(profile_path, 'w') as f:
                json.dump(profile.get_profile(), f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving profile {profile.id}: {str(e)}")
            return False
    
    def get_profile(self, profile_id: str) -> Optional[AthleteProfile]:
        """
        Retrieve an athlete profile by ID.
        
        Args:
            profile_id: Unique identifier for the profile
            
        Returns:
            AthleteProfile instance or None if not found
        """
        profile_path = self._get_profile_path(profile_id)
        
        if not os.path.exists(profile_path):
            logger.warning(f"Profile not found: {profile_id}")
            return None
        
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
                
            # Convert dictionary to AthleteProfile object
            return AthleteProfile(**profile_data)
        except Exception as e:
            logger.error(f"Error loading profile {profile_id}: {str(e)}")
            return None
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        List all available athlete profiles.
        
        Returns:
            List of profile dictionaries with basic info
        """
        profiles = []
        
        # Get all JSON files in the profiles directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                profile_id = filename.replace('.json', '')
                
                profile = self.get_profile(profile_id)
                if profile:
                    # Include only basic info in the list
                    profiles.append({
                        'id': profile.id,
                        'name': profile.name,
                        'athlete_type': profile.athlete_type,
                        'age': profile.age,
                        'gender': profile.gender,
                        'created_at': profile.created_at
                    })
        
        return profiles
    
    def update_profile(self, profile_id: str, 
                       attributes: Dict[str, Any]) -> Optional[AthleteProfile]:
        """
        Update an existing athlete profile.
        
        Args:
            profile_id: Unique identifier for the profile
            attributes: Dictionary of attributes to update
            
        Returns:
            Updated AthleteProfile instance or None if not found
        """
        profile = self.get_profile(profile_id)
        
        if not profile:
            logger.warning(f"Cannot update: Profile not found: {profile_id}")
            return None
        
        # Update the profile attributes
        profile.update_profile(attributes)
        
        # Save the updated profile
        if self._save_profile(profile):
            logger.info(f"Updated profile: {profile.id}")
            return profile
        
        return None
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete an athlete profile.
        
        Args:
            profile_id: ID of the profile to delete
            
        Returns:
            True if successful, False otherwise
        """
        profile_path = self._get_profile_path(profile_id)
        
        if not os.path.exists(profile_path):
            logger.warning(f"Profile not found: {profile_id}")
            return False
        
        try:
            os.remove(profile_path)
            logger.info(f"Deleted profile: {profile_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting profile {profile_id}: {str(e)}")
            return False
    
    def calculate_average_metrics(self, athlete_type: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate average metrics across all athlete profiles or specific athlete type.
        
        Args:
            athlete_type: Optional filter to only include specific athlete types
            
        Returns:
            Dictionary of average metrics
        """
        profiles = self.list_profiles()
        if athlete_type:
            profiles = [p for p in profiles if p.get('athlete_type') == athlete_type]
        
        if not profiles:
            return {}
        
        # Metrics to calculate
        metrics = [
            'age', 'height', 'weight', 'strength', 'endurance', 'speed', 'agility',
            'vo2max', 'lactate_threshold', 'critical_power', 'anaerobic_capacity'
        ]
        
        # Initialize sums
        sums = {metric: 0.0 for metric in metrics}
        counts = {metric: 0 for metric in metrics}
        
        # Calculate sums
        for profile in profiles:
            for metric in metrics:
                if metric in profile and profile[metric] is not None:
                    sums[metric] += float(profile[metric])
                    counts[metric] += 1
        
        # Calculate averages
        averages = {}
        for metric in metrics:
            if counts[metric] > 0:
                averages[metric] = sums[metric] / counts[metric]
        
        return averages
    
    def generate_random_profile(self, athlete_type: Optional[str] = None) -> AthleteProfile:
        """
        Generate a random athlete profile for testing.
        
        Args:
            athlete_type: Type of athlete (random if not specified)
            
        Returns:
            Random AthleteProfile instance
        """
        # Select random athlete type if not specified
        if not athlete_type:
            athlete_type = random.choice(self.athlete_types)
        
        # Generate random name
        first_names = [
            'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard',
            'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan'
        ]
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller',
            'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White'
        ]
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        # Generate random gender
        gender = random.choice(['Male', 'Female', 'Other'])
        
        # Generate random age within appropriate range (18-45)
        age = random.randint(18, 45)
        
        # Generate random height and weight based on gender
        if gender == 'Male':
            height = random.uniform(165.0, 195.0)  # cm
            weight = random.uniform(65.0, 95.0)    # kg
        else:
            height = random.uniform(155.0, 180.0)  # cm
            weight = random.uniform(50.0, 75.0)    # kg
        
        # Generate random performance attributes
        attributes = {
            'strength': random.uniform(0.4, 0.9),
            'endurance': random.uniform(0.4, 0.9),
            'speed': random.uniform(0.4, 0.9),
            'agility': random.uniform(0.4, 0.9),
            'vo2max': random.uniform(35.0, 65.0),
            'lactate_threshold': random.uniform(0.65, 0.85),
            'fatigue_resistance': random.uniform(0.4, 0.8),
            'recovery_rate': random.uniform(0.4, 0.8)
        }
        
        # Create the profile
        return self.create_profile(
            name=name,
            athlete_type=athlete_type,
            age=age,
            gender=gender,
            height=height,
            weight=weight,
            attributes=attributes
        ) 