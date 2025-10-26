"""
Military Simulation Module
Handles military-specific simulation extensions, including tactical movement
and equipment load effects.
"""

import math
import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from dt_project.config import ConfigManager
from dt_project.physics.terrain import TerrainType
from dt_project.physics.biomechanics import BiomechanicalModel

logger = logging.getLogger(__name__)

# Equipment load classification
class EquipmentLoad:
    """Military equipment load classification."""
    FIGHTING_LOAD = "fighting_load"  # Basic combat load (~20-30kg)
    APPROACH_LOAD = "approach_load"  # Sustained operations load (~30-40kg)
    EMERGENCY_LOAD = "emergency_load"  # Heavy mission load (~40-55kg)
    
    @staticmethod
    def get_weight(load_type: str) -> float:
        """Get the weight in kg for a load type."""
        weights = {
            EquipmentLoad.FIGHTING_LOAD: 25.0,
            EquipmentLoad.APPROACH_LOAD: 35.0,
            EquipmentLoad.EMERGENCY_LOAD: 45.0
        }
        return weights.get(load_type, 25.0)
    
    @staticmethod
    def get_speed_factor(load_type: str) -> float:
        """Get the speed adjustment factor for a load type."""
        factors = {
            EquipmentLoad.FIGHTING_LOAD: 0.8,
            EquipmentLoad.APPROACH_LOAD: 0.7,
            EquipmentLoad.EMERGENCY_LOAD: 0.6
        }
        return factors.get(load_type, 0.8)
    
    @staticmethod
    def get_endurance_factor(load_type: str) -> float:
        """Get the endurance adjustment factor for a load type."""
        factors = {
            EquipmentLoad.FIGHTING_LOAD: 0.85,
            EquipmentLoad.APPROACH_LOAD: 0.7,
            EquipmentLoad.EMERGENCY_LOAD: 0.55
        }
        return factors.get(load_type, 0.85)

# Tactical movement types
class MovementType:
    """Military tactical movement types."""
    NORMAL = "normal"  # Standard marching
    RUSH = "rush"  # Rapid movement (burst)
    PATROL = "patrol"  # Vigilant movement
    STEALTH = "stealth"  # Concealed movement
    
    @staticmethod
    def get_speed_factor(movement_type: str) -> float:
        """Get the speed adjustment factor for a movement type."""
        factors = {
            MovementType.NORMAL: 1.0,
            MovementType.RUSH: 1.5,
            MovementType.PATROL: 0.8,
            MovementType.STEALTH: 0.5
        }
        return factors.get(movement_type, 1.0)
    
    @staticmethod
    def get_fatigue_factor(movement_type: str) -> float:
        """Get the fatigue adjustment factor for a movement type."""
        factors = {
            MovementType.NORMAL: 1.0,
            MovementType.RUSH: 2.0,
            MovementType.PATROL: 1.2,
            MovementType.STEALTH: 1.4
        }
        return factors.get(movement_type, 1.0)

class MilitarySimulation:
    """Military-specific simulation extensions."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the military simulation.
        
        Args:
            config: Configuration manager. If None, creates a new one.
        """
        self.config = config or ConfigManager()
        
        # Load configuration
        self._load_config()
        
        # Initialize biomechanical model for baseline calculations
        self.biomech_model = BiomechanicalModel(config)
        
    def _load_config(self) -> None:
        """Load configuration parameters for the simulation."""
        # Currently no specific military config parameters
        pass
    
    def calculate_military_movement(self, soldier_profile: Dict[str, Any],
                                 terrain_point: Dict[str, Any],
                                 environmental_conditions: Dict[str, Any],
                                 current_fatigue: float,
                                 equipment_load: str = EquipmentLoad.FIGHTING_LOAD,
                                 movement_type: str = MovementType.NORMAL,
                                 is_night: bool = False) -> Dict[str, Any]:
        """
        Calculate military movement metrics based on terrain, environment, and load.
        
        Args:
            soldier_profile: Soldier profile with physical attributes
            terrain_point: Terrain information
            environmental_conditions: Environmental conditions
            current_fatigue: Current fatigue level (0-1)
            equipment_load: Type of equipment load
            movement_type: Type of tactical movement
            is_night: Whether the movement is at night
            
        Returns:
            Dictionary with movement metrics
        """
        # Create a modified profile that accounts for equipment load
        modified_profile = self._adjust_profile_for_equipment(soldier_profile, equipment_load)
        
        # Get base speed calculation from biomechanical model
        base_metrics = self.biomech_model.calculate_speed(
            modified_profile, terrain_point, environmental_conditions, current_fatigue)
        
        # Adjust for tactical movement type
        movement_speed_factor = MovementType.get_speed_factor(movement_type)
        movement_fatigue_factor = MovementType.get_fatigue_factor(movement_type)
        
        # Night adjustment (reduced speed at night)
        night_factor = 0.7 if is_night else 1.0
        
        # Calculate adjusted metrics
        speed = base_metrics["speed"] * movement_speed_factor * night_factor
        speed_kmh = speed * 3.6
        
        # Adjust fatigue increase for tactical movement
        fatigue_increase = base_metrics["fatigue_increase"] * movement_fatigue_factor
        
        # Adjust energy expenditure for tactical movement and load
        energy_rate = base_metrics["energy_rate"] * (1.0 + 0.2 * movement_fatigue_factor)
        
        # Create military-specific result
        result = {
            "speed": round(speed, 2),  # m/s
            "speed_kmh": round(speed_kmh, 2),  # km/h
            "energy_rate": round(energy_rate, 2),  # joules/second
            "fatigue_increase": round(fatigue_increase, 4),  # fatigue units/second
            "equipment_load": equipment_load,
            "equipment_weight": round(EquipmentLoad.get_weight(equipment_load), 1),  # kg
            "movement_type": movement_type,
            "stride_length": round(base_metrics["stride_length"] * movement_speed_factor, 2),  # meters
            "is_night": is_night,
            "tactical_adjustments": {
                "load_factor": round(EquipmentLoad.get_speed_factor(equipment_load), 2),
                "movement_factor": round(movement_speed_factor, 2),
                "night_factor": round(night_factor, 2)
            }
        }
        
        return result
    
    def simulate_mission(self, soldier_profile: Dict[str, Any],
                      terrain_profile: List[Dict[str, Any]],
                      environmental_conditions: Dict[str, Any],
                      equipment_load: str = EquipmentLoad.FIGHTING_LOAD,
                      movement_type: str = MovementType.NORMAL,
                      is_night: bool = False) -> List[Dict[str, Any]]:
        """
        Simulate a military mission across a terrain profile.
        
        Args:
            soldier_profile: Soldier profile
            terrain_profile: List of terrain points
            environmental_conditions: Environmental conditions
            equipment_load: Type of equipment load
            movement_type: Type of tactical movement
            is_night: Whether the mission is at night
            
        Returns:
            List of mission performance data points
        """
        mission_data = []
        current_fatigue = 0.0
        total_distance = 0.0
        total_time = 0.0
        total_energy = 0.0
        
        # Get equipment weight
        equipment_weight = EquipmentLoad.get_weight(equipment_load)
        total_weight = soldier_profile.get("weight", 70.0) + equipment_weight
        
        for i, terrain_point in enumerate(terrain_profile):
            # Calculate movement metrics at this point
            movement_data = self.calculate_military_movement(
                soldier_profile,
                terrain_point,
                environmental_conditions,
                current_fatigue,
                equipment_load,
                movement_type,
                is_night
            )
            
            # Extract values
            speed = movement_data["speed"]  # m/s
            energy_rate = movement_data["energy_rate"]  # joules/second
            fatigue_increase = movement_data["fatigue_increase"]
            
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
                fatigue_change = fatigue_increase * segment_time
                current_fatigue = min(1.0, max(0.0, current_fatigue + fatigue_change))
            
            # Calculate operational effectiveness based on fatigue
            # (decreases as fatigue increases, especially affecting cognitive function)
            operational_effectiveness = self._calculate_operational_effectiveness(
                current_fatigue, equipment_load, is_night)
            
            # Calculate time to max fatigue at current pace
            time_to_exhaustion = 0.0
            if fatigue_increase > 0:
                time_to_exhaustion = (1.0 - current_fatigue) / fatigue_increase
            
            # Create mission data point
            mission_point = {
                "index": i,
                "distance": round(total_distance, 2),  # meters
                "time": round(total_time, 2),  # seconds
                "time_hours": round(total_time / 3600, 2),  # hours
                "speed": round(speed, 2),  # m/s
                "speed_kmh": round(speed * 3.6, 2),  # km/h
                "energy": round(total_energy / 1000, 2),  # kilojoules
                "fatigue": round(current_fatigue, 2),  # 0-1 scale
                "operational_effectiveness": round(operational_effectiveness, 2),  # 0-1 scale
                "time_to_exhaustion_hours": round(time_to_exhaustion / 3600, 2) if time_to_exhaustion > 0 else float('inf'),
                "altitude": terrain_point["altitude"],
                "gradient": terrain_point["gradient"],
                "terrain_type": terrain_point["terrain_type"],
                "equipment_load": equipment_load,
                "movement_type": movement_type,
                "is_night": is_night
            }
            
            mission_data.append(mission_point)
        
        # Add pace and mission completion data
        if mission_data:
            last_point = mission_data[-1]
            total_distance = last_point["distance"]
            total_time = last_point["time"]
            
            # Add summary data to each point
            for point in mission_data:
                point["percent_complete"] = round(100 * point["distance"] / total_distance, 1)
                
                # Pace in min/km
                if point["time"] > 0:
                    avg_speed = point["distance"] / point["time"]
                    pace_mins_per_km = (1000 / avg_speed) / 60 if avg_speed > 0 else 0
                    point["pace_mins_per_km"] = round(pace_mins_per_km, 2)
                else:
                    point["pace_mins_per_km"] = 0
        
        return mission_data
    
    def calculate_equipment_impacts(self, base_weight_kg: float, 
                                 equipment_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate impacts of specific equipment items on movement and performance.
        
        Args:
            base_weight_kg: Base weight of the soldier without equipment
            equipment_items: List of equipment items with weight and properties
            
        Returns:
            Dictionary with equipment impact metrics
        """
        # Calculate total weight
        total_weight = base_weight_kg
        item_weights = {}
        
        for item in equipment_items:
            item_name = item.get("name", "Unknown item")
            item_weight = item.get("weight", 0.0)
            total_weight += item_weight
            item_weights[item_name] = item_weight
        
        # Determine load classification
        equipment_weight = total_weight - base_weight_kg
        load_classification = EquipmentLoad.FIGHTING_LOAD
        
        if equipment_weight > 40:
            load_classification = EquipmentLoad.EMERGENCY_LOAD
        elif equipment_weight > 30:
            load_classification = EquipmentLoad.APPROACH_LOAD
        
        # Calculate weight distribution
        weight_distribution = {}
        for item_name, weight in item_weights.items():
            weight_distribution[item_name] = round(100 * weight / equipment_weight, 1)
        
        # Calculate weight as percentage of body weight
        weight_percentage = round(100 * equipment_weight / base_weight_kg, 1)
        
        # Recommended max is typically 30% of body weight
        is_overloaded = weight_percentage > 30
        
        # Calculate impacts
        speed_impact = EquipmentLoad.get_speed_factor(load_classification)
        endurance_impact = EquipmentLoad.get_endurance_factor(load_classification)
        
        # Calculate metabolic cost increase (approximately 1.5% per kg of load)
        metabolic_increase = round(equipment_weight * 1.5, 1)
        
        return {
            "total_weight": round(total_weight, 1),
            "equipment_weight": round(equipment_weight, 1),
            "weight_percentage": weight_percentage,
            "load_classification": load_classification,
            "is_overloaded": is_overloaded,
            "weight_distribution": weight_distribution,
            "performance_impacts": {
                "speed_factor": round(speed_impact, 2),
                "endurance_factor": round(endurance_impact, 2),
                "metabolic_increase_percent": metabolic_increase
            },
            "recommendations": self._generate_load_recommendations(weight_percentage, equipment_items)
        }
    
    def _adjust_profile_for_equipment(self, soldier_profile: Dict[str, Any], 
                                  equipment_load: str) -> Dict[str, Any]:
        """
        Adjust a soldier profile to account for equipment load.
        
        Args:
            soldier_profile: Original soldier profile
            equipment_load: Equipment load type
            
        Returns:
            Adjusted soldier profile
        """
        # Create a copy of the profile
        adjusted_profile = soldier_profile.copy()
        
        # Get adjustment factors for the equipment load
        speed_factor = EquipmentLoad.get_speed_factor(equipment_load)
        endurance_factor = EquipmentLoad.get_endurance_factor(equipment_load)
        
        # Get equipment weight
        equipment_weight = EquipmentLoad.get_weight(equipment_load)
        
        # Adjust profile attributes
        if "max_speed" in adjusted_profile:
            adjusted_profile["max_speed"] = adjusted_profile["max_speed"] * speed_factor
            
        if "endurance" in adjusted_profile:
            adjusted_profile["endurance"] = adjusted_profile["endurance"] * endurance_factor
            
        # Adjust weight to include equipment
        adjusted_profile["total_weight"] = adjusted_profile.get("weight", 70.0) + equipment_weight
        
        # Adjust biomechanics
        if "biomechanics" in adjusted_profile:
            biomech = adjusted_profile["biomechanics"].copy()
            
            # Equipment load reduces stride length and increases ground contact time
            if "stride_length" in biomech:
                biomech["stride_length"] = biomech["stride_length"] * (0.9 + 0.1 * speed_factor)
                
            if "ground_contact_time" in biomech:
                biomech["ground_contact_time"] = biomech["ground_contact_time"] * (1.0 + 0.2 * (1.0 - speed_factor))
                
            adjusted_profile["biomechanics"] = biomech
        
        return adjusted_profile
    
    def _calculate_operational_effectiveness(self, fatigue: float, 
                                        equipment_load: str, 
                                        is_night: bool) -> float:
        """
        Calculate operational effectiveness based on fatigue and conditions.
        
        Args:
            fatigue: Current fatigue level (0-1)
            equipment_load: Equipment load type
            is_night: Whether operation is at night
            
        Returns:
            Operational effectiveness (0-1)
        """
        # Base effectiveness starts at 1.0 (100%)
        effectiveness = 1.0
        
        # Fatigue has a significant impact on effectiveness
        # Cognitive functions deteriorate more rapidly with fatigue
        fatigue_impact = fatigue ** 1.5  # Non-linear relationship
        effectiveness -= fatigue_impact * 0.7
        
        # Equipment load reduces effectiveness
        load_factor = 1.0 - (1.0 - EquipmentLoad.get_speed_factor(equipment_load)) * 0.5
        effectiveness *= load_factor
        
        # Night operations are more challenging
        if is_night:
            effectiveness *= 0.8
        
        # Ensure effectiveness is between 0 and 1
        return max(0.0, min(1.0, effectiveness))
    
    def _generate_load_recommendations(self, weight_percentage: float,
                                   equipment_items: List[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations for load adjustment.
        
        Args:
            weight_percentage: Equipment weight as percentage of body weight
            equipment_items: List of equipment items
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if weight_percentage > 45:
            recommendations.append("CRITICAL: Load exceeds emergency threshold. Reduce weight immediately.")
        elif weight_percentage > 35:
            recommendations.append("WARNING: Load approaches emergency threshold. Consider redistribution.")
        elif weight_percentage > 25:
            recommendations.append("NOTICE: Load is within approach march limits.")
        else:
            recommendations.append("OPTIMAL: Load is within fighting load limits.")
        
        # Sort items by weight to identify heaviest items
        sorted_items = sorted(equipment_items, key=lambda x: x.get("weight", 0), reverse=True)
        
        if weight_percentage > 30 and len(sorted_items) > 0:
            heaviest = sorted_items[0]
            recommendations.append(
                f"Consider redistributing or reducing {heaviest.get('name', 'heaviest item')} "
                f"({heaviest.get('weight', 0)} kg)."
            )
        
        return recommendations 