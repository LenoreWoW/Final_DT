#!/usr/bin/env python3
"""
Military Mission Simulation Demo
This script demonstrates the impact of different equipment loads on military mission performance.
"""

import json
from dt_project.physics.military import MilitarySimulation, EquipmentLoad, MovementType
from dt_project.physics.terrain import TerrainType

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def main():
    """Run the demonstration."""
    print_section("MILITARY MISSION SIMULATION DEMO")
    
    # Initialize military simulation
    military_sim = MilitarySimulation()
    
    # Define a simple soldier profile
    soldier_profile = {
        "id": "soldier1",
        "name": "John Doe",
        "age": 28,
        "gender": "M",
        "height": 180.0,  # cm
        "weight": 75.0,   # kg
        "fitness_level": 8.5,  # 1-10 scale
        "max_speed": 5.2,  # m/s (~18.7 km/h)
        "endurance": 0.85,  # 0-1 scale
        "biomechanics": {
            "stride_length": 1.8,  # meters
            "ground_contact_time": 0.22  # seconds
        }
    }
    
    # Define a simple terrain profile (1km with varied terrain)
    terrain_profile = [
        {"distance": 0, "altitude": 100, "gradient": 0.0, "terrain_type": TerrainType.ROAD},
        {"distance": 200, "altitude": 110, "gradient": 0.05, "terrain_type": TerrainType.ROAD},
        {"distance": 400, "altitude": 130, "gradient": 0.1, "terrain_type": TerrainType.TRAIL},
        {"distance": 600, "altitude": 125, "gradient": -0.025, "terrain_type": TerrainType.TRAIL},
        {"distance": 800, "altitude": 115, "gradient": -0.05, "terrain_type": TerrainType.GRASS},
        {"distance": 1000, "altitude": 105, "gradient": -0.05, "terrain_type": TerrainType.GRASS}
    ]
    
    # Define environmental conditions
    environmental_conditions = {
        "temperature": 22.0,  # Celsius
        "humidity": 60.0,     # Percentage
        "wind_speed": 8.0,    # km/h
        "wind_direction": 90  # Degrees (90 = East)
    }
    
    # Run simulation with different equipment loads
    print_section("MISSION PERFORMANCE COMPARISON")
    
    # Equipment load types
    load_types = [
        EquipmentLoad.FIGHTING_LOAD,
        EquipmentLoad.APPROACH_LOAD,
        EquipmentLoad.EMERGENCY_LOAD
    ]
    
    results = {}
    
    for load_type in load_types:
        # Simulate mission with this load type
        mission_data = military_sim.simulate_mission(
            soldier_profile,
            terrain_profile,
            environmental_conditions,
            equipment_load=load_type,
            movement_type=MovementType.NORMAL,
            is_night=False
        )
        
        # Store the last data point which has the mission summary
        results[load_type] = mission_data[-1]
    
    # Display comparison table
    print("EQUIPMENT LOAD IMPACT ON 1KM MISSION\n")
    print("Load Type       | Weight | Time (min) | Avg Speed | Energy (kJ) | Final Fatigue")
    print("-" * 80)
    
    for load_type, data in results.items():
        load_weight = EquipmentLoad.get_weight(load_type)
        mission_time_min = data["time"] / 60
        avg_speed_kmh = data["speed_kmh"]
        energy_kj = data["energy"]
        fatigue = data["fatigue"]
        
        print(f"{load_type.replace('_', ' ').title():15} | {load_weight:6.1f} | {mission_time_min:9.2f} | {avg_speed_kmh:9.2f} | {energy_kj:11.1f} | {fatigue:13.2f}")
    
    # Show performance metrics over the mission path for one load type
    detailed_load = EquipmentLoad.APPROACH_LOAD
    detailed_mission = military_sim.simulate_mission(
        soldier_profile,
        terrain_profile,
        environmental_conditions,
        equipment_load=detailed_load,
        movement_type=MovementType.NORMAL,
        is_night=False
    )
    
    print_section(f"DETAILED METRICS FOR {detailed_load.replace('_', ' ').upper()}")
    print("Distance (m) | Altitude | Terrain   | Speed (km/h) | Fatigue | Effect. | Energy")
    print("-" * 80)
    
    for point in detailed_mission:
        print(f"{point['distance']:12.0f} | {terrain_profile[point['index']]['altitude']:8.0f} | {terrain_profile[point['index']]['terrain_type']:9} | {point['speed_kmh']:12.2f} | {point['fatigue']:7.2f} | {point['operational_effectiveness']:7.2f} | {point['energy']:6.1f}")
    
    # Movement type comparison (keeping the same equipment load)
    print_section("MOVEMENT TYPE COMPARISON")
    
    movement_types = [
        MovementType.NORMAL,
        MovementType.RUSH,
        MovementType.PATROL,
        MovementType.STEALTH
    ]
    
    movement_results = {}
    
    for move_type in movement_types:
        # Simulate mission with this movement type
        mission_data = military_sim.simulate_mission(
            soldier_profile,
            terrain_profile,
            environmental_conditions,
            equipment_load=EquipmentLoad.FIGHTING_LOAD,
            movement_type=move_type,
            is_night=False
        )
        
        # Store the last data point
        movement_results[move_type] = mission_data[-1]
    
    # Display comparison table
    print("MOVEMENT TYPE IMPACT WITH FIGHTING LOAD\n")
    print("Movement Type   | Time (min) | Avg Speed | Energy (kJ) | Final Fatigue")
    print("-" * 75)
    
    for move_type, data in movement_results.items():
        mission_time_min = data["time"] / 60
        avg_speed_kmh = data["speed_kmh"]
        energy_kj = data["energy"]
        fatigue = data["fatigue"]
        
        print(f"{move_type.replace('_', ' ').title():15} | {mission_time_min:9.2f} | {avg_speed_kmh:9.2f} | {energy_kj:11.1f} | {fatigue:13.2f}")

if __name__ == "__main__":
    main() 