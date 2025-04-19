#!/usr/bin/env python3
"""
Night Operations and Equipment Impact Demo
This script demonstrates how night operations and specific equipment loadouts 
affect military performance.
"""

from dt_project.physics.military import MilitarySimulation, EquipmentLoad, MovementType
from dt_project.physics.terrain import TerrainType

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def main():
    """Run the demonstration."""
    print_section("NIGHT OPERATIONS AND EQUIPMENT IMPACT DEMO")
    
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
        "fitness_level": 8.0,  # 1-10 scale
        "max_speed": 5.0,  # m/s (~18 km/h)
        "endurance": 0.8,  # 0-1 scale
        "biomechanics": {
            "stride_length": 1.75,  # meters
            "ground_contact_time": 0.24  # seconds
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
        "temperature": 18.0,  # Celsius
        "humidity": 70.0,     # Percentage
        "wind_speed": 5.0,    # km/h
        "wind_direction": 45  # Degrees (45 = Northeast)
    }
    
    # Compare day vs night operations
    print_section("DAY VS NIGHT OPERATIONS")
    
    # Day mission
    day_mission = military_sim.simulate_mission(
        soldier_profile,
        terrain_profile,
        environmental_conditions,
        equipment_load=EquipmentLoad.FIGHTING_LOAD,
        movement_type=MovementType.NORMAL,
        is_night=False
    )
    
    # Night mission without night vision
    night_mission = military_sim.simulate_mission(
        soldier_profile,
        terrain_profile,
        environmental_conditions,
        equipment_load=EquipmentLoad.FIGHTING_LOAD,
        movement_type=MovementType.NORMAL,
        is_night=True
    )
    
    # Display comparison table
    print("DAY VS NIGHT PERFORMANCE COMPARISON (FIGHTING LOAD)\n")
    print("Operation     | Time (min) | Avg Speed | Energy (kJ) | Final Fatigue | Operational Effectiveness")
    print("-" * 100)
    
    # Day mission stats
    day_data = day_mission[-1]
    day_time_min = day_data["time"] / 60
    day_speed_kmh = day_data["speed_kmh"]
    day_energy_kj = day_data["energy"]
    day_fatigue = day_data["fatigue"]
    day_effectiveness = day_data["operational_effectiveness"]
    
    print(f"Day Operation  | {day_time_min:9.2f} | {day_speed_kmh:9.2f} | {day_energy_kj:11.1f} | {day_fatigue:13.2f} | {day_effectiveness:25.2f}")
    
    # Night mission stats
    night_data = night_mission[-1]
    night_time_min = night_data["time"] / 60
    night_speed_kmh = night_data["speed_kmh"]
    night_energy_kj = night_data["energy"]
    night_fatigue = night_data["fatigue"]
    night_effectiveness = night_data["operational_effectiveness"]
    
    print(f"Night Operation| {night_time_min:9.2f} | {night_speed_kmh:9.2f} | {night_energy_kj:11.1f} | {night_fatigue:13.2f} | {night_effectiveness:25.2f}")
    
    # Calculate percentage differences
    time_diff = (night_time_min / day_time_min - 1) * 100
    speed_diff = (night_speed_kmh / day_speed_kmh - 1) * 100
    energy_diff = (night_energy_kj / day_energy_kj - 1) * 100
    effect_diff = (night_effectiveness / day_effectiveness - 1) * 100
    
    print(f"Difference (%) | {time_diff:+9.1f} | {speed_diff:+9.1f} | {energy_diff:+11.1f} | {'-':13} | {effect_diff:+25.1f}")
    
    # Custom equipment loadouts
    print_section("SPECIALIZED EQUIPMENT LOADOUTS")
    
    # Define equipment items
    standard_items = [
        {"id": 1, "name": "Tactical Vest", "weight": 4.5},
        {"id": 2, "name": "Helmet", "weight": 1.4},
        {"id": 3, "name": "Boots", "weight": 1.8},
        {"id": 4, "name": "Weapon", "weight": 3.6},
        {"id": 5, "name": "Water Supply", "weight": 2.0},
        {"id": 6, "name": "First Aid", "weight": 0.8}
    ]
    
    # Standard gear without night vision
    standard_gear_analysis = military_sim.calculate_equipment_impacts(
        soldier_profile["weight"], standard_items
    )
    
    # Standard gear with night vision
    with_night_vision = standard_items + [
        {"id": 7, "name": "Night Vision", "weight": 1.2}
    ]
    
    night_vision_analysis = military_sim.calculate_equipment_impacts(
        soldier_profile["weight"], with_night_vision
    )
    
    # Fully equipped (all gear)
    full_gear = with_night_vision + [
        {"id": 8, "name": "Communication", "weight": 1.5},
        {"id": 9, "name": "MRE Rations", "weight": 1.8}
    ]
    
    full_gear_analysis = military_sim.calculate_equipment_impacts(
        soldier_profile["weight"], full_gear
    )
    
    # Display equipment comparisons
    print("EQUIPMENT LOADOUT COMPARISON\n")
    print("Loadout               | Weight | % Body | Speed Factor | Endurance Factor | Load Type")
    print("-" * 90)
    
    print(f"Standard Gear         | {standard_gear_analysis['equipment_weight']:6.1f} | {standard_gear_analysis['weight_percentage']:6.1f} | {standard_gear_analysis['performance_impacts']['speed_factor']:12.2f} | {standard_gear_analysis['performance_impacts']['endurance_factor']:16.2f} | {standard_gear_analysis['load_classification'].replace('_', ' ').title()}")
    print(f"With Night Vision     | {night_vision_analysis['equipment_weight']:6.1f} | {night_vision_analysis['weight_percentage']:6.1f} | {night_vision_analysis['performance_impacts']['speed_factor']:12.2f} | {night_vision_analysis['performance_impacts']['endurance_factor']:16.2f} | {night_vision_analysis['load_classification'].replace('_', ' ').title()}")
    print(f"Full Combat Load      | {full_gear_analysis['equipment_weight']:6.1f} | {full_gear_analysis['weight_percentage']:6.1f} | {full_gear_analysis['performance_impacts']['speed_factor']:12.2f} | {full_gear_analysis['performance_impacts']['endurance_factor']:16.2f} | {full_gear_analysis['load_classification'].replace('_', ' ').title()}")
    
    # Night mission with night vision equipment
    print_section("NIGHT OPERATIONS WITH SPECIALIZED EQUIPMENT")
    
    # Simulate night mission with and without night vision
    # For demonstration, we'll use the equipment weight but add a "benefit factor" for night vision equipment
    
    # Night mission with standard gear (no night vision)
    night_standard = military_sim.simulate_mission(
        soldier_profile,
        terrain_profile,
        environmental_conditions,
        equipment_load=standard_gear_analysis['load_classification'],
        movement_type=MovementType.NORMAL,
        is_night=True
    )
    
    # Night mission with night vision
    # In a more complex simulation, night vision would provide direct benefits
    # Here we'll simulate it by using the same weight but reducing "is_night" penalties
    # (this is a simplified approach for demonstration)
    night_with_nv_data = []
    
    # Get the base movement calculation
    for i, terrain_point in enumerate(terrain_profile):
        # Apply a "night vision benefit" by manually adjusting the calculations
        # This simulates having night vision equipment partially offsetting night penalties
        # In a real implementation, this would be handled within the simulation model
        
        # Calculate with partial night penalty (50% of normal night penalty)
        # This is a mock-up just for demonstration purposes
        movement_data = military_sim.calculate_military_movement(
            soldier_profile,
            terrain_point,
            environmental_conditions,
            0 if i == 0 else night_with_nv_data[-1]["fatigue"],
            equipment_load=night_vision_analysis['load_classification'],
            movement_type=MovementType.NORMAL,
            is_night=True  # Still night, but we'll adjust the effectiveness below
        )
        
        # For demonstration, assume night vision improves operational effectiveness by 20%
        # A proper implementation would handle this within the model
        if i > 0:
            prev_point = terrain_profile[i-1]
            segment_distance = terrain_point["distance"] - prev_point["distance"]
            segment_time = segment_distance / movement_data["speed"] if movement_data["speed"] > 0 else 0
            
            night_with_nv_data.append({
                "index": i,
                "distance": night_with_nv_data[-1]["distance"] + segment_distance if i > 0 else 0,
                "time": night_with_nv_data[-1]["time"] + segment_time if i > 0 else 0, 
                "speed": movement_data["speed"] * 1.2,  # 20% speed boost with night vision
                "speed_kmh": movement_data["speed_kmh"] * 1.2,
                "energy": movement_data["energy_rate"] * segment_time + (night_with_nv_data[-1]["energy"] if i > 0 else 0),
                "fatigue": min(1.0, movement_data["fatigue_increase"] * segment_time + (night_with_nv_data[-1]["fatigue"] if i > 0 else 0)),
                "operational_effectiveness": min(0.8, movement_data["tactical_adjustments"]["night_factor"] * 1.25)  # Improved effectiveness with night vision
            })
        else:
            night_with_nv_data.append({
                "index": i,
                "distance": 0,
                "time": 0,
                "speed": movement_data["speed"] * 1.2,
                "speed_kmh": movement_data["speed_kmh"] * 1.2,
                "energy": 0,
                "fatigue": 0,
                "operational_effectiveness": min(0.8, movement_data["tactical_adjustments"]["night_factor"] * 1.25)
            })
    
    # Display comparison table
    print("EFFECT OF NIGHT VISION EQUIPMENT ON NIGHT OPERATIONS\n")
    print("Equipment Config | Time (min) | Avg Speed | Energy (kJ) | Final Fatigue | Operational Effectiveness")
    print("-" * 100)
    
    # Night without night vision stats
    std_data = night_standard[-1]
    std_time_min = std_data["time"] / 60
    std_speed_kmh = std_data["speed_kmh"]
    std_energy_kj = std_data["energy"]
    std_fatigue = std_data["fatigue"]
    std_effectiveness = std_data["operational_effectiveness"]
    
    print(f"No Night Vision   | {std_time_min:9.2f} | {std_speed_kmh:9.2f} | {std_energy_kj:11.1f} | {std_fatigue:13.2f} | {std_effectiveness:25.2f}")
    
    # Night with night vision stats
    nv_data = night_with_nv_data[-1]
    nv_time_min = nv_data["time"] / 60
    nv_speed_kmh = nv_data["speed_kmh"]
    nv_energy_kj = nv_data["energy"]
    nv_fatigue = nv_data["fatigue"]
    nv_effectiveness = nv_data["operational_effectiveness"]
    
    print(f"With Night Vision | {nv_time_min:9.2f} | {nv_speed_kmh:9.2f} | {nv_energy_kj:11.1f} | {nv_fatigue:13.2f} | {nv_effectiveness:25.2f}")
    
    # Calculate percentage improvements
    time_improve = ((std_time_min / nv_time_min) - 1) * 100
    speed_improve = ((nv_speed_kmh / std_speed_kmh) - 1) * 100
    effect_improve = ((nv_effectiveness / std_effectiveness) - 1) * 100
    
    print(f"Improvement (%)  | {time_improve:+9.1f} | {speed_improve:+9.1f} | {'-':11} | {'-':13} | {effect_improve:+25.1f}")

if __name__ == "__main__":
    main() 