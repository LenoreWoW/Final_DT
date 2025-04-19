#!/usr/bin/env python3
"""
Military Equipment Simulation Demo
This script demonstrates the equipment impact calculation functionality
from the military simulation module.
"""

import json
from dt_project.physics.military import MilitarySimulation, EquipmentLoad

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def main():
    """Run the demonstration."""
    print_section("MILITARY EQUIPMENT SIMULATION DEMO")
    
    # Initialize military simulation
    military_sim = MilitarySimulation()
    
    # Define soldier base weight (kg)
    soldier_weight = 70.0
    print(f"Soldier base weight: {soldier_weight} kg\n")
    
    # Define standard equipment items
    equipment_items = [
        {"id": 1, "name": "Tactical Vest", "weight": 4.5, "essential": True},
        {"id": 2, "name": "Helmet", "weight": 1.4, "essential": True},
        {"id": 3, "name": "Boots", "weight": 1.8, "essential": True},
        {"id": 4, "name": "Weapon", "weight": 3.6, "essential": True},
        {"id": 5, "name": "Water Supply", "weight": 2.0, "essential": False},
        {"id": 6, "name": "First Aid", "weight": 0.8, "essential": True},
        {"id": 7, "name": "Night Vision", "weight": 1.2, "essential": False},
        {"id": 8, "name": "Communication", "weight": 1.5, "essential": False},
        {"id": 9, "name": "MRE Rations", "weight": 1.8, "essential": False}
    ]
    
    # Calculate equipment impacts
    equipment_analysis = military_sim.calculate_equipment_impacts(soldier_weight, equipment_items)
    
    # Display results in a readable format
    print_section("EQUIPMENT ANALYSIS")
    print(f"Total equipment weight: {equipment_analysis['equipment_weight']} kg")
    print(f"Total weight with soldier: {equipment_analysis['total_weight']} kg")
    print(f"Equipment as percentage of body weight: {equipment_analysis['weight_percentage']}%")
    print(f"Load classification: {equipment_analysis['load_classification']}")
    print(f"Overloaded: {equipment_analysis['is_overloaded']}")
    
    print_section("WEIGHT DISTRIBUTION")
    for item, percentage in equipment_analysis['weight_distribution'].items():
        print(f"{item}: {percentage}%")
    
    print_section("PERFORMANCE IMPACTS")
    impacts = equipment_analysis['performance_impacts']
    print(f"Speed factor: {impacts['speed_factor']} (lower is slower)")
    print(f"Endurance factor: {impacts['endurance_factor']} (lower is less endurance)")
    print(f"Metabolic increase: {impacts['metabolic_increase_percent']}% (higher energy expenditure)")
    
    print_section("RECOMMENDATIONS")
    for recommendation in equipment_analysis['recommendations']:
        print(f"- {recommendation}")
    
    # Compare different load configurations
    print_section("LOAD COMPARISON")
    
    # Fighting load (essential items only)
    essential_items = [item for item in equipment_items if item["essential"]]
    fighting_analysis = military_sim.calculate_equipment_impacts(soldier_weight, essential_items)
    
    # Approach load (add some non-essential items)
    approach_items = essential_items + [equipment_items[4], equipment_items[7]]  # Water + Communication
    approach_analysis = military_sim.calculate_equipment_impacts(soldier_weight, approach_items)
    
    # Emergency load (all items)
    emergency_items = equipment_items
    emergency_analysis = military_sim.calculate_equipment_impacts(soldier_weight, emergency_items)
    
    print("Load Type       | Weight | % Body | Speed Factor | Endurance Factor")
    print("-" * 65)
    print(f"Fighting Load   | {fighting_analysis['equipment_weight']:6.1f} | {fighting_analysis['weight_percentage']:6.1f} | {fighting_analysis['performance_impacts']['speed_factor']:12.2f} | {fighting_analysis['performance_impacts']['endurance_factor']:15.2f}")
    print(f"Approach Load   | {approach_analysis['equipment_weight']:6.1f} | {approach_analysis['weight_percentage']:6.1f} | {approach_analysis['performance_impacts']['speed_factor']:12.2f} | {approach_analysis['performance_impacts']['endurance_factor']:15.2f}")
    print(f"Emergency Load  | {emergency_analysis['equipment_weight']:6.1f} | {emergency_analysis['weight_percentage']:6.1f} | {emergency_analysis['performance_impacts']['speed_factor']:12.2f} | {emergency_analysis['performance_impacts']['endurance_factor']:15.2f}")

if __name__ == "__main__":
    main() 