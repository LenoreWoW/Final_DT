#!/usr/bin/env python3
"""
Demo script to showcase AthleteManager functionality including average metrics calculation.
"""

import os
import sys
import random
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dt_project.data_acquisition.athlete import AthleteManager, AthleteProfile


def generate_test_profiles(manager: AthleteManager, count: int = 20) -> None:
    """Generate a set of test athlete profiles"""
    athlete_types = ['runner', 'cyclist', 'swimmer', 'triathlete']
    
    print(f"Generating {count} random athlete profiles...")
    for i in range(count):
        athlete_type = random.choice(athlete_types)
        profile = manager.generate_random_profile(athlete_type)
        manager.update_profile(profile)
    
    print(f"Generated {count} profiles successfully.")


def display_average_metrics(metrics: Dict[str, float]) -> None:
    """Display average metrics in a formatted way"""
    if not metrics:
        print("No metrics available.")
        return
    
    print("\nAverage Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
    print("-" * 50)


def plot_metrics_by_athlete_type(manager: AthleteManager) -> None:
    """Create a comparison plot of metrics across different athlete types"""
    athlete_types = ['runner', 'cyclist', 'swimmer', 'triathlete']
    metrics_to_plot = ['strength', 'endurance', 'speed', 'agility', 'vo2max']
    
    # Collect data for each athlete type
    data = {}
    for athlete_type in athlete_types:
        metrics = manager.calculate_average_metrics(athlete_type)
        if metrics:  # Only include if we have data
            data[athlete_type] = metrics
    
    if not data:
        print("No data available for plotting.")
        return
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Width of each bar
    bar_width = 0.2
    
    # Position on x-axis
    positions = np.arange(len(metrics_to_plot))
    
    # Plot bars for each athlete type
    for i, (athlete_type, metrics) in enumerate(data.items()):
        values = [metrics.get(metric, 0) for metric in metrics_to_plot]
        ax.bar(positions + (i * bar_width), values, bar_width, label=athlete_type.title())
    
    # Add labels, title and legend
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Average Value')
    ax.set_title('Average Metrics by Athlete Type')
    ax.set_xticks(positions + bar_width * (len(data) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('athlete_metrics_comparison.png')
    print("Plot saved as 'athlete_metrics_comparison.png'")


def main():
    """Main function to demonstrate AthleteManager functionality"""
    # Create an AthleteManager instance with a custom data directory
    data_dir = "data/test_profiles"
    manager = AthleteManager(data_dir=data_dir)
    
    # Delete existing profiles if needed and generate new ones
    profiles = manager.list_profiles()
    if not profiles:
        generate_test_profiles(manager)
    else:
        print(f"Found {len(profiles)} existing profiles.")
    
    # Calculate and display overall average metrics
    print("\n=== Overall Average Metrics ===")
    avg_metrics = manager.calculate_average_metrics()
    display_average_metrics(avg_metrics)
    
    # Calculate and display average metrics by athlete type
    athlete_types = ['runner', 'cyclist', 'swimmer', 'triathlete']
    for athlete_type in athlete_types:
        print(f"\n=== Average Metrics for {athlete_type.title()}s ===")
        type_metrics = manager.calculate_average_metrics(athlete_type)
        display_average_metrics(type_metrics)
    
    # Create a comparison plot
    plot_metrics_by_athlete_type(manager)


if __name__ == "__main__":
    main() 