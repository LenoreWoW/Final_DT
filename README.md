# Athlete Data Tracking System

A Python-based system for managing and analyzing athlete data profiles.

## Features

- Create, read, update, and delete athlete profiles
- Generate random athlete profiles for testing
- Calculate average metrics across all athletes or by athlete type
- Visualize athlete performance metrics with matplotlib

## Project Structure

```
dt_project/
├── data_acquisition/     # Core functionality for data handling
│   ├── athlete.py        # AthleteManager and AthleteProfile classes
│   └── ...
├── examples/             # Example scripts showing usage
│   ├── athlete_stats_demo.py  # Demo for average metrics calculation
│   └── ...
└── ...
```

## Usage Example

```python
# Import the necessary classes
from dt_project.data_acquisition.athlete import AthleteManager, AthleteProfile

# Create an AthleteManager instance
manager = AthleteManager(data_dir="data/profiles")

# Create or update an athlete profile
profile = AthleteProfile(
    name="John Doe",
    age=28,
    height=185,
    weight=75,
    athlete_type="runner",
    metrics={
        "strength": 7.5,
        "endurance": 8.9,
        "speed": 8.2
    }
)
manager.update_profile(profile)

# Calculate average metrics
avg_metrics = manager.calculate_average_metrics()
print(avg_metrics)

# Calculate average metrics for runners only
runner_metrics = manager.calculate_average_metrics(athlete_type="runner")
print(runner_metrics)
```

## Running the Demo

To run the athlete statistics demo:

```bash
# Navigate to project root
cd /path/to/project

# Run the demo script
python dt_project/examples/athlete_stats_demo.py
```

The demo will:
1. Generate random athlete profiles if none exist
2. Calculate and display average metrics across all athletes
3. Calculate and display average metrics by athlete type
4. Create a bar chart comparing key metrics across athlete types

## Requirements

- Python 3.6+
- matplotlib (for visualization)
- numpy (for data processing)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/athlete-tracking.git
cd athlete-tracking

# Install dependencies
pip install -r requirements.txt
``` 