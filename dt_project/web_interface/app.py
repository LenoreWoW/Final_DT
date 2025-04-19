"""
Flask Web Application
Main entry point for the Digital Twin web interface.
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

from dt_project.config import ConfigManager
from dt_project.data_acquisition.weather import WeatherService
from dt_project.data_acquisition.location import LocationService
from dt_project.data_acquisition.athlete import AthleteManager
from dt_project.physics.environment import EnvironmentalSimulation
from dt_project.physics.terrain import TerrainSimulation, Point
from dt_project.physics.biomechanics import BiomechanicalModel
from dt_project.physics.military import MilitarySimulation, EquipmentLoad, MovementType
from dt_project.quantum import initialize_quantum_components

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Load configuration
config_manager = ConfigManager()
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'development_key')
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize services
weather_service = WeatherService(config_manager)
location_service = LocationService(config_manager)
athlete_manager = AthleteManager(config_manager)
environmental_simulation = EnvironmentalSimulation(config_manager)
terrain_simulation = TerrainSimulation(config_manager)
biomechanical_model = BiomechanicalModel(config_manager)
military_simulation = MilitarySimulation(config_manager)
quantum_components = initialize_quantum_components(config_manager)
quantum_monte_carlo = quantum_components["monte_carlo"]
quantum_ml = quantum_components["machine_learning"]

# Main routes
@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html', 
                          title='Digital Twin Dashboard',
                          current_time=datetime.now())

@app.route('/athlete')
def athlete_page():
    """Render the athlete management page."""
    # Get list of available athlete profiles
    athlete_profiles = athlete_manager.list_profiles()
    
    return render_template('athlete.html',
                          title='Athlete Profiles',
                          athlete_profiles=athlete_profiles)

@app.route('/simulation')
def simulation_page():
    """Render the simulation configuration page."""
    # Get list of available athlete profiles
    athlete_profiles = athlete_manager.list_profiles()
    
    # Get major cities for location dropdown
    cities = location_service.get_major_cities()
    
    return render_template('simulation.html',
                          title='Simulation Configuration',
                          athlete_profiles=athlete_profiles,
                          cities=cities)

@app.route('/military')
def military_page():
    """Render the military simulation page."""
    # Get list of available athlete profiles
    athlete_profiles = athlete_manager.list_profiles()
    
    # Get major cities for location dropdown
    cities = location_service.get_major_cities()
    
    # Equipment load types
    equipment_loads = [
        {"id": EquipmentLoad.FIGHTING_LOAD, "name": "Fighting Load (~25kg)"},
        {"id": EquipmentLoad.APPROACH_LOAD, "name": "Approach Load (~35kg)"},
        {"id": EquipmentLoad.EMERGENCY_LOAD, "name": "Emergency Load (~45kg)"}
    ]
    
    # Movement types
    movement_types = [
        {"id": MovementType.NORMAL, "name": "Normal March"},
        {"id": MovementType.RUSH, "name": "Rush (Rapid Movement)"},
        {"id": MovementType.PATROL, "name": "Patrol (Vigilant)"},
        {"id": MovementType.STEALTH, "name": "Stealth (Concealed)"}
    ]
    
    # Equipment items
    equipment_items = [
        {"id": 1, "name": "Tactical Vest", "weight": 4.5, "icon": "shield-alt", "essential": True},
        {"id": 2, "name": "Helmet", "weight": 1.4, "icon": "helmet-battle", "essential": True},
        {"id": 3, "name": "Boots", "weight": 1.8, "icon": "hiking", "essential": True},
        {"id": 4, "name": "Weapon", "weight": 3.6, "icon": "bullseye", "essential": True},
        {"id": 5, "name": "Water Supply", "weight": 2.0, "icon": "tint", "essential": False},
        {"id": 6, "name": "First Aid", "weight": 0.8, "icon": "first-aid", "essential": True},
        {"id": 7, "name": "Night Vision", "weight": 1.2, "icon": "eye", "essential": False},
        {"id": 8, "name": "Communication", "weight": 1.5, "icon": "satellite-dish", "essential": False},
        {"id": 9, "name": "MRE Rations", "weight": 1.8, "icon": "utensils", "essential": False}
    ]
    
    # Default equipment values
    default_equipment = {
        "max_load": 30.0,
        "current_load": 12.1,
        "fighting_load": 25.0,
        "approach_load": 35.0,
        "emergency_load": 45.0
    }
    
    return render_template('military.html',
                          title='Military Simulation',
                          athlete_profiles=athlete_profiles,
                          cities=cities,
                          equipment_loads=equipment_loads,
                          movement_types=movement_types,
                          equipment_items=equipment_items,
                          default_equipment=default_equipment)

@app.route('/visualization')
def visualization_page():
    """Render the data visualization page."""
    return render_template('visualization.html',
                          title='Data Visualization')

@app.route('/quantum')
def quantum_page():
    """Render the quantum features page."""
    # Get quantum status
    quantum_status = {
        "qmc_available": quantum_components["qmc_available"],
        "qml_available": quantum_components["qml_available"],
        "enabled": config_manager.get("quantum.enabled", False),
        "backend": config_manager.get("quantum.backend", "simulator"),
        "error_mitigation": config_manager.get("quantum.error_mitigation", False)
    }
    
    return render_template('quantum.html',
                          title='Quantum Features',
                          quantum_status=quantum_status)

# API routes for data acquisition
@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Get weather data for a location."""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400
    
    # Get current weather
    current_weather = weather_service.get_current_weather(lat, lon)
    
    # Get forecast
    forecast = weather_service.get_forecast(lat, lon, days=3)
    
    # Get historical data
    historical = weather_service.get_historical_weather(lat, lon, days=7)
    
    return jsonify({
        "current": current_weather,
        "forecast": forecast,
        "historical": historical
    })

@app.route('/api/location/search', methods=['GET'])
def search_location():
    """Search for locations by name."""
    query = request.args.get('q', '')
    limit = request.args.get('limit', 5, type=int)
    
    if not query:
        return jsonify([])
    
    # Search for matching places
    places = location_service.search_places(query, limit)
    
    return jsonify(places)

@app.route('/api/location/reverse', methods=['GET'])
def reverse_geocode():
    """Convert coordinates to a location name."""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400
    
    # Perform reverse geocoding
    location = location_service.reverse_geocode(lat, lon)
    
    if not location:
        return jsonify({"error": "Location not found"}), 404
    
    return jsonify(location)

@app.route('/api/location/cities', methods=['GET'])
def get_cities():
    """Get list of major cities."""
    cities = location_service.get_major_cities()
    return jsonify(cities)

# API routes for athlete management
@app.route('/api/athletes', methods=['GET'])
def get_athletes():
    """Get list of athlete profiles."""
    athlete_profiles = athlete_manager.list_profiles()
    return jsonify(athlete_profiles)

@app.route('/api/athletes/<profile_id>', methods=['GET'])
def get_athlete(profile_id):
    """Get specific athlete profile."""
    athlete = athlete_manager.get_profile(profile_id)
    
    if not athlete:
        return jsonify({"error": "Athlete profile not found"}), 404
    
    return jsonify(athlete.get_profile())

@app.route('/api/athletes', methods=['POST'])
def create_athlete():
    """Create a new athlete profile."""
    data = request.json
    
    # Validate required fields
    required_fields = ['name', 'athlete_type', 'age', 'gender', 'height', 'weight']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Create athlete profile
    athlete = athlete_manager.create_profile(
        name=data['name'],
        athlete_type=data['athlete_type'],
        age=data['age'],
        gender=data['gender'],
        height=data['height'],
        weight=data['weight']
    )
    
    return jsonify(athlete.get_profile()), 201

@app.route('/api/athletes/random', methods=['POST'])
def create_random_athlete():
    """Create a random athlete profile."""
    data = request.json or {}
    athlete_type = data.get('athlete_type')
    
    # Generate random profile
    athlete = athlete_manager.generate_random_profile(athlete_type)
    
    return jsonify(athlete.get_profile()), 201

@app.route('/api/athletes/<profile_id>', methods=['DELETE'])
def delete_athlete(profile_id):
    """Delete an athlete profile."""
    success = athlete_manager.delete_profile(profile_id)
    
    if not success:
        return jsonify({"error": "Failed to delete athlete profile"}), 500
    
    return jsonify({"message": "Athlete profile deleted successfully"})

# API routes for simulations
@app.route('/api/simulation/environment', methods=['POST'])
def simulate_environment():
    """Simulate environmental conditions."""
    data = request.json
    
    # Get required parameters
    lat = data.get('latitude')
    lon = data.get('longitude')
    start_time_str = data.get('start_time')
    duration_hours = data.get('duration_hours', 24)
    interval_minutes = data.get('interval_minutes', 60)
    
    if not all([lat, lon, start_time_str]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Parse start time
    try:
        start_time = datetime.fromisoformat(start_time_str)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid start time format"}), 400
    
    # Set up simulation
    environmental_simulation.set_base_conditions(
        temperature=data.get('base_temperature', 20.0),
        humidity=data.get('base_humidity', 50.0),
        wind_speed=data.get('base_wind_speed', 5.0),
        wind_direction=data.get('base_wind_direction', 0.0)
    )
    
    environmental_simulation.set_time(start_time)
    
    # Generate conditions at intervals
    results = []
    current_time = start_time
    end_time = start_time + timedelta(hours=duration_hours)
    interval_seconds = interval_minutes * 60
    
    while current_time <= end_time:
        conditions = environmental_simulation.get_conditions(current_time)
        results.append(conditions)
        current_time += timedelta(seconds=interval_seconds)
    
    return jsonify(results)

@app.route('/api/simulation/terrain', methods=['POST'])
def simulate_terrain():
    """Generate terrain profile."""
    data = request.json
    
    # Get required parameters
    start_lat = data.get('start_latitude')
    start_lon = data.get('start_longitude')
    end_lat = data.get('end_latitude')
    end_lon = data.get('end_longitude')
    
    route_type = data.get('route_type', 'point_to_point')
    length_meters = data.get('length_meters')
    num_points = data.get('num_points', 100)
    
    # Validate parameters based on route type
    if route_type == 'point_to_point':
        if not all([start_lat, start_lon, end_lat, end_lon]):
            return jsonify({"error": "Missing required coordinates"}), 400
            
        # Create start and end points
        start_point = Point(start_lat, start_lon)
        end_point = Point(end_lat, end_lon)
        
        # Generate terrain profile
        terrain_profile = terrain_simulation.generate_terrain_profile(
            start_point, end_point, num_points)
            
    else:  # loop, out_and_back, random
        if not all([start_lat, start_lon, length_meters]):
            return jsonify({"error": "Missing required parameters"}), 400
            
        # Create start point
        start_point = Point(start_lat, start_lon)
        
        # Generate route
        terrain_profile = terrain_simulation.generate_route(
            start_point, length_meters, route_type, num_points)
    
    return jsonify(terrain_profile)

@app.route('/api/simulation/performance', methods=['POST'])
def simulate_performance():
    """Simulate athletic performance."""
    data = request.json
    
    # Get required parameters
    athlete_id = data.get('athlete_id')
    terrain_profile = data.get('terrain_profile')
    environmental_conditions = data.get('environmental_conditions')
    
    if not all([athlete_id, terrain_profile, environmental_conditions]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Get athlete profile
    athlete = athlete_manager.get_profile(athlete_id)
    if not athlete:
        return jsonify({"error": "Athlete profile not found"}), 404
    
    athlete_profile = athlete.get_profile()
    
    # Simulate performance
    performance_data = biomechanical_model.simulate_performance(
        athlete_profile, terrain_profile, environmental_conditions)
    
    return jsonify(performance_data)

@app.route('/api/simulation/military', methods=['POST'])
def simulate_military():
    """Simulate military mission."""
    data = request.json
    
    # Get required parameters
    soldier_id = data.get('soldier_id')
    terrain_profile = data.get('terrain_profile')
    environmental_conditions = data.get('environmental_conditions')
    equipment_load = data.get('equipment_load', EquipmentLoad.FIGHTING_LOAD)
    movement_type = data.get('movement_type', MovementType.NORMAL)
    is_night = data.get('is_night', False)
    
    if not all([soldier_id, terrain_profile, environmental_conditions]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Get soldier profile (using athlete profile)
    soldier = athlete_manager.get_profile(soldier_id)
    if not soldier:
        return jsonify({"error": "Soldier profile not found"}), 404
    
    soldier_profile = soldier.get_profile()
    
    # Simulate military mission
    mission_data = military_simulation.simulate_mission(
        soldier_profile, terrain_profile, environmental_conditions,
        equipment_load, movement_type, is_night)
    
    return jsonify(mission_data)

@app.route('/api/simulation/equipment', methods=['POST'])
def analyze_equipment():
    """Analyze equipment impacts."""
    data = request.json
    
    # Get required parameters
    base_weight = data.get('base_weight')
    equipment_items = data.get('equipment_items', [])
    
    if base_weight is None:
        return jsonify({"error": "Missing base weight"}), 400
    
    # Calculate equipment impacts
    equipment_analysis = military_simulation.calculate_equipment_impacts(
        base_weight, equipment_items)
    
    return jsonify(equipment_analysis)

# API routes for quantum features
@app.route('/api/quantum/settings', methods=['POST'])
def update_quantum_settings():
    """Update quantum computing settings."""
    data = request.json
    
    try:
        # Update configuration
        config_manager.update("quantum.enabled", data.get("enabled", False))
        config_manager.update("quantum.backend", data.get("backend", "simulator"))
        config_manager.update("quantum.error_mitigation", data.get("error_mitigation", False))
        
        # Save configuration
        config_manager.save()
        
        return jsonify({"message": "Settings updated successfully"})
        
    except Exception as e:
        logger.error(f"Error updating quantum settings: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/quantum/monte-carlo', methods=['POST'])
def run_quantum_monte_carlo():
    """Run quantum Monte Carlo simulation."""
    data = request.json
    
    try:
        # Get parameters
        distribution_type = data.get("distribution_type", "uniform")
        iterations = data.get("iterations", 1000)
        run_comparison = data.get("run_comparison", True)
        
        # Define a simple target function (quadratic function)
        def target_function(x, y):
            return x**2 + y**2 + 0.5 * x * y
        
        # Define parameter ranges
        param_ranges = {
            'x': (-2.0, 2.0),
            'y': (-2.0, 2.0)
        }
        
        # Run quantum Monte Carlo
        quantum_result = quantum_monte_carlo.run_quantum_monte_carlo(
            param_ranges, 
            iterations=iterations,
            target_function=lambda x, y: target_function(x, y),
            distribution_type=distribution_type
        )
        
        response = {
            "quantum": {
                "mean": quantum_result["mean"],
                "std": quantum_result["std"],
                "execution_time": quantum_result["execution_time"]
            }
        }
        
        # Run classical comparison if requested
        if run_comparison:
            comparison = quantum_monte_carlo.compare_with_classical(
                param_ranges,
                lambda x, y: target_function(x, y),
                iterations=iterations
            )
            
            response["classical"] = {
                "mean": comparison["classical"]["mean"],
                "std": comparison["classical"]["std"],
                "execution_time": comparison["classical"]["execution_time"]
            }
            
            response["speedup"] = comparison["speedup"]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in quantum Monte Carlo: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/quantum/train-model', methods=['POST'])
def train_quantum_model():
    """Train a quantum machine learning model."""
    data = request.json
    
    try:
        # Get parameters
        feature_map = data.get("feature_map", "zz")
        ansatz_type = data.get("ansatz_type", "strongly_entangling")
        n_layers = data.get("n_layers", 2)
        compare_encodings = data.get("compare_encodings", False)
        
        # Generate synthetic data
        np.random.seed(42)
        x = np.random.rand(100, 3)  # 100 samples, 3 features
        y = 0.5 * x[:, 0] + 0.3 * x[:, 1] + 0.2 * x[:, 2] + 0.1 * np.sin(x[:, 0] * 10) + 0.05 * np.random.rand(100)
        
        # Compare encoding strategies if requested
        if compare_encodings:
            quantum_ml.compare_encoding_strategies(x, y, test_size=0.2, iterations=15)
        
        # Configure quantum ML
        quantum_ml.feature_map = feature_map
        quantum_ml.ansatz_type = ansatz_type
        quantum_ml.n_layers = n_layers
        
        # Train quantum model
        quantum_result = quantum_ml.train_model(x, y, test_size=0.2, verbose=True)
        
        # Train classical model for comparison
        from sklearn.linear_model import LinearRegression
        classical_model = LinearRegression()
        comparison = quantum_ml.compare_with_classical(x, y, classical_model, test_size=0.2)
        
        response = {
            "quantum": {
                "mse": comparison["quantum_mse"],
                "training_time": quantum_result["training_time"]
            },
            "classical": {
                "mse": comparison["classical_mse"],
                "training_time": comparison["classical_training_time"]
            },
            "improvement": comparison["relative_improvement"],
            "history": quantum_ml.get_training_history()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in quantum ML: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', 
                          title='Page Not Found',
                          error_code=404,
                          error_message="The page you requested could not be found."), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('error.html',
                          title='Server Error',
                          error_code=500,
                          error_message="A server error occurred. Please try again later."), 500

# Main entry point
if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Get debug mode from environment
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Start the application
    app.run(host='0.0.0.0', port=port, debug=debug) 