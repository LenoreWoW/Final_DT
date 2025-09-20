#!/usr/bin/env python3
"""
Add quantum lab route to the Flask application
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def add_quantum_lab_route():
    """Add the quantum lab route to the main routes file"""
    
    routes_file = Path(__file__).parent / 'dt_project' / 'web_interface' / 'routes' / 'main_routes.py'
    
    # Read the current content
    with open(routes_file, 'r') as f:
        content = f.read()
    
    # Check if quantum lab route already exists
    if '/quantum-lab' in content:
        print("✅ Quantum lab route already exists")
        return
    
    # Find the position to insert the new route (after existing routes, before create_main_routes)
    insert_position = content.find('@main_bp.route(\'/circuit-designer\')')
    
    if insert_position == -1:
        # Try another position
        insert_position = content.find('@main_bp.route(\'/api/status\')')
    
    if insert_position == -1:
        print("❌ Could not find insertion point in routes file")
        return
    
    # Find the end of the previous route function
    end_of_previous_function = content.find('\n\n@', insert_position)
    if end_of_previous_function == -1:
        end_of_previous_function = content.find('\n@main_bp.route', insert_position + 20)
    
    # New route code
    new_route = '''
@main_bp.route('/quantum-lab')
def quantum_lab():
    """Interactive quantum research laboratory"""
    try:
        from flask import render_template
        return render_template('quantum_lab.html')
    except Exception as e:
        app.logger.error(f"Quantum lab error: {e}")
        return jsonify({'error': 'Quantum lab temporarily unavailable'}), 500

'''
    
    # Insert the new route
    new_content = content[:end_of_previous_function] + new_route + content[end_of_previous_function:]
    
    # Write back to file
    with open(routes_file, 'w') as f:
        f.write(new_content)
    
    print("✅ Added quantum lab route to main_routes.py")

if __name__ == "__main__":
    add_quantum_lab_route()