#!/usr/bin/env python3
"""
🌐 UNIVERSAL QUANTUM FACTORY WEB ROUTES
======================================

Flask routes for the Universal Quantum Digital Twin Factory.
Provides web interface for:
- Automatic quantum twin creation from any data
- Conversational AI-guided quantum twin development  
- Specialized domain quantum optimization
- Real-time quantum advantage demonstration

Author: Hassan Al-Sahli
Purpose: Web interface for Universal Quantum Computing Platform
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from flask import current_app as app
import json
import asyncio
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import quantum factory systems
try:
    from dt_project.quantum.quantum_digital_twin_factory_master import (
        quantum_digital_twin_factory_master,
        ProcessingRequest, ProcessingMode, UserInterface, UserExpertise
    )
    from dt_project.quantum.specialized_quantum_domains import SpecializedDomain
    QUANTUM_FACTORY_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"Quantum factory not available: {e}")
    QUANTUM_FACTORY_AVAILABLE = False

# Create blueprint
universal_quantum_bp = Blueprint('universal_quantum', __name__, url_prefix='/quantum-factory')


@universal_quantum_bp.route('/')
def quantum_factory_home():
    """🏭 Universal Quantum Factory main page"""
    try:
        return render_template('quantum_factory_home.html')
    except Exception as e:
        app.logger.error(f"Quantum factory home error: {e}")
        return render_template_fallback('quantum_factory_home_fallback.html')


@universal_quantum_bp.route('/upload')
def upload_interface():
    """📤 Data upload interface"""
    try:
        return render_template('quantum_upload.html')
    except Exception as e:
        app.logger.error(f"Upload interface error: {e}")
        return render_template_fallback('quantum_upload_fallback.html')


@universal_quantum_bp.route('/conversation')
def conversation_interface():
    """💬 Conversational AI interface"""
    try:
        return render_template('quantum_conversation.html')
    except Exception as e:
        app.logger.error(f"Conversation interface error: {e}")
        return render_template_fallback('quantum_conversation_fallback.html')


@universal_quantum_bp.route('/domains')
def domains_interface():
    """🏢 Specialized domains interface"""
    try:
        return render_template('quantum_domains.html')
    except Exception as e:
        app.logger.error(f"Domains interface error: {e}")
        return render_template_fallback('quantum_domains_fallback.html')


# API Routes

@universal_quantum_bp.route('/api/analyze-data', methods=['POST'])
def analyze_data():
    """🔍 Analyze uploaded data and provide quantum recommendations"""
    try:
        if not QUANTUM_FACTORY_AVAILABLE:
            return jsonify({
                'error': 'Quantum factory not available',
                'status': 'error'
            }), 500
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Process uploaded file
            data = process_uploaded_file(file)
            metadata = {
                'filename': file.filename,
                'file_type': file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown'
            }
        else:
            # Handle JSON data
            json_data = request.get_json()
            if not json_data:
            return jsonify({'error': 'No data provided'}), 400
            
            data = json_data.get('data')
            metadata = json_data.get('metadata', {})
        
        # Run async data analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            analysis_result = loop.run_until_complete(
                quantum_digital_twin_factory_master.analyze_data_preview(data, metadata)
            )
        finally:
            loop.close()
        
        return jsonify({
            'status': 'success',
            'analysis': analysis_result,
            'recommendations': generate_user_recommendations(analysis_result),
            'next_steps': generate_next_steps(analysis_result)
        })
        
    except Exception as e:
        import logging
        logging.error(f"Data analysis error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@universal_quantum_bp.route('/api/create-quantum-twin', methods=['POST'])
def create_quantum_twin():
    """🚀 Create quantum digital twin automatically"""
    try:
        if not QUANTUM_FACTORY_AVAILABLE:
        return jsonify({
                'error': 'Quantum factory not available',
                'status': 'error'
            }), 500
        
        request_data = request.get_json()
        if not request_data:
        return jsonify({'error': 'No request data provided'}), 400
        
        # Extract data and preferences
        data = request_data.get('data')
        user_preferences = request_data.get('preferences', {})
        processing_mode = request_data.get('mode', 'automatic')
        
        # Determine user expertise
        expertise_str = user_preferences.get('expertise', 'intermediate')
        try:
            user_expertise = UserExpertise(expertise_str)
        except ValueError:
            user_expertise = UserExpertise.INTERMEDIATE
        
        # Create processing request
        processing_request = ProcessingRequest(
            request_id=f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=session.get('user_id'),
            data=data,
            data_description=request_data.get('description'),
            processing_mode=ProcessingMode(processing_mode),
            user_interface=UserInterface.WEB,
            user_expertise=user_expertise,
            primary_goal=request_data.get('goal'),
            use_case=request_data.get('use_case'),
            performance_requirements=user_preferences.get('requirements', {}),
            explanation_level=user_preferences.get('explanation_level', 'moderate')
        )
        
        # Run async quantum twin creation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                quantum_digital_twin_factory_master.process_request(processing_request)
            )
        finally:
            loop.close()
        
        # Format result for web response
        web_result = format_result_for_web(result, user_expertise)
        
        return jsonify(web_result)
        
    except Exception as e:
        import logging
        logging.error(f"Quantum twin creation error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@universal_quantum_bp.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    """💬 Start conversational quantum twin creation"""
    try:
        if not QUANTUM_FACTORY_AVAILABLE:
        return jsonify({
                'error': 'Quantum factory not available',
                'status': 'error'
            }), 500
        
        user_id = session.get('user_id', f"anon_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Run async conversation start
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            session_id, initial_response = loop.run_until_complete(
                quantum_digital_twin_factory_master.start_conversational_session(user_id)
            )
        finally:
            loop.close()
        
        # Store session ID
        session['quantum_conversation_id'] = session_id
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message': initial_response.message,
            'suggestions': initial_response.suggestions,
            'questions': initial_response.questions,
            'options': initial_response.options,
            'requires_input': initial_response.requires_input,
            'quantum_insights': initial_response.quantum_insights
        })
        
    except Exception as e:
        import logging
        logging.error(f"Conversation start error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@universal_quantum_bp.route('/api/conversation/continue', methods=['POST'])
def continue_conversation():
    """💬 Continue conversational quantum twin creation"""
    try:
        if not QUANTUM_FACTORY_AVAILABLE:
        return jsonify({
                'error': 'Quantum factory not available',
                'status': 'error'
            }), 500
        
        request_data = request.get_json()
        if not request_data:
        return jsonify({'error': 'No request data provided'}), 400
        
        session_id = session.get('quantum_conversation_id') or request_data.get('session_id')
        user_input = request_data.get('input', '')
        uploaded_data = request_data.get('uploaded_data')
        
        if not session_id:
        return jsonify({'error': 'No active conversation session'}), 400
        
        # Run async conversation continuation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                quantum_digital_twin_factory_master.continue_conversation(
                    session_id, user_input, uploaded_data
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            'status': 'success',
            'message': response.message,
            'suggestions': response.suggestions,
            'questions': response.questions,
            'options': response.options,
            'requires_input': response.requires_input,
            'quantum_insights': response.quantum_insights,
            'educational_content': response.educational_content
        })
        
    except Exception as e:
        import logging
        logging.error(f"Conversation continue error: {e}")
        return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500


@universal_quantum_bp.route('/api/domains')
def get_domains():
    """🏢 Get available specialized domains"""
    try:
        if not QUANTUM_FACTORY_AVAILABLE:
        return jsonify({
                'error': 'Quantum factory not available',
                'domains': []
            }), 500
        
        # Run async domain retrieval
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            domains = loop.run_until_complete(
                quantum_digital_twin_factory_master.get_supported_domains()
            )
        finally:
            loop.close()
        
        return jsonify({
            'status': 'success',
            'domains': domains
        })
        
    except Exception as e:
        import logging
        logging.error(f"Get domains error: {e}")
        return jsonify({
                'error': str(e),
                'domains': []
            }), 500


@universal_quantum_bp.route('/api/quantum-advantages')
def get_quantum_advantages():
    """⚡ Get supported quantum advantages"""
    try:
        if not QUANTUM_FACTORY_AVAILABLE:
        return jsonify({
                'error': 'Quantum factory not available',
                'advantages': {}
            }), 500
        
        # Run async advantages retrieval
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            advantages = loop.run_until_complete(
                quantum_digital_twin_factory_master.get_supported_quantum_advantages()
            )
        finally:
            loop.close()
        
        return jsonify({
            'status': 'success',
            'advantages': advantages
        })
        
    except Exception as e:
        import logging
        logging.error(f"Get quantum advantages error: {e}")
        return jsonify({
                'error': str(e),
                'advantages': {}
            }), 500


@universal_quantum_bp.route('/api/specialized-twin', methods=['POST'])
def create_specialized_twin():
    """🏢 Create specialized domain quantum twin"""
    try:
        if not QUANTUM_FACTORY_AVAILABLE:
        return jsonify({
                'error': 'Quantum factory not available',
                'status': 'error'
            }), 500
        
        request_data = request.get_json()
        if not request_data:
        return jsonify({'error': 'No request data provided'}), 400
        
        # Extract parameters
        domain_str = request_data.get('domain')
        data = request_data.get('data')
        requirements = request_data.get('requirements', {})
        
        try:
            domain = SpecializedDomain(domain_str)
        except ValueError:
        return jsonify({'error': f'Invalid domain: {domain_str}'}), 400
        
        # Run async specialized twin creation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                quantum_digital_twin_factory_master.create_specialized_twin(
                    domain, data, requirements, session.get('user_id')
                )
            )
        finally:
            loop.close()
        
        # Format result for web response
        web_result = format_result_for_web(result, UserExpertise.INTERMEDIATE)
        
        return jsonify(web_result)
        
    except Exception as e:
        import logging
        logging.error(f"Specialized twin creation error: {e}")
        return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500


@universal_quantum_bp.route('/api/factory-stats')
def get_factory_stats():
    """📊 Get quantum factory usage statistics"""
    try:
        if not QUANTUM_FACTORY_AVAILABLE:
        return jsonify({
                'error': 'Quantum factory not available',
                'stats': {}
            }), 500
        
        stats = quantum_digital_twin_factory_master.get_factory_statistics()
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        import logging
        logging.error(f"Get factory stats error: {e}")
        return jsonify({
                'error': str(e),
                'stats': {}
            }), 500


@universal_quantum_bp.route('/api/upload-data', methods=['POST'])
def upload_data():
    """📤 Handle data file uploads"""
    try:
        if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
        # Process the uploaded file
        try:
            data = process_uploaded_file(file)
            
            # Get basic data info
            data_info = get_data_info(data, file.filename)
            
        return jsonify({
                'status': 'success',
                'filename': file.filename,
                'data_info': data_info,
                'message': f'Successfully uploaded {file.filename}'
            })
            
        except Exception as e:
        return jsonify({
                'error': f'Failed to process file: {str(e)}',
                'status': 'error'
            }), 400
        
    except Exception as e:
        import logging
        logging.error(f"Data upload error: {e}")
        return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500


# Helper Functions

def process_uploaded_file(file) -> Any:
    """Process uploaded file and return data in appropriate format"""
    
    filename = file.filename.lower()
    file_content = file.read()
    
    try:
        if filename.endswith('.csv'):
            # Process CSV file
            data = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
            return data
        
        elif filename.endswith('.json'):
            # Process JSON file
            data = json.loads(file_content.decode('utf-8'))
            return data
        
        elif filename.endswith(('.xlsx', '.xls')):
            # Process Excel file
            data = pd.read_excel(io.BytesIO(file_content))
            return data
        
        elif filename.endswith('.txt'):
            # Process text file
            data = file_content.decode('utf-8')
            return data
        
        elif filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            # Process image file
            # For now, return file info - in production would use PIL/OpenCV
            return {
                'type': 'image',
                'filename': filename,
                'size_bytes': len(file_content),
                'format': filename.split('.')[-1]
            }
        
        else:
            # Unknown file type - return as bytes info
            return {
                'type': 'binary',
                'filename': filename,
                'size_bytes': len(file_content),
                'content_preview': str(file_content[:200]) + "..." if len(file_content) > 200 else str(file_content)
            }
    
    except Exception as e:
        raise ValueError(f"Failed to process {filename}: {str(e)}")


def get_data_info(data: Any, filename: str) -> Dict[str, Any]:
    """Get basic information about the uploaded data"""
    
    info = {
        'filename': filename,
        'type': type(data).__name__
    }
    
    if isinstance(data, pd.DataFrame):
        info.update({
            'rows': len(data),
            'columns': len(data.columns),
            'column_names': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB"
        })
    
    elif isinstance(data, dict):
        info.update({
            'keys': list(data.keys()),
            'structure': 'dictionary'
        })
    
    elif isinstance(data, list):
        info.update({
            'length': len(data),
            'structure': 'list'
        })
    
    elif isinstance(data, str):
        info.update({
            'length': len(data),
            'word_count': len(data.split()),
            'structure': 'text'
        })
    
    return info


def generate_user_recommendations(analysis_result: Dict[str, Any]) -> List[str]:
    """Generate user-friendly recommendations based on analysis"""
    
    recommendations = []
    
    if 'error' in analysis_result:
        recommendations.append("❌ Unable to analyze data - please check format and try again")
        return recommendations
    
    # Quantum advantage recommendations
    top_advantages = analysis_result.get('top_quantum_advantages', [])
    if top_advantages:
        best_advantage = top_advantages[0]
        advantage_name = best_advantage['advantage'].replace('_', ' ').title()
        suitability = best_advantage['suitability']
        
        if suitability > 0.8:
            recommendations.append(f"🚀 Excellent quantum advantage potential! {advantage_name} could give you significant improvements")
        elif suitability > 0.6:
            recommendations.append(f"⚡ Good quantum advantage potential with {advantage_name}")
        elif suitability > 0.4:
            recommendations.append(f"💡 Moderate quantum advantage possible with {advantage_name}")
        else:
            recommendations.append("📊 Quantum processing available, though classical methods may be sufficient")
    
    # Domain recommendations
    detected_domain = analysis_result.get('detected_domain', 'general_purpose')
    if detected_domain != 'general_purpose':
        domain_name = detected_domain.replace('_', ' ').title()
        recommendations.append(f"🎯 Your data is perfect for our {domain_name} specialization!")
    
    # Complexity recommendations
    complexity = analysis_result.get('complexity_score', 0.5)
    if complexity > 0.8:
        recommendations.append("🧠 High complexity data detected - quantum computing will excel here!")
    elif complexity > 0.6:
        recommendations.append("⚡ Medium complexity data - good candidate for quantum enhancement")
    
    return recommendations


def generate_next_steps(analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate next steps for the user"""
    
    steps = []
    
    if 'error' in analysis_result:
        steps.append({
            'action': 'fix_data',
            'title': 'Fix Data Issues',
            'description': 'Please check your data format and try uploading again'
        })
        return steps
    
    # Primary recommendation
    confidence = analysis_result.get('confidence_score', 0.5)
    if confidence > 0.7:
        steps.append({
            'action': 'create_quantum_twin',
            'title': '🚀 Create Quantum Twin',
            'description': 'Your data is perfect for quantum processing! Create your quantum digital twin now.'
        })
    else:
        steps.append({
            'action': 'start_conversation',
            'title': '💬 Talk to Quantum AI',
            'description': 'Let our AI guide you through creating the perfect quantum solution for your needs.'
        })
    
    # Domain-specific step
    detected_domain = analysis_result.get('detected_domain', 'general_purpose')
    if detected_domain != 'general_purpose':
        domain_name = detected_domain.replace('_', ' ').title()
        steps.append({
            'action': 'explore_domain',
            'title': f'🏢 Explore {domain_name}',
            'description': f'Learn about specialized quantum advantages for {domain_name.lower()} applications.'
        })
    
    # Learning step
    steps.append({
        'action': 'learn_quantum',
        'title': '📚 Learn About Quantum Advantages',
        'description': 'Discover how quantum computing can benefit your specific use case.'
    })
    
    return steps


def format_result_for_web(result, user_expertise: UserExpertise) -> Dict[str, Any]:
    """Format processing result for web interface"""
    
    # Convert result to dict if needed
    if hasattr(result, 'to_dict'):
        result_dict = result.to_dict()
    else:
        result_dict = result
    
    # Customize based on user expertise
    if user_expertise == UserExpertise.BEGINNER:
        # Simplify technical details for beginners
        if 'technical_details' in result_dict:
            result_dict['technical_details'] = simplify_technical_details(result_dict['technical_details'])
    
    elif user_expertise == UserExpertise.EXPERT:
        # Add more detailed information for experts
        if 'twin_configuration' in result_dict and result_dict['twin_configuration']:
            result_dict['expert_analysis'] = generate_expert_analysis(result_dict)
    
    # Add user-friendly formatting
    result_dict['formatted_insights'] = format_insights_for_web(result_dict.get('insights', []))
    result_dict['formatted_recommendations'] = format_recommendations_for_web(result_dict.get('recommendations', []))
    
    return result_dict


def simplify_technical_details(technical_details: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify technical details for beginner users"""
    
    simplified = {}
    
    if 'quantum_circuit' in technical_details:
        circuit = technical_details['quantum_circuit']
        simplified['quantum_specs'] = {
            'quantum_bits': circuit.get('qubits', 0),
            'circuit_complexity': 'Simple' if circuit.get('depth', 0) < 5 else 'Advanced',
            'algorithm_type': circuit.get('algorithm', 'Quantum Optimization')
        }
    
    if 'simulation_details' in technical_details:
        simulation = technical_details['simulation_details']
        simplified['performance'] = {
            'processing_time': f"{simulation.get('execution_time', 0):.2f} seconds",
            'confidence_level': f"{simulation.get('confidence', 0.8) * 100:.0f}%"
        }
    
    return simplified


def generate_expert_analysis(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed expert analysis"""
    
    analysis = {}
    
    if 'twin_configuration' in result_dict and result_dict['twin_configuration']:
        twin = result_dict['twin_configuration']
        analysis['circuit_analysis'] = {
            'hilbert_space_dimension': 2**twin.get('qubit_count', 10),
            'quantum_volume': twin.get('qubit_count', 10) * twin.get('circuit_depth', 6),
            'theoretical_complexity': f"O(2^{twin.get('qubit_count', 10)})",
            'implementation_notes': twin.get('implementation_strategy', 'Standard quantum processing')
        }
    
    if 'simulation_results' in result_dict and result_dict['simulation_results']:
        sim = result_dict['simulation_results']
        analysis['performance_analysis'] = {
            'quantum_fidelity': sim.get('quantum_performance', 0.8),
            'advantage_significance': 'High' if sim.get('quantum_advantage_achieved', 0) > 0.5 else 'Moderate',
            'statistical_confidence': f"{sim.get('confidence', 0.8) * 100:.1f}%",
            'error_bounds': f"±{sim.get('quantum_advantage_achieved', 0.3) * 0.1:.3f}"
        }
    
    return analysis


def format_insights_for_web(insights: List[str]) -> List[Dict[str, str]]:
    """Format insights for web display"""
    
    formatted = []
    for insight in insights:
        # Extract emoji and categorize
        if '🚀' in insight or '✨' in insight:
            category = 'success'
            icon = '🚀'
        elif '⚡' in insight or '💡' in insight:
            category = 'info'
            icon = '⚡'
        elif '📊' in insight or '🔬' in insight:
            category = 'technical'
            icon = '📊'
        else:
            category = 'general'
            icon = '💭'
        
        formatted.append({
            'text': insight,
            'category': category,
            'icon': icon
        })
    
    return formatted


def format_recommendations_for_web(recommendations: List[str]) -> List[Dict[str, str]]:
    """Format recommendations for web display"""
    
    formatted = []
    for rec in recommendations:
        # Determine recommendation type
        if '✅' in rec or 'recommended' in rec.lower():
            rec_type = 'success'
            icon = '✅'
        elif '⚡' in rec or 'optimize' in rec.lower():
            rec_type = 'optimization'
            icon = '⚡'
        elif '🔧' in rec or 'consider' in rec.lower():
            rec_type = 'suggestion'
            icon = '🔧'
        else:
            rec_type = 'general'
            icon = '💡'
        
        formatted.append({
            'text': rec,
            'type': rec_type,
            'icon': icon
        })
    
    return formatted


def render_template_fallback(template_name: str) -> str:
    """Render fallback template if main template fails"""
    
    fallback_templates = {
        'quantum_factory_home_fallback.html': """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🏭 Universal Quantum Factory</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
                .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; }
                .header { text-align: center; margin-bottom: 40px; }
                .feature { margin: 20px 0; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; }
                .btn { display: inline-block; padding: 15px 30px; background: #4CAF50; color: white; text-decoration: none; border-radius: 25px; margin: 10px; }
                .btn:hover { background: #45a049; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏭 Welcome to the Universal Quantum Factory</h1>
                    <p>Transform any data into quantum-enhanced insights with proven quantum advantages!</p>
                </div>
                
                <div class="feature">
                    <h3>🤖 Automatic Quantum Twin Creation</h3>
                    <p>Upload any data and get a custom quantum digital twin with proven quantum advantages automatically.</p>
                    <a href="/quantum-factory/upload" class="btn">Upload Data</a>
                </div>
                
                <div class="feature">
                    <h3>💬 AI-Guided Quantum Optimization</h3>
                    <p>Talk to our quantum AI to create the perfect quantum solution for your specific needs.</p>
                    <a href="/quantum-factory/conversation" class="btn">Start Conversation</a>
                </div>
                
                <div class="feature">
                    <h3>🏢 Specialized Quantum Domains</h3>
                    <p>Explore quantum advantages for Financial Services, IoT, Healthcare, and more specialized domains.</p>
                    <a href="/quantum-factory/domains" class="btn">Explore Domains</a>
                </div>
                
                <div class="feature">
                    <h3>⚡ Proven Quantum Advantages</h3>
                    <ul>
                        <li>🎯 <strong>98% Quantum Sensing Advantage</strong> - Revolutionary precision improvements</li>
                        <li>🚀 <strong>24% Optimization Speedup</strong> - Faster solutions to complex problems</li>
                        <li>🧠 <strong>Universal Data Processing</strong> - Any data type, optimal quantum approach</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """,
        
        'quantum_upload_fallback.html': """
        <!DOCTYPE html>
        <html>
        <head>
            <title>📤 Upload Data - Quantum Factory</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
                .container { max-width: 600px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; }
                .upload-area { border: 2px dashed #fff; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
                .btn { display: inline-block; padding: 15px 30px; background: #4CAF50; color: white; text-decoration: none; border-radius: 25px; margin: 10px; }
                input[type="file"] { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📤 Upload Your Data</h1>
                <p>Upload any type of data file and let our quantum AI create the perfect quantum digital twin for you!</p>
                
                <div class="upload-area">
                    <h3>🎯 Drag & Drop or Select File</h3>
                    <input type="file" id="fileInput" accept=".csv,.json,.xlsx,.txt,.jpg,.png">
                    <p>Supported formats: CSV, JSON, Excel, Text, Images</p>
                </div>
                
                <div>
                    <h3>✨ What happens next?</h3>
                    <ul>
                        <li>🔍 Automatic data analysis and pattern detection</li>
                        <li>🎯 Optimal quantum algorithm selection</li>
                        <li>🚀 Quantum digital twin creation with proven advantages</li>
                        <li>📊 Performance comparison with classical methods</li>
                    </ul>
                </div>
                
                <a href="/quantum-factory/" class="btn">← Back to Factory</a>
            </div>
            
            <script>
                document.getElementById('fileInput').addEventListener('change', function(e) {
                    if (e.target.files.length > 0) {
                        alert('File upload functionality requires full quantum factory setup. This is a demo interface.');
                    }
                });
            </script>
        </body>
        </html>
        """,
        
        'quantum_conversation_fallback.html': """
        <!DOCTYPE html>
        <html>
        <head>
            <title>💬 Quantum AI Chat - Quantum Factory</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
                .chat-container { max-width: 800px; margin: 20px auto; background: rgba(255,255,255,0.95); border-radius: 20px; overflow: hidden; height: 80vh; display: flex; flex-direction: column; }
                .chat-header { background: #4CAF50; color: white; padding: 20px; text-align: center; }
                .chat-messages { flex: 1; padding: 20px; overflow-y: auto; }
                .message { margin: 15px 0; padding: 15px; border-radius: 15px; max-width: 80%; }
                .ai-message { background: #e3f2fd; margin-right: auto; }
                .user-message { background: #4CAF50; color: white; margin-left: auto; }
                .chat-input { padding: 20px; border-top: 1px solid #ddd; display: flex; }
                .chat-input input { flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 25px; margin-right: 10px; }
                .chat-input button { padding: 15px 30px; background: #4CAF50; color: white; border: none; border-radius: 25px; cursor: pointer; }
                .chat-input button:hover { background: #45a049; }
            </style>
        </head>
        <body>
            <div class="chat-container">
                <div class="chat-header">
                    <h2>🤖 Quantum AI Assistant</h2>
                    <p>Let me help you create the perfect quantum digital twin!</p>
                </div>
                
                <div class="chat-messages" id="chatMessages">
                    <div class="message ai-message">
                        <strong>🌟 Quantum AI:</strong> Welcome to the Universal Quantum AI! I'm here to help you harness proven quantum advantages for your data. Let's create the perfect quantum digital twin for your needs!<br><br>
                        
                        To get started, how familiar are you with quantum computing?<br>
                        <button onclick="selectExpertise('beginner')" style="margin: 5px; padding: 10px; background: #2196F3; color: white; border: none; border-radius: 15px; cursor: pointer;">Beginner - New to quantum computing</button><br>
                        <button onclick="selectExpertise('intermediate')" style="margin: 5px; padding: 10px; background: #2196F3; color: white; border: none; border-radius: 15px; cursor: pointer;">Intermediate - Some quantum knowledge</button><br>
                        <button onclick="selectExpertise('expert')" style="margin: 5px; padding: 10px; background: #2196F3; color: white; border: none; border-radius: 15px; cursor: pointer;">Expert - Advanced quantum understanding</button>
                    </div>
                </div>
                
                <div class="chat-input">
                    <input type="text" id="messageInput" placeholder="Type your message or select an option above..." disabled>
                    <button onclick="sendMessage()" disabled>Send</button>
                </div>
            </div>
            
            <script>
                function selectExpertise(level) {
                    addUserMessage('I am a ' + level + ' user');
                    setTimeout(() => {
                        if (level === 'beginner') {
                            addAIMessage("Perfect! I'll explain everything in simple terms and guide you through each step. Don't worry - quantum computing might sound complex, but I'll make it easy to understand.\\n\\nNow, let's talk about your data! What type of information are you working with?");
                        } else if (level === 'intermediate') {
                            addAIMessage("Great! I'll provide technical details where helpful and explain the quantum advantages clearly.\\n\\nLet's dive into understanding your data and requirements. Can you tell me about the type of data you're working with?");
                        } else {
                            addAIMessage("Excellent! I can discuss technical implementation details and quantum algorithms directly.\\n\\nLet's analyze your data characteristics and quantum suitability. What type of dataset are you looking to optimize?");
                        }
                        document.getElementById('messageInput').disabled = false;
                        document.querySelector('.chat-input button').disabled = false;
                    }, 1000);
                }
                
                function addUserMessage(message) {
                    const chatMessages = document.getElementById('chatMessages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message user-message';
                    messageDiv.innerHTML = '<strong>You:</strong> ' + message;
                    chatMessages.appendChild(messageDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                function addAIMessage(message) {
                    const chatMessages = document.getElementById('chatMessages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ai-message';
                    messageDiv.innerHTML = '<strong>🤖 Quantum AI:</strong> ' + message;
                    chatMessages.appendChild(messageDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                function sendMessage() {
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    if (message) {
                        addUserMessage(message);
                        input.value = '';
                        
                        // Simulate AI response
                        setTimeout(() => {
                            addAIMessage("Thank you for that information! For a full conversational experience with real quantum twin creation, the complete quantum factory system would analyze your input and provide customized quantum solutions. This is a demo interface showing the conversation flow.");
                        }, 1500);
                    }
                }
                
                document.getElementById('messageInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            </script>
        </body>
        </html>
        """,
        
        'quantum_domains_fallback.html': """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🏢 Specialized Domains - Quantum Factory</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }
                .container { max-width: 1200px; margin: 0 auto; padding: 40px; }
                .header { text-align: center; margin-bottom: 40px; }
                .domains-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px; }
                .domain-card { background: rgba(255,255,255,0.1); padding: 30px; border-radius: 20px; backdrop-filter: blur(10px); }
                .domain-card h3 { color: #fff; margin-top: 0; }
                .advantage-list { list-style: none; padding: 0; }
                .advantage-list li { margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px; }
                .btn { display: inline-block; padding: 12px 25px; background: #4CAF50; color: white; text-decoration: none; border-radius: 20px; margin-top: 15px; }
                .btn:hover { background: #45a049; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏢 Specialized Quantum Domains</h1>
                    <p>Discover quantum advantages tailored to your industry and use case</p>
                </div>
                
                <div class="domains-grid">
                    <div class="domain-card">
                        <h3>🏦 Financial Services</h3>
                        <p>Quantum advantages for trading, risk management, and portfolio optimization</p>
                        <ul class="advantage-list">
                            <li>📈 Portfolio optimization with 25.6x speedup</li>
                            <li>🛡️ Real-time fraud detection</li>
                            <li>📊 Quantum Monte Carlo risk modeling</li>
                            <li>🤖 Algorithmic trading strategies</li>
                        </ul>
                        <a href="#" class="btn">Explore Financial Quantum</a>
                    </div>
                    
                    <div class="domain-card">
                        <h3>🌐 IoT & Smart Systems</h3>
                        <p>Quantum-enhanced IoT with 98% sensing precision improvement</p>
                        <ul class="advantage-list">
                            <li>🎯 Quantum sensor fusion (98% advantage)</li>
                            <li>🔧 Predictive maintenance with quantum ML</li>
                            <li>🌐 Network optimization for IoT</li>
                            <li>🚨 Real-time anomaly detection</li>
                        </ul>
                        <a href="#" class="btn">Explore IoT Quantum</a>
                    </div>
                    
                    <div class="domain-card">
                        <h3>🏥 Healthcare & Life Sciences</h3>
                        <p>Quantum computing for medical breakthroughs and drug discovery</p>
                        <ul class="advantage-list">
                            <li>💊 Quantum drug discovery simulation</li>
                            <li>🧬 Genomic analysis acceleration</li>
                            <li>📸 Medical imaging enhancement</li>
                            <li>🎯 Personalized medicine optimization</li>
                        </ul>
                        <a href="#" class="btn">Explore Healthcare Quantum</a>
                    </div>
                    
                    <div class="domain-card">
                        <h3>🏭 Manufacturing & Supply Chain</h3>
                        <p>Quantum optimization for production and logistics</p>
                        <ul class="advantage-list">
                            <li>⚙️ Production scheduling optimization</li>
                            <li>🚚 Supply chain route optimization</li>
                            <li>🔍 Quality control enhancement</li>
                            <li>📦 Inventory management</li>
                        </ul>
                        <a href="#" class="btn">Coming Soon</a>
                    </div>
                    
                    <div class="domain-card">
                        <h3>⚡ Energy & Utilities</h3>
                        <p>Quantum solutions for smart grid and energy optimization</p>
                        <ul class="advantage-list">
                            <li>🔋 Smart grid optimization</li>
                            <li>🌱 Renewable energy planning</li>
                            <li>📊 Demand forecasting</li>
                            <li>⚡ Load balancing</li>
                        </ul>
                        <a href="#" class="btn">Coming Soon</a>
                    </div>
                    
                    <div class="domain-card">
                        <h3>🎯 General Purpose</h3>
                        <p>Universal quantum computing for any data type</p>
                        <ul class="advantage-list">
                            <li>🧠 Automatic quantum algorithm selection</li>
                            <li>📊 Pattern recognition enhancement</li>
                            <li>🔍 Search acceleration</li>
                            <li>⚡ General optimization speedup</li>
                        </ul>
                        <a href="/quantum-factory/upload" class="btn">Try Universal Quantum</a>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 40px;">
                    <a href="/quantum-factory/" class="btn">← Back to Quantum Factory</a>
                </div>
            </div>
        </body>
        </html>
        """
    }
    
    return fallback_templates.get(template_name, "<h1>Template not found</h1>")


# Register the blueprint with error handlers
@universal_quantum_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@universal_quantum_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# Export the blueprint
def create_universal_quantum_routes():
    """Create and return the universal quantum factory blueprint"""
    return universal_quantum_bp
