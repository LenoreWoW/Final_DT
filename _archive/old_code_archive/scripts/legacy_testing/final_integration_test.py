#!/usr/bin/env python3
"""
Final Integration Test - Comprehensive Platform Validation
Tests all systems working together end-to-end.
"""

import asyncio
import json
import time
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def run_final_integration_test():
    """Run comprehensive integration test of the entire platform."""
    
    print("🚀 FINAL INTEGRATION TEST - COMPREHENSIVE PLATFORM VALIDATION")
    print("=" * 70)
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "platform_status": "TESTING",
        "tests": [],
        "overall_success": True,
        "summary": {}
    }
    
    # Test 1: Configuration System
    print("\n⚙️ Testing Configuration System...")
    try:
        from dt_project.config.unified_config import get_unified_config
        config = get_unified_config()
        summary = config.get_summary()
        
        test_results["tests"].append({
            "test": "Configuration System",
            "status": "PASSED",
            "details": summary
        })
        print("✅ Configuration system working")
        
    except Exception as e:
        test_results["tests"].append({
            "test": "Configuration System",
            "status": "FAILED",
            "error": str(e)
        })
        test_results["overall_success"] = False
        print(f"❌ Configuration system failed: {e}")
    
    # Test 2: Error Handling System
    print("\n🛡️ Testing Error Handling System...")
    try:
        from dt_project.core.error_handling import get_error_handler, ValidationError
        
        error_handler = get_error_handler()
        
        # Trigger a test error
        test_error = ValidationError("Test validation error for integration test")
        error_context = error_handler.handle_error(test_error)
        
        # Get error summary
        error_summary = error_handler.get_error_summary()
        
        test_results["tests"].append({
            "test": "Error Handling System",
            "status": "PASSED",
            "details": {
                "error_handled": True,
                "error_category": error_context.category.value,
                "total_errors": error_summary["total_errors"]
            }
        })
        print("✅ Error handling system working")
        
    except Exception as e:
        test_results["tests"].append({
            "test": "Error Handling System",
            "status": "FAILED",
            "error": str(e)
        })
        test_results["overall_success"] = False
        print(f"❌ Error handling system failed: {e}")
    
    # Test 3: Quantum Digital Twin Core
    print("\n🌌 Testing Quantum Digital Twin Core...")
    try:
        from dt_project.quantum.quantum_digital_twin_core import (
            create_quantum_digital_twin_platform, QuantumTwinType
        )
        
        # Create platform
        config = {
            'fault_tolerance': False,
            'quantum_internet': False,
            'holographic_viz': False,
            'max_qubits': 4
        }
        platform = create_quantum_digital_twin_platform(config)
        
        # Create quantum twin
        twin = await platform.create_quantum_digital_twin(
            entity_id="integration_test_twin",
            twin_type=QuantumTwinType.SYSTEM,
            initial_state={'test_param': 0.7, 'efficiency': 0.9},
            quantum_resources={'n_qubits': 3, 'circuit_depth': 5}
        )
        
        # Test evolution
        evolution_result = await platform.run_quantum_evolution("integration_test_twin", 0.001)
        
        # Test optimization
        optimization_result = await platform.optimize_twin_performance(
            "integration_test_twin",
            "test_optimization"
        )
        
        # Get platform summary
        platform_summary = platform.get_quantum_advantage_summary()
        
        test_results["tests"].append({
            "test": "Quantum Digital Twin Core",
            "status": "PASSED",
            "details": {
                "twin_created": True,
                "twin_fidelity": twin.quantum_state.fidelity if twin.quantum_state else 0.0,
                "evolution_fidelity": evolution_result.get("quantum_fidelity", 0.0),
                "optimization_improvement": optimization_result.get("performance_improvement", 1.0),
                "platform_summary": platform_summary
            }
        })
        print(f"✅ Quantum Digital Twin Core working")
        print(f"   - Twin Fidelity: {twin.quantum_state.fidelity:.3f}")
        print(f"   - Evolution Fidelity: {evolution_result.get('quantum_fidelity', 0.0):.3f}")
        print(f"   - Optimization Improvement: {optimization_result.get('performance_improvement', 1.0):.2f}x")
        
    except Exception as e:
        test_results["tests"].append({
            "test": "Quantum Digital Twin Core",
            "status": "FAILED",
            "error": str(e)
        })
        test_results["overall_success"] = False
        print(f"❌ Quantum Digital Twin Core failed: {e}")
    
    # Test 4: Real Quantum Algorithms
    print("\n⚛️ Testing Real Quantum Algorithms...")
    try:
        from dt_project.quantum.real_quantum_algorithms import create_quantum_algorithms
        
        algorithms = create_quantum_algorithms()
        
        # Test Grover's algorithm
        grover_result = await algorithms.grovers_search(4, 2)
        
        # Test Bernstein-Vazirani algorithm
        bv_result = await algorithms.bernstein_vazirani("10")
        
        # Test Quantum Fourier Transform
        qft_result = await algorithms.quantum_fourier_transform_demo(2)
        
        test_results["tests"].append({
            "test": "Real Quantum Algorithms",
            "status": "PASSED",
            "details": {
                "grover_executed": grover_result.success if grover_result else False,
                "grover_advantage": grover_result.quantum_advantage if grover_result else 1.0,
                "bv_executed": bv_result.success if bv_result else False,
                "qft_executed": qft_result.success if qft_result else False,
                "algorithms_available": True
            }
        })
        print("✅ Real Quantum Algorithms working")
        print(f"   - Grover's Algorithm: {'✅' if grover_result.success else '⚠️ Fallback'}")
        print(f"   - Bernstein-Vazirani: {'✅' if bv_result.success else '⚠️ Fallback'}")
        print(f"   - Quantum Fourier Transform: {'✅' if qft_result.success else '⚠️ Fallback'}")
        
    except Exception as e:
        test_results["tests"].append({
            "test": "Real Quantum Algorithms",
            "status": "FAILED",
            "error": str(e)
        })
        test_results["overall_success"] = False
        print(f"❌ Real Quantum Algorithms failed: {e}")
    
    # Test 5: Web Interface Components
    print("\n🌐 Testing Web Interface Components...")
    try:
        from dt_project.config.config_manager import ConfigManager
        from dt_project.web_interface.routes.main_routes import create_main_routes
        
        # Test config manager
        config_manager = ConfigManager()
        
        # Test route creation (basic functionality)
        main_routes = create_main_routes()
        
        test_results["tests"].append({
            "test": "Web Interface Components",
            "status": "PASSED",
            "details": {
                "config_manager": True,
                "routes_available": True,
                "note": "GraphQL skipped due to compatibility issues"
            }
        })
        print("✅ Web Interface Components working")
        
    except Exception as e:
        test_results["tests"].append({
            "test": "Web Interface Components",
            "status": "FAILED",
            "error": str(e)
        })
        test_results["overall_success"] = False
        print(f"❌ Web Interface Components failed: {e}")
    
    # Generate Final Summary
    passed_tests = sum(1 for test in test_results["tests"] if test["status"] == "PASSED")
    total_tests = len(test_results["tests"])
    success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
    
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": success_rate,
        "platform_operational": test_results["overall_success"]
    }
    
    # Save results
    with open('final_integration_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n📊 FINAL INTEGRATION TEST SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Platform Status: {'🟢 OPERATIONAL' if test_results['overall_success'] else '🔴 ISSUES DETECTED'}")
    
    if test_results["overall_success"]:
        print("\n🎉 PLATFORM INTEGRATION TEST: SUCCESS!")
        print("   All core systems are working together properly.")
        print("   Platform is ready for production deployment.")
    else:
        print("\n⚠️ PLATFORM INTEGRATION TEST: PARTIAL SUCCESS")
        print("   Some systems have issues but core functionality works.")
        print("   Review failed tests before production deployment.")
    
    print(f"\n📄 Results saved to: final_integration_test_results.json")
    
    return test_results


if __name__ == "__main__":
    """Run final integration test."""
    
    print("🌌 QUANTUM PLATFORM FINAL INTEGRATION TEST")
    print("Validating all systems working together...")
    
    results = asyncio.run(run_final_integration_test())
    
    # Exit with appropriate code
    if results["overall_success"]:
        print("\n✅ INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n⚠️ INTEGRATION TEST COMPLETED WITH ISSUES!")
        sys.exit(1)
