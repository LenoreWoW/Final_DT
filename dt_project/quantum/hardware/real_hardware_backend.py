#!/usr/bin/env python3
"""
Quantum Backend Integration
Provides quantum simulation via Qiskit Aer simulator.
"""

import os
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

# Qiskit Aer imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False
    NoiseModel = Any
    QuantumCircuit = Any

# Rigetti imports
try:
    from pyquil import Program, get_qc
    from pyquil.gates import *
    from pyquil.api import QVMConnection, QPUConnection
    from pyquil.quil import address_qubits
    from pyquil.noise import decoherence_noise_with_asymmetric_ro
    RIGETTI_AVAILABLE = True
except ImportError:
    RIGETTI_AVAILABLE = False
    Program = Any

# IonQ imports
try:
    import requests
    IONQ_AVAILABLE = True
except ImportError:
    IONQ_AVAILABLE = False

@dataclass
class HardwareJob:
    """Container for quantum hardware job information"""
    job_id: str
    backend_name: str
    provider: str
    circuit: Any
    status: str
    created_at: datetime
    shots: int
    result: Optional[Dict] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    queue_position: Optional[int] = None

@dataclass
class BackendInfo:
    """Information about quantum hardware backend"""
    name: str
    provider: str
    num_qubits: int
    coupling_map: List[List[int]]
    gate_times: Dict[str, float]
    gate_errors: Dict[str, float]
    readout_error: float
    coherence_times: Dict[str, float]
    availability: bool
    queue_length: int

class QuantumHardwareManager:
    """Manage connections to quantum simulation backends"""

    def __init__(self):
        self.providers = {}
        self.backends = {}
        self.jobs = {}
        self.noise_models = {}

        # Initialize providers
        self._initialize_aer()
        self._initialize_rigetti()
        self._initialize_ionq()

    def _initialize_aer(self):
        """Initialize Qiskit Aer Simulator"""
        if not AER_AVAILABLE:
            print("Qiskit Aer not available")
            return

        try:
            simulator = AerSimulator()

            self.providers['aer'] = simulator

            backend_info = BackendInfo(
                name='aer_simulator',
                provider='aer',
                num_qubits=30,
                coupling_map=[],
                gate_times={},
                gate_errors={},
                readout_error=0.0,
                coherence_times={},
                availability=True,
                queue_length=0
            )
            self.backends['aer_simulator'] = {
                'backend': simulator,
                'info': backend_info,
                'provider': 'aer'
            }

            print("Aer Simulator initialized")

        except Exception as e:
            print(f"Failed to initialize Aer Simulator: {e}")
    
    def _initialize_rigetti(self):
        """Initialize Rigetti QCS provider"""
        if not RIGETTI_AVAILABLE:
            print("⚠️ Rigetti PyQuil not available")
            return
        
        try:
            # Check for Rigetti credentials
            if not os.environ.get('RIGETTI_API_KEY'):
                print("⚠️ Rigetti API key not found in environment")
                return
            
            # Get available quantum computers
            qc = get_qc('9q-square-qvm')  # Use QVM for testing
            
            self.providers['rigetti'] = qc
            
            # Add Rigetti backends
            rigetti_backends = ['9q-square-qvm', 'Aspen-11', 'Aspen-M-1']
            for backend_name in rigetti_backends:
                try:
                    backend = get_qc(backend_name)
                    backend_info = self._get_rigetti_backend_info(backend, backend_name)
                    self.backends[f"rigetti_{backend_name}"] = {
                        'backend': backend,
                        'info': backend_info,
                        'provider': 'rigetti'
                    }
                except Exception as e:
                    print(f"⚠️ Could not connect to Rigetti {backend_name}: {e}")
            
            print(f"✅ Connected to Rigetti with {len([b for b in self.backends.keys() if b.startswith('rigetti_')])} backends")
            
        except Exception as e:
            print(f"❌ Failed to connect to Rigetti: {e}")
    
    def _initialize_ionq(self):
        """Initialize IonQ provider"""
        if not IONQ_AVAILABLE:
            print("⚠️ IonQ requests not available")
            return
        
        try:
            api_key = os.environ.get('IONQ_API_KEY')
            if not api_key:
                print("⚠️ IonQ API key not found in environment")
                return
            
            # Test IonQ connection
            headers = {
                'Authorization': f'apiKey {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get('https://api.ionq.co/v0.3/backends', headers=headers)
            if response.status_code == 200:
                ionq_backends = response.json()
                
                for backend_data in ionq_backends:
                    backend_name = backend_data['backend']
                    backend_info = self._get_ionq_backend_info(backend_data)
                    self.backends[f"ionq_{backend_name}"] = {
                        'backend': backend_data,
                        'info': backend_info,
                        'provider': 'ionq',
                        'api_key': api_key
                    }
                
                print(f"✅ Connected to IonQ with {len([b for b in self.backends.keys() if b.startswith('ionq_')])} backends")
            else:
                print(f"❌ Failed to connect to IonQ: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Failed to connect to IonQ: {e}")
    
    def _get_rigetti_backend_info(self, backend, backend_name: str) -> BackendInfo:
        """Get Rigetti backend information"""
        # Mock implementation - would need actual Rigetti API calls
        return BackendInfo(
            name=backend_name,
            provider='rigetti',
            num_qubits=9 if '9q' in backend_name else 32,
            coupling_map=[[0,1], [1,2], [0,3], [1,4], [2,5], [3,4], [4,5]],
            gate_times={'RX': 25e-9, 'RY': 25e-9, 'CZ': 150e-9},
            gate_errors={'RX': 0.001, 'RY': 0.001, 'CZ': 0.02},
            readout_error=0.05,
            coherence_times={'T1': 20e-6, 'T2': 15e-6},
            availability=True,
            queue_length=0
        )
    
    def _get_ionq_backend_info(self, backend_data: Dict) -> BackendInfo:
        """Get IonQ backend information"""
        return BackendInfo(
            name=backend_data['backend'],
            provider='ionq',
            num_qubits=backend_data.get('qubits', 32),
            coupling_map=[[i, j] for i in range(32) for j in range(i+1, 32)],  # All-to-all
            gate_times={'single': 10e-6, 'two': 200e-6},
            gate_errors={'single': 0.0001, 'two': 0.005},
            readout_error=0.002,
            coherence_times={'T1': 1e10, 'T2': 1e10},  # Ion trap - very long coherence
            availability=backend_data.get('status') == 'available',
            queue_length=backend_data.get('queue_length', 0)
        )
    
    def list_backends(self) -> Dict[str, BackendInfo]:
        """List all available quantum hardware backends"""
        return {name: info['info'] for name, info in self.backends.items()}
    
    def get_backend_info(self, backend_name: str) -> Optional[BackendInfo]:
        """Get information about a specific backend"""
        if backend_name in self.backends:
            return self.backends[backend_name]['info']
        return None
    
    def get_best_backend(self, num_qubits: int, provider: Optional[str] = None) -> Optional[str]:
        """Find the best available backend for given requirements"""
        candidates = []
        
        for name, backend_data in self.backends.items():
            info = backend_data['info']
            
            # Filter by provider if specified
            if provider and not name.startswith(f"{provider}_"):
                continue
            
            # Must have enough qubits and be available
            if info.num_qubits >= num_qubits and info.availability:
                # Score based on queue length, gate errors, etc.
                score = (
                    -info.queue_length * 10 +  # Prefer shorter queues
                    -info.readout_error * 1000 +  # Prefer lower readout error
                    info.num_qubits  # Prefer more qubits (flexibility)
                )
                candidates.append((name, score))
        
        if candidates:
            # Return backend with highest score
            return max(candidates, key=lambda x: x[1])[0]
        
        return None
    
    async def submit_job(
        self, 
        circuit: Union[QuantumCircuit, Program, str], 
        backend_name: str, 
        shots: int = 1024,
        optimization_level: int = 1
    ) -> HardwareJob:
        """Submit a job to quantum hardware"""
        
        if backend_name not in self.backends:
            raise ValueError(f"Backend {backend_name} not available")
        
        backend_data = self.backends[backend_name]
        provider = backend_data['provider']
        
        try:
            if provider == 'aer':
                return await self._submit_aer_job(circuit, backend_data, shots, optimization_level)
            elif provider == 'rigetti':
                return await self._submit_rigetti_job(circuit, backend_data, shots)
            elif provider == 'ionq':
                return await self._submit_ionq_job(circuit, backend_data, shots)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            # Create failed job record
            job_id = f"failed_{int(time.time())}"
            return HardwareJob(
                job_id=job_id,
                backend_name=backend_name,
                provider=provider,
                circuit=circuit,
                status='FAILED',
                created_at=datetime.now(),
                shots=shots,
                error=str(e)
            )
    
    async def _submit_aer_job(
        self,
        circuit: QuantumCircuit,
        backend_data: Dict,
        shots: int,
        optimization_level: int
    ) -> HardwareJob:
        """Submit job to Aer Simulator"""

        backend = backend_data['backend']

        # Transpile circuit for simulator
        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level
        )

        # Submit job
        job = backend.run(transpiled, shots=shots)

        # Create job record
        hardware_job = HardwareJob(
            job_id=job.job_id(),
            backend_name='aer_simulator',
            provider='aer',
            circuit=transpiled,
            status='COMPLETED',
            created_at=datetime.now(),
            shots=shots,
            result=self._format_aer_result(job.result())
        )

        self.jobs[hardware_job.job_id] = {
            'hardware_job': hardware_job,
            'provider_job': job
        }

        return hardware_job
    
    async def _submit_rigetti_job(
        self, 
        circuit: Program, 
        backend_data: Dict, 
        shots: int
    ) -> HardwareJob:
        """Submit job to Rigetti QCS"""
        
        backend = backend_data['backend']
        
        # Wrap and execute
        executable = backend.compiler.native_quil_to_executable(circuit)
        job = backend.run(executable, shots=shots)
        
        job_id = f"rigetti_{int(time.time())}_{hash(str(circuit))}"
        
        hardware_job = HardwareJob(
            job_id=job_id,
            backend_name=backend_data['info'].name,
            provider='rigetti',
            circuit=circuit,
            status='RUNNING',
            created_at=datetime.now(),
            shots=shots,
            result=job  # Rigetti returns results immediately
        )
        
        self.jobs[job_id] = {
            'hardware_job': hardware_job,
            'provider_job': job
        }
        
        return hardware_job
    
    async def _submit_ionq_job(
        self, 
        circuit: Dict, 
        backend_data: Dict, 
        shots: int
    ) -> HardwareJob:
        """Submit job to IonQ"""
        
        backend = backend_data['backend']
        api_key = backend_data['api_key']
        
        headers = {
            'Authorization': f'apiKey {api_key}',
            'Content-Type': 'application/json'
        }
        
        job_data = {
            'target': backend['backend'],
            'shots': shots,
            'input': circuit
        }
        
        response = requests.post(
            'https://api.ionq.co/v0.3/jobs',
            headers=headers,
            json=job_data
        )
        
        if response.status_code == 200:
            job_info = response.json()
            job_id = job_info['id']
            
            hardware_job = HardwareJob(
                job_id=job_id,
                backend_name=backend['backend'],
                provider='ionq',
                circuit=circuit,
                status='SUBMITTED',
                created_at=datetime.now(),
                shots=shots
            )
            
            self.jobs[job_id] = {
                'hardware_job': hardware_job,
                'provider_job': job_info
            }
            
            return hardware_job
        else:
            raise Exception(f"IonQ job submission failed: {response.text}")
    
    async def get_job_status(self, job_id: str) -> Optional[HardwareJob]:
        """Get status of a submitted job"""
        
        if job_id not in self.jobs:
            return None
        
        job_data = self.jobs[job_id]
        hardware_job = job_data['hardware_job']
        provider_job = job_data['provider_job']
        
        try:
            if hardware_job.provider == 'aer':
                # Aer simulator jobs complete immediately
                hardware_job.status = 'COMPLETED'
                if hardware_job.result is None:
                    result = provider_job.result()
                    hardware_job.result = self._format_aer_result(result)

            elif hardware_job.provider == 'rigetti':
                # Rigetti jobs complete immediately in QVM
                hardware_job.status = 'COMPLETED'
                if hardware_job.result is None:
                    hardware_job.result = self._format_rigetti_result(provider_job)
                    
            elif hardware_job.provider == 'ionq':
                # Check IonQ job status via API
                api_key = self.backends[f"ionq_{hardware_job.backend_name}"]['api_key']
                headers = {'Authorization': f'apiKey {api_key}'}
                
                response = requests.get(
                    f'https://api.ionq.co/v0.3/jobs/{job_id}',
                    headers=headers
                )
                
                if response.status_code == 200:
                    job_info = response.json()
                    hardware_job.status = job_info['status'].upper()
                    
                    if hardware_job.status == 'COMPLETED':
                        hardware_job.result = self._format_ionq_result(job_info)
                    elif hardware_job.status == 'FAILED':
                        hardware_job.error = job_info.get('failure', {}).get('error', 'Unknown error')
            
            return hardware_job
            
        except Exception as e:
            hardware_job.status = 'ERROR'
            hardware_job.error = str(e)
            return hardware_job
    
    def _format_aer_result(self, result) -> Dict:
        """Format Aer Simulator result"""
        counts = result.get_counts()
        return {
            'counts': counts,
            'shots': sum(counts.values()),
            'success': True,
            'backend': 'aer_simulator'
        }
    
    def _format_rigetti_result(self, result) -> Dict:
        """Format Rigetti result"""
        if hasattr(result, 'shape'):
            # Convert numpy array to counts
            unique, counts = np.unique(result, axis=0, return_counts=True)
            count_dict = {}
            for bitstring, count in zip(unique, counts):
                key = ''.join(map(str, bitstring))
                count_dict[key] = int(count)
        else:
            count_dict = {'0': 1}  # Default for single execution
        
        return {
            'counts': count_dict,
            'shots': sum(count_dict.values()),
            'success': True,
            'backend': 'rigetti_qvm'
        }
    
    def _format_ionq_result(self, job_info: Dict) -> Dict:
        """Format IonQ result"""
        return {
            'counts': job_info.get('data', {}).get('histogram', {}),
            'shots': job_info.get('shots', 0),
            'success': True,
            'backend': job_info.get('target', 'ionq'),
            'job_id': job_info.get('id')
        }
    
    async def wait_for_job(self, job_id: str, timeout: int = 3600) -> HardwareJob:
        """Wait for job completion with timeout"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job = await self.get_job_status(job_id)
            
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status in ['COMPLETED', 'DONE']:
                return job
            elif job.status in ['FAILED', 'ERROR', 'CANCELLED']:
                raise JobError(f"Job failed with status {job.status}: {job.error}")
            
            # Wait before checking again
            await asyncio.sleep(10)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def get_noise_model(self, backend_name: str) -> Optional[NoiseModel]:
        """Get noise model for backend simulation"""
        
        if backend_name not in self.backends:
            return None
        
        backend_data = self.backends[backend_name]
        
        if backend_data['provider'] == 'aer' and backend_name not in self.noise_models:
            # Aer simulator can optionally use noise models
            # but by default runs ideal simulation
            pass
        
        return self.noise_models.get(backend_name)
    
    async def run_with_error_mitigation(
        self, 
        circuit: QuantumCircuit, 
        backend_name: str, 
        shots: int = 1024
    ) -> Dict:
        """Run circuit with readout error mitigation"""
        
        # For Aer simulator, run directly (no error mitigation needed for ideal simulation)
        job = await self.submit_job(circuit, backend_name, shots)
        return await self.wait_for_job(job.job_id)

# Global hardware manager instance
hardware_manager = QuantumHardwareManager()

if __name__ == "__main__":
    import asyncio

    async def test_simulator_connections():
        """Test connections to quantum simulator backends"""
        print("Testing Quantum Simulator Connections")
        print("=" * 50)

        # List available backends
        backends = hardware_manager.list_backends()

        if not backends:
            print("No quantum backends available")
            return

        print(f"Found {len(backends)} quantum backends:")

        for name, info in backends.items():
            status = "Available" if info.availability else "Unavailable"
            print(f"   {status} - {name}: {info.num_qubits} qubits")

        # Test with a simple circuit on Aer
        if 'aer_simulator' in backends and AER_AVAILABLE:
            print(f"\nTesting job submission to aer_simulator...")

            from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

            qr = QuantumRegister(2)
            cr = ClassicalRegister(2)
            circuit = QuantumCircuit(qr, cr)
            circuit.h(qr[0])
            circuit.cx(qr[0], qr[1])
            circuit.measure(qr, cr)

            try:
                job = await hardware_manager.submit_job(circuit, 'aer_simulator', shots=100)
                print(f"Job submitted: {job.job_id}")
                print(f"   Status: {job.status}")

                if job.result:
                    print(f"   Results: {job.result.get('counts', {})}")

            except Exception as e:
                print(f"Job submission failed: {e}")

    # Run the test
    asyncio.run(test_simulator_connections())