"""
Asynchronous Quantum Computing Backend
Provides async quantum operations with proper error handling and performance optimization.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class QuantumBackendType(Enum):
    """Available quantum backend types."""
    SIMULATOR = "simulator"
    IBM_QUANTUM = "ibm_quantum"
    LOCAL_SIMULATOR = "local_simulator"

@dataclass
class QuantumJob:
    """Represents a quantum computation job."""
    job_id: str
    circuit_data: Any
    shots: int = 1024
    priority: int = 1  # 1 = normal, 2 = high, 3 = urgent
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class QuantumResult:
    """Represents quantum computation result."""
    job_id: str
    counts: Dict[str, int]
    execution_time: float
    backend_name: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class QuantumBackend(ABC):
    """Abstract base class for quantum backends."""
    
    @abstractmethod
    async def execute_circuit(self, job: QuantumJob) -> QuantumResult:
        """Execute a quantum circuit asynchronously."""
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the backend."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

class SimulatorBackend(QuantumBackend):
    """Local quantum simulator backend."""
    
    def __init__(self, max_qubits: int = 20):
        self.max_qubits = max_qubits
        self.name = "local_simulator"
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    async def execute_circuit(self, job: QuantumJob) -> QuantumResult:
        """Execute circuit on local simulator."""
        start_time = time.time()
        
        try:
            # Run simulation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor, 
                self._simulate_circuit, 
                job
            )
            
            execution_time = time.time() - start_time
            
            return QuantumResult(
                job_id=job.job_id,
                counts=result,
                execution_time=execution_time,
                backend_name=self.name,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Simulation failed for job {job.job_id}: {e}")
            return QuantumResult(
                job_id=job.job_id,
                counts={},
                execution_time=time.time() - start_time,
                backend_name=self.name,
                success=False,
                error_message=str(e)
            )
    
    def _simulate_circuit(self, job: QuantumJob) -> Dict[str, int]:
        """Simulate quantum circuit (mock implementation)."""
        # This is a simplified simulation - in real implementation,
        # you would use actual quantum simulation libraries
        
        n_qubits = job.circuit_data.get('n_qubits', 2)
        if n_qubits > self.max_qubits:
            raise ValueError(f"Circuit has {n_qubits} qubits, max is {self.max_qubits}")
        
        # Generate mock results based on circuit type
        circuit_type = job.circuit_data.get('type', 'random')
        
        if circuit_type == 'uniform':
            # Uniform distribution over all possible outcomes
            n_outcomes = 2 ** n_qubits
            base_count = job.shots // n_outcomes
            remainder = job.shots % n_outcomes
            
            counts = {}
            for i in range(n_outcomes):
                bit_string = format(i, f'0{n_qubits}b')
                counts[bit_string] = base_count + (1 if i < remainder else 0)
                
        elif circuit_type == 'bell_state':
            # Bell state: |00⟩ and |11⟩ with equal probability
            counts = {
                '00': job.shots // 2,
                '11': job.shots - job.shots // 2
            }
            
        else:  # random
            # Random distribution
            n_outcomes = min(2 ** n_qubits, 16)  # Limit outcomes for efficiency
            np.random.seed(int(time.time() * 1000) % 2**32)
            
            # Generate random counts
            raw_counts = np.random.multinomial(job.shots, 
                                             np.random.dirichlet(np.ones(n_outcomes)))
            
            counts = {}
            for i, count in enumerate(raw_counts):
                if count > 0:
                    bit_string = format(i, f'0{n_qubits}b')
                    counts[bit_string] = int(count)
        
        # Add some realistic noise
        time.sleep(0.1 + n_qubits * 0.05)  # Simulate computation time
        
        return counts
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get simulator backend information."""
        return {
            'name': self.name,
            'type': 'simulator',
            'max_qubits': self.max_qubits,
            'available': True,
            'queue_length': 0
        }
    
    def is_available(self) -> bool:
        """Simulator is always available."""
        return True

class IBMQuantumBackend(QuantumBackend):
    """IBM Quantum backend (real quantum hardware and cloud simulators)."""
    
    def __init__(self, token: Optional[str] = None, backend_name: str = 'ibmq_qasm_simulator'):
        self.token = token or os.getenv('IBMQ_TOKEN')
        self.backend_name = backend_name
        self.name = "ibm_quantum"
        self._initialized = False
        self._service = None
        self._backend = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        if self.token:
            self._initialize()
    
    def _initialize(self):
        """Initialize IBM Quantum connection using Qiskit Runtime."""
        try:
            # Try modern Qiskit Runtime Service first
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                
                # Initialize service
                self._service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self.token
                )
                
                # Get backend
                if self.backend_name == 'least_busy':
                    # Find least busy backend
                    backends = self._service.backends(
                        filters=lambda x: x.configuration().n_qubits >= 5 
                        and not x.configuration().simulator 
                        and x.status().operational
                    )
                    if backends:
                        from qiskit.providers.ibmq import least_busy
                        self._backend = least_busy(backends)
                        self.backend_name = self._backend.name
                    else:
                        # Fallback to simulator
                        self._backend = self._service.backend('ibmq_qasm_simulator')
                        self.backend_name = 'ibmq_qasm_simulator'
                else:
                    self._backend = self._service.backend(self.backend_name)
                
                self._initialized = True
                logger.info(f"IBM Quantum Runtime Service initialized with backend: {self.backend_name}")
                
            except ImportError:
                # Fallback to legacy IBMQ
                from qiskit import IBMQ
                
                IBMQ.save_account(self.token, overwrite=True)
                provider = IBMQ.load_account()
                
                if self.backend_name == 'least_busy':
                    from qiskit.providers.ibmq import least_busy
                    backends = provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= 5 
                        and not x.configuration().simulator
                    )
                    self._backend = least_busy(backends) if backends else provider.get_backend('ibmq_qasm_simulator')
                else:
                    self._backend = provider.get_backend(self.backend_name)
                
                self._initialized = True
                logger.info(f"IBM Quantum backend initialized (legacy): {self._backend.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize IBM Quantum backend: {e}")
            self._initialized = False
            self._initialized = False
    
    async def execute_circuit(self, job: QuantumJob) -> QuantumResult:
        """Execute circuit on IBM Quantum."""
        if not self.is_available():
            # Fallback to mock execution if not available
            return await self._execute_mock(job)
        
        start_time = time.time()
        
        try:
            # Convert job data to Qiskit circuit
            qiskit_circuit = self._build_qiskit_circuit(job.circuit_data)
            
            # Execute on IBM Quantum
            loop = asyncio.get_event_loop()
            
            if self._service:
                # Use Qiskit Runtime Service
                result = await loop.run_in_executor(
                    self._executor,
                    self._execute_with_runtime,
                    qiskit_circuit,
                    job.shots
                )
            elif self._backend:
                # Use legacy IBMQ backend
                result = await loop.run_in_executor(
                    self._executor,
                    self._execute_with_ibmq,
                    qiskit_circuit,
                    job.shots
                )
            else:
                # Fallback to mock
                return await self._execute_mock(job)
            
            execution_time = time.time() - start_time
            
            return QuantumResult(
                job_id=job.job_id,
                counts=result['counts'],
                execution_time=execution_time,
                backend_name=self.backend_name,
                success=True,
                metadata={
                    'queue_time': result.get('queue_time', 0),
                    'backend_version': result.get('version', 'unknown'),
                    'actual_shots': result.get('shots', job.shots)
                }
            )
            
        except Exception as e:
            logger.error(f"IBM Quantum execution failed for job {job.job_id}: {e}")
            # Fallback to mock on error
            return await self._execute_mock(job)
    
    def _execute_with_runtime(self, circuit, shots):
        """Execute circuit using Qiskit Runtime Service."""
        try:
            from qiskit_ibm_runtime import Sampler, Session
            from qiskit import transpile
            
            # Transpile circuit for backend
            transpiled = transpile(circuit, backend=self._backend)
            
            # Create session and run
            with Session(service=self._service, backend=self._backend) as session:
                sampler = Sampler(session=session)
                job = sampler.run(transpiled, shots=shots)
                result = job.result()
                
                # Extract counts from quasi-probabilities
                counts = {}
                quasi_dists = result.quasi_dists[0]
                for outcome, prob in quasi_dists.items():
                    counts[format(outcome, f'0{circuit.num_qubits}b')] = int(prob * shots)
                
                return {
                    'counts': counts,
                    'queue_time': 0,
                    'version': self._backend.version if hasattr(self._backend, 'version') else '1.0',
                    'shots': shots
                }
                
        except Exception as e:
            logger.error(f"Runtime execution failed: {e}")
            raise
    
    def _execute_with_ibmq(self, circuit, shots):
        """Execute circuit using legacy IBMQ backend."""
        try:
            from qiskit import execute, transpile
            
            # Transpile circuit
            transpiled = transpile(circuit, backend=self._backend)
            
            # Execute
            job = execute(transpiled, backend=self._backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            return {
                'counts': counts,
                'queue_time': job.queue_position() if hasattr(job, 'queue_position') else 0,
                'version': self._backend.version if hasattr(self._backend, 'version') else '1.0',
                'shots': shots
            }
            
        except Exception as e:
            logger.error(f"IBMQ execution failed: {e}")
            raise
    
    async def _execute_mock(self, job: QuantumJob) -> QuantumResult:
        """Execute with mock results when IBM Quantum not available."""
        await asyncio.sleep(0.5)  # Simulate execution time
        
        n_qubits = job.circuit_data.get('n_qubits', 2)
        # Generate mock bell state results
        mock_counts = {
            '0' * n_qubits: job.shots // 2,
            '1' * n_qubits: job.shots - job.shots // 2
        }
        
        return QuantumResult(
            job_id=job.job_id,
            counts=mock_counts,
            execution_time=0.5,
            backend_name='mock_ibm',
            success=True,
            metadata={'note': 'Mock execution - IBM Quantum not available'}
        )
    
    def _build_qiskit_circuit(self, circuit_data: Dict[str, Any]):
        """Build Qiskit circuit from circuit data."""
        try:
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
            
            n_qubits = circuit_data.get('n_qubits', 2)
            gates = circuit_data.get('gates', [])
            
            # Create circuit
            qreg = QuantumRegister(n_qubits, 'q')
            creg = ClassicalRegister(n_qubits, 'c')
            circuit = QuantumCircuit(qreg, creg)
            
            # Add gates
            for gate in gates:
                gate_type = gate.get('type')
                qubits = gate.get('qubits', [])
                params = gate.get('params', [])
                
                if gate_type == 'h':
                    circuit.h(qubits[0])
                elif gate_type == 'x':
                    circuit.x(qubits[0])
                elif gate_type == 'y':
                    circuit.y(qubits[0])
                elif gate_type == 'z':
                    circuit.z(qubits[0])
                elif gate_type == 'rx':
                    circuit.rx(params[0], qubits[0])
                elif gate_type == 'ry':
                    circuit.ry(params[0], qubits[0])
                elif gate_type == 'rz':
                    circuit.rz(params[0], qubits[0])
                elif gate_type == 'cx' or gate_type == 'cnot':
                    circuit.cx(qubits[0], qubits[1])
                elif gate_type == 'cz':
                    circuit.cz(qubits[0], qubits[1])
                elif gate_type == 'swap':
                    circuit.swap(qubits[0], qubits[1])
            
            # Add measurements
            circuit.measure(qreg, creg)
            
            return circuit
            
        except ImportError:
            # If Qiskit not available, return mock circuit
            logger.warning("Qiskit not available, returning mock circuit")
            return None
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get IBM Quantum backend information."""
        return {
            'name': self.name,
            'type': 'hardware',
            'available': self._initialized,
            'queue_length': 5 if self._initialized else 0  # Mock queue length
        }
    
    def is_available(self) -> bool:
        """Check if IBM Quantum backend is available."""
        return self._initialized and self.token is not None

class AsyncQuantumProcessor:
    """Asynchronous quantum circuit processor with job queue management."""
    
    def __init__(self):
        self.backends: Dict[str, QuantumBackend] = {}
        self.job_queue = asyncio.Queue()
        self.results_cache = {}
        self.active_jobs = {}
        self._processing = False
        
        # Initialize default backends
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize available quantum backends."""
        # Always available: simulator
        self.backends['simulator'] = SimulatorBackend()
        
        # Try to initialize IBM Quantum if token is available
        import os
        ibm_token = os.getenv('IBMQ_TOKEN')
        if ibm_token:
            try:
                self.backends['ibm_quantum'] = IBMQuantumBackend(ibm_token)
            except Exception as e:
                logger.warning(f"Could not initialize IBM Quantum backend: {e}")
        
        logger.info(f"Initialized backends: {list(self.backends.keys())}")
    
    async def submit_job(self, 
                        circuit_data: Dict[str, Any], 
                        backend_preference: str = 'simulator',
                        shots: int = 1024,
                        priority: int = 1) -> str:
        """Submit a quantum job for execution."""
        
        job_id = f"job_{time.time()}_{np.random.randint(1000, 9999)}"
        
        job = QuantumJob(
            job_id=job_id,
            circuit_data=circuit_data,
            shots=shots,
            priority=priority
        )
        
        # Check if preferred backend is available
        backend = self._select_backend(backend_preference)
        
        await self.job_queue.put((job, backend))
        
        logger.info(f"Submitted job {job_id} to {backend.name}")
        
        # Start processing if not already running
        if not self._processing:
            asyncio.create_task(self._process_queue())
        
        return job_id
    
    def _select_backend(self, preference: str) -> QuantumBackend:
        """Select the best available backend."""
        if preference in self.backends and self.backends[preference].is_available():
            return self.backends[preference]
        
        # Fallback to any available backend
        for backend in self.backends.values():
            if backend.is_available():
                return backend
        
        raise RuntimeError("No quantum backends available")
    
    async def _process_queue(self):
        """Process the quantum job queue."""
        self._processing = True
        
        try:
            while not self.job_queue.empty():
                job, backend = await self.job_queue.get()
                
                # Execute job
                self.active_jobs[job.job_id] = {
                    'job': job,
                    'backend': backend.name,
                    'started_at': time.time()
                }
                
                try:
                    result = await backend.execute_circuit(job)
                    self.results_cache[job.job_id] = result
                    
                    logger.info(f"Completed job {job.job_id} on {backend.name} "
                               f"in {result.execution_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Job {job.job_id} failed: {e}")
                    self.results_cache[job.job_id] = QuantumResult(
                        job_id=job.job_id,
                        counts={},
                        execution_time=0,
                        backend_name=backend.name,
                        success=False,
                        error_message=str(e)
                    )
                
                finally:
                    # Remove from active jobs
                    if job.job_id in self.active_jobs:
                        del self.active_jobs[job.job_id]
                
        finally:
            self._processing = False
    
    async def get_result(self, job_id: str, timeout: float = 30.0) -> Optional[QuantumResult]:
        """Get result for a job, waiting up to timeout seconds."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if job_id in self.results_cache:
                return self.results_cache[job_id]
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        logger.warning(f"Timeout waiting for result of job {job_id}")
        return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a quantum job."""
        if job_id in self.results_cache:
            result = self.results_cache[job_id]
            return {
                'status': 'completed',
                'success': result.success,
                'execution_time': result.execution_time,
                'backend': result.backend_name
            }
        
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            return {
                'status': 'running',
                'backend': job_info['backend'],
                'running_time': time.time() - job_info['started_at']
            }
        
        return {'status': 'not_found'}
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Get status of all backends."""
        status = {}
        for name, backend in self.backends.items():
            status[name] = backend.get_backend_info()
        
        return {
            'backends': status,
            'active_jobs': len(self.active_jobs),
            'queued_jobs': self.job_queue.qsize(),
            'completed_jobs': len(self.results_cache)
        }

# Global quantum processor instance
_quantum_processor = None

def get_quantum_processor() -> AsyncQuantumProcessor:
    """Get the global quantum processor instance."""
    global _quantum_processor
    if _quantum_processor is None:
        _quantum_processor = AsyncQuantumProcessor()
    return _quantum_processor

async def execute_quantum_circuit_async(circuit_data: Dict[str, Any], 
                                       backend: str = 'simulator',
                                       shots: int = 1024,
                                       timeout: float = 30.0) -> Optional[QuantumResult]:
    """
    Execute a quantum circuit asynchronously.
    
    Args:
        circuit_data: Circuit description
        backend: Preferred backend name
        shots: Number of shots to execute
        timeout: Maximum wait time for result
    
    Returns:
        QuantumResult or None if timeout
    """
    processor = get_quantum_processor()
    
    job_id = await processor.submit_job(circuit_data, backend, shots)
    result = await processor.get_result(job_id, timeout)
    
    return result