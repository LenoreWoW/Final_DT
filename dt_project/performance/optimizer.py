"""
Resource Optimizer for Quantum Trail.
Handles resource allocation, load balancing, and performance optimization.
"""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import structlog

from dt_project.performance.profiler import profiler
from dt_project.performance.cache import quantum_cache, ml_cache, general_cache
from dt_project.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    QUANTUM_BACKEND = "quantum_backend"
    CELERY_WORKER = "celery_worker"

class OptimizationStrategy(Enum):
    """Resource optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

@dataclass
class ResourceLimit:
    """Resource usage limits."""
    resource_type: ResourceType
    soft_limit: float  # Warning threshold
    hard_limit: float  # Critical threshold
    unit: str = "percent"  # percent, MB, GB, count
    
@dataclass
class OptimizationAction:
    """Represents an optimization action."""
    action_type: str
    resource_type: ResourceType
    priority: int
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_impact: float = 0.0  # Expected improvement (0-1)
    cost: float = 0.0  # Cost of action (0-1)

@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_percent: float
    network_io_mbps: float
    active_connections: int
    quantum_jobs_queued: int
    celery_queue_lengths: Dict[str, int] = field(default_factory=dict)

class LoadBalancer:
    """Intelligent load balancer for distributing work across resources."""
    
    def __init__(self):
        self.backends: Dict[str, Dict[str, Any]] = {}
        self.routing_weights: Dict[str, float] = {}
        self.health_checks: Dict[str, bool] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Load balancing algorithms
        self.algorithm = "weighted_round_robin"
        self.current_backend_index = 0
        
        logger.info("Load balancer initialized")
    
    def add_backend(self, backend_id: str, endpoint: str, weight: float = 1.0, 
                   max_connections: int = 100):
        """Add a backend to the load balancer."""
        self.backends[backend_id] = {
            'endpoint': endpoint,
            'weight': weight,
            'max_connections': max_connections,
            'active_connections': 0,
            'total_requests': 0,
            'failed_requests': 0,
            'last_health_check': None
        }
        self.routing_weights[backend_id] = weight
        self.health_checks[backend_id] = True
        
        logger.info(f"Added backend {backend_id}: {endpoint}")
    
    def remove_backend(self, backend_id: str) -> bool:
        """Remove a backend from the load balancer."""
        if backend_id in self.backends:
            del self.backends[backend_id]
            del self.routing_weights[backend_id]
            del self.health_checks[backend_id]
            if backend_id in self.request_counts:
                del self.request_counts[backend_id]
            if backend_id in self.response_times:
                del self.response_times[backend_id]
            logger.info(f"Removed backend {backend_id}")
            return True
        return False
    
    def get_backend(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Select the best backend for a request."""
        available_backends = [
            backend_id for backend_id, healthy in self.health_checks.items()
            if healthy and self.backends[backend_id]['active_connections'] < self.backends[backend_id]['max_connections']
        ]
        
        if not available_backends:
            logger.warning("No available backends")
            return None
        
        if self.algorithm == "round_robin":
            return self._round_robin_selection(available_backends)
        elif self.algorithm == "weighted_round_robin":
            return self._weighted_round_robin_selection(available_backends)
        elif self.algorithm == "least_connections":
            return self._least_connections_selection(available_backends)
        elif self.algorithm == "least_response_time":
            return self._least_response_time_selection(available_backends)
        elif self.algorithm == "adaptive":
            return self._adaptive_selection(available_backends, request_context)
        else:
            return available_backends[0]
    
    def _round_robin_selection(self, backends: List[str]) -> str:
        """Simple round-robin selection."""
        backend = backends[self.current_backend_index % len(backends)]
        self.current_backend_index += 1
        return backend
    
    def _weighted_round_robin_selection(self, backends: List[str]) -> str:
        """Weighted round-robin selection based on backend weights."""
        total_weight = sum(self.routing_weights[b] for b in backends)
        cumulative_weight = 0
        selection_point = (self.current_backend_index * 0.618) % total_weight  # Golden ratio for distribution
        
        for backend_id in backends:
            cumulative_weight += self.routing_weights[backend_id]
            if selection_point <= cumulative_weight:
                self.current_backend_index += 1
                return backend_id
        
        return backends[0]
    
    def _least_connections_selection(self, backends: List[str]) -> str:
        """Select backend with fewest active connections."""
        return min(backends, key=lambda b: self.backends[b]['active_connections'])
    
    def _least_response_time_selection(self, backends: List[str]) -> str:
        """Select backend with lowest average response time."""
        def avg_response_time(backend_id: str) -> float:
            times = self.response_times[backend_id]
            return sum(times) / len(times) if times else 0.0
        
        return min(backends, key=avg_response_time)
    
    def _adaptive_selection(self, backends: List[str], request_context: Optional[Dict[str, Any]]) -> str:
        """Adaptive selection based on multiple factors."""
        scores = {}
        
        for backend_id in backends:
            backend = self.backends[backend_id]
            
            # Connection load factor
            connection_load = backend['active_connections'] / backend['max_connections']
            
            # Response time factor
            avg_response_time = 0.0
            if self.response_times[backend_id]:
                avg_response_time = sum(self.response_times[backend_id]) / len(self.response_times[backend_id])
            
            # Error rate factor
            total_requests = backend['total_requests']
            error_rate = backend['failed_requests'] / max(total_requests, 1)
            
            # Weight factor
            weight_factor = self.routing_weights[backend_id]
            
            # Combined score (lower is better)
            score = (
                connection_load * 0.3 +
                (avg_response_time / 1000) * 0.3 +  # Convert ms to seconds
                error_rate * 0.2 +
                (1 / weight_factor) * 0.2
            )
            
            scores[backend_id] = score
        
        return min(scores.items(), key=lambda x: x[1])[0]
    
    def record_request(self, backend_id: str, response_time_ms: float, success: bool):
        """Record request metrics for a backend."""
        if backend_id in self.backends:
            self.backends[backend_id]['total_requests'] += 1
            if not success:
                self.backends[backend_id]['failed_requests'] += 1
            
            self.response_times[backend_id].append(response_time_ms)
            self.request_counts[backend_id] += 1
    
    def update_backend_connections(self, backend_id: str, active_connections: int):
        """Update active connection count for a backend."""
        if backend_id in self.backends:
            self.backends[backend_id]['active_connections'] = active_connections
    
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        stats = {}
        
        for backend_id, backend in self.backends.items():
            avg_response_time = 0.0
            if self.response_times[backend_id]:
                avg_response_time = sum(self.response_times[backend_id]) / len(self.response_times[backend_id])
            
            error_rate = 0.0
            if backend['total_requests'] > 0:
                error_rate = backend['failed_requests'] / backend['total_requests']
            
            stats[backend_id] = {
                'endpoint': backend['endpoint'],
                'healthy': self.health_checks[backend_id],
                'active_connections': backend['active_connections'],
                'max_connections': backend['max_connections'],
                'total_requests': backend['total_requests'],
                'failed_requests': backend['failed_requests'],
                'error_rate': error_rate,
                'avg_response_time_ms': avg_response_time,
                'weight': self.routing_weights[backend_id]
            }
        
        return stats

class ResourceOptimizer:
    """Main resource optimizer that manages system resources and performance."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.resource_limits: Dict[ResourceType, ResourceLimit] = {}
        self.load_balancer = LoadBalancer()
        
        # Monitoring
        self.resource_history: deque = deque(maxlen=1000)
        self.optimization_actions: List[OptimizationAction] = []
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Auto-scaling
        self.auto_scaling_enabled = False
        self.scaling_cooldown = 300  # seconds
        self.last_scaling_action: Optional[datetime] = None
        
        # Default resource limits
        self._setup_default_limits()
        
        logger.info(f"Resource optimizer initialized with {strategy.value} strategy")
    
    def _setup_default_limits(self):
        """Set up default resource limits."""
        self.resource_limits = {
            ResourceType.CPU: ResourceLimit(ResourceType.CPU, 80.0, 95.0, "percent"),
            ResourceType.MEMORY: ResourceLimit(ResourceType.MEMORY, 85.0, 95.0, "percent"),
            ResourceType.DISK: ResourceLimit(ResourceType.DISK, 80.0, 90.0, "percent"),
            ResourceType.NETWORK: ResourceLimit(ResourceType.NETWORK, 80.0, 90.0, "percent"),
        }
    
    def set_resource_limit(self, resource_type: ResourceType, soft_limit: float, 
                          hard_limit: float, unit: str = "percent"):
        """Set resource limits."""
        self.resource_limits[resource_type] = ResourceLimit(resource_type, soft_limit, hard_limit, unit)
        logger.info(f"Set resource limit for {resource_type.value}: {soft_limit}/{hard_limit} {unit}")
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start resource monitoring and optimization."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Started resource monitoring with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped resource monitoring")
    
    async def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current resource usage
                usage = self._collect_resource_usage()
                self.resource_history.append(usage)
                
                # Analyze resource usage and identify issues
                issues = self._analyze_resource_usage(usage)
                
                # Generate optimization actions if needed
                if issues:
                    actions = self._generate_optimization_actions(issues, usage)
                    
                    # Execute high-priority actions
                    for action in actions:
                        if action.priority >= 8:  # High priority threshold
                            await self._execute_optimization_action(action, usage)
                
                # Auto-scaling check
                if self.auto_scaling_enabled:
                    await self._check_auto_scaling(usage)
                
                # Cache optimization
                await self._optimize_caches(usage)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(10.0)
    
    def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Calculate network throughput (rough estimate)
            network_mbps = 0.0
            if hasattr(self, '_last_network_io'):
                time_delta = time.time() - self._last_network_time
                bytes_delta = (network.bytes_sent + network.bytes_recv) - self._last_network_io
                network_mbps = (bytes_delta / time_delta) / (1024 * 1024) if time_delta > 0 else 0.0
            
            self._last_network_io = network.bytes_sent + network.bytes_recv
            self._last_network_time = time.time()
            
            # Application-specific metrics
            active_connections = len(psutil.net_connections())
            
            # Mock quantum and celery metrics (would integrate with actual systems)
            quantum_jobs_queued = 0
            celery_queue_lengths = {}
            
            return ResourceUsage(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                disk_percent=(disk.used / disk.total) * 100,
                network_io_mbps=network_mbps,
                active_connections=active_connections,
                quantum_jobs_queued=quantum_jobs_queued,
                celery_queue_lengths=celery_queue_lengths
            )
            
        except Exception as e:
            logger.error(f"Failed to collect resource usage: {e}")
            return ResourceUsage(
                timestamp=datetime.utcnow(),
                cpu_percent=0, memory_percent=0, memory_available_mb=0,
                disk_percent=0, network_io_mbps=0, active_connections=0,
                quantum_jobs_queued=0
            )
    
    def _analyze_resource_usage(self, usage: ResourceUsage) -> List[Dict[str, Any]]:
        """Analyze resource usage and identify issues."""
        issues = []
        
        # CPU analysis
        cpu_limit = self.resource_limits.get(ResourceType.CPU)
        if cpu_limit and usage.cpu_percent > cpu_limit.hard_limit:
            issues.append({
                'type': 'cpu_critical',
                'severity': 'critical',
                'value': usage.cpu_percent,
                'limit': cpu_limit.hard_limit,
                'description': f"CPU usage at {usage.cpu_percent:.1f}% exceeds critical limit"
            })
        elif cpu_limit and usage.cpu_percent > cpu_limit.soft_limit:
            issues.append({
                'type': 'cpu_warning',
                'severity': 'warning',
                'value': usage.cpu_percent,
                'limit': cpu_limit.soft_limit,
                'description': f"CPU usage at {usage.cpu_percent:.1f}% exceeds warning limit"
            })
        
        # Memory analysis
        memory_limit = self.resource_limits.get(ResourceType.MEMORY)
        if memory_limit and usage.memory_percent > memory_limit.hard_limit:
            issues.append({
                'type': 'memory_critical',
                'severity': 'critical',
                'value': usage.memory_percent,
                'limit': memory_limit.hard_limit,
                'description': f"Memory usage at {usage.memory_percent:.1f}% exceeds critical limit"
            })
        elif memory_limit and usage.memory_percent > memory_limit.soft_limit:
            issues.append({
                'type': 'memory_warning',
                'severity': 'warning',
                'value': usage.memory_percent,
                'limit': memory_limit.soft_limit,
                'description': f"Memory usage at {usage.memory_percent:.1f}% exceeds warning limit"
            })
        
        # Disk analysis
        disk_limit = self.resource_limits.get(ResourceType.DISK)
        if disk_limit and usage.disk_percent > disk_limit.hard_limit:
            issues.append({
                'type': 'disk_critical',
                'severity': 'critical',
                'value': usage.disk_percent,
                'limit': disk_limit.hard_limit,
                'description': f"Disk usage at {usage.disk_percent:.1f}% exceeds critical limit"
            })
        elif disk_limit and usage.disk_percent > disk_limit.soft_limit:
            issues.append({
                'type': 'disk_warning',
                'severity': 'warning',
                'value': usage.disk_percent,
                'limit': disk_limit.soft_limit,
                'description': f"Disk usage at {usage.disk_percent:.1f}% exceeds warning limit"
            })
        
        return issues
    
    def _generate_optimization_actions(self, issues: List[Dict[str, Any]], 
                                     usage: ResourceUsage) -> List[OptimizationAction]:
        """Generate optimization actions based on identified issues."""
        actions = []
        
        for issue in issues:
            issue_type = issue['type']
            severity = issue['severity']
            
            if 'cpu' in issue_type:
                # CPU optimization actions
                if usage.cpu_percent > 90:
                    actions.append(OptimizationAction(
                        action_type="scale_workers",
                        resource_type=ResourceType.CPU,
                        priority=9,
                        description="Scale up workers to distribute CPU load",
                        parameters={'worker_type': 'celery', 'scale_factor': 1.5},
                        estimated_impact=0.3,
                        cost=0.2
                    ))
                
                actions.append(OptimizationAction(
                    action_type="optimize_processes",
                    resource_type=ResourceType.CPU,
                    priority=7,
                    description="Optimize process priorities and affinity",
                    estimated_impact=0.15,
                    cost=0.1
                ))
            
            elif 'memory' in issue_type:
                # Memory optimization actions
                actions.append(OptimizationAction(
                    action_type="clear_caches",
                    resource_type=ResourceType.MEMORY,
                    priority=8 if severity == 'critical' else 6,
                    description="Clear cache to free memory",
                    estimated_impact=0.2,
                    cost=0.1
                ))
                
                actions.append(OptimizationAction(
                    action_type="force_garbage_collection",
                    resource_type=ResourceType.MEMORY,
                    priority=7,
                    description="Force garbage collection",
                    estimated_impact=0.1,
                    cost=0.05
                ))
            
            elif 'disk' in issue_type:
                # Disk optimization actions
                actions.append(OptimizationAction(
                    action_type="cleanup_temp_files",
                    resource_type=ResourceType.DISK,
                    priority=6,
                    description="Clean up temporary files",
                    estimated_impact=0.15,
                    cost=0.05
                ))
        
        # Sort by priority and estimated impact
        actions.sort(key=lambda a: (a.priority, a.estimated_impact), reverse=True)
        
        return actions
    
    async def _execute_optimization_action(self, action: OptimizationAction, 
                                         usage: ResourceUsage) -> bool:
        """Execute an optimization action."""
        try:
            logger.info(f"Executing optimization action: {action.description}")
            
            if action.action_type == "clear_caches":
                # Clear application caches
                quantum_cache.clear()
                ml_cache.clear()
                general_cache.clear()
                logger.info("Cleared all application caches")
            
            elif action.action_type == "force_garbage_collection":
                # Force garbage collection
                profiler.force_garbage_collection()
                logger.info("Forced garbage collection")
            
            elif action.action_type == "cleanup_temp_files":
                # Clean up temporary files
                # This would integrate with actual cleanup mechanisms
                logger.info("Cleaned up temporary files")
            
            elif action.action_type == "optimize_processes":
                # Optimize process priorities
                # This would adjust process priorities and CPU affinity
                logger.info("Optimized process priorities")
            
            elif action.action_type == "scale_workers":
                # Scale workers (would integrate with actual scaling system)
                logger.info("Initiated worker scaling")
            
            self.optimization_actions.append(action)
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute optimization action {action.action_type}: {e}")
            return False
    
    async def _check_auto_scaling(self, usage: ResourceUsage):
        """Check if auto-scaling is needed."""
        if not self.auto_scaling_enabled:
            return
        
        # Cooldown check
        if (self.last_scaling_action and 
            (datetime.utcnow() - self.last_scaling_action).total_seconds() < self.scaling_cooldown):
            return
        
        # Scale up conditions
        should_scale_up = (
            usage.cpu_percent > 80 or
            usage.memory_percent > 85 or
            usage.quantum_jobs_queued > 10
        )
        
        # Scale down conditions
        should_scale_down = (
            usage.cpu_percent < 20 and
            usage.memory_percent < 40 and
            usage.quantum_jobs_queued < 2
        )
        
        if should_scale_up:
            await self._scale_up()
        elif should_scale_down:
            await self._scale_down()
    
    async def _scale_up(self):
        """Scale up resources."""
        logger.info("Initiating scale-up operation")
        # This would integrate with container orchestration or cloud auto-scaling
        self.last_scaling_action = datetime.utcnow()
    
    async def _scale_down(self):
        """Scale down resources."""
        logger.info("Initiating scale-down operation")
        # This would integrate with container orchestration or cloud auto-scaling
        self.last_scaling_action = datetime.utcnow()
    
    async def _optimize_caches(self, usage: ResourceUsage):
        """Optimize cache configurations based on resource usage."""
        if usage.memory_percent > 85:
            # Reduce cache sizes if memory is high
            quantum_cache.max_size_bytes = int(quantum_cache.max_size_bytes * 0.8)
            ml_cache.max_size_bytes = int(ml_cache.max_size_bytes * 0.8)
            logger.info("Reduced cache sizes due to high memory usage")
        elif usage.memory_percent < 60:
            # Increase cache sizes if memory is available
            quantum_cache.max_size_bytes = int(quantum_cache.max_size_bytes * 1.1)
            ml_cache.max_size_bytes = int(ml_cache.max_size_bytes * 1.1)
            logger.info("Increased cache sizes due to available memory")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        recent_actions = [a for a in self.optimization_actions 
                         if (datetime.utcnow() - a.timestamp if hasattr(a, 'timestamp') else datetime.utcnow()).total_seconds() < 3600]
        
        current_usage = self._collect_resource_usage() if self.resource_history else None
        
        return {
            'strategy': self.strategy.value,
            'monitoring_active': self.monitoring_active,
            'auto_scaling_enabled': self.auto_scaling_enabled,
            'resource_limits': {rt.value: {'soft': rl.soft_limit, 'hard': rl.hard_limit, 'unit': rl.unit} 
                              for rt, rl in self.resource_limits.items()},
            'current_usage': {
                'cpu_percent': current_usage.cpu_percent if current_usage else 0,
                'memory_percent': current_usage.memory_percent if current_usage else 0,
                'disk_percent': current_usage.disk_percent if current_usage else 0,
                'active_connections': current_usage.active_connections if current_usage else 0
            } if current_usage else {},
            'recent_actions': len(recent_actions),
            'load_balancer_stats': self.load_balancer.get_backend_stats(),
            'optimization_history': len(self.optimization_actions)
        }

# Global resource optimizer instance
resource_optimizer = ResourceOptimizer()

# Convenience functions
async def start_optimization(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED, 
                           monitoring_interval: float = 30.0):
    """Start resource optimization."""
    resource_optimizer.strategy = strategy
    await resource_optimizer.start_monitoring(monitoring_interval)

async def stop_optimization():
    """Stop resource optimization."""
    await resource_optimizer.stop_monitoring()

def enable_auto_scaling(cooldown: int = 300):
    """Enable auto-scaling with specified cooldown."""
    resource_optimizer.auto_scaling_enabled = True
    resource_optimizer.scaling_cooldown = cooldown
    logger.info(f"Auto-scaling enabled with {cooldown}s cooldown")

def disable_auto_scaling():
    """Disable auto-scaling."""
    resource_optimizer.auto_scaling_enabled = False
    logger.info("Auto-scaling disabled")

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    return {
        'optimization_stats': resource_optimizer.get_optimization_stats(),
        'profiler_stats': profiler.get_system_stats(timedelta(hours=1)),
        'cache_stats': {
            'quantum_cache': quantum_cache.get_stats(),
            'ml_cache': ml_cache.get_stats(),
            'general_cache': general_cache.get_stats()
        }
    }