"""
Celery application configuration for async task processing.
Handles quantum computations, simulations, and background tasks.
"""

import os
import ssl
from celery import Celery
from celery.schedules import crontab
from datetime import timedelta
import structlog
from kombu import Queue

from dt_project.config.secure_config import get_config

logger = structlog.get_logger(__name__)

def make_celery(app_name='quantum_trail'):
    """Create and configure Celery application."""
    config = get_config()
    
    # Redis configuration with security
    redis_url = config.get('REDIS.URL', 'redis://localhost:6379/0')
    
    # Create Celery instance
    celery = Celery(app_name)
    
    # Celery configuration
    celery.conf.update(
        # Broker and result backend
        broker_url=redis_url,
        result_backend=redis_url,
        
        # Security settings
        broker_use_ssl={
            'ssl_cert_reqs': ssl.CERT_NONE,
            'ssl_ca_certs': None,
            'ssl_certfile': None,
            'ssl_keyfile': None,
        } if config.get('REDIS.USE_SSL', False) else None,
        
        # Task serialization
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        
        # Task routing and queues
        task_routes={
            'dt_project.tasks.quantum.*': {'queue': 'quantum'},
            'dt_project.tasks.simulation.*': {'queue': 'simulation'},
            'dt_project.tasks.ml.*': {'queue': 'ml'},
            'dt_project.tasks.monitoring.*': {'queue': 'monitoring'},
        },
        
        # Define queues with priorities
        task_default_queue='default',
        task_queues=(
            Queue('default', routing_key='default'),
            Queue('quantum', routing_key='quantum', delivery_mode=2),
            Queue('simulation', routing_key='simulation', delivery_mode=2),
            Queue('ml', routing_key='ml', delivery_mode=2),
            Queue('monitoring', routing_key='monitoring', delivery_mode=1),
            Queue('high_priority', routing_key='high_priority', delivery_mode=2),
        ),
        
        # Task execution
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_reject_on_worker_lost=True,
        
        # Result settings
        result_expires=3600,  # 1 hour
        result_compression='gzip',
        
        # Worker settings
        worker_concurrency=4,
        worker_max_tasks_per_child=1000,
        worker_disable_rate_limits=False,
        
        # Beat schedule for periodic tasks
        beat_schedule={
            'cleanup-expired-results': {
                'task': 'dt_project.tasks.monitoring.cleanup_expired_results',
                'schedule': crontab(minute=0),  # Every hour
            },
            'health-check': {
                'task': 'dt_project.tasks.monitoring.system_health_check',
                'schedule': timedelta(minutes=5),
            },
            'quantum-backend-status': {
                'task': 'dt_project.tasks.quantum.check_backend_status',
                'schedule': timedelta(minutes=10),
            },
            'process-queued-simulations': {
                'task': 'dt_project.tasks.simulation.process_queued_simulations',
                'schedule': timedelta(minutes=2),
            },
        },
        
        # Error handling
        task_soft_time_limit=300,  # 5 minutes
        task_time_limit=600,       # 10 minutes hard limit
        task_max_retries=3,
        task_default_retry_delay=60,
        
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Security
        worker_hijack_root_logger=False,
        worker_log_color=False,
    )
    
    logger.info("Celery application configured", 
                broker=redis_url, 
                queues=['default', 'quantum', 'simulation', 'ml', 'monitoring'])
    
    return celery

# Create global Celery instance
celery_app = make_celery()

class CeleryConfig:
    """Centralized Celery configuration class."""
    
    @staticmethod
    def get_queue_info():
        """Get information about configured queues."""
        return {
            'default': {'description': 'General purpose tasks', 'priority': 'normal'},
            'quantum': {'description': 'Quantum circuit execution and optimization', 'priority': 'high'},
            'simulation': {'description': 'Digital twin simulations', 'priority': 'high'},
            'ml': {'description': 'Machine learning pipelines', 'priority': 'normal'},
            'monitoring': {'description': 'System monitoring and cleanup', 'priority': 'low'},
            'high_priority': {'description': 'Critical tasks requiring immediate attention', 'priority': 'critical'}
        }
    
    @staticmethod
    def get_task_info():
        """Get information about registered tasks."""
        return {
            'quantum': [
                'execute_quantum_circuit',
                'run_optimization_algorithm',
                'check_backend_status',
                'process_quantum_job_queue'
            ],
            'simulation': [
                'create_quantum_twin',
                'update_twin_from_sensors',
                'run_simulation_step',
                'process_queued_simulations'
            ],
            'ml': [
                'train_quantum_ml_model',
                'predict_with_quantum_model',
                'optimize_hyperparameters'
            ],
            'monitoring': [
                'cleanup_expired_results',
                'system_health_check',
                'generate_performance_report'
            ]
        }

# Task decorators with common configurations
def quantum_task(func):
    """Decorator for quantum computing tasks."""
    return celery_app.task(
        bind=True,
        queue='quantum',
        max_retries=2,
        default_retry_delay=30,
        soft_time_limit=180,
        time_limit=300
    )(func)

def simulation_task(func):
    """Decorator for simulation tasks."""
    return celery_app.task(
        bind=True,
        queue='simulation',
        max_retries=3,
        default_retry_delay=60,
        soft_time_limit=240,
        time_limit=400
    )(func)

def ml_task(func):
    """Decorator for machine learning tasks."""
    return celery_app.task(
        bind=True,
        queue='ml',
        max_retries=2,
        default_retry_delay=120,
        soft_time_limit=600,
        time_limit=900
    )(func)

def monitoring_task(func):
    """Decorator for monitoring tasks."""
    return celery_app.task(
        bind=True,
        queue='monitoring',
        max_retries=1,
        default_retry_delay=300
    )(func)

# Task result management
class TaskResultManager:
    """Manages task results and provides utilities for task monitoring."""
    
    def __init__(self, celery_instance=None):
        self.celery = celery_instance or celery_app
    
    def get_task_status(self, task_id):
        """Get detailed status of a task."""
        result = self.celery.AsyncResult(task_id)
        return {
            'id': task_id,
            'status': result.status,
            'result': result.result if result.ready() else None,
            'traceback': result.traceback,
            'info': result.info,
            'successful': result.successful() if result.ready() else None,
            'failed': result.failed() if result.ready() else None,
        }
    
    def get_active_tasks(self):
        """Get list of currently active tasks."""
        inspect = self.celery.control.inspect()
        active_tasks = inspect.active()
        
        if not active_tasks:
            return []
        
        all_active = []
        for worker, tasks in active_tasks.items():
            for task in tasks:
                task['worker'] = worker
                all_active.append(task)
        
        return all_active
    
    def get_queue_lengths(self):
        """Get lengths of all queues."""
        inspect = self.celery.control.inspect()
        reserved = inspect.reserved()
        
        queue_lengths = {}
        if reserved:
            for worker, tasks in reserved.items():
                for task in tasks:
                    queue = task.get('delivery_info', {}).get('routing_key', 'unknown')
                    queue_lengths[queue] = queue_lengths.get(queue, 0) + 1
        
        return queue_lengths
    
    def purge_queue(self, queue_name):
        """Purge all tasks from a specific queue."""
        try:
            purged = self.celery.control.purge()
            logger.info(f"Purged {purged} tasks from queue {queue_name}")
            return purged
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return None
    
    def revoke_task(self, task_id, terminate=False):
        """Revoke a specific task."""
        try:
            self.celery.control.revoke(task_id, terminate=terminate)
            logger.info(f"Revoked task {task_id}, terminate={terminate}")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke task {task_id}: {e}")
            return False

# Global task result manager instance
task_manager = TaskResultManager()

# Health check utilities
def get_celery_health():
    """Get Celery cluster health information."""
    try:
        inspect = celery_app.control.inspect()
        
        # Get worker stats
        stats = inspect.stats()
        active = inspect.active()
        reserved = inspect.reserved()
        
        if not stats:
            return {
                'status': 'unhealthy',
                'message': 'No workers responding',
                'workers': 0,
                'active_tasks': 0,
                'reserved_tasks': 0
            }
        
        total_active = sum(len(tasks) for tasks in (active or {}).values())
        total_reserved = sum(len(tasks) for tasks in (reserved or {}).values())
        
        return {
            'status': 'healthy',
            'workers': len(stats),
            'active_tasks': total_active,
            'reserved_tasks': total_reserved,
            'worker_stats': stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get Celery health: {e}")
        return {
            'status': 'unhealthy',
            'message': str(e),
            'workers': 0,
            'active_tasks': 0,
            'reserved_tasks': 0
        }