#!/usr/bin/env python3
"""
Celery worker startup script for Quantum Trail.
Starts Celery workers with proper configuration and task discovery.
"""

import os
import sys
import signal
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dt_project.celery_app import celery_app, get_celery_health
from dt_project.config.secure_config import get_config
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        celery_app.control.broadcast('shutdown')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point for Celery worker."""
    parser = argparse.ArgumentParser(description='Quantum Trail Celery Worker')
    parser.add_argument('--loglevel', default='info', 
                       choices=['debug', 'info', 'warning', 'error', 'critical'],
                       help='Logging level')
    parser.add_argument('--concurrency', type=int, default=4,
                       help='Number of concurrent worker processes')
    parser.add_argument('--queues', default='default,quantum,simulation,ml,monitoring',
                       help='Comma-separated list of queues to consume from')
    parser.add_argument('--hostname', default=None,
                       help='Custom hostname for this worker')
    parser.add_argument('--autoscale', default=None,
                       help='Enable autoscaling (format: max,min)')
    parser.add_argument('--beat', action='store_true',
                       help='Also run Celery beat scheduler')
    parser.add_argument('--flower', action='store_true',
                       help='Also start Flower monitoring')
    parser.add_argument('--health-check', action='store_true',
                       help='Run health check and exit')
    
    args = parser.parse_args()
    
    # Health check mode
    if args.health_check:
        logger.info("Running Celery health check...")
        health = get_celery_health()
        print(f"Celery Health Status: {health['status']}")
        print(f"Workers: {health['workers']}")
        print(f"Active Tasks: {health['active_tasks']}")
        print(f"Reserved Tasks: {health['reserved_tasks']}")
        sys.exit(0 if health['status'] == 'healthy' else 1)
    
    # Set up signal handlers
    setup_signal_handlers()
    
    # Validate configuration
    config = get_config()
    redis_url = config.get('REDIS.URL', 'redis://localhost:6379/0')
    
    logger.info("Starting Celery worker", 
                redis_url=redis_url,
                queues=args.queues,
                concurrency=args.concurrency,
                loglevel=args.loglevel)
    
    # Prepare worker arguments
    worker_args = [
        '--loglevel', args.loglevel,
        '--concurrency', str(args.concurrency),
        '--queues', args.queues,
        '--without-gossip',
        '--without-mingle',
        '--without-heartbeat'
    ]
    
    if args.hostname:
        worker_args.extend(['--hostname', args.hostname])
    
    if args.autoscale:
        worker_args.extend(['--autoscale', args.autoscale])
        # Remove concurrency when using autoscale
        if '--concurrency' in worker_args:
            idx = worker_args.index('--concurrency')
            worker_args.pop(idx)  # Remove --concurrency
            worker_args.pop(idx)  # Remove the value
    
    # Import task modules to ensure they're registered
    try:
        import dt_project.tasks.quantum
        import dt_project.tasks.simulation
        import dt_project.tasks.ml
        import dt_project.tasks.monitoring
        logger.info("Task modules loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import task modules: {e}")
        sys.exit(1)
    
    # Start additional services if requested
    processes = []
    
    if args.beat:
        logger.info("Starting Celery Beat scheduler...")
        import subprocess
        beat_process = subprocess.Popen([
            sys.executable, '-m', 'celery', 
            '--app', 'dt_project.celery_app:celery_app',
            'beat',
            '--loglevel', args.loglevel
        ])
        processes.append(beat_process)
    
    if args.flower:
        logger.info("Starting Flower monitoring...")
        import subprocess
        flower_process = subprocess.Popen([
            sys.executable, '-m', 'flower',
            '--broker', redis_url,
            '--port=5555',
            '--logging=info'
        ])
        processes.append(flower_process)
    
    try:
        # Start the main worker
        logger.info("Starting main Celery worker with args: " + " ".join(worker_args))
        celery_app.worker_main(worker_args)
        
    except KeyboardInterrupt:
        logger.info("Worker shutdown requested")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        raise
    finally:
        # Clean shutdown of additional processes
        for process in processes:
            logger.info(f"Terminating process {process.pid}")
            process.terminate()
            process.wait()

class CeleryWorkerManager:
    """Manager class for running multiple Celery workers."""
    
    def __init__(self):
        self.workers = []
        self.beat_process = None
        self.flower_process = None
    
    def start_worker(self, queue_name, concurrency=2, loglevel='info'):
        """Start a worker for a specific queue."""
        import subprocess
        
        worker_cmd = [
            sys.executable, '-m', 'celery',
            '--app', 'dt_project.celery_app:celery_app',
            'worker',
            '--loglevel', loglevel,
            '--concurrency', str(concurrency),
            '--queues', queue_name,
            '--hostname', f'{queue_name}-worker@%h',
            '--without-gossip',
            '--without-mingle'
        ]
        
        logger.info(f"Starting {queue_name} worker", concurrency=concurrency)
        process = subprocess.Popen(worker_cmd)
        self.workers.append((queue_name, process))
        return process
    
    def start_beat(self):
        """Start Celery Beat scheduler."""
        import subprocess
        
        beat_cmd = [
            sys.executable, '-m', 'celery',
            '--app', 'dt_project.celery_app:celery_app',
            'beat',
            '--loglevel', 'info'
        ]
        
        logger.info("Starting Celery Beat scheduler")
        self.beat_process = subprocess.Popen(beat_cmd)
        return self.beat_process
    
    def start_flower(self, port=5555):
        """Start Flower monitoring interface."""
        import subprocess
        
        config = get_config()
        redis_url = config.get('REDIS.URL', 'redis://localhost:6379/0')
        
        flower_cmd = [
            sys.executable, '-m', 'flower',
            '--broker', redis_url,
            f'--port={port}',
            '--logging=info'
        ]
        
        logger.info(f"Starting Flower monitoring on port {port}")
        self.flower_process = subprocess.Popen(flower_cmd)
        return self.flower_process
    
    def start_all_queues(self):
        """Start workers for all queues with optimal configuration."""
        queue_configs = [
            ('quantum', 2),      # 2 workers for quantum tasks
            ('simulation', 3),   # 3 workers for simulations
            ('ml', 2),          # 2 workers for ML tasks
            ('monitoring', 1),   # 1 worker for monitoring
            ('default', 2)       # 2 workers for general tasks
        ]
        
        for queue_name, concurrency in queue_configs:
            self.start_worker(queue_name, concurrency)
        
        # Start beat scheduler
        self.start_beat()
        
        # Start flower monitoring
        self.start_flower()
        
        logger.info(f"Started {len(self.workers)} workers across all queues")
    
    def stop_all(self):
        """Stop all workers and processes."""
        logger.info("Stopping all Celery processes...")
        
        # Stop workers
        for queue_name, process in self.workers:
            logger.info(f"Stopping {queue_name} worker")
            process.terminate()
            process.wait()
        
        # Stop beat
        if self.beat_process:
            logger.info("Stopping Celery Beat")
            self.beat_process.terminate()
            self.beat_process.wait()
        
        # Stop flower
        if self.flower_process:
            logger.info("Stopping Flower")
            self.flower_process.terminate()
            self.flower_process.wait()
        
        logger.info("All Celery processes stopped")

def run_multi_worker_setup():
    """Run a multi-worker setup with dedicated queues."""
    manager = CeleryWorkerManager()
    
    try:
        manager.start_all_queues()
        
        # Keep the main process alive
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        manager.stop_all()

if __name__ == '__main__':
    # Check if we should run in multi-worker mode
    if '--multi' in sys.argv:
        sys.argv.remove('--multi')
        run_multi_worker_setup()
    else:
        main()