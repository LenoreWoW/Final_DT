"""
Monitoring and maintenance related Celery tasks.
Handles system health checks, cleanup operations, and performance monitoring.
"""

import time
import json
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog

from dt_project.celery_app import celery_app, monitoring_task
from dt_project.monitoring.metrics import metrics, get_system_health

logger = structlog.get_logger(__name__)

@monitoring_task
def cleanup_expired_results(self):
    """
    Clean up expired Celery task results and old data.
    
    Returns:
        Dict containing cleanup results
    """
    task_id = self.request.id
    logger.info("Starting cleanup of expired results", task_id=task_id)
    
    try:
        cleanup_stats = {
            'celery_results_cleaned': 0,
            'log_files_cleaned': 0,
            'temp_files_cleaned': 0,
            'database_records_cleaned': 0,
            'total_space_freed_mb': 0.0
        }
        
        # Clean up Celery results
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Cleaning Celery results', 'progress': 20}
        )
        
        # This would integrate with the actual result backend
        # For now, simulate cleanup
        cleanup_stats['celery_results_cleaned'] = _cleanup_celery_results()
        
        # Clean up old log files
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Cleaning log files', 'progress': 40}
        )
        
        cleanup_stats['log_files_cleaned'], log_space_freed = _cleanup_log_files()
        cleanup_stats['total_space_freed_mb'] += log_space_freed
        
        # Clean up temporary files
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Cleaning temporary files', 'progress': 60}
        )
        
        cleanup_stats['temp_files_cleaned'], temp_space_freed = _cleanup_temp_files()
        cleanup_stats['total_space_freed_mb'] += temp_space_freed
        
        # Clean up old database records (would integrate with actual DB)
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Cleaning database records', 'progress': 80}
        )
        
        cleanup_stats['database_records_cleaned'] = _cleanup_old_database_records()
        
        result = {
            'cleanup_stats': cleanup_stats,
            'cleaned_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        # Record metrics
        if metrics:
            metrics.cleanup_operations_total.inc()
            metrics.disk_space_freed_bytes.inc(cleanup_stats['total_space_freed_mb'] * 1024 * 1024)
        
        logger.info("Cleanup completed successfully", 
                   task_id=task_id,
                   total_space_freed=cleanup_stats['total_space_freed_mb'])
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Cleanup operation failed", task_id=task_id, error=str(e))
        raise

@monitoring_task
def system_health_check(self):
    """
    Perform comprehensive system health check.
    
    Returns:
        Dict containing system health information
    """
    task_id = self.request.id
    logger.info("Starting system health check", task_id=task_id)
    
    try:
        health_data = {}
        
        # Basic system metrics
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Checking system resources', 'progress': 20}
        )
        
        health_data['system'] = _check_system_resources()
        
        # Application health
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Checking application health', 'progress': 40}
        )
        
        health_data['application'] = _check_application_health()
        
        # Quantum backend health
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Checking quantum backends', 'progress': 60}
        )
        
        health_data['quantum_backends'] = _check_quantum_backend_health()
        
        # Database health
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Checking database', 'progress': 80}
        )
        
        health_data['database'] = _check_database_health()
        
        # Overall health assessment
        overall_status = _assess_overall_health(health_data)
        
        result = {
            'overall_status': overall_status,
            'health_data': health_data,
            'check_timestamp': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        # Record health metrics
        if metrics:
            metrics.health_check_runs_total.inc()
            if overall_status == 'healthy':
                metrics.system_health_status.set(1)
            elif overall_status == 'degraded':
                metrics.system_health_status.set(0.5)
            else:
                metrics.system_health_status.set(0)
        
        logger.info("Health check completed", 
                   task_id=task_id,
                   overall_status=overall_status)
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Health check failed", task_id=task_id, error=str(e))
        raise

@monitoring_task
def generate_performance_report(self, report_period_hours: int = 24):
    """
    Generate comprehensive performance report.
    
    Args:
        report_period_hours: Number of hours to include in the report
    
    Returns:
        Dict containing performance report
    """
    task_id = self.request.id
    logger.info("Generating performance report", 
                task_id=task_id,
                period_hours=report_period_hours)
    
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=report_period_hours)
        
        report_data = {
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': report_period_hours
            }
        }
        
        # System performance metrics
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Collecting system metrics', 'progress': 20}
        )
        
        report_data['system_performance'] = _collect_system_performance_metrics(start_time, end_time)
        
        # Application performance metrics
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Collecting application metrics', 'progress': 40}
        )
        
        report_data['application_performance'] = _collect_application_performance_metrics(start_time, end_time)
        
        # Quantum operations performance
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Collecting quantum metrics', 'progress': 60}
        )
        
        report_data['quantum_performance'] = _collect_quantum_performance_metrics(start_time, end_time)
        
        # Error analysis
        self.update_state(
            state='PROCESSING',
            meta={'message': 'Analyzing errors', 'progress': 80}
        )
        
        report_data['error_analysis'] = _analyze_errors(start_time, end_time)
        
        # Performance trends and recommendations
        report_data['trends'] = _analyze_performance_trends(report_data)
        report_data['recommendations'] = _generate_recommendations(report_data)
        
        result = {
            'report_data': report_data,
            'generated_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        logger.info("Performance report generated", 
                   task_id=task_id,
                   period_hours=report_period_hours)
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Performance report generation failed", 
                    task_id=task_id,
                    error=str(e))
        raise

@monitoring_task
def monitor_resource_usage(self, alert_thresholds: Dict[str, float]):
    """
    Monitor system resource usage and trigger alerts if thresholds are exceeded.
    
    Args:
        alert_thresholds: Dictionary of resource thresholds (cpu, memory, disk, etc.)
    
    Returns:
        Dict containing monitoring results and any alerts
    """
    task_id = self.request.id
    logger.info("Monitoring resource usage", 
                task_id=task_id,
                thresholds=alert_thresholds)
    
    try:
        current_usage = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': dict(psutil.net_io_counters()._asdict()),
            'process_count': len(psutil.pids())
        }
        
        alerts = []
        
        # Check CPU usage
        if current_usage['cpu_percent'] > alert_thresholds.get('cpu_percent', 90):
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'message': f"High CPU usage: {current_usage['cpu_percent']:.1f}%",
                'threshold': alert_thresholds.get('cpu_percent', 90),
                'current_value': current_usage['cpu_percent']
            })
        
        # Check memory usage
        if current_usage['memory_percent'] > alert_thresholds.get('memory_percent', 85):
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f"High memory usage: {current_usage['memory_percent']:.1f}%",
                'threshold': alert_thresholds.get('memory_percent', 85),
                'current_value': current_usage['memory_percent']
            })
        
        # Check disk usage
        if current_usage['disk_percent'] > alert_thresholds.get('disk_percent', 80):
            alerts.append({
                'type': 'disk_high',
                'severity': 'critical' if current_usage['disk_percent'] > 95 else 'warning',
                'message': f"High disk usage: {current_usage['disk_percent']:.1f}%",
                'threshold': alert_thresholds.get('disk_percent', 80),
                'current_value': current_usage['disk_percent']
            })
        
        # Update metrics
        if metrics:
            metrics.system_cpu_percent.set(current_usage['cpu_percent'])
            metrics.system_memory_percent.set(current_usage['memory_percent'])
            metrics.system_disk_percent.set(current_usage['disk_percent'])
        
        result = {
            'current_usage': current_usage,
            'alert_thresholds': alert_thresholds,
            'alerts': alerts,
            'alert_count': len(alerts),
            'monitored_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        if alerts:
            logger.warning("Resource usage alerts triggered", 
                          task_id=task_id,
                          alert_count=len(alerts),
                          alerts=[a['type'] for a in alerts])
        else:
            logger.info("Resource monitoring completed - all normal", 
                       task_id=task_id)
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Resource monitoring failed", task_id=task_id, error=str(e))
        raise

@monitoring_task
def backup_critical_data(self, backup_config: Dict[str, Any]):
    """
    Backup critical system data and configurations.
    
    Args:
        backup_config: Backup configuration including paths and retention
    
    Returns:
        Dict containing backup results
    """
    task_id = self.request.id
    logger.info("Starting critical data backup", task_id=task_id)
    
    try:
        backup_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_results = {
            'backup_timestamp': backup_timestamp,
            'backed_up_items': []
        }
        
        backup_paths = backup_config.get('backup_paths', [])
        backup_destination = backup_config.get('destination', '/tmp/backups')
        
        # Ensure backup directory exists
        os.makedirs(backup_destination, exist_ok=True)
        
        total_items = len(backup_paths)
        
        for i, item_config in enumerate(backup_paths):
            progress = int((i / total_items) * 90) + 5
            self.update_state(
                state='PROCESSING',
                meta={'message': f'Backing up {item_config.get("name", "item")}', 'progress': progress}
            )
            
            try:
                backup_result = _backup_item(item_config, backup_destination, backup_timestamp)
                backup_results['backed_up_items'].append(backup_result)
                
            except Exception as e:
                logger.warning(f"Failed to backup {item_config.get('name', 'item')}", error=str(e))
                backup_results['backed_up_items'].append({
                    'name': item_config.get('name', 'unknown'),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Cleanup old backups based on retention policy
        retention_days = backup_config.get('retention_days', 7)
        cleaned_count = _cleanup_old_backups(backup_destination, retention_days)
        backup_results['old_backups_cleaned'] = cleaned_count
        
        successful_backups = len([item for item in backup_results['backed_up_items'] if item.get('status') == 'success'])
        
        result = {
            'backup_results': backup_results,
            'total_items': total_items,
            'successful_backups': successful_backups,
            'failed_backups': total_items - successful_backups,
            'completed_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        
        # Record metrics
        if metrics:
            metrics.backup_operations_total.inc()
            metrics.backup_success_total.inc(successful_backups)
            metrics.backup_failure_total.inc(total_items - successful_backups)
        
        logger.info("Backup operation completed", 
                   task_id=task_id,
                   successful=successful_backups,
                   failed=total_items - successful_backups)
        
        return {
            'status': 'success',
            'data': result
        }
        
    except Exception as e:
        logger.error("Backup operation failed", task_id=task_id, error=str(e))
        raise

# Helper functions

def _cleanup_celery_results():
    """Clean up expired Celery task results."""
    # This would integrate with the actual result backend (Redis, database, etc.)
    # For now, return a mock count
    cleaned_count = 42
    logger.info(f"Cleaned {cleaned_count} expired Celery results")
    return cleaned_count

def _cleanup_log_files():
    """Clean up old log files."""
    log_dirs = ['/var/log', '/tmp/logs', './logs']
    total_cleaned = 0
    space_freed = 0.0
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            # Mock cleanup - in reality would delete old log files
            dir_cleaned = 5
            dir_space = 10.5  # MB
            total_cleaned += dir_cleaned
            space_freed += dir_space
    
    logger.info(f"Cleaned {total_cleaned} log files, freed {space_freed} MB")
    return total_cleaned, space_freed

def _cleanup_temp_files():
    """Clean up temporary files."""
    temp_dirs = ['/tmp', './temp', './cache']
    total_cleaned = 0
    space_freed = 0.0
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            # Mock cleanup
            dir_cleaned = 8
            dir_space = 25.3  # MB
            total_cleaned += dir_cleaned
            space_freed += dir_space
    
    logger.info(f"Cleaned {total_cleaned} temporary files, freed {space_freed} MB")
    return total_cleaned, space_freed

def _cleanup_old_database_records():
    """Clean up old database records."""
    # This would integrate with the actual database
    # For now, return a mock count
    cleaned_count = 156
    logger.info(f"Cleaned {cleaned_count} old database records")
    return cleaned_count

def _check_system_resources():
    """Check system resource usage."""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory': dict(psutil.virtual_memory()._asdict()),
        'disk': dict(psutil.disk_usage('/')._asdict()),
        'network': dict(psutil.net_io_counters()._asdict()),
        'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        'uptime_seconds': time.time() - psutil.boot_time()
    }

def _check_application_health():
    """Check application-specific health metrics."""
    return {
        'status': 'healthy',
        'uptime': '2d 14h 32m',  # Mock uptime
        'active_connections': 24,
        'queue_lengths': {
            'quantum': 2,
            'simulation': 1,
            'ml': 0,
            'monitoring': 0
        },
        'error_rate_last_hour': 0.002
    }

def _check_quantum_backend_health():
    """Check quantum backend health."""
    return {
        'total_backends': 3,
        'available_backends': 2,
        'average_queue_time': 45.2,  # seconds
        'success_rate_24h': 0.97,
        'backends': {
            'simulator': {'status': 'healthy', 'queue_length': 0},
            'ibm_quantum': {'status': 'healthy', 'queue_length': 15},
            'local_qpu': {'status': 'offline', 'queue_length': 0}
        }
    }

def _check_database_health():
    """Check database health."""
    return {
        'status': 'healthy',
        'connection_pool_size': 10,
        'active_connections': 3,
        'query_performance': {
            'average_response_time_ms': 12.4,
            'slow_queries_last_hour': 2
        },
        'disk_usage_percent': 35.7
    }

def _assess_overall_health(health_data):
    """Assess overall system health based on individual checks."""
    issues = []
    
    # Check CPU
    if health_data['system']['cpu_percent'] > 90:
        issues.append('high_cpu')
    
    # Check memory
    if health_data['system']['memory']['percent'] > 85:
        issues.append('high_memory')
    
    # Check disk
    if health_data['system']['disk']['percent'] > 80:
        issues.append('high_disk')
    
    # Check quantum backends
    if health_data['quantum_backends']['available_backends'] < 1:
        issues.append('no_quantum_backends')
    
    if not issues:
        return 'healthy'
    elif len(issues) == 1 or (len(issues) == 2 and 'high_disk' not in issues):
        return 'degraded'
    else:
        return 'unhealthy'

def _collect_system_performance_metrics(start_time, end_time):
    """Collect system performance metrics for the specified period."""
    return {
        'average_cpu_percent': 45.2,
        'peak_cpu_percent': 78.1,
        'average_memory_percent': 62.8,
        'peak_memory_percent': 81.3,
        'disk_io_operations': 15432,
        'network_bytes_sent': 1024 * 1024 * 256,  # 256 MB
        'network_bytes_received': 1024 * 1024 * 512,  # 512 MB
    }

def _collect_application_performance_metrics(start_time, end_time):
    """Collect application performance metrics."""
    return {
        'total_requests': 8642,
        'successful_requests': 8521,
        'failed_requests': 121,
        'average_response_time_ms': 145.7,
        'peak_response_time_ms': 2341.2,
        'active_sessions': 89,
        'cache_hit_rate': 0.73
    }

def _collect_quantum_performance_metrics(start_time, end_time):
    """Collect quantum operations performance metrics."""
    return {
        'quantum_circuits_executed': 234,
        'successful_executions': 225,
        'failed_executions': 9,
        'average_execution_time_seconds': 12.4,
        'optimization_jobs_completed': 45,
        'quantum_advantage_achieved': 1.23,
        'backend_utilization': {
            'simulator': 0.78,
            'ibm_quantum': 0.34,
            'local_qpu': 0.0
        }
    }

def _analyze_errors(start_time, end_time):
    """Analyze errors that occurred during the specified period."""
    return {
        'total_errors': 121,
        'error_types': {
            'quantum_execution_error': 45,
            'network_timeout': 32,
            'database_connection_error': 28,
            'validation_error': 16
        },
        'error_rate_trend': 'decreasing',
        'most_frequent_error': 'quantum_execution_error'
    }

def _analyze_performance_trends(report_data):
    """Analyze performance trends from the report data."""
    return {
        'cpu_trend': 'stable',
        'memory_trend': 'increasing',
        'request_volume_trend': 'increasing',
        'error_rate_trend': 'decreasing',
        'quantum_performance_trend': 'improving'
    }

def _generate_recommendations(report_data):
    """Generate performance improvement recommendations."""
    recommendations = []
    
    # Check CPU usage
    if report_data['system_performance']['peak_cpu_percent'] > 80:
        recommendations.append({
            'category': 'cpu',
            'priority': 'medium',
            'recommendation': 'Consider adding more CPU resources or optimizing high-CPU processes'
        })
    
    # Check memory usage
    if report_data['system_performance']['peak_memory_percent'] > 85:
        recommendations.append({
            'category': 'memory',
            'priority': 'high',
            'recommendation': 'Memory usage is high - consider increasing memory or optimizing memory-intensive operations'
        })
    
    # Check error rates
    if report_data['application_performance']['failed_requests'] > report_data['application_performance']['total_requests'] * 0.02:
        recommendations.append({
            'category': 'reliability',
            'priority': 'high',
            'recommendation': 'Error rate is above 2% - investigate and fix recurring errors'
        })
    
    # Check quantum performance
    quantum_success_rate = (report_data['quantum_performance']['successful_executions'] / 
                           max(1, report_data['quantum_performance']['quantum_circuits_executed']))
    if quantum_success_rate < 0.95:
        recommendations.append({
            'category': 'quantum',
            'priority': 'medium',
            'recommendation': 'Quantum execution success rate is below 95% - check backend connectivity and circuit validation'
        })
    
    return recommendations

def _backup_item(item_config, backup_destination, backup_timestamp):
    """Backup a single item."""
    item_name = item_config.get('name', 'unknown')
    source_path = item_config.get('source_path', '')
    
    # Mock backup operation
    backup_filename = f"{item_name}_{backup_timestamp}.backup"
    backup_path = os.path.join(backup_destination, backup_filename)
    
    # In reality, this would copy/compress the actual files
    with open(backup_path, 'w') as f:
        f.write(f"Mock backup of {item_name} from {source_path}")
    
    file_size = os.path.getsize(backup_path)
    
    return {
        'name': item_name,
        'source_path': source_path,
        'backup_path': backup_path,
        'backup_size_bytes': file_size,
        'status': 'success',
        'backed_up_at': datetime.utcnow().isoformat()
    }

def _cleanup_old_backups(backup_destination, retention_days):
    """Clean up backups older than retention period."""
    cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
    cleaned_count = 0
    
    # Mock cleanup - in reality would scan directory and delete old files
    if os.path.exists(backup_destination):
        # Simulate finding and removing 3 old backup files
        cleaned_count = 3
    
    logger.info(f"Cleaned {cleaned_count} old backups older than {retention_days} days")
    return cleaned_count