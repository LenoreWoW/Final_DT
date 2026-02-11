"""
Data Broker for Quantum Trail.
Coordinates data flow between collectors, processors, and quantum digital twins.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog

from dt_project.data_acquisition.data_collector import DataCollector, DataPoint, DataSourceConfig, DataSourceType
from dt_project.data_acquisition.stream_processor import StreamProcessor, ProcessingRule
from dt_project.celery_app import celery_app
from dt_project.tasks.simulation import update_twin_from_sensors
from dt_project.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

class DataRoute(Enum):
    """Types of data routing destinations."""
    QUANTUM_TWIN = "quantum_twin"
    CELERY_TASK = "celery_task"
    WEBSOCKET = "websocket"
    DATABASE = "database"
    FILE = "file"
    WEBHOOK = "webhook"

@dataclass
class RoutingRule:
    """Rule for routing processed data to destinations."""
    rule_id: str
    route_type: DataRoute
    destination: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class TwinBinding:
    """Binding between data sources and quantum twins."""
    twin_id: str
    source_ids: List[str]
    data_mapping: Dict[str, str] = field(default_factory=dict)  # source_field -> twin_field
    update_frequency: float = 1.0  # seconds
    last_update: Optional[datetime] = None

class DataBroker:
    """Central data broker that coordinates all data flow."""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.stream_processor = StreamProcessor()
        self.routing_rules: List[RoutingRule] = []
        self.twin_bindings: Dict[str, TwinBinding] = {}
        self.data_handlers: Dict[str, Callable] = {}
        
        # WebSocket connections (would integrate with actual WebSocket handler)
        self.websocket_clients: Set[str] = set()
        
        # Statistics
        self.stats = {
            'data_points_routed': 0,
            'routing_errors': 0,
            'twin_updates': 0,
            'start_time': datetime.utcnow()
        }
        
        # Background tasks
        self.is_running = False
        self.routing_task: Optional[asyncio.Task] = None
        
        # Set up data processing pipeline
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Set up the data processing pipeline."""
        # Add data collector handler
        self.data_collector.add_data_handler(self._handle_collected_data)
        
        # Add stream processor
        self.stream_processor.add_processor(self._route_processed_data)
        
        logger.info("Data processing pipeline configured")
    
    def add_data_source(self, config: DataSourceConfig) -> bool:
        """Add a data source to the collector."""
        return self.data_collector.add_data_source(config)
    
    def remove_data_source(self, source_id: str) -> bool:
        """Remove a data source from the collector."""
        return self.data_collector.remove_data_source(source_id)
    
    def add_processing_rule(self, rule: ProcessingRule):
        """Add a stream processing rule."""
        self.stream_processor.add_processing_rule(rule)
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a data routing rule."""
        self.routing_rules.append(rule)
        logger.info(f"Added routing rule: {rule.rule_id}")
    
    def remove_routing_rule(self, rule_id: str) -> bool:
        """Remove a routing rule."""
        for i, rule in enumerate(self.routing_rules):
            if rule.rule_id == rule_id:
                del self.routing_rules[i]
                logger.info(f"Removed routing rule: {rule_id}")
                return True
        return False
    
    def bind_twin_to_sources(self, twin_id: str, source_ids: List[str], 
                           data_mapping: Dict[str, str] = None, 
                           update_frequency: float = 1.0) -> bool:
        """Bind a quantum twin to data sources."""
        try:
            binding = TwinBinding(
                twin_id=twin_id,
                source_ids=source_ids,
                data_mapping=data_mapping or {},
                update_frequency=update_frequency
            )
            
            self.twin_bindings[twin_id] = binding
            logger.info(f"Bound quantum twin {twin_id} to sources {source_ids}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bind twin {twin_id} to sources: {e}")
            return False
    
    def unbind_twin(self, twin_id: str) -> bool:
        """Unbind a quantum twin from data sources."""
        if twin_id in self.twin_bindings:
            del self.twin_bindings[twin_id]
            logger.info(f"Unbound quantum twin {twin_id}")
            return True
        return False
    
    async def start(self) -> bool:
        """Start the data broker and all components."""
        try:
            # Start data collection
            if not await self.data_collector.start_collection():
                logger.error("Failed to start data collection")
                return False
            
            # Start stream processing
            self.stream_processor.start_background_processing()
            
            # Start routing
            self.is_running = True
            self.routing_task = asyncio.create_task(self._routing_loop())
            
            # Record start time
            self.stats['start_time'] = datetime.utcnow()
            
            logger.info("Data broker started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start data broker: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the data broker and all components."""
        try:
            # Stop routing
            self.is_running = False
            if self.routing_task:
                self.routing_task.cancel()
                try:
                    await self.routing_task
                except asyncio.CancelledError:
                    pass
            
            # Stop stream processing
            self.stream_processor.stop_background_processing()
            
            # Stop data collection
            await self.data_collector.stop_collection()
            
            logger.info("Data broker stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop data broker: {e}")
            return False
    
    async def _handle_collected_data(self, data_point: DataPoint):
        """Handle data point from collector."""
        try:
            # Process through stream processor
            processed_points = await self.stream_processor.process_data_point(data_point)
            
            # Route processed points
            for point in processed_points:
                await self._route_data_point(point)
                
        except Exception as e:
            logger.error(f"Failed to handle collected data: {e}")
    
    def _route_processed_data(self, data_point: DataPoint) -> DataPoint:
        """Process data point for routing (called by stream processor)."""
        # This is called synchronously by the stream processor
        # We'll schedule the actual routing asynchronously
        asyncio.create_task(self._route_data_point(data_point))
        return data_point
    
    async def _route_data_point(self, data_point: DataPoint):
        """Route a data point to appropriate destinations."""
        try:
            routed = False
            
            # Apply routing rules
            for rule in self.routing_rules:
                if not rule.enabled:
                    continue
                
                if self._check_routing_conditions(data_point, rule.conditions):
                    await self._execute_routing(data_point, rule)
                    routed = True
            
            # Check for quantum twin bindings
            for twin_id, binding in self.twin_bindings.items():
                if data_point.source_id in binding.source_ids:
                    if self._should_update_twin(binding):
                        await self._update_quantum_twin(data_point, binding)
                        routed = True
            
            if routed:
                self.stats['data_points_routed'] += 1
            
            # Record metrics
            if metrics:
                metrics.data_points_routed_total.labels(source=data_point.source_id).inc()
                
        except Exception as e:
            self.stats['routing_errors'] += 1
            logger.error(f"Failed to route data point from {data_point.source_id}: {e}")
    
    def _check_routing_conditions(self, data_point: DataPoint, conditions: Dict[str, Any]) -> bool:
        """Check if data point meets routing conditions."""
        if not conditions:
            return True
        
        # Check source ID
        if 'source_id' in conditions:
            allowed_sources = conditions['source_id']
            if isinstance(allowed_sources, str):
                allowed_sources = [allowed_sources]
            if data_point.source_id not in allowed_sources:
                return False
        
        # Check data conditions
        if 'data_conditions' in conditions:
            for field, condition in conditions['data_conditions'].items():
                if field not in data_point.data:
                    return False
                
                value = data_point.data[field]
                
                if 'equals' in condition and value != condition['equals']:
                    return False
                if 'min' in condition and value < condition['min']:
                    return False
                if 'max' in condition and value > condition['max']:
                    return False
        
        return True
    
    async def _execute_routing(self, data_point: DataPoint, rule: RoutingRule):
        """Execute a specific routing rule."""
        try:
            if rule.route_type == DataRoute.QUANTUM_TWIN:
                await self._route_to_quantum_twin(data_point, rule)
            elif rule.route_type == DataRoute.CELERY_TASK:
                await self._route_to_celery_task(data_point, rule)
            elif rule.route_type == DataRoute.WEBSOCKET:
                await self._route_to_websocket(data_point, rule)
            elif rule.route_type == DataRoute.DATABASE:
                await self._route_to_database(data_point, rule)
            elif rule.route_type == DataRoute.FILE:
                await self._route_to_file(data_point, rule)
            elif rule.route_type == DataRoute.WEBHOOK:
                await self._route_to_webhook(data_point, rule)
            else:
                logger.warning(f"Unknown routing type: {rule.route_type}")
                
        except Exception as e:
            logger.error(f"Failed to execute routing rule {rule.rule_id}: {e}")
    
    async def _route_to_quantum_twin(self, data_point: DataPoint, rule: RoutingRule):
        """Route data to a quantum twin."""
        twin_id = rule.destination
        
        # Map data fields if specified
        field_mapping = rule.parameters.get('field_mapping', {})
        mapped_data = {}
        
        for source_field, twin_field in field_mapping.items():
            if source_field in data_point.data:
                mapped_data[twin_field] = data_point.data[source_field]
        
        # Use original data if no mapping specified
        if not mapped_data:
            mapped_data = data_point.data
        
        # Send to Celery task for twin update
        update_twin_from_sensors.delay(twin_id, mapped_data)
        logger.debug(f"Routed data to quantum twin {twin_id}")
    
    async def _route_to_celery_task(self, data_point: DataPoint, rule: RoutingRule):
        """Route data to a Celery task."""
        task_name = rule.destination
        task_params = rule.parameters.get('task_params', {})
        
        # Get task function
        task_func = getattr(celery_app, task_name, None)
        if task_func:
            task_func.delay(data_point.to_dict(), **task_params)
            logger.debug(f"Routed data to Celery task {task_name}")
        else:
            logger.warning(f"Celery task {task_name} not found")
    
    async def _route_to_websocket(self, data_point: DataPoint, rule: RoutingRule):
        """Route data to WebSocket clients."""
        channel = rule.destination
        
        # Format message
        message = {
            'channel': channel,
            'data': data_point.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # This would integrate with the actual WebSocket handler
        logger.debug(f"Would route data to WebSocket channel {channel}")
    
    async def _route_to_database(self, data_point: DataPoint, rule: RoutingRule):
        """Route data to database."""
        table_name = rule.destination
        
        # This would integrate with the actual database
        logger.debug(f"Would route data to database table {table_name}")
    
    async def _route_to_file(self, data_point: DataPoint, rule: RoutingRule):
        """Route data to file."""
        file_path = rule.destination
        file_format = rule.parameters.get('format', 'json')
        
        try:
            if file_format == 'json':
                # Append to JSON lines file
                with open(file_path, 'a') as f:
                    f.write(json.dumps(data_point.to_dict()) + '\n')
            elif file_format == 'csv':
                # Would implement CSV writing
                pass
            
            logger.debug(f"Routed data to file {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to write to file {file_path}: {e}")
    
    async def _route_to_webhook(self, data_point: DataPoint, rule: RoutingRule):
        """Route data to webhook."""
        webhook_url = rule.destination
        
        # This would make HTTP POST to webhook
        logger.debug(f"Would route data to webhook {webhook_url}")
    
    def _should_update_twin(self, binding: TwinBinding) -> bool:
        """Check if twin should be updated based on frequency."""
        if not binding.last_update:
            return True
        
        elapsed = (datetime.utcnow() - binding.last_update).total_seconds()
        return elapsed >= binding.update_frequency
    
    async def _update_quantum_twin(self, data_point: DataPoint, binding: TwinBinding):
        """Update quantum twin with data."""
        try:
            # Map data fields
            mapped_data = {}
            if binding.data_mapping:
                for source_field, twin_field in binding.data_mapping.items():
                    if source_field in data_point.data:
                        mapped_data[twin_field] = data_point.data[source_field]
            else:
                mapped_data = data_point.data
            
            # Send to Celery task
            update_twin_from_sensors.delay(binding.twin_id, mapped_data)
            
            # Update binding
            binding.last_update = datetime.utcnow()
            self.stats['twin_updates'] += 1
            
            logger.debug(f"Updated quantum twin {binding.twin_id}")
            
        except Exception as e:
            logger.error(f"Failed to update quantum twin {binding.twin_id}: {e}")
    
    async def _routing_loop(self):
        """Background loop for periodic routing tasks."""
        while self.is_running:
            try:
                # Perform any periodic routing tasks
                await self._check_twin_bindings()
                
                # Sleep briefly
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in routing loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_twin_bindings(self):
        """Check and maintain quantum twin bindings."""
        current_time = datetime.utcnow()
        
        for twin_id, binding in self.twin_bindings.items():
            # Check if binding is stale
            if binding.last_update:
                elapsed = (current_time - binding.last_update).total_seconds()
                if elapsed > binding.update_frequency * 10:  # 10x update frequency
                    logger.warning(f"Quantum twin {twin_id} has stale data binding")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get broker statistics."""
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'data_collector_stats': self.data_collector.get_statistics(),
            'stream_processor_stats': self.stream_processor.get_statistics(),
            'active_routing_rules': len([r for r in self.routing_rules if r.enabled]),
            'total_routing_rules': len(self.routing_rules),
            'active_twin_bindings': len(self.twin_bindings),
            'routing_rate': self.stats['data_points_routed'] / max(uptime, 1)
        }
    
    def get_data_flow_status(self) -> Dict[str, Any]:
        """Get current data flow status."""
        return {
            'data_sources': self.data_collector.get_source_status(),
            'twin_bindings': {
                twin_id: {
                    'source_ids': binding.source_ids,
                    'update_frequency': binding.update_frequency,
                    'last_update': binding.last_update.isoformat() if binding.last_update else None
                }
                for twin_id, binding in self.twin_bindings.items()
            },
            'routing_rules': [
                {
                    'rule_id': rule.rule_id,
                    'route_type': rule.route_type.value,
                    'destination': rule.destination,
                    'enabled': rule.enabled
                }
                for rule in self.routing_rules
            ]
        }

# Utility functions for setting up common configurations

def create_athlete_data_broker() -> DataBroker:
    """Create a data broker configured for athlete monitoring."""
    broker = DataBroker()
    
    # Add athlete sensor data source
    athlete_sensor_config = DataSourceConfig(
        source_id="athlete_sensors",
        source_type=DataSourceType.MOCK,
        polling_interval=1.0,
        parameters={'data_type': 'athlete'}
    )
    broker.add_data_source(athlete_sensor_config)
    
    # Add processing rules
    from dt_project.data_acquisition.stream_processor import create_athlete_processing_rules
    for rule in create_athlete_processing_rules():
        broker.add_processing_rule(rule)
    
    # Add routing rule for quantum twin updates
    routing_rule = RoutingRule(
        rule_id="athlete_to_twin",
        route_type=DataRoute.QUANTUM_TWIN,
        destination="athlete_001",
        conditions={'source_id': 'athlete_sensors'}
    )
    broker.add_routing_rule(routing_rule)
    
    return broker

def create_military_data_broker() -> DataBroker:
    """Create a data broker configured for military unit monitoring."""
    broker = DataBroker()
    
    # Add military sensor data source
    military_sensor_config = DataSourceConfig(
        source_id="military_sensors",
        source_type=DataSourceType.MOCK,
        polling_interval=2.0,
        parameters={'data_type': 'military'}
    )
    broker.add_data_source(military_sensor_config)
    
    # Add processing rules
    from dt_project.data_acquisition.stream_processor import create_military_processing_rules
    for rule in create_military_processing_rules():
        broker.add_processing_rule(rule)
    
    # Add routing rule for quantum twin updates
    routing_rule = RoutingRule(
        rule_id="military_to_twin",
        route_type=DataRoute.QUANTUM_TWIN,
        destination="unit_001",
        conditions={'source_id': 'military_sensors'}
    )
    broker.add_routing_rule(routing_rule)
    
    return broker