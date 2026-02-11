"""
Stream Processor for Quantum Trail.
Handles real-time data stream processing, filtering, aggregation, and routing.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import structlog

from dt_project.data_acquisition.data_collector import DataPoint, DataQuality
from dt_project.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

class ProcessingOperation(Enum):
    """Types of stream processing operations."""
    FILTER = "filter"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"
    ROUTE = "route"
    VALIDATE = "validate"

class AggregationType(Enum):
    """Types of aggregation operations."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    STD = "std"
    PERCENTILE = "percentile"

@dataclass
class ProcessingRule:
    """Rule for stream processing operations."""
    rule_id: str
    operation: ProcessingOperation
    conditions: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0

@dataclass
class StreamWindow:
    """Sliding window for stream processing."""
    window_size: int  # Number of data points
    window_duration: Optional[timedelta] = None  # Time-based window
    data_points: deque = field(default_factory=deque)
    created_at: datetime = field(default_factory=datetime.utcnow)

class StreamProcessor:
    """Main stream processor for real-time data processing."""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.processing_rules: List[ProcessingRule] = []
        self.data_buffer: deque = deque(maxlen=buffer_size)
        self.windows: Dict[str, StreamWindow] = {}
        self.aggregations: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.processors: List[Callable] = []
        
        # Statistics
        self.stats = {
            'processed_count': 0,
            'filtered_count': 0,
            'error_count': 0,
            'start_time': datetime.utcnow()
        }
        
        # Background processing
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
    
    def add_processing_rule(self, rule: ProcessingRule):
        """Add a processing rule."""
        self.processing_rules.append(rule)
        self.processing_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added processing rule: {rule.rule_id}")
    
    def remove_processing_rule(self, rule_id: str) -> bool:
        """Remove a processing rule."""
        for i, rule in enumerate(self.processing_rules):
            if rule.rule_id == rule_id:
                del self.processing_rules[i]
                logger.info(f"Removed processing rule: {rule_id}")
                return True
        return False
    
    def add_processor(self, processor: Callable[[DataPoint], Union[DataPoint, List[DataPoint], None]]):
        """Add a custom processor function."""
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.__name__}")
    
    def create_window(self, window_id: str, window_size: int, window_duration: Optional[timedelta] = None):
        """Create a sliding window for aggregations."""
        self.windows[window_id] = StreamWindow(
            window_size=window_size,
            window_duration=window_duration
        )
        logger.info(f"Created window {window_id}: size={window_size}, duration={window_duration}")
    
    async def process_data_point(self, data_point: DataPoint) -> List[DataPoint]:
        """Process a single data point through all rules and processors."""
        try:
            processed_points = [data_point]
            
            # Apply processing rules
            for rule in self.processing_rules:
                if not rule.enabled:
                    continue
                
                new_processed_points = []
                for point in processed_points:
                    result = await self._apply_rule(point, rule)
                    if result is not None:
                        if isinstance(result, list):
                            new_processed_points.extend(result)
                        else:
                            new_processed_points.append(result)
                
                processed_points = new_processed_points
                
                # If no points remain after filtering, stop processing
                if not processed_points:
                    self.stats['filtered_count'] += 1
                    break
            
            # Apply custom processors
            final_points = []
            for point in processed_points:
                for processor in self.processors:
                    try:
                        result = processor(point)
                        if result is not None:
                            if isinstance(result, list):
                                final_points.extend(result)
                            else:
                                final_points.append(result)
                    except Exception as e:
                        logger.error(f"Processor {processor.__name__} failed: {e}")
                        final_points.append(point)  # Keep original on error
            
            # Update windows
            for point in (final_points if final_points else processed_points):
                self._update_windows(point)
            
            self.stats['processed_count'] += 1
            
            # Record metrics
            if metrics:
                metrics.stream_data_processed_total.inc()
                metrics.stream_processing_latency.observe(
                    (datetime.utcnow() - data_point.timestamp).total_seconds()
                )
            
            return final_points if final_points else processed_points
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Failed to process data point: {e}")
            return [data_point]  # Return original on error
    
    async def _apply_rule(self, data_point: DataPoint, rule: ProcessingRule) -> Union[DataPoint, List[DataPoint], None]:
        """Apply a specific processing rule to a data point."""
        
        # Check conditions
        if not self._check_conditions(data_point, rule.conditions):
            return data_point
        
        if rule.operation == ProcessingOperation.FILTER:
            return self._apply_filter(data_point, rule)
        elif rule.operation == ProcessingOperation.TRANSFORM:
            return self._apply_transform(data_point, rule)
        elif rule.operation == ProcessingOperation.AGGREGATE:
            return self._apply_aggregation(data_point, rule)
        elif rule.operation == ProcessingOperation.ENRICH:
            return self._apply_enrichment(data_point, rule)
        elif rule.operation == ProcessingOperation.VALIDATE:
            return self._apply_validation(data_point, rule)
        elif rule.operation == ProcessingOperation.ROUTE:
            return self._apply_routing(data_point, rule)
        else:
            logger.warning(f"Unknown processing operation: {rule.operation}")
            return data_point
    
    def _check_conditions(self, data_point: DataPoint, conditions: Dict[str, Any]) -> bool:
        """Check if data point meets rule conditions."""
        if not conditions:
            return True
        
        # Check source conditions
        if 'source_id' in conditions:
            allowed_sources = conditions['source_id']
            if isinstance(allowed_sources, str):
                allowed_sources = [allowed_sources]
            if data_point.source_id not in allowed_sources:
                return False
        
        # Check data quality conditions
        if 'min_quality' in conditions:
            quality_levels = {
                DataQuality.LOW: 1,
                DataQuality.MEDIUM: 2,
                DataQuality.HIGH: 3
            }
            min_level = quality_levels.get(conditions['min_quality'], 0)
            current_level = quality_levels.get(data_point.quality, 0)
            if current_level < min_level:
                return False
        
        # Check data field conditions
        if 'data_conditions' in conditions:
            for field, condition in conditions['data_conditions'].items():
                if field not in data_point.data:
                    return False
                
                value = data_point.data[field]
                
                if 'min' in condition and value < condition['min']:
                    return False
                if 'max' in condition and value > condition['max']:
                    return False
                if 'equals' in condition and value != condition['equals']:
                    return False
                if 'contains' in condition and condition['contains'] not in str(value):
                    return False
        
        return True
    
    def _apply_filter(self, data_point: DataPoint, rule: ProcessingRule) -> Optional[DataPoint]:
        """Apply filtering rule."""
        filter_type = rule.parameters.get('type', 'pass')
        
        if filter_type == 'pass':
            return data_point
        elif filter_type == 'block':
            return None
        elif filter_type == 'sample':
            sample_rate = rule.parameters.get('rate', 0.1)
            if np.random.random() < sample_rate:
                return data_point
            return None
        else:
            return data_point
    
    def _apply_transform(self, data_point: DataPoint, rule: ProcessingRule) -> DataPoint:
        """Apply transformation rule."""
        transform_type = rule.parameters.get('type', 'identity')
        
        if transform_type == 'identity':
            return data_point
        
        elif transform_type == 'scale':
            # Scale numeric values
            scale_factor = rule.parameters.get('factor', 1.0)
            fields = rule.parameters.get('fields', [])
            
            new_data = data_point.data.copy()
            for field in fields:
                if field in new_data and isinstance(new_data[field], (int, float)):
                    new_data[field] = new_data[field] * scale_factor
            
            return DataPoint(
                timestamp=data_point.timestamp,
                source_id=data_point.source_id,
                source_type=data_point.source_type,
                data=new_data,
                quality=data_point.quality,
                metadata={**data_point.metadata, 'transformed': True}
            )
        
        elif transform_type == 'normalize':
            # Normalize values to [0, 1]
            fields = rule.parameters.get('fields', [])
            ranges = rule.parameters.get('ranges', {})
            
            new_data = data_point.data.copy()
            for field in fields:
                if field in new_data and isinstance(new_data[field], (int, float)):
                    min_val, max_val = ranges.get(field, (0, 100))
                    normalized = (new_data[field] - min_val) / (max_val - min_val)
                    new_data[field] = max(0, min(1, normalized))
            
            return DataPoint(
                timestamp=data_point.timestamp,
                source_id=data_point.source_id,
                source_type=data_point.source_type,
                data=new_data,
                quality=data_point.quality,
                metadata={**data_point.metadata, 'normalized': True}
            )
        
        elif transform_type == 'extract':
            # Extract specific fields
            fields = rule.parameters.get('fields', [])
            new_data = {field: data_point.data[field] for field in fields if field in data_point.data}
            
            return DataPoint(
                timestamp=data_point.timestamp,
                source_id=data_point.source_id,
                source_type=data_point.source_type,
                data=new_data,
                quality=data_point.quality,
                metadata={**data_point.metadata, 'extracted': True}
            )
        
        return data_point
    
    def _apply_aggregation(self, data_point: DataPoint, rule: ProcessingRule) -> Optional[DataPoint]:
        """Apply aggregation rule."""
        window_id = rule.parameters.get('window_id', 'default')
        aggregation_type = rule.parameters.get('type', AggregationType.AVERAGE)
        fields = rule.parameters.get('fields', [])
        
        if window_id not in self.windows:
            logger.warning(f"Window {window_id} not found for aggregation")
            return data_point
        
        window = self.windows[window_id]
        
        # Check if we have enough data for aggregation
        if len(window.data_points) < 2:
            return None  # Need more data
        
        # Perform aggregation
        aggregated_data = {}
        for field in fields:
            values = []
            for dp in window.data_points:
                if field in dp.data and isinstance(dp.data[field], (int, float)):
                    values.append(dp.data[field])
            
            if values:
                if aggregation_type == AggregationType.SUM:
                    aggregated_data[field] = sum(values)
                elif aggregation_type == AggregationType.AVERAGE:
                    aggregated_data[field] = sum(values) / len(values)
                elif aggregation_type == AggregationType.MIN:
                    aggregated_data[field] = min(values)
                elif aggregation_type == AggregationType.MAX:
                    aggregated_data[field] = max(values)
                elif aggregation_type == AggregationType.COUNT:
                    aggregated_data[field] = len(values)
                elif aggregation_type == AggregationType.MEDIAN:
                    aggregated_data[field] = np.median(values)
                elif aggregation_type == AggregationType.STD:
                    aggregated_data[field] = np.std(values)
        
        # Create aggregated data point
        return DataPoint(
            timestamp=datetime.utcnow(),
            source_id=f"aggregated_{window_id}",
            source_type=data_point.source_type,
            data=aggregated_data,
            quality=DataQuality.HIGH,
            metadata={
                'aggregation_type': aggregation_type.value,
                'window_size': len(window.data_points),
                'original_source': data_point.source_id
            }
        )
    
    def _apply_enrichment(self, data_point: DataPoint, rule: ProcessingRule) -> DataPoint:
        """Apply data enrichment rule."""
        enrichment_type = rule.parameters.get('type', 'timestamp')
        
        new_data = data_point.data.copy()
        new_metadata = data_point.metadata.copy()
        
        if enrichment_type == 'timestamp':
            new_data['processing_timestamp'] = datetime.utcnow().isoformat()
        
        elif enrichment_type == 'geolocation':
            # Add geolocation data if GPS coordinates exist
            if 'gps_position' in new_data:
                lat, lon = new_data['gps_position'][:2]
                # Mock reverse geocoding
                new_data['location'] = {
                    'city': 'Unknown',
                    'country': 'Unknown',
                    'timezone': 'UTC'
                }
        
        elif enrichment_type == 'statistics':
            # Add statistical information
            numeric_fields = [k for k, v in new_data.items() if isinstance(v, (int, float))]
            if numeric_fields:
                values = [new_data[field] for field in numeric_fields]
                new_data['statistics'] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        elif enrichment_type == 'derived':
            # Add derived fields
            derivations = rule.parameters.get('derivations', {})
            for field_name, formula in derivations.items():
                try:
                    # Simple formula evaluation (secure subset)
                    if '+' in formula:
                        terms = formula.split('+')
                        value = sum(new_data.get(term.strip(), 0) for term in terms if term.strip() in new_data)
                        new_data[field_name] = value
                    elif '*' in formula:
                        terms = formula.split('*')
                        value = 1
                        for term in terms:
                            value *= new_data.get(term.strip(), 1)
                        new_data[field_name] = value
                except Exception as e:
                    logger.warning(f"Failed to derive field {field_name}: {e}")
        
        return DataPoint(
            timestamp=data_point.timestamp,
            source_id=data_point.source_id,
            source_type=data_point.source_type,
            data=new_data,
            quality=data_point.quality,
            metadata={**new_metadata, 'enriched': True}
        )
    
    def _apply_validation(self, data_point: DataPoint, rule: ProcessingRule) -> Optional[DataPoint]:
        """Apply validation rule."""
        validation_rules = rule.parameters.get('rules', {})
        strict = rule.parameters.get('strict', False)
        
        is_valid = True
        validation_errors = []
        
        for field, constraints in validation_rules.items():
            if field not in data_point.data:
                if constraints.get('required', False):
                    is_valid = False
                    validation_errors.append(f"Required field {field} missing")
                continue
            
            value = data_point.data[field]
            
            # Type validation
            if 'type' in constraints:
                expected_type = constraints['type']
                if expected_type == 'number' and not isinstance(value, (int, float)):
                    is_valid = False
                    validation_errors.append(f"Field {field} should be number")
                elif expected_type == 'string' and not isinstance(value, str):
                    is_valid = False
                    validation_errors.append(f"Field {field} should be string")
            
            # Range validation
            if isinstance(value, (int, float)):
                if 'min' in constraints and value < constraints['min']:
                    is_valid = False
                    validation_errors.append(f"Field {field} below minimum {constraints['min']}")
                if 'max' in constraints and value > constraints['max']:
                    is_valid = False
                    validation_errors.append(f"Field {field} above maximum {constraints['max']}")
            
            # Pattern validation for strings
            if isinstance(value, str) and 'pattern' in constraints:
                import re
                if not re.match(constraints['pattern'], value):
                    is_valid = False
                    validation_errors.append(f"Field {field} doesn't match pattern")
        
        if not is_valid:
            if strict:
                logger.warning(f"Data validation failed for {data_point.source_id}: {validation_errors}")
                return None
            else:
                # Add validation errors to metadata
                new_metadata = data_point.metadata.copy()
                new_metadata['validation_errors'] = validation_errors
                
                return DataPoint(
                    timestamp=data_point.timestamp,
                    source_id=data_point.source_id,
                    source_type=data_point.source_type,
                    data=data_point.data,
                    quality=DataQuality.LOW,
                    metadata=new_metadata
                )
        
        return data_point
    
    def _apply_routing(self, data_point: DataPoint, rule: ProcessingRule) -> List[DataPoint]:
        """Apply routing rule."""
        routing_type = rule.parameters.get('type', 'duplicate')
        destinations = rule.parameters.get('destinations', [])
        
        if routing_type == 'duplicate':
            # Create copies for each destination
            routed_points = []
            for destination in destinations:
                new_metadata = data_point.metadata.copy()
                new_metadata['routed_to'] = destination
                
                routed_point = DataPoint(
                    timestamp=data_point.timestamp,
                    source_id=data_point.source_id,
                    source_type=data_point.source_type,
                    data=data_point.data,
                    quality=data_point.quality,
                    metadata=new_metadata
                )
                routed_points.append(routed_point)
            
            return routed_points
        
        return [data_point]
    
    def _update_windows(self, data_point: DataPoint):
        """Update all sliding windows with new data point."""
        for window_id, window in self.windows.items():
            # Add to window
            window.data_points.append(data_point)
            
            # Remove old data based on size limit
            while len(window.data_points) > window.window_size:
                window.data_points.popleft()
            
            # Remove old data based on time limit
            if window.window_duration:
                cutoff_time = datetime.utcnow() - window.window_duration
                while (window.data_points and 
                       window.data_points[0].timestamp < cutoff_time):
                    window.data_points.popleft()
    
    def get_window_data(self, window_id: str) -> List[DataPoint]:
        """Get current data in a window."""
        if window_id in self.windows:
            return list(self.windows[window_id].data_points)
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'processing_rate': self.stats['processed_count'] / max(uptime, 1),
            'buffer_usage': len(self.data_buffer),
            'buffer_capacity': self.buffer_size,
            'active_rules': len([r for r in self.processing_rules if r.enabled]),
            'total_rules': len(self.processing_rules),
            'active_windows': len(self.windows)
        }
    
    def start_background_processing(self):
        """Start background processing task."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._background_processor())
        logger.info("Started background stream processing")
    
    def stop_background_processing(self):
        """Stop background processing task."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
        logger.info("Stopped background stream processing")
    
    async def _background_processor(self):
        """Background task for continuous processing."""
        while self.is_running:
            try:
                # Process any buffered data
                if self.data_buffer:
                    data_point = self.data_buffer.popleft()
                    await self.process_data_point(data_point)
                else:
                    # No data to process, sleep briefly
                    await asyncio.sleep(0.01)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
                await asyncio.sleep(1)

# Utility functions for common stream processing patterns

def create_athlete_processing_rules() -> List[ProcessingRule]:
    """Create common processing rules for athlete data."""
    return [
        ProcessingRule(
            rule_id="athlete_filter_quality",
            operation=ProcessingOperation.FILTER,
            conditions={'min_quality': DataQuality.MEDIUM},
            parameters={'type': 'pass'}
        ),
        ProcessingRule(
            rule_id="athlete_normalize_heart_rate",
            operation=ProcessingOperation.TRANSFORM,
            conditions={'data_conditions': {'heart_rate': {'min': 0, 'max': 250}}},
            parameters={
                'type': 'normalize',
                'fields': ['heart_rate'],
                'ranges': {'heart_rate': (50, 200)}
            }
        ),
        ProcessingRule(
            rule_id="athlete_enrich_performance",
            operation=ProcessingOperation.ENRICH,
            parameters={
                'type': 'derived',
                'derivations': {
                    'exertion_index': 'heart_rate + speed * 10'
                }
            }
        )
    ]

def create_military_processing_rules() -> List[ProcessingRule]:
    """Create common processing rules for military data."""
    return [
        ProcessingRule(
            rule_id="military_validate_position",
            operation=ProcessingOperation.VALIDATE,
            parameters={
                'rules': {
                    'position': {'required': True, 'type': 'list'},
                    'personnel_count': {'required': True, 'type': 'number', 'min': 1, 'max': 50}
                },
                'strict': False
            }
        ),
        ProcessingRule(
            rule_id="military_aggregate_status",
            operation=ProcessingOperation.AGGREGATE,
            parameters={
                'window_id': 'military_5min',
                'type': AggregationType.AVERAGE,
                'fields': ['threat_level', 'personnel_count']
            }
        )
    ]