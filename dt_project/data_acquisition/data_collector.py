"""
Data Collector Service for Quantum Trail.
Handles real-time data collection from various sources including sensors,
APIs, databases, and streaming services.
"""

import asyncio
import time
import json
import aiohttp
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import structlog

from dt_project.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)

class DataSourceType(Enum):
    """Types of data sources."""
    SENSOR = "sensor"
    API = "api"
    DATABASE = "database" 
    WEBSOCKET = "websocket"
    FILE = "file"
    STREAM = "stream"
    MOCK = "mock"

class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

@dataclass
class DataPoint:
    """Represents a single data point from a source."""
    timestamp: datetime
    source_id: str
    source_type: DataSourceType
    data: Dict[str, Any]
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'source_id': self.source_id,
            'source_type': self.source_type.value,
            'data': self.data,
            'quality': self.quality.value,
            'metadata': self.metadata
        }

@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    source_id: str
    source_type: DataSourceType
    endpoint: Optional[str] = None
    polling_interval: float = 1.0  # seconds
    timeout: float = 10.0
    max_retries: int = 3
    enabled: bool = True
    authentication: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_validation: Dict[str, Any] = field(default_factory=dict)

class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.is_connected = False
        self.last_data_time = None
        self.error_count = 0
        self.total_data_points = 0
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    async def collect_data(self) -> Optional[DataPoint]:
        """Collect a single data point."""
        pass
    
    async def validate_data(self, data: Dict[str, Any]) -> DataQuality:
        """Validate data quality."""
        validation_rules = self.config.data_validation
        
        if not validation_rules:
            return DataQuality.UNKNOWN
        
        quality_score = 1.0
        
        # Check required fields
        required_fields = validation_rules.get('required_fields', [])
        for field in required_fields:
            if field not in data:
                quality_score -= 0.3
        
        # Check data ranges
        ranges = validation_rules.get('ranges', {})
        for field, (min_val, max_val) in ranges.items():
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        quality_score -= 0.2
        
        # Check data freshness
        max_age = validation_rules.get('max_age_seconds', 300)
        if self.last_data_time:
            age = (datetime.utcnow() - self.last_data_time).total_seconds()
            if age > max_age:
                quality_score -= 0.2
        
        # Determine quality level
        if quality_score >= 0.8:
            return DataQuality.HIGH
        elif quality_score >= 0.5:
            return DataQuality.MEDIUM
        else:
            return DataQuality.LOW

class SensorDataSource(DataSource):
    """Data source for physical sensors (IoT devices, wearables, etc.)."""
    
    async def connect(self) -> bool:
        """Connect to sensor via HTTP/MQTT/etc."""
        try:
            # Mock sensor connection
            await asyncio.sleep(0.1)
            self.is_connected = True
            logger.info(f"Connected to sensor {self.config.source_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to sensor {self.config.source_id}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from sensor."""
        self.is_connected = False
        logger.info(f"Disconnected from sensor {self.config.source_id}")
        return True
    
    async def collect_data(self) -> Optional[DataPoint]:
        """Collect sensor data."""
        if not self.is_connected:
            return None
        
        try:
            # Simulate sensor data collection
            if self.config.source_id.startswith('athlete'):
                data = self._generate_athlete_sensor_data()
            elif self.config.source_id.startswith('military'):
                data = self._generate_military_sensor_data()
            elif self.config.source_id.startswith('environmental'):
                data = self._generate_environmental_data()
            else:
                data = self._generate_generic_sensor_data()
            
            quality = await self.validate_data(data)
            self.last_data_time = datetime.utcnow()
            self.total_data_points += 1
            
            return DataPoint(
                timestamp=self.last_data_time,
                source_id=self.config.source_id,
                source_type=DataSourceType.SENSOR,
                data=data,
                quality=quality,
                metadata={'sensor_type': self.config.parameters.get('sensor_type', 'unknown')}
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to collect data from sensor {self.config.source_id}: {e}")
            return None
    
    def _generate_athlete_sensor_data(self) -> Dict[str, Any]:
        """Generate realistic athlete sensor data."""
        base_heart_rate = 70
        base_speed = 0
        
        # Add realistic variation
        heart_rate = base_heart_rate + np.random.normal(0, 10)
        speed = max(0, base_speed + np.random.exponential(2))
        
        return {
            'heart_rate': max(50, min(200, int(heart_rate))),
            'speed': round(speed, 2),
            'acceleration': [
                round(np.random.normal(0, 0.5), 3),
                round(np.random.normal(0, 0.5), 3), 
                round(np.random.normal(9.8, 0.3), 3)
            ],
            'gps_position': [
                40.7589 + np.random.normal(0, 0.001),  # NYC coordinates with noise
                -73.9851 + np.random.normal(0, 0.001)
            ],
            'body_temperature': round(37.0 + np.random.normal(0, 0.5), 1),
            'hydration_level': round(np.random.uniform(0.6, 1.0), 2),
            'calories_burned': int(np.random.uniform(0, 10)),
            'step_count': int(np.random.uniform(0, 20))
        }
    
    def _generate_military_sensor_data(self) -> Dict[str, Any]:
        """Generate military unit sensor data."""
        return {
            'position': [
                35.6762 + np.random.normal(0, 0.001),  # Mock military coordinates
                139.6503 + np.random.normal(0, 0.001)
            ],
            'heading': int(np.random.uniform(0, 360)),
            'elevation': int(np.random.normal(100, 50)),
            'equipment_status': {
                'radio': np.random.choice(['operational', 'degraded', 'offline'], p=[0.8, 0.15, 0.05]),
                'gps': np.random.choice(['operational', 'degraded'], p=[0.95, 0.05]),
                'weapon_system': np.random.choice(['ready', 'maintenance'], p=[0.9, 0.1]),
                'vehicle': np.random.choice(['operational', 'maintenance', 'offline'], p=[0.85, 0.1, 0.05])
            },
            'communication_strength': round(np.random.uniform(0.3, 1.0), 2),
            'threat_level': int(np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.08, 0.02])),
            'personnel_count': int(np.random.normal(8, 2)),
            'fuel_level': round(np.random.uniform(0.2, 1.0), 2)
        }
    
    def _generate_environmental_data(self) -> Dict[str, Any]:
        """Generate environmental sensor data."""
        return {
            'temperature': round(np.random.normal(22, 5), 1),
            'humidity': round(np.random.uniform(30, 90), 1),
            'pressure': round(np.random.normal(1013.25, 10), 2),
            'wind_speed': round(np.random.exponential(5), 1),
            'wind_direction': int(np.random.uniform(0, 360)),
            'precipitation': round(max(0, np.random.exponential(0.5)), 2),
            'visibility': round(np.random.uniform(1, 15), 1),
            'air_quality_index': int(np.random.uniform(20, 150)),
            'uv_index': int(np.random.uniform(0, 11))
        }
    
    def _generate_generic_sensor_data(self) -> Dict[str, Any]:
        """Generate generic sensor data."""
        return {
            'value': round(np.random.normal(0, 1), 3),
            'status': np.random.choice(['normal', 'warning', 'error'], p=[0.85, 0.12, 0.03]),
            'battery_level': round(np.random.uniform(0.1, 1.0), 2),
            'signal_strength': round(np.random.uniform(0.3, 1.0), 2)
        }

class APIDataSource(DataSource):
    """Data source for REST APIs and web services."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.session = None
    
    async def connect(self) -> bool:
        """Establish HTTP session."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            self.is_connected = True
            logger.info(f"Connected to API {self.config.source_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to API {self.config.source_id}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
        self.is_connected = False
        logger.info(f"Disconnected from API {self.config.source_id}")
        return True
    
    async def collect_data(self) -> Optional[DataPoint]:
        """Fetch data from API endpoint."""
        if not self.is_connected or not self.session:
            return None
        
        try:
            # Prepare request
            headers = self.config.authentication.get('headers', {})
            params = self.config.parameters.get('query_params', {})
            
            async with self.session.get(
                self.config.endpoint,
                headers=headers,
                params=params
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    quality = await self.validate_data(data)
                    self.last_data_time = datetime.utcnow()
                    self.total_data_points += 1
                    
                    return DataPoint(
                        timestamp=self.last_data_time,
                        source_id=self.config.source_id,
                        source_type=DataSourceType.API,
                        data=data,
                        quality=quality,
                        metadata={
                            'status_code': response.status,
                            'response_headers': dict(response.headers)
                        }
                    )
                else:
                    logger.warning(f"API {self.config.source_id} returned status {response.status}")
                    return None
                    
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to collect data from API {self.config.source_id}: {e}")
            return None

class WebSocketDataSource(DataSource):
    """Data source for WebSocket streams."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.websocket = None
    
    async def connect(self) -> bool:
        """Connect to WebSocket."""
        try:
            self.websocket = await websockets.connect(self.config.endpoint)
            self.is_connected = True
            logger.info(f"Connected to WebSocket {self.config.source_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket {self.config.source_id}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info(f"Disconnected from WebSocket {self.config.source_id}")
        return True
    
    async def collect_data(self) -> Optional[DataPoint]:
        """Receive data from WebSocket."""
        if not self.is_connected or not self.websocket:
            return None
        
        try:
            # Wait for message with timeout
            message = await asyncio.wait_for(
                self.websocket.recv(), 
                timeout=self.config.timeout
            )
            
            data = json.loads(message)
            quality = await self.validate_data(data)
            self.last_data_time = datetime.utcnow()
            self.total_data_points += 1
            
            return DataPoint(
                timestamp=self.last_data_time,
                source_id=self.config.source_id,
                source_type=DataSourceType.WEBSOCKET,
                data=data,
                quality=quality,
                metadata={'message_length': len(message)}
            )
            
        except asyncio.TimeoutError:
            # No data received within timeout - this is normal
            return None
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to collect data from WebSocket {self.config.source_id}: {e}")
            return None

class MockDataSource(DataSource):
    """Mock data source for testing and development."""
    
    async def connect(self) -> bool:
        """Mock connection."""
        self.is_connected = True
        logger.info(f"Connected to mock source {self.config.source_id}")
        return True
    
    async def disconnect(self) -> bool:
        """Mock disconnection."""
        self.is_connected = False
        logger.info(f"Disconnected from mock source {self.config.source_id}")
        return True
    
    async def collect_data(self) -> Optional[DataPoint]:
        """Generate mock data."""
        if not self.is_connected:
            return None
        
        # Generate different types of mock data
        data_type = self.config.parameters.get('data_type', 'random')
        
        if data_type == 'athlete':
            data = {
                'heart_rate': int(np.random.normal(75, 15)),
                'speed': round(np.random.exponential(3), 2),
                'distance': round(np.random.uniform(0, 1000), 2)
            }
        elif data_type == 'military':
            data = {
                'position': [np.random.uniform(-90, 90), np.random.uniform(-180, 180)],
                'status': np.random.choice(['green', 'yellow', 'red'], p=[0.8, 0.15, 0.05]),
                'personnel': int(np.random.uniform(5, 20))
            }
        else:
            data = {
                'value': round(np.random.normal(0, 1), 3),
                'timestamp_offset': int(time.time())
            }
        
        self.last_data_time = datetime.utcnow()
        self.total_data_points += 1
        
        return DataPoint(
            timestamp=self.last_data_time,
            source_id=self.config.source_id,
            source_type=DataSourceType.MOCK,
            data=data,
            quality=DataQuality.HIGH,
            metadata={'generated': True}
        )

class DataCollector:
    """Main data collector that manages multiple data sources."""
    
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {}
        self.data_handlers: List[Callable[[DataPoint], None]] = []
        self.is_running = False
        self.collection_tasks: Dict[str, asyncio.Task] = {}
        self.stats = {
            'total_data_points': 0,
            'data_points_by_source': {},
            'errors_by_source': {},
            'start_time': None
        }
    
    def add_data_source(self, config: DataSourceConfig) -> bool:
        """Add a new data source."""
        try:
            # Create appropriate data source instance
            if config.source_type == DataSourceType.SENSOR:
                source = SensorDataSource(config)
            elif config.source_type == DataSourceType.API:
                source = APIDataSource(config)
            elif config.source_type == DataSourceType.WEBSOCKET:
                source = WebSocketDataSource(config)
            elif config.source_type == DataSourceType.MOCK:
                source = MockDataSource(config)
            else:
                logger.error(f"Unsupported data source type: {config.source_type}")
                return False
            
            self.data_sources[config.source_id] = source
            self.stats['data_points_by_source'][config.source_id] = 0
            self.stats['errors_by_source'][config.source_id] = 0
            
            logger.info(f"Added data source {config.source_id} of type {config.source_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add data source {config.source_id}: {e}")
            return False
    
    def remove_data_source(self, source_id: str) -> bool:
        """Remove a data source."""
        if source_id in self.data_sources:
            # Stop collection task if running
            if source_id in self.collection_tasks:
                self.collection_tasks[source_id].cancel()
                del self.collection_tasks[source_id]
            
            # Disconnect and remove source
            asyncio.create_task(self.data_sources[source_id].disconnect())
            del self.data_sources[source_id]
            
            logger.info(f"Removed data source {source_id}")
            return True
        else:
            logger.warning(f"Data source {source_id} not found")
            return False
    
    def add_data_handler(self, handler: Callable[[DataPoint], None]):
        """Add a data handler function."""
        self.data_handlers.append(handler)
        logger.info(f"Added data handler: {handler.__name__}")
    
    async def start_collection(self) -> bool:
        """Start data collection from all sources."""
        if self.is_running:
            logger.warning("Data collection is already running")
            return False
        
        try:
            # Connect to all data sources
            connection_results = await asyncio.gather(
                *[source.connect() for source in self.data_sources.values()],
                return_exceptions=True
            )
            
            connected_sources = []
            for i, (source_id, source) in enumerate(self.data_sources.items()):
                if isinstance(connection_results[i], bool) and connection_results[i]:
                    connected_sources.append(source_id)
                else:
                    logger.error(f"Failed to connect to {source_id}: {connection_results[i]}")
            
            if not connected_sources:
                logger.error("No data sources connected successfully")
                return False
            
            # Start collection tasks for connected sources
            for source_id in connected_sources:
                task = asyncio.create_task(self._collect_from_source(source_id))
                self.collection_tasks[source_id] = task
            
            self.is_running = True
            self.stats['start_time'] = datetime.utcnow()
            
            logger.info(f"Started data collection from {len(connected_sources)} sources")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start data collection: {e}")
            return False
    
    async def stop_collection(self) -> bool:
        """Stop data collection from all sources."""
        if not self.is_running:
            logger.warning("Data collection is not running")
            return False
        
        try:
            # Cancel all collection tasks
            for task in self.collection_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.collection_tasks.values(), return_exceptions=True)
            self.collection_tasks.clear()
            
            # Disconnect from all data sources
            await asyncio.gather(
                *[source.disconnect() for source in self.data_sources.values()],
                return_exceptions=True
            )
            
            self.is_running = False
            logger.info("Stopped data collection from all sources")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop data collection: {e}")
            return False
    
    async def _collect_from_source(self, source_id: str):
        """Collect data from a specific source continuously."""
        source = self.data_sources[source_id]
        
        logger.info(f"Starting data collection from {source_id}")
        
        while self.is_running:
            try:
                # Collect data point
                data_point = await source.collect_data()
                
                if data_point:
                    # Update statistics
                    self.stats['total_data_points'] += 1
                    self.stats['data_points_by_source'][source_id] += 1
                    
                    # Process data through handlers
                    for handler in self.data_handlers:
                        try:
                            await self._run_handler(handler, data_point)
                        except Exception as e:
                            logger.error(f"Data handler {handler.__name__} failed: {e}")
                    
                    # Record metrics
                    if metrics:
                        metrics.data_points_collected_total.labels(
                            source=source_id, 
                            source_type=source.config.source_type.value
                        ).inc()
                        
                        metrics.data_quality.labels(
                            source=source_id, 
                            quality=data_point.quality.value
                        ).inc()
                
                # Update error statistics
                self.stats['errors_by_source'][source_id] = source.error_count
                
                # Wait before next collection
                await asyncio.sleep(source.config.polling_interval)
                
            except asyncio.CancelledError:
                logger.info(f"Data collection cancelled for {source_id}")
                break
            except Exception as e:
                logger.error(f"Error in data collection from {source_id}: {e}")
                await asyncio.sleep(min(source.config.polling_interval * 2, 10))
    
    async def _run_handler(self, handler: Callable, data_point: DataPoint):
        """Run a data handler, supporting both sync and async handlers."""
        if asyncio.iscoroutinefunction(handler):
            await handler(data_point)
        else:
            handler(data_point)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = self.stats.copy()
        
        # Add runtime statistics
        if stats['start_time']:
            stats['uptime_seconds'] = (datetime.utcnow() - stats['start_time']).total_seconds()
        
        # Add source statistics
        stats['active_sources'] = len([s for s in self.data_sources.values() if s.is_connected])
        stats['total_sources'] = len(self.data_sources)
        
        return stats
    
    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources."""
        status = {}
        
        for source_id, source in self.data_sources.items():
            status[source_id] = {
                'connected': source.is_connected,
                'source_type': source.config.source_type.value,
                'total_data_points': source.total_data_points,
                'error_count': source.error_count,
                'last_data_time': source.last_data_time.isoformat() if source.last_data_time else None,
                'enabled': source.config.enabled
            }
        
        return status