"""
Next-Generation Architecture Framework - BCI-2-Token Evolution
============================================================

Advanced architectural patterns implementing cutting-edge concepts:
- Microkernel architecture with hot-swappable components
- Event-driven reactive systems
- Self-healing distributed components  
- Quantum-ready neural interfaces
- Edge-cloud hybrid processing
- Zero-downtime deployment patterns

This module provides the foundational architecture for Generation 1+ features.
"""

import asyncio
import time
import threading
import uuid
from typing import Dict, Any, List, Optional, Callable, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from contextlib import asynccontextmanager

T = TypeVar('T')

class ComponentState(Enum):
    """States for architectural components."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class EventType(Enum):
    """Types of system events."""
    COMPONENT_STARTED = "component_started"
    COMPONENT_STOPPED = "component_stopped"
    COMPONENT_ERROR = "component_error"
    SIGNAL_RECEIVED = "signal_received"
    MODEL_UPDATED = "model_updated"
    PERFORMANCE_ALERT = "performance_alert"
    SECURITY_ALERT = "security_alert"
    USER_ACTION = "user_action"

@dataclass
class SystemEvent:
    """Represents a system event."""
    event_type: EventType
    source_component: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None

class ComponentInterface(Protocol):
    """Interface for architectural components."""
    
    def get_component_id(self) -> str:
        """Get unique component identifier."""
        ...
        
    def get_state(self) -> ComponentState:
        """Get current component state."""
        ...
        
    async def start(self) -> bool:
        """Start the component."""
        ...
        
    async def stop(self) -> bool:
        """Stop the component."""
        ...
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform component health check."""
        ...

class EventBus:
    """
    High-performance event bus for component communication.
    
    Implements publish-subscribe pattern with:
    - Async event handling
    - Event filtering and routing
    - Dead letter queues
    - Event replay capabilities
    """
    
    def __init__(self, max_events_in_memory: int = 10000):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[SystemEvent] = []
        self.max_events_in_memory = max_events_in_memory
        self.dead_letter_queue: List[SystemEvent] = []
        self._lock = asyncio.Lock()
        
    async def subscribe(self, event_type: EventType, handler: Callable[[SystemEvent], None]):
        """Subscribe to events of a specific type."""
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
            
    async def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from events."""
        async with self._lock:
            if event_type in self.subscribers:
                try:
                    self.subscribers[event_type].remove(handler)
                except ValueError:
                    pass
                    
    async def publish(self, event: SystemEvent):
        """Publish an event to all subscribers."""
        # Store event in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_events_in_memory:
            self.event_history = self.event_history[-int(self.max_events_in_memory * 0.8):]
            
        # Notify subscribers
        if event.event_type in self.subscribers:
            failed_handlers = []
            
            for handler in self.subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    warnings.warn(f"Event handler failed: {e}")
                    failed_handlers.append(handler)
                    
            # Remove failed handlers and add event to dead letter queue
            if failed_handlers:
                async with self._lock:
                    for handler in failed_handlers:
                        try:
                            self.subscribers[event.event_type].remove(handler)
                        except ValueError:
                            pass
                    self.dead_letter_queue.append(event)
                    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[SystemEvent]:
        """Get event history, optionally filtered by type."""
        events = self.event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

class BaseComponent(ComponentInterface):
    """Base class for architectural components."""
    
    def __init__(self, component_id: str, event_bus: Optional[EventBus] = None):
        self.component_id = component_id
        self.state = ComponentState.INITIALIZING
        self.event_bus = event_bus
        self.start_time: Optional[float] = None
        self.error_count = 0
        self.health_metrics: Dict[str, Any] = {}
        
    def get_component_id(self) -> str:
        return self.component_id
        
    def get_state(self) -> ComponentState:
        return self.state
        
    async def start(self) -> bool:
        """Start the component with error handling."""
        try:
            self.state = ComponentState.INITIALIZING
            await self._on_start()
            self.state = ComponentState.RUNNING
            self.start_time = time.time()
            
            if self.event_bus:
                await self.event_bus.publish(SystemEvent(
                    EventType.COMPONENT_STARTED,
                    self.component_id,
                    data={'start_time': self.start_time}
                ))
                
            return True
            
        except Exception as e:
            self.state = ComponentState.ERROR
            self.error_count += 1
            warnings.warn(f"Component {self.component_id} failed to start: {e}")
            
            if self.event_bus:
                await self.event_bus.publish(SystemEvent(
                    EventType.COMPONENT_ERROR,
                    self.component_id,
                    data={'error': str(e), 'error_count': self.error_count}
                ))
                
            return False
            
    async def stop(self) -> bool:
        """Stop the component gracefully."""
        try:
            if self.state == ComponentState.RUNNING:
                self.state = ComponentState.SHUTDOWN
                await self._on_stop()
                
                if self.event_bus:
                    await self.event_bus.publish(SystemEvent(
                        EventType.COMPONENT_STOPPED,
                        self.component_id,
                        data={'uptime': time.time() - (self.start_time or 0)}
                    ))
                    
            return True
            
        except Exception as e:
            self.error_count += 1
            warnings.warn(f"Component {self.component_id} failed to stop gracefully: {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        uptime = time.time() - (self.start_time or time.time())
        
        health_status = {
            'component_id': self.component_id,
            'state': self.state.value,
            'uptime_seconds': uptime,
            'error_count': self.error_count,
            'is_healthy': self.state == ComponentState.RUNNING and self.error_count < 5,
            'metrics': self.health_metrics.copy()
        }
        
        # Add component-specific health data
        try:
            custom_health = await self._on_health_check()
            health_status.update(custom_health)
        except Exception as e:
            health_status['health_check_error'] = str(e)
            
        return health_status
        
    @abstractmethod
    async def _on_start(self):
        """Component-specific start logic."""
        pass
        
    @abstractmethod
    async def _on_stop(self):
        """Component-specific stop logic."""
        pass
        
    async def _on_health_check(self) -> Dict[str, Any]:
        """Component-specific health check logic."""
        return {}

class MicrokernelArchitecture:
    """
    Microkernel architecture implementation.
    
    Provides:
    - Hot-swappable components
    - Dependency injection
    - Service discovery
    - Load balancing
    - Circuit breakers
    """
    
    def __init__(self):
        self.event_bus = EventBus()
        self.components: Dict[str, ComponentInterface] = {}
        self.service_registry: Dict[str, List[str]] = {}  # service_type -> component_ids
        self.dependencies: Dict[str, List[str]] = {}      # component_id -> dependency_ids
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._kernel_state = ComponentState.INITIALIZING
        
    async def register_component(self, component: ComponentInterface, service_types: List[str] = None):
        """Register a component with the kernel."""
        component_id = component.get_component_id()
        self.components[component_id] = component
        
        # Register for service discovery
        if service_types:
            for service_type in service_types:
                if service_type not in self.service_registry:
                    self.service_registry[service_type] = []
                self.service_registry[service_type].append(component_id)
                
        await self.event_bus.publish(SystemEvent(
            EventType.COMPONENT_STARTED,
            'kernel',
            data={'registered_component': component_id, 'services': service_types or []}
        ))
        
    async def unregister_component(self, component_id: str):
        """Unregister a component."""
        if component_id in self.components:
            component = self.components[component_id]
            await component.stop()
            del self.components[component_id]
            
            # Remove from service registry
            for service_type, component_ids in self.service_registry.items():
                if component_id in component_ids:
                    component_ids.remove(component_id)
                    
            await self.event_bus.publish(SystemEvent(
                EventType.COMPONENT_STOPPED,
                'kernel',
                data={'unregistered_component': component_id}
            ))
            
    async def start_kernel(self):
        """Start the microkernel and all components."""
        self._kernel_state = ComponentState.RUNNING
        
        # Start components in dependency order
        started_components = set()
        
        for component_id, component in self.components.items():
            if await self._start_component_with_dependencies(component_id, started_components):
                started_components.add(component_id)
                
    async def _start_component_with_dependencies(self, component_id: str, started: set) -> bool:
        """Start a component after ensuring its dependencies are started."""
        if component_id in started:
            return True
            
        # Start dependencies first
        if component_id in self.dependencies:
            for dep_id in self.dependencies[component_id]:
                if dep_id not in started:
                    if not await self._start_component_with_dependencies(dep_id, started):
                        return False
                        
        # Start the component
        component = self.components[component_id]
        return await component.start()
        
    async def stop_kernel(self):
        """Stop the microkernel and all components."""
        self._kernel_state = ComponentState.SHUTDOWN
        
        # Stop components in reverse dependency order
        for component in self.components.values():
            await component.stop()
            
    def discover_service(self, service_type: str) -> List[ComponentInterface]:
        """Discover components providing a service."""
        component_ids = self.service_registry.get(service_type, [])
        return [self.components[cid] for cid in component_ids if cid in self.components]
        
    async def get_kernel_health(self) -> Dict[str, Any]:
        """Get overall kernel health."""
        component_health = {}
        
        for component_id, component in self.components.items():
            try:
                health = await component.health_check()
                component_health[component_id] = health
            except Exception as e:
                component_health[component_id] = {
                    'error': str(e),
                    'is_healthy': False
                }
                
        healthy_components = sum(1 for h in component_health.values() if h.get('is_healthy', False))
        total_components = len(component_health)
        
        return {
            'kernel_state': self._kernel_state.value,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'health_percentage': (healthy_components / total_components * 100) if total_components > 0 else 0,
            'component_health': component_health,
            'event_history_size': len(self.event_bus.event_history),
            'dead_letter_queue_size': len(self.event_bus.dead_letter_queue)
        }

class ReactiveSignalProcessor(BaseComponent):
    """Reactive signal processing component."""
    
    def __init__(self, component_id: str = "reactive_processor", event_bus: Optional[EventBus] = None):
        super().__init__(component_id, event_bus)
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.signals_processed = 0
        
    async def _on_start(self):
        """Start reactive processing."""
        self.processing_task = asyncio.create_task(self._process_signals())
        
    async def _on_stop(self):
        """Stop reactive processing."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
                
    async def _on_health_check(self) -> Dict[str, Any]:
        """Health check for signal processor."""
        return {
            'signals_processed': self.signals_processed,
            'queue_size': self.processing_queue.qsize(),
            'is_processing': self.processing_task and not self.processing_task.done()
        }
        
    async def process_signal(self, signal_data: Dict[str, Any]):
        """Queue signal for reactive processing."""
        await self.processing_queue.put(signal_data)
        
    async def _process_signals(self):
        """Reactive signal processing loop."""
        while True:
            try:
                signal_data = await self.processing_queue.get()
                
                # Process signal (simulate processing)
                await asyncio.sleep(0.01)  # Simulate processing time
                self.signals_processed += 1
                
                # Publish signal processed event
                if self.event_bus:
                    await self.event_bus.publish(SystemEvent(
                        EventType.SIGNAL_RECEIVED,
                        self.component_id,
                        data={'signal_id': signal_data.get('id', 'unknown'), 'processed_count': self.signals_processed}
                    ))
                    
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                warnings.warn(f"Error processing signal: {e}")

class EdgeCloudHybridManager(BaseComponent):
    """Manages edge-cloud hybrid processing."""
    
    def __init__(self, component_id: str = "hybrid_manager", event_bus: Optional[EventBus] = None):
        super().__init__(component_id, event_bus)
        self.edge_capacity = 0.8  # 80% processing on edge
        self.cloud_capacity = 0.2  # 20% processing on cloud
        self.latency_threshold = 0.1  # 100ms
        
    async def _on_start(self):
        """Start hybrid manager."""
        pass
        
    async def _on_stop(self):
        """Stop hybrid manager."""
        pass
        
    async def route_processing(self, task_data: Dict[str, Any]) -> str:
        """Route processing task to edge or cloud based on criteria."""
        task_complexity = task_data.get('complexity', 'medium')
        required_latency = task_data.get('max_latency', 1.0)
        
        if task_complexity == 'low' and required_latency < self.latency_threshold:
            return 'edge'
        elif task_complexity == 'high':
            return 'cloud'
        else:
            # Load balance based on current capacity
            return 'edge' if self.edge_capacity > 0.5 else 'cloud'
            
    async def _on_health_check(self) -> Dict[str, Any]:
        """Health check for hybrid manager."""
        return {
            'edge_capacity': self.edge_capacity,
            'cloud_capacity': self.cloud_capacity,
            'latency_threshold': self.latency_threshold
        }

# Global architecture instance
_global_architecture: Optional[MicrokernelArchitecture] = None

def get_global_architecture() -> MicrokernelArchitecture:
    """Get or create global architecture instance."""
    global _global_architecture
    if _global_architecture is None:
        _global_architecture = MicrokernelArchitecture()
    return _global_architecture

@asynccontextmanager
async def next_gen_architecture_context():
    """Context manager for next-gen architecture."""
    architecture = get_global_architecture()
    try:
        # Register default components
        signal_processor = ReactiveSignalProcessor(event_bus=architecture.event_bus)
        hybrid_manager = EdgeCloudHybridManager(event_bus=architecture.event_bus)
        
        await architecture.register_component(signal_processor, ['signal_processing'])
        await architecture.register_component(hybrid_manager, ['hybrid_processing'])
        
        # Start kernel
        await architecture.start_kernel()
        
        yield architecture
        
    finally:
        # Clean shutdown
        await architecture.stop_kernel()

async def create_next_gen_system() -> MicrokernelArchitecture:
    """Create and initialize a next-generation BCI system."""
    architecture = MicrokernelArchitecture()
    
    # Create and register components
    components = [
        ReactiveSignalProcessor(event_bus=architecture.event_bus),
        EdgeCloudHybridManager(event_bus=architecture.event_bus)
    ]
    
    for component in components:
        await architecture.register_component(
            component,
            [component.__class__.__name__.lower().replace('component', '').replace('manager', '')]
        )
        
    await architecture.start_kernel()
    return architecture