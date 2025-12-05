"""
Event scheduling system for simulation.

Supports:
- Time-based events (executed at specific hours)
- Recurring events (periodic execution)
- Component modification events
- Custom event handlers
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod

from h2_plant.core.component_registry import ComponentRegistry

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """
    Simulation event.
    
    Example:
        # Component state change event
        event = Event(
            hour=100,
            event_type="maintenance",
            handler=lambda reg: reg.get("electrolyzer").set_state(ProductionState.MAINTENANCE),
            metadata={"component": "electrolyzer", "duration": 24}
        )
    """
    hour: int                                    # Hour to execute event
    event_type: str                              # Event type identifier
    handler: Callable[[ComponentRegistry], None] # Event execution function
    metadata: Dict[str, Any] = field(default_factory=dict) # Additional event data
    recurring: bool = False                      # If True, reschedule after execution
    recurrence_interval: int = 0                 # Hours between recurrences


class EventScheduler:
    """
    Manages scheduled events during simulation.
    
    Example:
        scheduler = EventScheduler()
        
        # Schedule maintenance
        scheduler.schedule(Event(
            hour=1000,
            event_type="maintenance",
            handler=lambda reg: reg.get("electrolyzer").shutdown()
        ))
        
        # Process events each timestep
        for hour in range(8760):
            scheduler.process_events(hour, registry)
    """
    
    def __init__(self):
        """Initialize event scheduler."""
        self._events: List[Event] = []
    
    def schedule(self, event: Event) -> None:
        """
        Schedule an event.
        
        Args:
            event: Event to schedule
        """
        self._events.append(event)
        self._events.sort(key=lambda e: e.hour)  # Keep sorted by hour
        
        logger.debug(f"Scheduled {event.event_type} event at hour {event.hour}")
    
    def process_events(self, current_hour: int, registry: ComponentRegistry) -> None:
        """
        Process all events scheduled for current hour.
        
        Args:
            current_hour: Current simulation hour
            registry: ComponentRegistry for event handlers
        """
        # Find events for this hour
        events_to_execute = [e for e in self._events if e.hour == current_hour]
        
        if not events_to_execute:
            return
        
        logger.info(f"Processing {len(events_to_execute)} events at hour {current_hour}")
        
        for event in events_to_execute:
            try:
                # Execute event handler
                event.handler(registry)
                
                logger.info(f"Executed {event.event_type} event")
                
                # Handle recurring events
                if event.recurring and event.recurrence_interval > 0:
                    next_event = Event(
                        hour=current_hour + event.recurrence_interval,
                        event_type=event.event_type,
                        handler=event.handler,
                        metadata=event.metadata,
                        recurring=True,
                        recurrence_interval=event.recurrence_interval
                    )
                    self.schedule(next_event)
                
            except Exception as e:
                logger.error(f"Event {event.event_type} failed at hour {current_hour}: {e}")
                # Continue processing other events
        
        # Remove executed events for the current hour
        self._events = [e for e in self._events if e.hour != current_hour]
    
    def get_pending_events(self) -> List[Event]:
        """Return list of all pending events."""
        return self._events.copy()
    
    def clear_events(self) -> None:
        """Clear all scheduled events."""
        self._events.clear()


# Common event factories

def create_maintenance_event(
    hour: int,
    component_id: str,
    duration_hours: int
) -> Event:
    """
    Create component maintenance event.
    
    Args:
        hour: Hour to start maintenance
        component_id: Component to put in maintenance
        duration_hours: Maintenance duration
        
    Returns:
        Maintenance start event
    """
    def handler(registry: ComponentRegistry):
        component = registry.get(component_id)
        if hasattr(component, 'state'):
            from h2_plant.core.enums import ProductionState
            component.state = ProductionState.MAINTENANCE
            logger.info(f"Component {component_id} entering maintenance")
    
    return Event(
        hour=hour,
        event_type="maintenance_start",
        handler=handler,
        metadata={"component": component_id, "duration": duration_hours}
    )


def create_price_update_event(
    hour: int,
    new_price: float
) -> Event:
    """
    Create energy price update event.
    
    Args:
        hour: Hour to update price
        new_price: New energy price ($/MWh)
        
    Returns:
        Price update event
    """
    def handler(registry: ComponentRegistry):
        if registry.has('energy_price_tracker'):
            tracker = registry.get('energy_price_tracker')
            tracker.current_price_per_mwh = new_price
            logger.info(f"Energy price updated to ${new_price:.2f}/MWh")
    
    return Event(
        hour=hour,
        event_type="price_update",
        handler=handler,
        metadata={"new_price": new_price}
    )
