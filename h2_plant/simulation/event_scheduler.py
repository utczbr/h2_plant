"""
Event Scheduling System for Simulation.

This module provides a time-based event scheduling system that enables
discrete events during continuous simulation. Supports one-time and
recurring events with custom handlers.

Event Types:
    - **Time-based events**: Execute at specific simulation hours.
    - **Recurring events**: Periodic execution with configurable interval.
    - **Component events**: Modify component state (maintenance, shutdown).
    - **System events**: Price updates, demand changes, setpoint modifications.

Integration:
    The SimulationEngine calls `process_events()` at each timestep,
    executing all events scheduled for the current hour.
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
    Simulation event with handler and metadata.

    Events encapsulate actions to be executed at specific simulation times.
    The handler function receives the ComponentRegistry for component access.

    Attributes:
        hour (int): Simulation hour to execute event.
        event_type (str): Event type identifier for logging/filtering.
        handler (Callable): Function receiving ComponentRegistry.
        metadata (Dict): Additional event data for debugging/analysis.
        recurring (bool): If True, reschedule after execution.
        recurrence_interval (int): Hours between recurrences.

    Example:
        >>> event = Event(
        ...     hour=100,
        ...     event_type="maintenance",
        ...     handler=lambda reg: reg.get("electrolyzer").shutdown(),
        ...     metadata={"component": "electrolyzer", "duration": 24}
        ... )
    """
    hour: int
    event_type: str
    handler: Callable[[ComponentRegistry], None]
    metadata: Dict[str, Any] = field(default_factory=dict)
    recurring: bool = False
    recurrence_interval: int = 0


class EventScheduler:
    """
    Manages scheduled events during simulation execution.

    Maintains a sorted list of pending events and processes them
    as simulation time advances. Handles recurring events by
    automatically rescheduling after execution.

    Thread Safety:
        Not thread-safe. Designed for single-threaded simulation loops.

    Example:
        >>> scheduler = EventScheduler()
        >>> scheduler.schedule(Event(
        ...     hour=1000,
        ...     event_type="maintenance",
        ...     handler=lambda reg: reg.get("electrolyzer").shutdown()
        ... ))
        >>> for hour in range(8760):
        ...     scheduler.process_events(hour, registry)
    """

    def __init__(self):
        """Initialize the event scheduler with empty event list."""
        self._events: List[Event] = []

    def schedule(self, event: Event) -> None:
        """
        Schedule an event for future execution.

        Events are maintained in sorted order by hour for efficient
        processing. Duplicate events at the same hour are allowed.

        Args:
            event (Event): Event to schedule.
        """
        self._events.append(event)
        self._events.sort(key=lambda e: e.hour)

        logger.debug(f"Scheduled {event.event_type} event at hour {event.hour}")

    def process_events(self, current_hour: int, registry: ComponentRegistry) -> None:
        """
        Process all events scheduled for the current hour.

        Executes handler for each matching event. Recurring events are
        automatically rescheduled. Failed events are logged but do not
        interrupt processing of remaining events.

        Args:
            current_hour (int): Current simulation hour.
            registry (ComponentRegistry): Registry for handler access.
        """
        events_to_execute = [e for e in self._events if e.hour == current_hour]

        if not events_to_execute:
            return

        logger.info(f"Processing {len(events_to_execute)} events at hour {current_hour}")

        for event in events_to_execute:
            try:
                event.handler(registry)

                logger.info(f"Executed {event.event_type} event")

                # Reschedule recurring events
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

        # Remove executed events
        self._events = [e for e in self._events if e.hour != current_hour]

    def get_pending_events(self) -> List[Event]:
        """
        Return list of all pending events.

        Returns:
            List[Event]: Copy of pending events list.
        """
        return self._events.copy()

    def clear_events(self) -> None:
        """Clear all scheduled events."""
        self._events.clear()


# =============================================================================
# EVENT FACTORIES
# =============================================================================

def create_maintenance_event(
    hour: int,
    component_id: str,
    duration_hours: int
) -> Event:
    """
    Create a component maintenance event.

    Sets component state to MAINTENANCE mode at the specified hour.
    Actual maintenance completion should be handled by a separate event.

    Args:
        hour (int): Hour to start maintenance.
        component_id (str): Component identifier in registry.
        duration_hours (int): Maintenance duration for metadata.

    Returns:
        Event: Configured maintenance start event.

    Example:
        >>> event = create_maintenance_event(1000, "electrolyzer", 24)
        >>> scheduler.schedule(event)
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
    Create an energy price update event.

    Modifies the energy price tracker's current price at the specified hour.

    Args:
        hour (int): Hour to update price.
        new_price (float): New energy price in EUR/MWh.

    Returns:
        Event: Configured price update event.

    Example:
        >>> event = create_price_update_event(500, 75.0)  # High price period
        >>> scheduler.schedule(event)
    """
    def handler(registry: ComponentRegistry):
        if registry.has('energy_price_tracker'):
            tracker = registry.get('energy_price_tracker')
            tracker.current_price_per_mwh = new_price
            logger.info(f"Energy price updated to €{new_price:.2f}/MWh")

    return Event(
        hour=hour,
        event_type="price_update",
        handler=handler,
        metadata={"new_price": new_price}
    )
