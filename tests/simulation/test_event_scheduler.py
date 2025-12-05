import pytest
from h2_plant.simulation.event_scheduler import Event, EventScheduler
from h2_plant.core.component_registry import ComponentRegistry

def test_event_creation():
    """Test the Event dataclass."""
    def dummy_handler(reg): pass
    event = Event(hour=10, event_type="test", handler=dummy_handler)
    assert event.hour == 10
    assert event.event_type == "test"
    assert not event.recurring

def test_schedule_event(mocker):
    """Test scheduling a single event."""
    scheduler = EventScheduler()
    handler = mocker.Mock()
    
    event = Event(hour=5, event_type="test_event", handler=handler)
    scheduler.schedule(event)

    pending = scheduler.get_pending_events()
    assert len(pending) == 1
    assert pending[0].hour == 5

def test_process_events_at_correct_hour(mocker):
    """Test that events are processed only at their scheduled hour."""
    scheduler = EventScheduler()
    handler_hr5 = mocker.Mock()
    registry = ComponentRegistry()

    event = Event(hour=5, event_type="test_event", handler=handler_hr5)
    scheduler.schedule(event)

    # Process at hour 4, should do nothing
    scheduler.process_events(current_hour=4, registry=registry)
    handler_hr5.assert_not_called()
    assert len(scheduler.get_pending_events()) == 1

    # Process at hour 5, should execute
    scheduler.process_events(current_hour=5, registry=registry)
    handler_hr5.assert_called_once_with(registry)
    assert len(scheduler.get_pending_events()) == 0

def test_recurring_event(mocker):
    """Test that recurring events are rescheduled."""
    scheduler = EventScheduler()
    handler = mocker.Mock()
    registry = ComponentRegistry()

    event = Event(
        hour=10,
        event_type="recurring_test",
        handler=handler,
        recurring=True,
        recurrence_interval=5
    )
    scheduler.schedule(event)

    # Process at hour 10
    scheduler.process_events(current_hour=10, registry=registry)
    handler.assert_called_once()
    
    # Check if it was rescheduled for hour 15
    pending = scheduler.get_pending_events()
    assert len(pending) == 1
    assert pending[0].hour == 15
    assert pending[0].event_type == "recurring_test"

    # Process at hour 15
    scheduler.process_events(current_hour=15, registry=registry)
    assert handler.call_count == 2
    
    # Check if it was rescheduled for hour 20
    pending = scheduler.get_pending_events()
    assert len(pending) == 1
    assert pending[0].hour == 20

def test_clear_events(mocker):
    """Test clearing all scheduled events."""
    scheduler = EventScheduler()
    scheduler.schedule(Event(hour=1, event_type="e1", handler=mocker.Mock()))
    scheduler.schedule(Event(hour=2, event_type="e2", handler=mocker.Mock()))
    
    assert len(scheduler.get_pending_events()) == 2
    scheduler.clear_events()
    assert len(scheduler.get_pending_events()) == 0
