"""
Formatted display handler
"""
import dateutil
from icalendar import Calendar, vDDDTypes, vInt


class Handler:
    """Handler class for formatted event information display"""
    def __call__(
        self,
        calendar: Calendar,
    ) -> None:
        """
        Format and output event information to console
        
        Args:
            calendar: icalendar Calendar object
        """
        events: list[dict[str, object]] = []
        
        # Iterate through all event components
        for component in calendar.subcomponents:
            x_alarm_trigger = component.get('X-ALARM-TRIGGER')
            trigger_time = dateutil.parser.parse(str(x_alarm_trigger))

            dtstart: vDDDTypes | None = component.get('DTSTART')
            start_time = dtstart.dt if dtstart and hasattr(dtstart, 'dt') else None

            summary = component.get('SUMMARY')
            description = component.get('DESCRIPTION')

            priority = component.get('PRIORITY')

            events.append({
                'trigger_time': trigger_time,
                'start_time': start_time,
                'summary': summary,
                'description': description,
                'priority': priority,
            })

        # Output formatted results
        print(f"\n{'='*80}")
        print(f"📅 Calendar Events List (Total {len(events)} events)")
        print(f"{'='*80}\n")
        
        for idx, event in enumerate(events, 1):
            """
            Print formatted information for a single event
            
            Args:
                index: Event index
                event: Event information dictionary
            """
            print(f"[{idx}] {event.get('summary', 'No Title')}")
            print(f"{'─' * 80}")
            print(f"⏰ Time: {event.get('start_time', 'Unknown')}")
            print(f"🔔 Alarm: {event.get('trigger_time')}")
            print(f"📝 Desc: {event.get('description', '')}")
            print(f"🔢 Priority: {event.get('priority', 'N/A')}")
            print()  # Empty line between events
    