""" 
ICS format output handler
"""
from icalendar import Calendar


class Handler:
    """Console output handler class"""
    
    def __call__(self, calendar: Calendar) -> None:
        """
        Output ICS content to console
        
        Args:
            calendar: icalendar Calendar object
        """
        ics_content = calendar.to_ical().decode('utf-8')
        print(f"\n{'='*60}")
        print("ICS Content:")
        print(f"{'='*60}")
        print(ics_content)
        print(f"{'='*60}")
