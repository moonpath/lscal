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
        print(calendar.to_ical().decode('utf-8'), flush=True, end='')
