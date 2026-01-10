"""
File output handler
"""
import logging
from pathlib import Path

from icalendar import Calendar


class Handler:
    """File writer handler class"""
    
    def __init__(self, output_file: str):
        """
        Initialize file output handler
        
        Args:
            output_file: Output file path
        """
        self.output_file = Path(output_file)
    
    def __call__(self, calendar: Calendar) -> None:
        """
        Write ICS content to file
        
        Args:
            calendar: icalendar Calendar object
        """
        try:
            ics_content = calendar.to_ical().decode('utf-8')
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(ics_content)
            logging.info(f"ICS content written to file: {self.output_file}")
            
        except Exception as e:
            logging.error(f"Failed to write file: {e}")
