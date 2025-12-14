"""
Ntfy handler
"""
import logging
from datetime import datetime
from typing import Any

import dateutil
import requests
from icalendar import Component, Calendar, vDDDTypes


class Handler:
    """Handler class for sending to Ntfy webhook"""
    
    def __init__(
        self,
        url: str = "https://ntfy.sh/calendar",
        headers: dict | None = None,
        individual: bool = False,
    ):
        """
        Initialize Ntfy webhook handler
        
        Args:
            url: Ntfy webhook URL
            headers: Custom headers for the request
            individual: Whether to send each event individually. False (default) sends summary message, True sends each event separately
        """
        self.url = url
        self.headers = headers or {}
        self.individual = individual
    
    def __call__(self, calendar: Calendar) -> None:
        """
        Send event information to specified Ntfy webhook
        
        Args:
            calendar: icalendar Calendar object
        """
        events: list[dict[str, object]] = []
        
        # Iterate through all event components
        for component in calendar.subcomponents:
            component: Component
            x_alarm_trigger = component.get('X-ALARM-TRIGGER')
            try:
                trigger_time = dateutil.parser.parse(str(x_alarm_trigger))
            except Exception:
                continue

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

        # Decide sending method based on individual parameter
        if self.individual:
            # Send each event individually
            self._send_individual_event(events)
        else:
            # Send summary message
            self._send_events(events)
    
    def _send_individual_event(self, events: list[dict[str, Any]]) -> None:
        """
        Send individual message for each event to webhook
        
        Args:
            events: Event list
        """
        for event in events:
            # Build message for single event
            lines = []

            summary = event['summary'] or "No Title"
            lines.append(f"📌 Title: {summary}")
            
            # Format time
            start_time = event['start_time'] or "N/A"
            lines.append(f"⏰ Time: {start_time}")
            
            # Description
            description = event['description']
            if description:
                description = str(description).strip()
                if len(description) > 200:
                    description = description[:197] + "..."
                lines.append(f"📝 Desc: {description}")
            
            message = "\n".join(lines).strip()

            priority = int(event.get('priority', 3))
            priority = max(1, min(5, priority))

            trigger_time: datetime = event['trigger_time']

            try:
                headers = {
                    "Content-Type": "text/plain; charset=utf-8",
                    "Title": f"Calendar Event",
                    "Tags": "calendar",
                    "Priority": str(priority),
                    "Delay": str(int(trigger_time.timestamp())),
                }
                headers.update(self.headers)
                
                response = requests.post(
                    self.url,
                    data=message.encode('utf-8'),
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logging.info(f"✅ Event {summary} sent successfully")
                    
                else:
                    logging.error(f"❌ Event {summary} failed to send: HTTP {response.status_code}, {response.text}")
                    
            except Exception as e:
                logging.error(f"❌ Error sending event {summary}: {e}")
    
    def _send_events(self, events: list[dict[str, Any]]) -> None:
        """
        Send event list to webhook
        
        Args:
            events: Event list
        """
        if events:
            priority = max(int(event.get('priority', 3)) for event in events)
            priority = max(1, min(5, priority))
            
            lines = []
            for event in events:
                summary = event['summary'] or "No Title"
                lines.append(f"📌 Title: {summary}")
                
                # 格式化时间
                start_time = event['start_time'] or "N/A"
                lines.append(f"⏰ Time: {start_time}")
                
                # Description
                description = event['description']
                if description:
                    description = str(description).strip()
                    if len(description) > 200:
                        description = description[:197] + "..."
                    lines.append(f"📝 Desc: {description}")
                lines.append("")  # Empty line to separate events
        else:
            priority = 3
            lines = ["No upcoming events."]
        
        message = "\n".join(lines).strip()

        try:
            headers = {
                "Content-Type": "text/plain; charset=utf-8",
                "Title": f"Calendar Events ({len(events)} events)",
                "Tags": "calendar",
                "Priority": str(priority),
            }
            headers.update(self.headers)
            
            response = requests.post(
                self.url,
                data=message.encode('utf-8'),
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logging.info(f"✅ Successfully sent {len(events)} events to {self.url}")
            else:
                logging.error(f"❌ Failed to send: HTTP {response.status_code}, {response.text}")
                
        except Exception as e:
            logging.error(f"❌ Error sending to webhook: {e}")
