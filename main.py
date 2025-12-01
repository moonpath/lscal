#!/usr/bin/env python
'''
@File    :   main.py
@Time    :   2025/11/30 20:59:09
@Author  :   hongyu zhang
@Version :   1.0
@Desc    :   ICS Calendar Parser and Reminder Tool
'''
import re
import json
import logging
import argparse
import importlib
import dateutil
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timedelta, date
from icalendar import Component, Calendar, Alarm, vDatetime, vDuration, vRecur, vInt
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class iDatetime(Protocol):
    @property
    def dt(self) -> datetime | date:
        ...

@runtime_checkable
class iTimedelta(Protocol):
    @property
    def dt(self) -> timedelta | datetime:
        ...


def parse_ics_files(path: Path) -> Calendar:
    """
    Parse and merge ICS files from a file or directory.

    Args:
        path: Path to a single .ics file or directory containing .ics files

    Returns:
        Merged Calendar object containing all events from parsed files
    """
    file_paths = []

    if path.is_file():
        if path.suffix.lower() == '.ics':
            file_paths = [path]
    elif path.is_dir():
        file_paths = list(path.glob('*.ics'))

    merged_cal = Calendar()
    merged_cal.add('prodid', '-//Merged Calendar//mxm.dk//')
    merged_cal.add('version', '2.0')

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            cal = Calendar.from_ical(content)
            for component in cal.subcomponents:
                merged_cal.add_component(component)
        except Exception as e:
            logging.warning(f"Error processing file {file_path}: {e}, skipped.")

    return merged_cal


def load_handler(handler_name: str | None, handler_params: dict | None = None) -> Callable:
    """Load and instantiate an event handler module.
    
    Args:
        handler_name: Name of the handler module in the handlers package
        handler_params: Optional parameters to pass to handler constructor
        
    Returns:
        Callable handler instance or default console output handler
    """
    if not handler_name:
        return lambda calendar: print(calendar.to_ical().decode('utf-8'), flush=True, end='')

    try:
        module = importlib.import_module(f'handlers.{handler_name}')
        handler_class = getattr(module, 'Handler')
        
        if handler_params:
            return handler_class(**handler_params)
        else:
            return handler_class()
    
    except (ImportError, AttributeError) as e:
        logging.error(f"Failed to load handler class '{handler_name}'")
        logging.error(f"      Please check if the class name is correct and exists in the handlers module")
        logging.error(f"      Details: {e}")
        raise
    except TypeError as e:
        logging.error(f"Failed to instantiate handler class '{handler_name}'")
        logging.error(f"      The provided parameters may not match the constructor signature")
        logging.error(f"      Details: {e}")
        raise


def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string to timedelta object.
    
    Args:
        duration_str: Duration string (e.g., '30m', '12h', '7d', '2w', '1M', '1y')
        
    Returns:
        timedelta object representing the duration
        
    Raises:
        ValueError: If duration format is invalid or unit is unsupported
    """
    pattern = r'^(\d+)([mhdwMy])$'
    match = re.match(pattern, duration_str)
    
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}. Supported formats: 30m, 12h, 7d, 2w, 1M, 1y")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == 'm':  # minutes
        return timedelta(minutes=value)
    elif unit == 'h':  # hours
        return timedelta(hours=value)
    elif unit == 'd':  # days
        return timedelta(days=value)
    elif unit == 'w':  # weeks
        return timedelta(weeks=value)
    elif unit == 'M':  # months (approximated as 30 days)
        return timedelta(days=value * 30)
    elif unit == 'y':  # years (approximated as 365 days)
        return timedelta(days=value * 365)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")


def normalize_datetime(dt: date) -> datetime:
    """Normalize date/datetime to naive local datetime.
    
    Args:
        dt: date or datetime object, optionally timezone-aware
        
    Returns:
        Naive datetime object in local timezone
    """
    local_tz = dateutil.tz.tzlocal()
    if isinstance(dt, datetime):
        if dt.tzinfo:
            return dt.astimezone(local_tz).replace(tzinfo=None)
        else:
            return dt
    return datetime(dt.year, dt.month, dt.day)


def adjust_time_range_by_alarms(
    component: Component,
    start_time: datetime,
    end_time: datetime
) -> tuple[datetime, datetime]:
    """Adjust time range to account for alarm triggers.
    
    Expands the query range to ensure events with alarms are captured
    even if the alarm fires outside the original time range.
    
    Args:
        component: Calendar component to analyze
        start_time: Original start time
        end_time: Original end time
        
    Returns:
        Tuple of (adjusted_start_time, adjusted_end_time)
    """
    base_start = normalize_datetime(start_time)
    base_end = normalize_datetime(end_time)
    
    candidate_starts: list[datetime] = [base_start]
    candidate_ends: list[datetime] = [base_end]

    for item in component.walk():
        if item.name not in ('VEVENT', 'VTODO'):
            continue

        if not hasattr(item, 'subcomponents'):
            continue

        for sub in item.subcomponents:
            if sub.name != 'VALARM':
                continue
                
            trigger = sub.get('TRIGGER')
            if trigger is None:
                continue
            
            trigger_value = trigger.dt if hasattr(trigger, 'dt') else trigger
            
            if isinstance(trigger_value, timedelta):
                adjustment = -trigger_value
                
                if adjustment > timedelta(0):
                    candidate_ends.append(base_end + adjustment)
                else:
                    candidate_starts.append(base_start + adjustment)

            elif isinstance(trigger_value, date):
                trigger_dt = normalize_datetime(trigger_value)
                item_dtstart_prop: iDatetime | None = item.get('DTSTART')
                
                if item_dtstart_prop and base_start <= trigger_dt <= base_end:
                        item_dtstart = normalize_datetime(item_dtstart_prop.dt)
                        candidate_starts.append(item_dtstart)
                        candidate_ends.append(item_dtstart)

    return min(candidate_starts), max(candidate_ends)


def get_occurrences(
    component: Component,
    start_range: datetime,
    end_range: datetime
) -> list[datetime]:
    """Get all occurrences of an event within a time range.
    
    Handles recurring events using RRULE, RDATE, and EXDATE properties.
    
    Args:
        component: Calendar component (VEVENT or VTODO)
        start_range: Start of the query range
        end_range: End of the query range
        
    Returns:
        List of datetime objects for each occurrence
    """
    start_range = normalize_datetime(start_range)
    end_range = normalize_datetime(end_range)
    
    dtstart_prop: iDatetime | None = component.get('DTSTART')
    if not dtstart_prop:
        due_prop: iDatetime | None = component.get('DUE')
        if due_prop:
            due_dt = normalize_datetime(due_prop.dt)
            if start_range <= due_dt <= end_range:
                return [due_dt]
        return []
    
    start_dt_native = normalize_datetime(dtstart_prop.dt)

    rules = dateutil.rrule.rruleset()

    has_rrule = False
    
    rrule_props: list[vRecur] | vRecur | None = component.get('RRULE')
    if rrule_props:
        if not isinstance(rrule_props, list):
            rrule_props = [rrule_props]
            
        for prop in rrule_props:
            try:
                rrule_str = prop.to_ical().decode('utf-8')
                rule = dateutil.rrule.rrulestr(
                    f"RRULE:{rrule_str}", 
                    dtstart=start_dt_native,
                    forceset=False,
                    ignoretz=True
                )
                if isinstance(rule, dateutil.rrule.rrule):
                    rules.rrule(rule)
                elif isinstance(rule, dateutil.rrule.rruleset):
                    pass
                has_rrule = True
            except ValueError:
                continue

    if not has_rrule:
        rules.rdate(start_dt_native)

    def extract_dates(prop_item) -> list[datetime]:
        extracted = []
        if hasattr(prop_item, 'dts'):
            for dt_obj in prop_item.dts:
                extracted.append(normalize_datetime(dt_obj.dt))
        elif hasattr(prop_item, 'dt'):
            extracted.append(normalize_datetime(prop_item.dt))
        elif isinstance(prop_item, date):
            extracted.append(normalize_datetime(prop_item))
        return extracted

    if 'RDATE' in component:
        rdates = component['RDATE']
        if not isinstance(rdates, list):
            rdates = [rdates]
        
        for rdate_item in rdates:
            for dt in extract_dates(rdate_item):
                rules.rdate(dt)

    if 'EXDATE' in component:
        exdates = component['EXDATE']
        if not isinstance(exdates, list):
            exdates = [exdates]
            
        for exdate_item in exdates:
            for dt in extract_dates(exdate_item):
                rules.exdate(dt)

    try:
        occurrences = list(rules.between(start_range, end_range, inc=True))
    except Exception:
        return []
        
    return occurrences


def calculate_alarm_trigger(component: Component):
    """Calculate the absolute trigger time for an event's alarm.
    
    Args:
        component: Calendar component containing alarm information
        
    Returns:
        datetime object for alarm trigger time, or None if no valid trigger
    """
    alarm: Alarm | None = None
    if component.subcomponents:
        for sub in component.subcomponents:
            if sub.name == 'VALARM':
                alarm = sub
                break
    
    trigger_value = timedelta(0) # Default value: 0 interval
    related = 'START'            # Default value: based on start time

    if alarm:
        trigger_prop: vDuration | vDatetime | None = alarm.get('TRIGGER')
        if trigger_prop:
            trigger_value = trigger_prop.dt
            related = trigger_prop.params.get('RELATED', 'START')

    if isinstance(trigger_value, datetime):
        return normalize_datetime(trigger_value)

    elif isinstance(trigger_value, timedelta):
        base_time = None

        dtstart_prop: iDatetime | None = component.get('DTSTART')
        dtend_prop: iDatetime | None = component.get('DTEND')
        due_prop: iDatetime | None = component.get('DUE') # Only used for VTODO
        duration_prop: vDuration | None = component.get('DURATION')

        dtstart_val = normalize_datetime(dtstart_prop.dt) if dtstart_prop else None
        dtend_val = normalize_datetime(dtend_prop.dt) if dtend_prop else None
        due_val = normalize_datetime(due_prop.dt) if due_prop else None
        duration_val = duration_prop.dt if duration_prop else None

        if related == 'START':
            if dtstart_val:
                base_time = dtstart_val
            elif due_val and duration_val:
                 base_time = due_val - duration_val
            elif due_val:
                 base_time = due_val
            elif dtend_val and duration_val:
                 base_time = dtend_val - duration_val
            elif dtend_val:
                 base_time = dtend_val

        elif related == 'END':
            if dtend_val:
                base_time = dtend_val
            elif due_val:
                base_time = due_val
            elif dtstart_val and duration_val:
                base_time = dtstart_val + duration_val
            elif dtstart_val:
                 base_time = dtstart_val

        if base_time:
            return base_time + trigger_value

    return None


def expand_alarm(component: Component) -> list[Component]:
    """Expand component with multiple alarms into separate components.
    
    Creates individual components for each alarm to simplify processing.
    
    Args:
        component: Calendar component with zero or more alarms
        
    Returns:
        List of components, each with at most one alarm
    """
    alarms = [sub for sub in component.subcomponents if sub.name == 'VALARM']

    if len(alarms) <= 1:
        return [component]

    base_component = deepcopy(component)

    base_component.subcomponents = [
        c for c in base_component.subcomponents 
        if c.name != 'VALARM'
    ]

    result_components = []

    for alarm in alarms:
        new_comp = deepcopy(base_component)
        new_comp.add_component(deepcopy(alarm))
        result_components.append(new_comp)
            
    return result_components


def filter_and_sort_components(
    components: list[Component], 
    start_time: datetime, 
    end_time: datetime
) -> list[Component]:
    """Filter components by alarm trigger time and sort chronologically.
    
    Removes duplicates based on UID and start time.
    
    Args:
        components: List of calendar components to filter
        start_time: Start of the time range
        end_time: End of the time range
        
    Returns:
        Sorted list of unique components within the time range
    """
    start_time = normalize_datetime(start_time)
    end_time = normalize_datetime(end_time)

    candidates: list[tuple[datetime, Component]] = []

    for comp in components:
        trigger_val = comp.get('X-ALARM-TRIGGER')
        if not trigger_val:
            continue

        trigger_dt = None

        if isinstance(trigger_val, datetime):
            trigger_dt = trigger_val
        elif hasattr(trigger_val, 'dt'):
            trigger_dt = trigger_val.dt
        else:
            try:
                trigger_dt = dateutil.parser.parse(str(trigger_val))
            except (ValueError, TypeError):
                continue

        trigger_dt = normalize_datetime(trigger_dt)

        if start_time <= trigger_dt <= end_time:
            candidates.append((trigger_dt, comp))

    candidates.sort(key=lambda x: x[0])

    unique_components = []
    seen_keys = set()

    for _, comp in candidates:
        uid_prop = comp.get('UID')
        uid = str(uid_prop) if uid_prop else None

        dtstart_prop: iDatetime | None = comp.get('DTSTART')
        dtstart_val = normalize_datetime(dtstart_prop.dt) if dtstart_prop and hasattr(dtstart_prop, 'dt') else None
            
        if uid:
            key = (uid, dtstart_val)
            
            if key in seen_keys:
                continue
            seen_keys.add(key)
        
        unique_components.append(comp)

    return unique_components


def filter_outdated_and_overridden_components(
    original_cal: Calendar,
    subset_list: list[Component]
) -> list[Component]:
    """Filter out outdated and overridden components from a subset list.
    
    Handles complex scenarios including:
    1. Global updates (higher sequence on UID).
    2. Exception updates (higher sequence on specific RECURRENCE-ID).
    3. Instance overrides (Calculated instance replaced by an Exception).
    
    Args:
        original_cal: The original calendar containing all components
        subset_list: A subset list of components to be filtered
        
    Returns:
        A filtered list of components with outdated and overridden ones removed
    """
    
    revision_map: dict[tuple[str, datetime | None], int] = {}

    for component in original_cal.walk():
        uid = component.get('UID')
        if uid is None:
            continue
        
        sequence = component.get('SEQUENCE', vInt(0))
        
        rid_prop: iDatetime | None = component.get('RECURRENCE-ID')
        rid_val = normalize_datetime(rid_prop.dt) if rid_prop else None

        key = (uid, rid_val)

        if key not in revision_map:
            revision_map[key] = sequence
        else:
            if sequence > revision_map[key]:
                revision_map[key] = sequence

    cleaned_list: list[Component] = []

    for comp in subset_list:
        sub_uid = comp.get('UID')
        if sub_uid is None:
            cleaned_list.append(comp)
            continue
        
        sub_sequence = comp.get('SEQUENCE', vInt(0))
        
        sub_rid_prop: iDatetime | None = comp.get('RECURRENCE-ID')
        sub_rid_val = normalize_datetime(sub_rid_prop.dt) if sub_rid_prop else None

        check_key = (sub_uid, sub_rid_val)
        
        if check_key in revision_map:
            max_existing_seq = revision_map[check_key]
            if max_existing_seq > sub_sequence:
                continue

        if sub_rid_val is None:
            dt_prop: iDatetime | None = comp.get('DTSTART') or comp.get('DUE')
            
            if dt_prop is not None:
                instance_time = normalize_datetime(dt_prop.dt)
                
                potential_exception_key = (sub_uid, instance_time)
                
                if potential_exception_key in revision_map:
                    continue

        cleaned_list.append(comp)

    return cleaned_list


def search_calendar_schedule(
    cal: Calendar,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> Calendar:
    """Search and expand calendar events within a time range.
    
    Expands recurring events, processes alarms, and returns a new calendar
    with only the events that have alarms triggering in the specified range.
    
    Args:
        cal: Input calendar to search
        start_time: Start of query range (default: now)
        end_time: End of query range (default: 7 days from start)
        
    Returns:
        New calendar containing filtered and expanded events
    """
    start_time = start_time or datetime.now(dateutil.tz.tzlocal())
    end_time = end_time or (start_time + timedelta(days=7))
    adjust_start_time, adjust_end_time = adjust_time_range_by_alarms(cal, start_time, end_time)

    components = []

    for component in cal.subcomponents:
        component: Component = deepcopy(component)
        
        dtstarts = get_occurrences(
            component,
            adjust_start_time,
            adjust_end_time
        )
        
        component.pop('RRULE')
        component.pop('RDATE')
        component.pop('EXDATE')
        
        duration = None
        if 'DTEND' in component and 'DTSTART' in component:
            old_start = component.decoded('dtstart', None)
            old_end = component.decoded('dtend', None)
            if isinstance(old_start, date) and isinstance(old_end, date):
                duration = normalize_datetime(old_end) - normalize_datetime(old_start)

        component.pop('DTSTART')
        component.pop('DTEND')

        for dtstart in dtstarts:
            base_component = deepcopy(component)
            base_component.add('DTSTART', dtstart)
            if duration:
                dtend = dtstart + duration
                base_component.add('DTEND', dtend)
            expand_alarm_components = expand_alarm(base_component)
            for expand_alarm_component in expand_alarm_components:
                trigger_time = calculate_alarm_trigger(expand_alarm_component)
                if trigger_time:
                    expand_alarm_component.add('X-ALARM-TRIGGER', trigger_time)
                components.append(expand_alarm_component)
    
    components = filter_and_sort_components(components, start_time, end_time)
    components = filter_outdated_and_overridden_components(cal, components)

    new_cal = Calendar()
    new_cal.add('VERSION', cal.get('VERSION', '2.0'))
    new_cal.add('PRODID', cal.get('PRODID', '-//Modified Calendar//mxm.dk//'))
    for comp in components:
        new_cal.add_component(comp)

    return new_cal


def main():
    """Main entry point for ICS calendar parser and reminder tool."""
    parser = argparse.ArgumentParser(
        description='ICS Calendar Parser and Reminder Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Path to ICS file or directory. Can be a single .ics file or a directory containing multiple .ics files (default: current directory)'
    )
    
    parser.add_argument(
        '-d', '--duration',
        type=str,
        default='7d',
        help='Time span to query from current time (default: 7d). Supported formats: 30m (30 minutes), 12h (12 hours), 7d (7 days), 2w (2 weeks), 1M (1 month), 1y (1 year). Overridden when -s or -e is specified'
    )
    
    parser.add_argument(
        '-s', '--start-time',
        type=str,
        help='Start date/time for the query. Supports flexible formats: 2025-01-01, 2025/01/01, 2025-01-01 08:30, 2025-01-01T08:30:45, Jan 1 2025, etc. Defaults to 00:00:00 when only date is specified. Takes precedence over --duration'
    )
    
    parser.add_argument(
        '-e', '--end-time',
        type=str,
        help='End date/time for the query. Supports flexible formats: 2025-12-31, 2025/12/31, 2025-12-31 18:00, 2025-12-31T18:30:45, Dec 31 2025, etc. Defaults to 23:59:59 when only date is specified. Takes precedence over --duration'
    )
    
    parser.add_argument(
        '-m', '--module',
        type=str,
        default=None,
        help='Event handler module name (default: output to console)'
    )
    
    parser.add_argument(
        '-p', '--params',
        type=str,
        default=None,
        help='Handler initialization parameters in JSON format'
    )
    
    args = parser.parse_args()
    
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format for handler parameters: {e}")
            exit(1)
    
    try:
        handler = load_handler(args.module, params)
    except Exception:
        exit(1)
    
    if args.start_time:
        try:
            start_time = dateutil.parser.parse(args.start_time)
            start_time = normalize_datetime(start_time)
        except (ValueError, TypeError) as e:
            logging.error(f"Failed to parse start time: {args.start_time}")
            logging.error(f"      Supported formats: 2025-01-01, 2025/01/01 08:30, 2025-01-01T08:30:45")
            logging.error(f"      Details: {e}")
            exit(1)
    else:
        start_time = datetime.now(dateutil.tz.tzlocal())
    
    if args.end_time:
        try:
            end_time = dateutil.parser.parse(args.end_time)
            end_time = normalize_datetime(end_time)
        except (ValueError, TypeError) as e:
            logging.error(f"Failed to parse end time: {args.end_time}")
            logging.error(f"      Supported formats: 2025-12-31, 2025/12/31 18:00, 2025-12-31T18:30:45")
            logging.error(f"      Details: {e}")
            exit(1)
    else:
        try:
            duration_delta = parse_duration(args.duration)
            end_time = start_time + duration_delta
        except ValueError as e:
            logging.error(f"Error: {e}")
            exit(1)
    
    cal = parse_ics_files(Path(args.path))
    cal = search_calendar_schedule(
        cal,
        start_time=start_time,
        end_time=end_time
    )
    
    print(f"Query time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
    handler(cal)
    return


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
