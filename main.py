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
from icalendar import Component, Calendar, Alarm, vDDDLists, vDDDTypes, vRecur, vInt
from typing import Callable, cast


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
    if isinstance(dt, datetime):
        if dt.tzinfo:
            local_tz = dateutil.tz.tzlocal()
            return dt.astimezone(local_tz).replace(tzinfo=None)
        return dt
    return datetime(dt.year, dt.month, dt.day)


def extract_datetime(ddd: list[vDDDTypes] | vDDDTypes | None) -> datetime | None:
    if not ddd:
        return None

    if isinstance(ddd, list):
        ddd = ddd[0]

    dt = ddd.dt

    if isinstance(dt, datetime):
        if dt.tzinfo:
            local_tz = dateutil.tz.tzlocal()
            return dt.astimezone(local_tz).replace(tzinfo=None)
        return dt

    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day)

    return None


def extract_trigger(ddd: list[vDDDTypes] | vDDDTypes | None) -> timedelta | datetime | None:
    if not ddd:
        return None

    if isinstance(ddd, list):
        ddd = ddd[0]

    dt = ddd.dt

    if isinstance(dt, timedelta):
        return dt

    if isinstance(dt, datetime):
        if dt.tzinfo:
            local_tz = dateutil.tz.tzlocal()
            return dt.astimezone(local_tz).replace(tzinfo=None)
        else:
            return dt

    return None


def extract_duration(ddd: list[vDDDTypes] | vDDDTypes | None) -> timedelta | None:
    if not ddd:
        return None

    if isinstance(ddd, list):
        ddd = ddd[0]

    dt = ddd.dt

    if isinstance(dt, timedelta):
        return dt

    return None


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

    for sub in component.walk():
        if sub.name not in ('VEVENT', 'VTODO'):
            continue

        if not hasattr(sub, 'subcomponents'):
            continue

        for alarm in sub.subcomponents:
            if alarm.name != 'VALARM':
                continue
                
            trigger_value = extract_trigger(alarm.get('TRIGGER'))
            if trigger_value is None:
                continue
            
            if isinstance(trigger_value, timedelta):
                if trigger_value < timedelta(0):
                    candidate_ends.append(base_end - trigger_value)
                else:
                    candidate_starts.append(base_start - trigger_value)

            elif isinstance(trigger_value, date):
                if base_start <= trigger_value <= base_end:
                    dtstart_value = extract_datetime(sub.get('DTSTART'))
                    due_value = extract_datetime(sub.get('DUE')) # Only for VTODO

                    if dtstart_value: 
                        candidate_starts.append(dtstart_value)
                        candidate_ends.append(dtstart_value)
                    if due_value:
                        candidate_starts.append(due_value)
                        candidate_ends.append(due_value)

    return min(candidate_starts), max(candidate_ends)


def get_occurrences(
    component: Component,
    start_time: datetime,
    end_time: datetime
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
    start_time = normalize_datetime(start_time)
    end_time = normalize_datetime(end_time)
    
    dtstart_value = extract_datetime(component.get('DTSTART'))
    if not dtstart_value:
        due_value = extract_datetime(component.get('DUE'))
        if due_value:
            if start_time <= due_value <= end_time:
                return [due_value]
        return []
    
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
                    s=rrule_str, 
                    dtstart=dtstart_value,
                    forceset=False,
                    ignoretz=True
                )
                rule = cast(dateutil.rrule.rrule, rule)
                rules.rrule(rule)
                has_rrule = True
            except ValueError:
                continue

    if not has_rrule:
        rules.rdate(dtstart_value)

    def extract_dates(props: vDDDLists | list[vDDDLists]) -> list[datetime]:
        extracted = []
        if not isinstance(props, list):
            props = [props]
        for prop in props:
            for ddd in prop.dts:
                dt = cast(date, ddd.dt)
                dt = normalize_datetime(dt)
                extracted.append(dt)
        return extracted

    rdates = component['RDATE']
    if rdates:
        for rdate in extract_dates(rdates):
            rules.rdate(rdate)

    exdates = component['EXDATE']
    if exdates:
        for exdate in extract_dates(exdates):
            rules.exdate(exdate)

    try:
        occurrences = list(rules.between(start_time, end_time, inc=True))
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

    if not alarm:
        return None

    trigger_prop: vDDDTypes = alarm.get('TRIGGER')

    trigger_value = extract_trigger(trigger_prop)
    if trigger_value is None:
        return None

    related = trigger_prop.params.get('RELATED', 'START')

    if isinstance(trigger_value, datetime):
        return trigger_value

    elif isinstance(trigger_value, timedelta):
        base_time = None

        dtstart_val = extract_datetime(component.get('DTSTART'))
        dtend_val = extract_datetime(component.get('DTEND'))
        due_val = extract_datetime(component.get('DUE')) # Only used for VTODO
        duration_val = extract_duration(component.get('DURATION'))

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
        if trigger_val is None:
            continue

        try:
            trigger_dt = dateutil.parser.parse(str(trigger_val))
        except Exception:
            continue

        trigger_dt = normalize_datetime(trigger_dt)

        if start_time <= trigger_dt <= end_time:
            candidates.append((trigger_dt, comp))

    candidates.sort(key=lambda x: x[0])

    unique_components = []
    seen_keys = set()

    for _, comp in candidates:
        uid = comp.get('UID')

        sequence_prop = comp.get('SEQUENCE', vInt(0))
        sequence_val = int(sequence_prop)

        dtstart_val = extract_datetime(comp.get('DTSTART'))
        due_val = extract_datetime(comp.get('DUE'))
        rid_val = extract_datetime(comp.get('RECURRENCE-ID'))
            
        key = (uid, dtstart_val, due_val, rid_val, sequence_val)
        
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
        
        sequence_prop = component.get('SEQUENCE', vInt(0))
        sequence_val = sequence_prop if isinstance(sequence_prop, vInt) else vInt(0)
        
        rid_val = extract_datetime(component.get('RECURRENCE-ID'))

        key = (uid, rid_val)

        if key not in revision_map:
            revision_map[key] = sequence_val
        else:
            if sequence_val > revision_map[key]:
                revision_map[key] = sequence_val

    cleaned_list: list[Component] = []

    for comp in subset_list:
        sub_uid = comp.get('UID')
        if sub_uid is None:
            cleaned_list.append(comp)
            continue
        
        sub_sequence_prop = comp.get('SEQUENCE', vInt(0))
        sub_sequence_val = sub_sequence_prop if isinstance(sub_sequence_prop, vInt) else vInt(0)
        
        sub_rid_val = extract_datetime(comp.get('RECURRENCE-ID'))

        check_key = (sub_uid, sub_rid_val)
        
        if check_key in revision_map:
            max_existing_seq = revision_map[check_key]
            if max_existing_seq > sub_sequence_val:
                continue

        if sub_rid_val is None:
            instance_time = extract_datetime(comp.get('DTSTART') or comp.get('DUE'))
            
            if instance_time:
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

    components = []

    for component in cal.subcomponents:
        component: Component = deepcopy(component)
        
        adjust_start_time, adjust_end_time = adjust_time_range_by_alarms(component, start_time, end_time)

        dtstarts = get_occurrences(
            component,
            adjust_start_time,
            adjust_end_time
        )
        
        component.pop('RRULE')
        component.pop('RDATE')
        component.pop('EXDATE')
        
        duration = None
        old_start = extract_datetime(component.get('DTSTART'))
        old_end = extract_datetime(component.get('DTEND'))
        if old_start and old_end:
            duration = old_end - old_start

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
                if trigger_time is not None:
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
