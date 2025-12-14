#!/usr/bin/env python
'''
@File    :   main.py
@Time    :   2025/11/30 20:59:09
@Author  :   hongyu zhang
@Version :   1.0
@Desc    :   ICS Calendar Parser and Reminder Tool
'''
import argparse
import importlib
import importlib.util
import json
import logging
import os
import re
import sys
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, cast, overload

import dateutil
from icalendar import Calendar, Component, vDDDLists, vDDDTypes, vInt, vRecur


class BaseHandler(Protocol):
    def __call__(self, calendar: Calendar) -> None: ...


def to_naive_local_datetime(dt: date) -> datetime:
    """Convert a date or datetime to a naive datetime in local timezone.
    
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


@overload
def extract_datetime(component: Component, key: Literal['DTSTART', 'DTEND', 'DUE', 'RECURRENCE-ID']) -> datetime | None: ...

@overload
def extract_datetime(component: Component, key: Literal['TRIGGER']) -> datetime | timedelta | None: ...

@overload
def extract_datetime(component: Component, key: Literal['DURATION']) -> timedelta | None: ...

def extract_datetime(component: Component, key: str | bytes) -> datetime | timedelta | None:
    if not isinstance(component, Component):
        return None
    
    if isinstance(key, bytes):
        key = key.decode("utf-8-sig", "replace")
    if isinstance(key, str):
        key = key.upper()
    else:
        return None

    ddd: list[vDDDTypes] | vDDDTypes | None = component.get(key)
    if not ddd:
        return None

    if isinstance(ddd, list):
        ddd = ddd[0]

    dt = ddd.dt

    if key in ['TRIGGER', 'DURATION'] and isinstance(dt, timedelta):
        return dt

    if key in ['DTSTART', 'DTEND', 'DUE', 'RECURRENCE-ID', 'TRIGGER'] and isinstance(dt, datetime):
        return to_naive_local_datetime(dt)

    if key in ['DTSTART', 'DTEND', 'DUE', 'RECURRENCE-ID'] and isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day)

    return None


def load_and_merge_calendars(input_paths: list[str]) -> Calendar:
    """Load and merge multiple ICS calendars from files, directories, or stdin.

    Args:
        input_paths: List of file paths or directories containing .ics files.
                     If empty, reads from stdin.

    Returns:
        Merged Calendar object containing all parsed events.
    """
    parsed_components: list[Component] = []

    if not input_paths:
        # Read from stdin
        try:
            content = sys.stdin.read().strip()
            if content:
                component = Calendar.from_ical(content)
                parsed_components.append(component)
            else:
                logging.warning("No content read from stdin.")
        except Exception as e:
            logging.warning(f"Error reading stdin: {e}")

    for path_str in input_paths:
        path = Path(path_str)
        
        if not path.exists():
            logging.warning(f"Path '{path}' does not exist, skipped.")
            continue
            
        if path.is_file():
            try:
                with path.open('r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    component = Calendar.from_ical(content)
                    parsed_components.append(component)
                else:
                    logging.warning(f"No content in file {path}, skipped.")
            except Exception as e:
                logging.warning(f"Error reading file {path}: {e}")
            
        elif path.is_dir():
            for ics_file in path.rglob('*.ics'):
                if ics_file.is_file():
                    try:
                        with ics_file.open('r', encoding='utf-8') as f:
                            content = f.read().strip()
                        if content:
                            component = Calendar.from_ical(content)
                            parsed_components.append(component)
                        else:
                            logging.warning(f"No content in file {ics_file}, skipped.")
                    except Exception as e:
                        logging.warning(f"Error reading file {ics_file}: {e}")
    
    if len(parsed_components) == 1 and isinstance(parsed_components[0], Calendar):
        return parsed_components[0]
    
    merged_calendar = Calendar()
    merged_calendar.add('VERSION', '2.0')
    merged_calendar.add('PRODID', '-//Nobugs Club//lscal 1.0.0//EN')
    for component in parsed_components:
        for subcomponent in component.walk():
            if subcomponent.name in ('VEVENT', 'VTODO', 'VJOURNAL', 'VFREEBUSY', 'VTIMEZONE'):
                merged_calendar.add_component(subcomponent)
    return merged_calendar


def load_handler(handler_name: str, handler_params: dict | None = None) -> BaseHandler:
    """Load and instantiate an event handler module.
    
    Args:
        handler_name: Module name or path to handler script
        handler_params: Optional parameters to pass to handler constructor
        
    Returns:
        Callable handler instance
    """
    handler_file = Path(handler_name)
    if handler_file.is_file() and handler_file.suffix == '.py':
        spec = importlib.util.spec_from_file_location(handler_file.stem, handler_name)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(f'handlers.{handler_name}')
        except ImportError:
            module = importlib.import_module(handler_name)

    handler_class = getattr(module, 'Handler')
    
    if handler_params:
        return handler_class(**handler_params)
    else:
        return handler_class()


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


def adjust_query_range_for_alarms(
    component: Component,
    start_time: datetime,
    end_time: datetime
) -> tuple[datetime, datetime]:
    """Adjust query time range based on alarm triggers in the component.
    
    Args:
        component: Calendar component to analyze
        start_time: Original start time
        end_time: Original end time
        
    Returns:
        Tuple of (adjusted_start_time, adjusted_end_time)
    """
    base_start = to_naive_local_datetime(start_time)
    base_end = to_naive_local_datetime(end_time)
    
    candidate_starts: list[datetime] = [base_start]
    candidate_ends: list[datetime] = [base_end]

    for sub in component.walk():
        if sub.name not in ('VEVENT', 'VTODO'):
            continue

        for alarm in filter(lambda sc: sc.name == 'VALARM', sub.subcomponents):
            trigger_value = extract_datetime(alarm, 'TRIGGER')
            if trigger_value is None:
                continue
            
            if isinstance(trigger_value, timedelta):
                if trigger_value < timedelta(0):
                    candidate_ends.append(base_end - trigger_value)
                else:
                    candidate_starts.append(base_start - trigger_value)

            elif isinstance(trigger_value, datetime):
                if base_start <= trigger_value <= base_end:
                    dtstart_value = extract_datetime(sub, 'DTSTART')
                    due_value = extract_datetime(sub, 'DUE') # Only for VTODO

                    if dtstart_value: 
                        candidate_starts.append(dtstart_value)
                        candidate_ends.append(dtstart_value)
                    if due_value:
                        candidate_starts.append(due_value)
                        candidate_ends.append(due_value)

    return min(candidate_starts), max(candidate_ends)


def get_occurrences_in_range(
    component: Component,
    start_time: datetime,
    end_time: datetime
) -> list[date]:
    """Get all occurrences of an event within a time range.
    
    Handles recurring events using RRULE, RDATE, and EXDATE properties.
    
    Args:
        component: Calendar component (VEVENT or VTODO)
        start_range: Start of the query range
        end_range: End of the query range
        
    Returns:
        List of datetime objects for each occurrence
    """
    start_time = to_naive_local_datetime(start_time)
    end_time = to_naive_local_datetime(end_time)

    is_date = (component.decoded('DTSTART', None) or component.decoded('DUE', None)).__class__ == date
    
    dtstart_value = extract_datetime(component, 'DTSTART')
    if not dtstart_value:
        due_value = extract_datetime(component, 'DUE')
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
                logging.warning(f"Invalid RRULE format: {prop.to_ical().decode('utf-8')}, skipped.")
                continue

    if not has_rrule:
        rules.rdate(dtstart_value)

    def extract_dates(props: list[vDDDLists] | vDDDLists) -> list[datetime]:
        extracted = []
        if not isinstance(props, list):
            props = [props]
        for prop in props:
            for ddd in prop.dts:
                ddd_dt = ddd.dt
                if isinstance(ddd_dt, tuple) and len(ddd_dt) == 2:
                    dt_start, dt_end = ddd_dt
                    dt_start = to_naive_local_datetime(dt_start)
                    if isinstance(dt_end, date):
                        dt_end = to_naive_local_datetime(dt_end)
                    dt_range = (dt_start, dt_end)
                    extracted.append(dt_range)
                elif isinstance(ddd_dt, date):
                    dt = to_naive_local_datetime(ddd_dt)
                    extracted.append(dt)
        return extracted

    rdates = component.get('RDATE')
    if rdates:
        for rdate in extract_dates(rdates):
            rules.rdate(rdate)

    exdates = component.get('EXDATE')
    if exdates:
        for exdate in extract_dates(exdates):
            rules.exdate(exdate)

    try:
        occurrences = list(rules.between(start_time, end_time, inc=True))
        if is_date:
            occurrences = [dt.date() for dt in occurrences]
    except Exception as e:
        logging.warning(f"Error expanding occurrences: {e}")
        return []
        
    return occurrences


def resolve_alarm_trigger(component: Component) -> datetime | None:
    """Resolve the trigger time for an alarm in a calendar component.
    
    Args:
        component: Calendar component containing alarm information
        
    Returns:
        datetime object for alarm trigger time, or None if no valid trigger
    """
    alarm = next(filter(lambda sub: sub.name == 'VALARM', component.subcomponents), None)

    if not alarm:
        return None

    trigger_value = extract_datetime(alarm, 'TRIGGER')

    if trigger_value is None:
        return None

    if isinstance(trigger_value, datetime):
        return trigger_value

    related = 'START'
    trigger_prop: list[vDDDTypes] | vDDDTypes | None = alarm.get('TRIGGER')
    if isinstance(trigger_prop, vDDDTypes):
        related_prop = trigger_prop.params.get('RELATED')
        if isinstance(related_prop, str):
            related = related_prop.upper()

    base_time = None

    dtstart_value = extract_datetime(component, 'DTSTART')
    dtend_value = extract_datetime(component, 'DTEND')
    due_value = extract_datetime(component, 'DUE') # Only used for VTODO
    duration_value = extract_datetime(component, 'DURATION')

    if related == 'START':
        if dtstart_value:
            base_time = dtstart_value
        elif due_value and duration_value:
                base_time = due_value - duration_value
        elif due_value:
                base_time = due_value
        elif dtend_value and duration_value:
                base_time = dtend_value - duration_value
        elif dtend_value:
                base_time = dtend_value

    elif related == 'END':
        if dtend_value:
            base_time = dtend_value
        elif due_value:
            base_time = due_value
        elif dtstart_value and duration_value:
            base_time = dtstart_value + duration_value
        elif dtstart_value:
                base_time = dtstart_value

    if base_time:
        return base_time + trigger_value

    return None


def explode_multi_alarm_component(component: Component) -> list[Component]:
    """Explode a calendar component with multiple alarms into separate components.
    
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
        new_component = deepcopy(base_component)
        new_component.add_component(deepcopy(alarm))
        result_components.append(new_component)
            
    return result_components


def filter_and_sort_by_trigger_time(
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
    start_time = to_naive_local_datetime(start_time)
    end_time = to_naive_local_datetime(end_time)

    candidates: list[tuple[datetime, Component]] = []

    for component in components:
        trigger_value = component.get('X-ALARM-TRIGGER')
        if trigger_value is None:
            continue

        try:
            trigger_dt = dateutil.parser.parse(str(trigger_value))
        except Exception as e:
            logging.warning(f"Error parsing trigger datetime: {e}")
            continue

        trigger_dt = to_naive_local_datetime(trigger_dt)

        if start_time <= trigger_dt <= end_time:
            candidates.append((trigger_dt, component))

    candidates.sort(key=lambda x: x[0])

    unique_components = []
    seen_keys = set()

    for _, component in candidates:
        uid = component.get('UID')

        sequence_value = 0
        sequence_prop = component.get('SEQUENCE')
        if isinstance(sequence_prop, vInt):
            sequence_value = int(sequence_prop)

        dtstart_value = extract_datetime(component, 'DTSTART')
        due_value = extract_datetime(component, 'DUE')
        rid_value = extract_datetime(component, 'RECURRENCE-ID')
            
        key = (uid, dtstart_value, due_value, rid_value, sequence_value)
        
        if key in seen_keys:
            continue
        seen_keys.add(key)
        
        unique_components.append(component)

    return unique_components


def filter_superseded_components(
    original_calendar: Calendar,
    subcomponents: list[Component]
) -> list[Component]:
    """Filter out outdated and overridden components in two distinct steps.

    Step 1: STRICT SEQUENCE FILTERING
    Regardless of source (original_calendar or subcomponents input), for any specific
    (UID, RECURRENCE-ID) pair, only the component with the highest SEQUENCE number survives.

    Step 2: RECURRENCE LOGIC
    Handle 'THISANDFUTURE' range overrides and single instance exceptions overriding
    master events based on the surviving components from Step 1.

    Args:
        original_calendar: The original calendar containing all components (master + exceptions)
        subcomponents: An exploded/expanded list of components to be filtered
        
    Returns:
        A filtered list of components.
    """
    
    # =========================================================================
    # Step 1: Handle SEQUENCE Logic
    # Objective: Ensure we only work with the latest version of any specific item.
    # =========================================================================

    # Dictionary to store the highest sequence found for each specific instance key.
    # Key: (UID, RECURRENCE-ID value), Value: (SEQUENCE, Component)
    latest_versions: dict[tuple[str, datetime | None], tuple[int, Component]] = {}

    def _get_sequence(comp: Component) -> int:
        seq_prop = comp.get('SEQUENCE')
        if isinstance(seq_prop, vInt):
            return int(seq_prop)
        return 0

    def _process_component_for_sequence(comp: Component):
        uid = comp.get('UID')
        if uid is None:
            return
        
        rid_value = extract_datetime(comp, 'RECURRENCE-ID')
        seq_value = _get_sequence(comp)
        key = (uid, rid_value)

        # If key doesn't exist, or we found a higher sequence, update the map
        if key not in latest_versions:
            latest_versions[key] = (seq_value, comp)
        else:
            current_max_seq, _ = latest_versions[key]
            if seq_value > current_max_seq:
                latest_versions[key] = (seq_value, comp)

    # 1.1 Process components from original_calendar (to get knowledge of all overrides)
    for component in original_calendar.subcomponents:
        # We generally skip VTIMEZONE or others without UID in this logic, 
        # but safely checking UID handles it.
        if component.name == 'VCALENDAR': continue 
        _process_component_for_sequence(component)

    # 1.2 Process components from the input list `subcomponents`
    # Note: subcomponents might be expanded recurrences (virtual) or actual components.
    # If they are duplicate (same UID/RID) as original, the one with higher sequence wins.
    for component in subcomponents:
        _process_component_for_sequence(component)

    # =========================================================================
    # Step 2: Handle RECURRENCE-ID Logic
    # Objective: Filter out events superseded by RANGE overrides or specific exceptions.
    # Now we only look at the 'survivors' from Step 1.
    # =========================================================================

    # We need to build a map of RANGE overrides based on the *latest* versions derived above.
    range_overrides: dict[str, list[tuple[datetime, int]]] = {}
    
    # We also need a set of existing specific exceptions to hide master instances.
    specific_exceptions: set[tuple[str, datetime]] = set()

    for (uid, rid_value), (seq, comp) in latest_versions.items():
        rid_prop = comp.get('RECURRENCE-ID')
        
        # Collect RANGE overrides
        if rid_prop is not None and 'RANGE' in rid_prop.params and rid_value is not None:
            if rid_prop.params['RANGE'] == 'THISANDFUTURE':
                if uid not in range_overrides:
                    range_overrides[uid] = []
                range_overrides[uid].append((rid_value, seq))

        # Collect specific exceptions (RID exists but is not None)
        if rid_value is not None:
            specific_exceptions.add((uid, rid_value))

    # The result list
    final_list: list[Component] = []

    # Iterate through the subcomponents input again. 
    # But strictly check if they match the 'latest_version' and satisfy recurrence logic.
    for component in subcomponents:
        uid = component.get('UID')
        if uid is None:
            final_list.append(component)
            continue

        rid_value = extract_datetime(component, 'RECURRENCE-ID')
        seq_value = _get_sequence(component)
        
        # 2.1 Consistency Check (Step 1 Enforcement)
        # Does this component match the "latest version" we decided on in Step 1?
        key = (uid, rid_value)
        if key in latest_versions:
            best_seq, best_comp = latest_versions[key]
            # If the best sequence is strictly higher, this component is stale.
            # Note: We compare sequence values. If they are equal, we assume 
            # the component in subcomponents is valid (or identical).
            if best_seq > seq_value:
                continue
        
        # Determine the effective time of this instance
        # If it's a master event (rid is None), its time is DTSTART.
        # If it's an exception, its time is usually the RID value (original start).
        instance_effective_time = rid_value or extract_datetime(component, 'DTSTART') or extract_datetime(component, 'DUE')

        # 2.2 Handle RANGE=THISANDFUTURE Overrides
        is_superseded_by_range = False
        if uid in range_overrides and instance_effective_time is not None:
            for range_start_time, range_seq in range_overrides[uid]:
                # Don't let an event supersede itself
                if rid_value == range_start_time and seq_value == range_seq:
                    continue

                # The logic: If this instance starts on or after the range start,
                # AND the range override has a higher or equal sequence (usually implying a newer definition logic),
                # AND this instance is not the range override itself (checked above).
                # Note: Usually, if a RANGE override exists, it kills existing master instances 
                # or older exceptions that fall into that timeframe.
                
                if instance_effective_time >= range_start_time:
                    # If this component is generated from the master rule (rid is None),
                    # it is definitely superseded by the RANGE exception.
                    if rid_value is None:
                        is_superseded_by_range = True
                        break
                    
                    # If this is an OLDER specific exception that falls into the range,
                    # checks if the range sequence is newer.
                    if range_seq > seq_value:
                         is_superseded_by_range = True
                         break
        
        if is_superseded_by_range:
            continue

        # 2.3 Handle Single Instance Exceptions Hiding Master
        # If this component is an instance from the master rule (RID is None),
        # but there is a specific exception (RID is not None) for this exact time,
        # the master instance should be hidden.
        if rid_value is None and instance_effective_time is not None:
            # Check if there is a specific exception for this UID at this Time
            if (uid, instance_effective_time) in specific_exceptions:
                continue

        final_list.append(component)

    return final_list


def extract_scheduled_events(
    calendar: Calendar,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> Calendar:
    """Extract scheduled events from calendar within a time range.
    
    Args:
        calendar: Input calendar to search
        start_time: Start of query range (default: now)
        end_time: End of query range (default: 7 days from start)
        
    Returns:
        New calendar containing filtered and expanded events
    """
    start_time = start_time or datetime.now(dateutil.tz.tzlocal())
    end_time = end_time or (start_time + timedelta(days=7))

    components = []

    for component in calendar.subcomponents:
        component: Component = deepcopy(component)
        
        adjust_start_time, adjust_end_time = adjust_query_range_for_alarms(component, start_time, end_time)

        dtstarts = get_occurrences_in_range(
            component,
            adjust_start_time,
            adjust_end_time
        )
        
        component.pop('RRULE', None)
        component.pop('RDATE', None)
        component.pop('EXDATE', None)
        
        old_start = extract_datetime(component, 'DTSTART')
        old_end = extract_datetime(component, 'DTEND')
        if old_start is not None and old_end is not None:
            duration = old_end - old_start
        else:
            duration = None

        component.pop('DTSTART', None)
        component.pop('DTEND', None)

        for dtstart in dtstarts:
            base_component = deepcopy(component)
            base_component.add('DTSTART', dtstart)
            if duration:
                dtend = dtstart + duration
                base_component.add('DTEND', dtend)
            expanded_alarm_components = explode_multi_alarm_component(base_component)
            for alarm_component in expanded_alarm_components:
                trigger_time = resolve_alarm_trigger(alarm_component)
                if trigger_time is not None:
                    alarm_component.add('X-ALARM-TRIGGER', trigger_time)
                components.append(alarm_component)
    
    components = filter_and_sort_by_trigger_time(components, start_time, end_time)
    components = filter_superseded_components(calendar, components)

    scheduled_calendar = Calendar()
    scheduled_calendar.add('VERSION', calendar.get('VERSION', '2.0'))
    scheduled_calendar.add('PRODID', calendar.get('PRODID', '-//Nobugs Club//lscal 1.0.0//EN'))
    for component in components:
        scheduled_calendar.add_component(component)

    return scheduled_calendar


def main():
    """Main entry point for ICS calendar parser and reminder tool."""
    parser = argparse.ArgumentParser(
        description='ICS Calendar Parser and Reminder Tool'
    )
    
    parser.add_argument(
        'path',
        nargs='*',
        help='path to .ics file or directory containing .ics files; if omitted, reads from stdin'
    )
    
    parser.add_argument(
        '-d', '--duration',
        type=str,
        default='7d',
        help='time span to query relative to the start time; used to calculate the end time if --end-time is not provided; formats: 30m, 12h, 7d, 2w, 1M, 1y. (Default: 7d)'
    )
    
    parser.add_argument(
        '-s', '--start-time',
        type=str,
        help='query start time (e.g., "2000-01-01 00:00"); defaults to current local time if not provided'
    )
    
    parser.add_argument(
        '-e', '--end-time',
        type=str,
        help='query end time (e.g., "2000-01-07 23:59"); if not provided, calculated using start time and duration'
    )
    
    parser.add_argument(
        '-m', '--module',
        type=str,
        help='event handler module name (default: output to console)'
    )
    
    parser.add_argument(
        '-p', '--params',
        type=str,
        help='handler initialization parameters in JSON format'
    )

    parser.add_argument(
        '-l', '--log-level',
        type=str.upper,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='set the logging level (default: INFO)'
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version='lscal 1.0.0',
        help='show program version and exit'
    )
    
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format for handler parameters: {e}")
            sys.exit(1)
        if not isinstance(params, dict):
            logging.error("Params must be a JSON object (dict)")
            sys.exit(1)
    else:
        params = None
    
    if args.module:
        try:
            handler = load_handler(args.module, params)
        except Exception as e:
            logging.error(f"Error loading handler module: {e}")
            sys.exit(1)
    else:
        handler: BaseHandler = lambda calendar: print(calendar.to_ical().decode('utf-8'), flush=True, end='', file=sys.stdout)
    
    if args.start_time:
        try:
            start_time = dateutil.parser.parse(args.start_time)
            start_time = to_naive_local_datetime(start_time)
        except Exception as e:
            logging.error(f"Error parsing start time: {e}")
            sys.exit(1)
    else:
        start_time = datetime.now(dateutil.tz.tzlocal())
    
    if args.end_time:
        try:
            end_time = dateutil.parser.parse(args.end_time)
            end_time = to_naive_local_datetime(end_time)
        except Exception as e:
            logging.error(f"Error parsing end time: {e}")
            sys.exit(1)
    else:
        try:
            duration_delta = parse_duration(args.duration)
            end_time = start_time + duration_delta
        except Exception as e:
            logging.error(f"Error parsing duration: {e}")
            sys.exit(1)
    
    if not args.path and sys.stdin.isatty():
        parser.print_help()
        sys.exit(1)
    calendar = load_and_merge_calendars(args.path)
    
    calendar = extract_scheduled_events(
        calendar,
        start_time=start_time,
        end_time=end_time
    )
    
    logging.info(f"Time Range: {start_time} to {end_time}")

    handler(calendar)
    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        sys.stderr = open(os.devnull, 'w')
        sys.exit(1)
