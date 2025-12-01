# lscal

A lightweight ICS calendar parser and reminder tool that processes iCalendar files and sends notifications for upcoming events.

## Features

- Parse single or multiple `.ics` files
- Support for recurring events (RRULE, RDATE, EXDATE)
- Event alarm/reminder processing
- Flexible time range queries
- Multiple output handlers (console, file, webhook)
- Timezone-aware datetime handling
- Docker deployment support

## Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

```bash
# Parse ICS files in current directory (7-day lookahead by default)
lscal

# Parse a specific ICS file
lscal path/to/calendar.ics

# Parse all ICS files in a directory
lscal path/to/ics/directory
```

### Time Range Options

```bash
# Query events in the next 30 minutes
lscal -d 30m

# Query events in the next 12 hours
lscal -d 12h

# Query events between specific dates
lscal -s "2025-01-01" -e "2025-12-31"

# Query events with specific datetime
lscal -s "2025-01-01 08:30" -e "2025-01-31 18:00"
```

### Output Handlers

```bash
# Output to console (default)
lscal calendar.ics

# Send to webhook
lscal calendar.ics -m webhook -p '{"url":"https://example.com/notify"}'

# Write to file
lscal calendar.ics -m file -p '{"path":"output.ics"}'
```

## Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# The container runs scheduled tasks via cron
# Configure your schedules in cron.d/crontab
```

## Time Span Formats

- `30m` - 30 minutes
- `12h` - 12 hours  
- `7d` - 7 days (default)
- `2w` - 2 weeks
- `1M` - 1 month (~30 days)
- `1y` - 1 year (~365 days)

## Handler Parameters

Handler parameters are passed as JSON via the `-p` flag:

```bash
# Webhook handler
-p '{"url":"https://api.example.com/notify","headers":{"Authorization":"Bearer TOKEN"}}'

# File handler  
-p '{"path":"/path/to/output.ics"}'
```
