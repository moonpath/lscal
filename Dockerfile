ARG BASE_IMAGE="ubuntu:24.04"
FROM $BASE_IMAGE AS base_image

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
	apt-get install -y \
	python3 \
	python3-pip \
	cron

RUN pip3 install --break-system-packages\
	icalendar \
	python-dateutil \
	requests

COPY handlers/ /app/handlers/
COPY main.py /app/main.py

RUN cat <<'EOF' > /usr/bin/docker-entrypoint && chmod 755 /usr/bin/docker-entrypoint
#!/bin/bash
set -e

if [ -d "/app/cron.d" ] && [ "$(ls -A /app/cron.d)" ]; then
    cp -L /app/cron.d/* /etc/cron.d/
    chmod 0644 /etc/cron.d/*
    chown root:root /etc/cron.d/*
fi

exec cron "$@"
EOF

WORKDIR "/app"

ENTRYPOINT ["docker-entrypoint"]

CMD ["-f"]
