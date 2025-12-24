ARG BASE_IMAGE="python:alpine"
FROM $BASE_IMAGE AS runtime

RUN apk add --no-cache \
    dcron \
    tzdata

RUN pip install --no-cache-dir \
    icalendar \
    python-dateutil \
    requests

COPY handlers/ /app/handlers/
COPY main.py /app/main.py

RUN cat <<'EOF' > /usr/local/bin/docker-entrypoint && chmod 755 /usr/local/bin/docker-entrypoint
#!/bin/sh
set -e

if [ -d "/app/crontabs" ] && [ "$(ls -A /app/crontabs)" ]; then
    cp /app/crontabs/* /etc/crontabs/
    chmod 0600 /etc/crontabs/*
fi

exec "$@"
EOF

HEALTHCHECK --interval=1m --timeout=5s --start-period=10s --retries=3 \
    CMD pidof crond > /dev/null

WORKDIR "/app"

ENTRYPOINT ["docker-entrypoint"]

CMD ["crond", "-f"]
