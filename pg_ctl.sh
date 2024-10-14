#!/bin/bash

# Get the PostgreSQL major version (e.g., 14)
PG_VERSION=$(psql --version | awk '{print $3}' | cut -d '.' -f 1)

if [ "$2" = "--pgdata" ]; then
    # Recursively change parent directories permissions so that pg_ctl can access the necessary files as non-root.
    f=$(dirname "$3")
    while [[ $f != / ]]; do chmod o+rwx "$f"; f=$(dirname "$f"); done;
fi

# Execute pg_ctl as postgres instead of root.
sudo -u postgres "/usr/lib/postgresql/$PG_VERSION/bin/pg_ctl" "$@"
