#!/bin/sh
set -e

echo "=== Running database init ==="
python -m src.db.init_db

echo "=== Starting: $@ ==="
exec "$@"
