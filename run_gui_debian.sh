#!/usr/bin/env bash

python3 -m h2_plant.gui.main
exit_code=$?

if [ "$exit_code" -ne 0 ]; then
    echo "ERROR: python3 -m h2_plant.gui.main failed (exit $exit_code)" >&2
fi

exit "$exit_code"
