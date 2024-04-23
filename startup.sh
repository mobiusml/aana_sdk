#!/bin/bash
# TODO: pass arguments to the docker to set target instead of environment variable
HF_HUB_ENABLE_HF_TRANSFER=1  poetry run python aana/projects/$TARGET/app.py --port 8000 --host 0.0.0.0