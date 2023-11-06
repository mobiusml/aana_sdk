#!/bin/bash
# TODO: pass arguments to the docker to set target instead of environment variable
poetry run aana --port 8000 --host 0.0.0.0 --target $TARGET
