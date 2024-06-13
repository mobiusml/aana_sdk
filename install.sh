#!/bin/sh
poetry install $1
poetry run pip install git+https://github.com/mobiusml/faster-whisper.git@v1.0.1_mobiusml_v1.1