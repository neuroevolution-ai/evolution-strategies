#!/bin/bash

HASHED_PASSWORD=`cat hashed_password.txt`

# Run Jupyter Lab with a fake display to allow rendering in the OpenAI Gym as suggested here:
# https://github.com/openai/gym#rendering-on-a-server
xvfb-run -a -s="-screen 0 1400x900x24" start.sh jupyter lab --NotebookApp.password="${HASHED_PASSWORD}"