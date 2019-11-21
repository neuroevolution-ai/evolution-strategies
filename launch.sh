#!/bin/bash

HASHED_PASSWORD=`cat hashed_password.txt`

xvfb-run -a -s="-screen 0 1400x900x24" start.sh jupyter lab --NotebookApp.password="${HASHED_PASSWORD}"