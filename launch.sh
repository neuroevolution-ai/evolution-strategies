#!/bin/bash

PASSWORD="sha1:9eeee5ad359d:b3a4cf67b0e0cbdf8ad4a63d8c2df3702bc26b33"

xvfb-run -a -s="-screen 0 1400x900x24" start.sh jupyter lab --NotebookApp.password="$PASSWORD"