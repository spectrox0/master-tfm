#!/bin/bash

# Check if Python is installed
if command -v python3 &>/dev/null; then
    echo "Python is installed. Running initialize.py..."
    python3 initialize.py
elif command -v python &>/dev/null; then
    echo "Python is installed. Running initialize.py..."
    python initialize.py
else
    echo "Python is not installed. Please install Python to proceed."
    exit 1
fi
