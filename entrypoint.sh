#!/bin/bash

# Get the argument number from the environment variable
CLIENT_NUM=${CLIENT_NUM:-1}  # Default to 1 if not set

# Check if NOISE environment variable exists and set the additional parameter
if [ -n "$NOISE" ]; then
    # Run the client script with the additional noise parameter
    python client.py --num=$CLIENT_NUM --noise
else
    # Run the client script without the noise parameter
    python client.py --num=$CLIENT_NUM
fi
