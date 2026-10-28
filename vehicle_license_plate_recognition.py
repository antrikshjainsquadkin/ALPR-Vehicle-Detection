#!/usr/bin/env python3
"""Vehicle License Plate Recognition (ALPR) application entrypoint.

Provides the CLI interface and interactive terminal menu for the ALPR system.
Delegates to the core pipeline in main.py.
"""

import os
import sys

# Ensure current directory is on sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import main

if __name__ == "__main__":
    main.main()
