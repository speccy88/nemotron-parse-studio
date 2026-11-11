#!/usr/bin/env python3
"""Compatibility entry point: now exclusively uses the private stdio worker."""
import sys
from validate_stdio import main
if __name__ == "__main__":
    sys.argv[1:1] = ["--scope", "cancel"]
    main()
