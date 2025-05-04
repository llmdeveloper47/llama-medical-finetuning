#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HealthCareMagic-100K dataset processor for fine-tuning LLaMA 3.1 8B model.
Converts the HealthCareMagic-100K dataset to an instruction-following format.
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

# Add the project root to the path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from datasets import Dataset, load_dataset
except ImportError:
    print("datasets library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import Dataset, load_dataset

try:
    from src.utils.logging_utils import get_logger
except ImportError:
    # Fallback if logging_utils is not available
    import logging
    def get_logger(name):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger(name)

logger = get_logger(__name__)