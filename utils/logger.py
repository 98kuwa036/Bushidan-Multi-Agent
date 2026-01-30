"""
Bushidan Multi-Agent System v9.1 - Logging Utilities

Centralized logging configuration for the system.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup centralized logger for Bushidan v9.1"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        # Formatter with Japanese-friendly encoding
        formatter = logging.Formatter(
            '%(asctime)s - 🏯 %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger or create new one"""
    return logging.getLogger(name) or setup_logger(name)