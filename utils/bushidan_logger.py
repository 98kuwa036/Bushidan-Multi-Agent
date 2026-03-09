```python
import logging
import os
from pathlib import Path


def setup_logger(name, log_file=None, level=logging.INFO):
    """Function to set up a logger with file and console handlers."""
    
    # Create logs directory if it doesn't exist
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """Get a configured logger instance."""
    import yaml
    from pathlib import Path
    
    config_path = Path("bushidan_config.yaml")
    if not config_path.exists():
        # Fallback to basic setup
        return logging.getLogger(name)
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        log_file = config.get('logging', {}).get('file')
        level_str = config.get('logging', {}).get('level', 'INFO')
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        level = level_map.get(level_str.upper(), logging.INFO)
        
        return setup_logger(name, log_file, level)
    except Exception as e:
        # Fallback to basic logger if config fails
        print(f"Failed to load configuration for logger {name}: {e}")
        return logging.getLogger(name)
```