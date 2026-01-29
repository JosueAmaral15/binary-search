"""
Logging configuration for math_toolkit package.

Provides centralized logging setup with sensible defaults.
Users can configure logging level and format as needed.
"""

import logging
import sys


def setup_logging(level=logging.INFO, format_string=None):
    """
    Setup logging for math_toolkit package.
    
    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_string : str, optional
        Custom format string for log messages
        
    Examples
    --------
    >>> from math_toolkit.logging_config import setup_logging
    >>> import logging
    >>> 
    >>> # Basic setup
    >>> setup_logging()
    >>> 
    >>> # Debug mode
    >>> setup_logging(level=logging.DEBUG)
    >>> 
    >>> # Custom format
    >>> setup_logging(format_string='%(levelname)s - %(message)s')
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger for math_toolkit
    logger = logging.getLogger('math_toolkit')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def get_logger(name):
    """
    Get a logger for a specific module.
    
    Parameters
    ----------
    name : str
        Module name (typically __name__)
        
    Returns
    -------
    logger : logging.Logger
        Configured logger instance
        
    Examples
    --------
    >>> from math_toolkit.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting optimization...")
    """
    return logging.getLogger(f'math_toolkit.{name}')


# Setup default logging configuration when module is imported
# Users can override by calling setup_logging()
_default_logger = setup_logging(level=logging.WARNING)  # Default: only warnings and errors
