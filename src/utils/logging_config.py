import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Configures basic logging for the application.
    
    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        stream=sys.stdout
    )

def get_logger(name: str):
    """
    Retrieves a logger instance with a specific name.
    
    Args:
        name (str): The name for the logger, typically __name__.
        
    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name) 