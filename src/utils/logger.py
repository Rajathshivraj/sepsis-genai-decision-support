import logging
import sys

def setup_logger(name="sepsis_genai"):
    """
    Sets up a structured logger for the project.
    
    Args:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Format: 2026-03-04 23:07:53 | INFO | module_name | message
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# Create a default logger instance for common use
logger = setup_logger()
