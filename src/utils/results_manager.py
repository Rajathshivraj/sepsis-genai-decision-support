"""
results_manager.py — Utilities for saving and loading experiment metrics and artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from configs.config import PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("results_manager")

_RESULTS_DIR = PROJECT_ROOT / "results"

def save_experiment_results(
    experiment_name: str,
    data: Dict[str, Any],
    filename: Optional[str] = None
) -> Path:
    """
    Save experiment results to a JSON file in the results directory.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment (e.g., 'baseline', 'hybrid').
    data : dict
        The data/metrics to save.
    filename : str, optional
        Custom filename. If None, uses timestamp.
        
    Returns
    -------
    Path
        The path to the saved file.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
    
    save_path = _RESULTS_DIR / filename
    
    try:
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Results saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        
    return save_path

def load_latest_results(experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load the most recent results for a given experiment name."""
    if not _RESULTS_DIR.exists():
        return None
        
    files = list(_RESULTS_DIR.glob(f"{experiment_name}_*.json"))
    if not files:
        return None
        
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results from {latest_file}: {e}")
        return None
