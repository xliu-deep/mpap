"""
Configuration management for MPAP model.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for MPAP model."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._expand_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _expand_paths(self):
        """Expand environment variables and make paths absolute."""
        if 'paths' in self._config:
            for key, value in self._config['paths'].items():
                if isinstance(value, str):
                    # Expand environment variables
                    value = os.path.expandvars(value)
                    # Convert to absolute path if relative
                    if not os.path.isabs(value):
                        value = os.path.abspath(value)
                    self._config['paths'][key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.dim' or 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config

