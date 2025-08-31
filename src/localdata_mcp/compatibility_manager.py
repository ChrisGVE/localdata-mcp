"""
Backward Compatibility Manager for LocalData MCP v1.3.1
========================================================

Ensures seamless migration from v1.3.0 while providing clear deprecation warnings
and migration guidance to users. Maintains API compatibility for external integrations.
"""

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DeprecationInfo:
    """Information about deprecated features and their replacements."""
    feature: str
    version_deprecated: str
    version_removed: Optional[str]
    replacement: str
    migration_guide: str
    impact_level: str  # 'low', 'medium', 'high', 'critical'

class CompatibilityManager:
    """
    Manages backward compatibility for LocalData MCP v1.3.1.
    
    Provides:
    - Deprecation warnings with clear migration guidance
    - API compatibility wrappers
    - Configuration migration detection and assistance
    - Response format compatibility
    """
    
    def __init__(self):
        self._deprecation_warnings_shown = set()
        self._legacy_config_detected = False
        self._compatibility_mode = False
        
        # Define deprecated features
        self._deprecated_features = {
            'env_only_config': DeprecationInfo(
                feature='Environment Variable Only Configuration',
                version_deprecated='1.3.1',
                version_removed='1.4.0',
                replacement='YAML Configuration Files',
                migration_guide='Create localdata.yaml config file. See documentation for examples.',
                impact_level='medium'
            ),
            'single_database_env_vars': DeprecationInfo(
                feature='Single Database Environment Variables (POSTGRES_HOST, etc.)',
                version_deprecated='1.3.1', 
                version_removed='1.4.0',
                replacement='Multi-database YAML configuration',
                migration_guide='Use databases section in localdata.yaml with named database configurations.',
                impact_level='low'
            ),
            'execute_query_no_analysis': DeprecationInfo(
                feature='execute_query without enable_analysis parameter',
                version_deprecated='1.3.1',
                version_removed=None,
                replacement='execute_query with enable_analysis=True',
                migration_guide='Add enable_analysis=True to execute_query calls for optimal performance.',
                impact_level='low'
            )
        }
        
    def warn_deprecated(self, feature_key: str, additional_context: Optional[str] = None) -> None:
        """
        Show deprecation warning for a feature.
        
        Args:
            feature_key: Key from self._deprecated_features
            additional_context: Optional additional context for the warning
        """
        if feature_key in self._deprecation_warnings_shown:
            return
            
        if feature_key not in self._deprecated_features:
            logger.warning(f"Unknown deprecated feature key: {feature_key}")
            return
            
        info = self._deprecated_features[feature_key]
        
        # Create warning message
        message = f"""
DEPRECATION WARNING: {info.feature}

This feature was deprecated in v{info.version_deprecated} and will be removed in {"a future version" if not info.version_removed else f"v{info.version_removed}"}.

Replacement: {info.replacement}
Migration Guide: {info.migration_guide}
Impact Level: {info.impact_level.upper()}
"""
        
        if additional_context:
            message += f"\nContext: {additional_context}"
            
        # Show warning based on impact level
        if info.impact_level in ['high', 'critical']:
            warnings.warn(message, DeprecationWarning, stacklevel=3)
            logger.warning(f"DEPRECATED: {info.feature}")
        else:
            logger.info(f"DEPRECATED: {info.feature}")
            
        # Only show each warning once per session
        self._deprecation_warnings_shown.add(feature_key)
        
    def detect_legacy_configuration(self) -> Dict[str, Any]:
        """
        Detect legacy configuration patterns and provide migration suggestions.
        
        Returns:
            Dict with detected legacy patterns and migration suggestions
        """
        legacy_patterns = {
            'detected': [],
            'suggestions': [],
            'migration_required': False
        }
        
        # Check for legacy single-database environment variables
        legacy_db_vars = [
            'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB',
            'MYSQL_HOST', 'MYSQL_PORT', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB',
            'SQLITE_PATH', 'DUCKDB_PATH', 'MONGODB_URL'
        ]
        
        found_legacy_vars = []
        for var in legacy_db_vars:
            if os.getenv(var):
                found_legacy_vars.append(var)
                
        if found_legacy_vars:
            legacy_patterns['detected'].append({
                'pattern': 'Legacy Environment Variables',
                'variables': found_legacy_vars,
                'impact': 'medium'
            })
            legacy_patterns['suggestions'].append(
                'Consider migrating to YAML configuration for better organization and multi-database support. '
                'Legacy environment variables will continue to work but are deprecated.'
            )
            self._legacy_config_detected = True
            
        # Check if only using environment variables (no YAML config)
        yaml_config_paths = [
            './localdata.yaml',
            os.path.expanduser('~/.localdata.yaml'),
            '/etc/localdata.yaml'
        ]
        
        has_yaml_config = any(Path(path).exists() for path in yaml_config_paths)
        
        if found_legacy_vars and not has_yaml_config:
            legacy_patterns['migration_required'] = True
            legacy_patterns['suggestions'].append(
                'No YAML configuration found. Create localdata.yaml for improved configuration management.'
            )
            
        return legacy_patterns
        
    def check_api_compatibility(self, function_name: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Check API call compatibility and provide warnings if needed.
        
        Args:
            function_name: Name of the API function being called
            args: Positional arguments passed to function
            kwargs: Keyword arguments passed to function
            
        Returns:
            Dict with compatibility info and any warnings
        """
        compatibility_info = {
            'compatible': True,
            'warnings': [],
            'suggestions': []
        }
        
        # Check execute_query compatibility
        if function_name == 'execute_query':
            # Check if using legacy 2-parameter call (name, query only)
            if len(args) == 2 and not kwargs:
                compatibility_info['warnings'].append(
                    'Using execute_query with only name and query parameters. '
                    'Consider using enable_analysis=True for improved performance.'
                )
                compatibility_info['suggestions'].append(
                    'Add enable_analysis=True parameter to enable pre-query optimization and memory management.'
                )
                
            # Check if explicitly disabling analysis
            if kwargs.get('enable_analysis') is False:
                self.warn_deprecated('execute_query_no_analysis', 
                                   'Analysis disabled - may impact performance on large queries')
                
        return compatibility_info
        
    def migrate_legacy_response_format(self, response: Any, format_version: str = 'v1.3.0') -> Any:
        """
        Ensure response format compatibility with older versions.
        
        Args:
            response: Response data from new system
            format_version: Target format version for compatibility
            
        Returns:
            Response formatted for backward compatibility
        """
        if format_version == 'v1.3.0':
            # v1.3.0 expected simple JSON string responses for execute_query
            # v1.3.1 might include additional metadata
            
            if isinstance(response, dict) and 'data' in response:
                # Extract just the data portion for backward compatibility
                return response['data']
            elif isinstance(response, dict) and 'result' in response:
                # Handle other metadata structures
                return response['result']
                
        return response
        
    def get_compatibility_status(self) -> Dict[str, Any]:
        """
        Get overall compatibility status and recommendations.
        
        Returns:
            Comprehensive compatibility report
        """
        legacy_config = self.detect_legacy_configuration()
        
        return {
            'version': '1.3.1',
            'compatibility_mode': self._compatibility_mode,
            'legacy_config_detected': self._legacy_config_detected,
            'legacy_patterns': legacy_config,
            'deprecation_warnings_shown': list(self._deprecation_warnings_shown),
            'recommendations': self._generate_recommendations(legacy_config)
        }
        
    def _generate_recommendations(self, legacy_config: Dict[str, Any]) -> List[str]:
        """Generate compatibility recommendations based on detected patterns."""
        recommendations = []
        
        if legacy_config['migration_required']:
            recommendations.append(
                'HIGH PRIORITY: Create YAML configuration file for improved multi-database support'
            )
            
        if self._legacy_config_detected:
            recommendations.append(
                'MEDIUM PRIORITY: Consider migrating environment variables to YAML configuration'
            )
            
        if not self._deprecation_warnings_shown:
            recommendations.append(
                'GOOD: No deprecated features detected in current usage'
            )
            
        return recommendations
        
    def enable_compatibility_mode(self, enabled: bool = True) -> None:
        """
        Enable strict compatibility mode for maximum backward compatibility.
        
        Args:
            enabled: Whether to enable compatibility mode
        """
        self._compatibility_mode = enabled
        
        if enabled:
            logger.info("Compatibility mode enabled - maximum backward compatibility active")
        else:
            logger.info("Compatibility mode disabled - using standard v1.3.1 behavior")
            
    def create_migration_script(self, output_path: Optional[str] = None) -> str:
        """
        Generate a migration script for upgrading from legacy configuration.
        
        Args:
            output_path: Optional path to save migration script
            
        Returns:
            Migration script content
        """
        legacy_config = self.detect_legacy_configuration()
        
        script_content = """#!/usr/bin/env python3
# LocalData MCP Configuration Migration Script
# Generated by CompatibilityManager v1.3.1

import os
import yaml
from pathlib import Path

def migrate_configuration():
    \"\"\"Migrate legacy environment variables to YAML configuration.\"\"\"
    
    config = {
        'databases': {},
        'logging': {'level': 'INFO'},
        'performance': {'memory_limit_mb': 512}
    }
    
"""
        
        # Add detected legacy variables to migration script
        for pattern in legacy_config['detected']:
            if pattern['pattern'] == 'Legacy Environment Variables':
                script_content += "    # Migrate detected environment variables\n"
                
                # Group by database type
                db_groups = {}
                for var in pattern['variables']:
                    if var.startswith('POSTGRES_'):
                        db_type = 'postgresql'
                    elif var.startswith('MYSQL_'):
                        db_type = 'mysql'
                    elif var.startswith('SQLITE_'):
                        db_type = 'sqlite'
                    elif var.startswith('DUCKDB_'):
                        db_type = 'duckdb'
                    elif var == 'MONGODB_URL':
                        db_type = 'mongodb'
                    else:
                        continue
                        
                    if db_type not in db_groups:
                        db_groups[db_type] = []
                    db_groups[db_type].append(var)
                    
                for db_type, vars_list in db_groups.items():
                    script_content += f"""
    # {db_type.title()} configuration
    if os.getenv('{vars_list[0]}'):  # Check if any {db_type} vars exist
        config['databases']['{db_type}'] = {{
            'type': '{db_type}',
"""
                    
                    if db_type in ['postgresql', 'mysql']:
                        script_content += """            'host': os.getenv('{}', 'localhost'),
            'port': int(os.getenv('{}', {})),
            'database': os.getenv('{}', ''),
            'user': os.getenv('{}', ''),
            'password': os.getenv('{}', '')
""".format(
                            f'{db_type.upper()}_HOST',
                            f'{db_type.upper()}_PORT',
                            5432 if db_type == 'postgresql' else 3306,
                            f'{db_type.upper()}_DB',
                            f'{db_type.upper()}_USER',
                            f'{db_type.upper()}_PASSWORD'
                        )
                    elif db_type in ['sqlite', 'duckdb']:
                        script_content += f"            'path': os.getenv('{db_type.upper()}_PATH', '')\n"
                    elif db_type == 'mongodb':
                        script_content += "            'url': os.getenv('MONGODB_URL', '')\n"
                        
                    script_content += "        }\n"
                    
        script_content += """
    # Write YAML configuration
    config_path = Path('./localdata.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration migrated to {config_path}")
    print("You can now use the new YAML configuration instead of environment variables.")
    print("Environment variables will still work but are deprecated.")

if __name__ == '__main__':
    migrate_configuration()
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(script_content)
            logger.info(f"Migration script written to {output_path}")
            
        return script_content

# Global compatibility manager instance
_compatibility_manager = None

def get_compatibility_manager() -> CompatibilityManager:
    """Get the global compatibility manager instance."""
    global _compatibility_manager
    if _compatibility_manager is None:
        _compatibility_manager = CompatibilityManager()
    return _compatibility_manager