"""
Dataset generators for LocalData MCP stress testing.

This module provides generators for creating large, realistic datasets
to test the performance and reliability of LocalData MCP under stress.
"""

from .social_media_generator import SocialMediaGenerator

__all__ = ['SocialMediaGenerator']