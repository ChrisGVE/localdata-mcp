"""Pipeline integration test configuration.

Skip test modules that reference unfinalized APIs.
"""

collect_ignore_glob = [
    "test_domain_combinations.py",
    "test_format_conversions.py",
]
