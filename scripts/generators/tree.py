"""Generate tree-structured test fixtures (YAML, TOML, INI, XML)."""

from __future__ import annotations

import os

from generators._common import sub_dir, write_text

_YAML = """\
app:
  name: test-application
  version: 1.2.3
  debug: false

database:
  host: localhost
  port: 5432
  name: testdb
  pool:
    min: 2
    max: 10
    timeout: 30

logging:
  level: info
  format: json
  outputs:
    - type: file
      path: /var/log/app.log
    - type: stdout

features:
  cache_enabled: true
  rate_limit: 100
  allowed_origins:
    - "https://example.com"
    - "https://api.example.com"
"""

_TOML = """\
[app]
name = "test-application"
version = "1.2.3"
debug = false

[database]
host = "localhost"
port = 5432
name = "testdb"

[database.pool]
min = 2
max = 10
timeout = 30

[logging]
level = "info"
format = "json"

[[logging.outputs]]
type = "file"
path = "/var/log/app.log"

[[logging.outputs]]
type = "stdout"

[features]
cache_enabled = true
rate_limit = 100
allowed_origins = ["https://example.com", "https://api.example.com"]
"""

_INI = """\
[app]
name = test-application
version = 1.2.3
debug = false

[database]
host = localhost
port = 5432
name = testdb

[database.pool]
min = 2
max = 10
timeout = 30

[logging]
level = info
format = json

[features]
cache_enabled = true
rate_limit = 100
"""

_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<config>
  <app>
    <name>test-application</name>
    <version>1.2.3</version>
    <debug>false</debug>
  </app>
  <database>
    <host>localhost</host>
    <port>5432</port>
    <name>testdb</name>
    <pool min="2" max="10" timeout="30"/>
  </database>
  <logging level="info" format="json">
    <output type="file" path="/var/log/app.log"/>
    <output type="stdout"/>
  </logging>
  <features>
    <cache_enabled>true</cache_enabled>
    <rate_limit>100</rate_limit>
    <allowed_origins>
      <origin>https://example.com</origin>
      <origin>https://api.example.com</origin>
    </allowed_origins>
  </features>
</config>
"""


def generate_tree(output_dir: str) -> list[str]:
    """Generate YAML, TOML, INI, and XML config files in tree/ subdirectory."""
    d = sub_dir(output_dir, "tree")
    return [
        write_text(os.path.join(d, "config.yaml"), _YAML),
        write_text(os.path.join(d, "config.toml"), _TOML),
        write_text(os.path.join(d, "config.ini"), _INI),
        write_text(os.path.join(d, "config.xml"), _XML),
    ]
