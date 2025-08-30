# InfluxDB Service Startup Fix

## Problem
InfluxDB service failed to start with error: `Bootstrap failed: 5: Input/output error`

The service was outputting: "command required, -h/--help/--help-all for help" repeatedly.

## Root Cause
The Homebrew-installed InfluxDB v3 (`influxdb3`) binary requires explicit commands to run. The original LaunchAgent plist file was calling the binary without the required `serve` command and parameters.

## Solution
Updated the LaunchAgent plist file at `~/Library/LaunchAgents/homebrew.mxcl.influxdb.plist` with the correct command line arguments:

```xml
<key>ProgramArguments</key>
<array>
    <string>/usr/local/opt/influxdb/bin/influxdb3</string>
    <string>serve</string>
    <string>--node-id</string>
    <string>localhost</string>
    <string>--object-store</string>
    <string>file</string>
    <string>--data-dir</string>
    <string>/usr/local/var/lib/influxdb3</string>
    <string>--http-bind</string>
    <string>127.0.0.1:8086</string>
    <string>--without-auth</string>
</array>
```

## Commands Used
1. Stop the service: `brew services stop influxdb`
2. Unload the problematic plist: `launchctl unload ~/Library/LaunchAgents/homebrew.mxcl.influxdb.plist`
3. Remove and recreate the plist file with correct parameters
4. Load the service: `launchctl load ~/Library/LaunchAgents/homebrew.mxcl.influxdb.plist`

## Verification
- Service status: `brew services list | grep influxdb` shows "started"
- Health check: `curl http://127.0.0.1:8086/health` returns "OK"
- InfluxDB3 server logs show proper startup without errors

## Result
InfluxDB service now starts successfully and is accessible on localhost:8086, completing the database connection suite for 100% success rate.