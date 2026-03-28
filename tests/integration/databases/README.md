# Database Integration Tests

Integration tests that verify LocalData MCP enterprise database connectors against
real database instances running in Docker containers.

## Quick start

```bash
# 1. Start the test containers
docker compose -f docker-compose.test.yml up -d

# 2. Wait for health checks to pass
./scripts/wait-for-databases.sh

# 3. Run the integration suite
pytest tests/integration/databases/ -v -m integration

# 4. Tear down
docker compose -f docker-compose.test.yml down -v
```

## Container details

| Database   | Image                                       | Host port | Default credentials         |
| ---------- | ------------------------------------------- | --------- | --------------------------- |
| PostgreSQL | `postgres:16-alpine`                        | 15432     | `testuser` / `testpass`     |
| MySQL      | `mysql:8.0`                                 | 13306     | `testuser` / `testpass`     |
| MS SQL     | `mcr.microsoft.com/mssql/server:2022-latest`| 11433     | `sa` / `TestPass123!`       |

Oracle Free is excluded from the default compose file because of its large image
size (~2 GB) and slow startup (~120 s). Oracle tests should be decorated with
`@pytest.mark.skip(reason="Requires Oracle container")`.

## Environment variables

Override the default connection URLs by setting these before running pytest:

- `TEST_POSTGRES_URL` (default: `postgresql://testuser:testpass@localhost:15432/testdb`)
- `TEST_MYSQL_URL` (default: `mysql+mysqlconnector://testuser:testpass@localhost:13306/testdb`)
- `TEST_MSSQL_URL` (default: `mssql+pymssql://sa:TestPass123!@localhost:11433/master`)

## CI

The GitHub Actions workflow at `.github/workflows/integration-tests.yml` runs
PostgreSQL and MySQL as service containers automatically. MS SQL is omitted from
CI because GitHub-hosted runners have limited resources; add it when a
self-hosted runner is available.
