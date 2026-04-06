"""Bulk-load NYC Taxi dataset into every supported database.

Provides load_dataset(), cleanup_dataset(), and get_connection_info() for
integration tests that need realistic data at scale.
"""

from __future__ import annotations

import os
import tempfile
import time

import pandas as pd

from tests.integration.large_dataset import get_create_table_sql, get_dataset

# ---------------------------------------------------------------------------
# Connection info (env vars with docker-compose.test.yml defaults)
# ---------------------------------------------------------------------------

_CONNECTION_DEFAULTS: dict[str, dict[str, str]] = {
    "postgresql": {
        "host": "localhost",
        "port": "15432",
        "user": "testuser",
        "password": "testpass",
        "database": "testdb",
    },
    "mysql": {
        "host": "localhost",
        "port": "13306",
        "user": "testuser",
        "password": "testpass",
        "database": "testdb",
    },
    "mssql": {
        "host": "localhost",
        "port": "11433",
        "user": "sa",
        "password": "TestPass123!",
        "database": "master",
    },
    "oracle": {
        "host": "localhost",
        "port": "11521",
        "user": "testuser",
        "password": "testpass",
        "dsn": "localhost:11521/FREEPDB1",
    },
    "sqlite": {"path": ""},
    "mongodb": {
        "host": "localhost",
        "port": "17017",
        "database": "testdb",
    },
    "redis": {"host": "localhost", "port": "16379", "db": "0"},
    "elasticsearch": {"host": "localhost", "port": "19200"},
    "influxdb": {
        "url": "http://localhost:18086",
        "token": "testtokenforlocaldata",
        "org": "testorg",
        "bucket": "testbucket",
    },
    "neo4j": {"host": "localhost", "port": "17687"},
    "couchdb": {
        "host": "localhost",
        "port": "15984",
        "user": "admin",
        "password": "admin",
    },
}

_ENV_PREFIX = "LOCALDATA_TEST_"

# Track the temp SQLite path so get_connection_info can return it
_sqlite_temp_path: str | None = None


def _env(db_type: str, key: str, default: str) -> str:
    """Read connection param from env with fallback to default."""
    env_key = f"{_ENV_PREFIX}{db_type.upper()}_{key.upper()}"
    return os.environ.get(env_key, default)


def get_connection_info(db_type: str) -> dict:
    """Return connection details for a database type."""
    db_type = db_type.lower()
    if db_type not in _CONNECTION_DEFAULTS:
        raise ValueError(
            f"Unsupported db_type '{db_type}'. "
            f"Supported: {', '.join(sorted(_CONNECTION_DEFAULTS))}"
        )
    defaults = _CONNECTION_DEFAULTS[db_type]
    info = {k: _env(db_type, k, v) for k, v in defaults.items()}

    # Use the temp path created by the loader if available
    if db_type == "sqlite" and _sqlite_temp_path and not info.get("path"):
        info["path"] = _sqlite_temp_path

    # Build convenience URIs
    if db_type == "postgresql":
        h, p = info["host"], info["port"]
        info["uri"] = (
            f"postgresql://{info['user']}:{info['password']}@{h}:{p}/{info['database']}"
        )
    elif db_type == "mysql":
        h, p = info["host"], info["port"]
        info["uri"] = (
            f"mysql+mysqlconnector://{info['user']}:{info['password']}@{h}:{p}/{info['database']}"
        )
    elif db_type == "mssql":
        h, p = info["host"], info["port"]
        info["uri"] = (
            f"mssql+pymssql://{info['user']}:{info['password']}@{h}:{p}/{info['database']}"
        )
    elif db_type == "mongodb":
        info["uri"] = f"mongodb://{info['host']}:{info['port']}/{info['database']}"
    elif db_type == "redis":
        info["uri"] = f"redis://{info['host']}:{info['port']}/{info['db']}"
    elif db_type == "elasticsearch":
        info["uri"] = f"http://{info['host']}:{info['port']}"
    elif db_type == "couchdb":
        info["uri"] = (
            f"http://{info['user']}:{info['password']}@{info['host']}:{info['port']}"
        )
    return info


# ---------------------------------------------------------------------------
# Row-count caps per database type
# ---------------------------------------------------------------------------

_MAX_ROWS_CAP: dict[str, int] = {
    "redis": 10_000,
    "neo4j": 50_000,
    "influxdb": 100_000,
    "couchdb": 100_000,
}


# ---------------------------------------------------------------------------
# SQL loaders
# ---------------------------------------------------------------------------


def _load_sql(db_type: str, df: pd.DataFrame, table_name: str) -> int:
    """Load into a SQL database via SQLAlchemy + pandas to_sql."""
    try:
        import sqlalchemy  # noqa: F401
    except ImportError as exc:
        raise ImportError("sqlalchemy is required for SQL loading") from exc

    info = get_connection_info(db_type)

    if db_type == "sqlite":
        global _sqlite_temp_path
        if info.get("path"):
            path = info["path"]
        else:
            fd, path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
        _sqlite_temp_path = path
        info["path"] = path
        uri = f"sqlite:///{path}"
    elif db_type == "oracle":
        try:
            import oracledb  # noqa: F401
        except ImportError as exc:
            raise ImportError("oracledb is required for Oracle loading") from exc
        host = _env("oracle", "host", "localhost")
        port = _env("oracle", "port", "11521")
        uri = f"oracle+oracledb://{info['user']}:{info['password']}@{host}:{port}/?service_name=FREEPDB1"
    else:
        uri = info["uri"]

    engine = sqlalchemy.create_engine(uri)
    dialect = db_type if db_type != "oracle" else "oracle"
    ddl = get_create_table_sql(dialect, table_name)

    with engine.connect() as conn:
        if db_type == "oracle":
            try:
                conn.execute(sqlalchemy.text(f"DROP TABLE {table_name}"))
                conn.commit()
            except Exception:
                conn.rollback()
        else:
            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()
        conn.execute(sqlalchemy.text(ddl))
        conn.commit()

    total = len(df)

    # PostgreSQL: use COPY via psycopg2 for 10-50x speedup over INSERT
    if db_type == "postgresql":
        return _load_pg_copy(engine, df, table_name)

    # SQLite has a 999 bind-variable limit, MSSQL has ~2100.
    # With 19 columns, max rows per multi-INSERT: SQLite ~52, MSSQL ~110.
    # Use executemany (method=None) for these; multi for MySQL/Oracle.
    if db_type in ("sqlite", "mssql", "oracle"):
        chunk = 50_000
        method = None  # default executemany — no bind-variable limit
    else:
        chunk = 50_000
        method = "multi"

    loaded = 0
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        df.iloc[start:end].to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            method=method,
        )
        loaded = end
        if loaded % 100_000 == 0 or loaded == total:
            print(f"  [{db_type}] {loaded:,}/{total:,} rows loaded")
    return loaded


def _load_pg_copy(engine, df: pd.DataFrame, table_name: str) -> int:
    """Fast PostgreSQL bulk load via COPY FROM STDIN (CSV)."""
    import io

    total = len(df)
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cols = ",".join(df.columns)
        chunk = 500_000
        loaded = 0
        for start in range(0, total, chunk):
            buf = io.StringIO()
            df.iloc[start : start + chunk].to_csv(buf, index=False, header=False)
            buf.seek(0)
            cur.copy_expert(f"COPY {table_name} ({cols}) FROM STDIN WITH CSV", buf)
            raw.commit()
            loaded = min(start + chunk, total)
            print(f"  [postgresql/COPY] {loaded:,}/{total:,} rows loaded")
        return loaded
    finally:
        raw.close()


# ---------------------------------------------------------------------------
# NoSQL loaders
# ---------------------------------------------------------------------------


def _load_mongodb(df: pd.DataFrame, table_name: str) -> int:
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise ImportError("pymongo is required for MongoDB loading") from exc

    info = get_connection_info("mongodb")
    client = MongoClient(info["uri"])
    db = client[info["database"]]
    db.drop_collection(table_name)
    col = db[table_name]

    records = df.to_dict("records")
    total = len(records)
    batch = 50_000
    loaded = 0
    for start in range(0, total, batch):
        end = min(start + batch, total)
        col.insert_many(records[start:end])
        loaded = end
        print(f"  [mongodb] {loaded:,}/{total:,} docs inserted")
    client.close()
    return loaded


def _load_redis(df: pd.DataFrame, table_name: str) -> int:
    try:
        import redis
    except ImportError as exc:
        raise ImportError("redis is required for Redis loading") from exc

    info = get_connection_info("redis")
    r = redis.from_url(info["uri"])
    r.delete(f"{table_name}:meta")
    # Delete existing row keys
    cursor, keys = r.scan(match=f"{table_name}:row:*", count=10_000)
    while keys:
        r.delete(*keys)
        if cursor == 0:
            break
        cursor, keys = r.scan(cursor=cursor, match=f"{table_name}:row:*", count=10_000)

    # Store rows as hashes
    pipe = r.pipeline()
    total = len(df)
    for i, row in df.iterrows():
        key = f"{table_name}:row:{i}"
        pipe.hset(key, mapping={k: str(v) for k, v in row.items()})
        if (i + 1) % 1_000 == 0:
            pipe.execute()
            pipe = r.pipeline()
            if (i + 1) % 5_000 == 0:
                print(f"  [redis] {i + 1:,}/{total:,} hashes stored")
    pipe.execute()

    # Store aggregated stats
    stats = {
        "row_count": str(total),
        "avg_fare": str(df["fare_amount"].mean()),
        "avg_distance": str(df["trip_distance"].mean()),
        "avg_total": str(df["total_amount"].mean()),
        "avg_tip": str(df["tip_amount"].mean()),
        "avg_passengers": str(df["passenger_count"].mean()),
    }
    r.hset(f"{table_name}:meta", mapping=stats)
    print(f"  [redis] {total:,} rows + meta stats stored")
    return total


def _load_elasticsearch(df: pd.DataFrame, table_name: str) -> int:
    try:
        from elasticsearch import Elasticsearch
        from elasticsearch.helpers import bulk
    except ImportError as exc:
        raise ImportError(
            "elasticsearch is required for Elasticsearch loading"
        ) from exc

    info = get_connection_info("elasticsearch")
    es = Elasticsearch(info["uri"])
    if es.indices.exists(index=table_name):
        es.indices.delete(index=table_name)

    mapping = {
        "mappings": {
            "properties": {
                "VendorID": {"type": "integer"},
                "tpep_pickup_datetime": {"type": "date"},
                "tpep_dropoff_datetime": {"type": "date"},
                "passenger_count": {"type": "float"},
                "trip_distance": {"type": "float"},
                "RatecodeID": {"type": "float"},
                "store_and_fwd_flag": {"type": "keyword"},
                "PULocationID": {"type": "integer"},
                "DOLocationID": {"type": "integer"},
                "payment_type": {"type": "integer"},
                "fare_amount": {"type": "float"},
                "extra": {"type": "float"},
                "mta_tax": {"type": "float"},
                "tip_amount": {"type": "float"},
                "tolls_amount": {"type": "float"},
                "improvement_surcharge": {"type": "float"},
                "total_amount": {"type": "float"},
                "congestion_surcharge": {"type": "float"},
                "Airport_fee": {"type": "float"},
            }
        }
    }
    es.indices.create(index=table_name, body=mapping)

    total = len(df)
    batch = 10_000
    loaded = 0
    for start in range(0, total, batch):
        end = min(start + batch, total)
        actions = [
            {"_index": table_name, "_source": row.to_dict()}
            for _, row in df.iloc[start:end].iterrows()
        ]
        bulk(es, actions)
        loaded = end
        print(f"  [elasticsearch] {loaded:,}/{total:,} docs indexed")
    es.indices.refresh(index=table_name)
    return loaded


def _load_influxdb(df: pd.DataFrame, table_name: str) -> int:
    try:
        from influxdb_client import InfluxDBClient, Point, WritePrecision
        from influxdb_client.client.write_api import SYNCHRONOUS
    except ImportError as exc:
        raise ImportError("influxdb-client is required for InfluxDB loading") from exc

    info = get_connection_info("influxdb")
    client = InfluxDBClient(
        url=info["url"],
        token=info["token"],
        org=info["org"],
    )
    delete_api = client.delete_api()
    delete_api.delete(
        "1970-01-01T00:00:00Z",
        "2100-01-01T00:00:00Z",
        predicate="",
        bucket=info["bucket"],
        org=info["org"],
    )

    write_api = client.write_api(write_options=SYNCHRONOUS)
    total = len(df)
    batch = 5_000
    loaded = 0
    for start in range(0, total, batch):
        end = min(start + batch, total)
        points = []
        for _, row in df.iloc[start:end].iterrows():
            p = (
                Point(table_name)
                .tag(
                    "VendorID",
                    str(int(row["VendorID"])) if pd.notna(row["VendorID"]) else "0",
                )
                .tag(
                    "PULocationID",
                    (
                        str(int(row["PULocationID"]))
                        if pd.notna(row["PULocationID"])
                        else "0"
                    ),
                )
                .tag(
                    "payment_type",
                    (
                        str(int(row["payment_type"]))
                        if pd.notna(row["payment_type"])
                        else "0"
                    ),
                )
                .field(
                    "trip_distance",
                    (
                        float(row["trip_distance"])
                        if pd.notna(row["trip_distance"])
                        else 0.0
                    ),
                )
                .field(
                    "fare_amount",
                    float(row["fare_amount"]) if pd.notna(row["fare_amount"]) else 0.0,
                )
                .field(
                    "tip_amount",
                    float(row["tip_amount"]) if pd.notna(row["tip_amount"]) else 0.0,
                )
                .field(
                    "total_amount",
                    (
                        float(row["total_amount"])
                        if pd.notna(row["total_amount"])
                        else 0.0
                    ),
                )
                .field(
                    "passenger_count",
                    (
                        float(row["passenger_count"])
                        if pd.notna(row["passenger_count"])
                        else 0.0
                    ),
                )
                .time(row["tpep_pickup_datetime"], WritePrecision.S)
            )
            points.append(p)
        write_api.write(bucket=info["bucket"], org=info["org"], record=points)
        loaded = end
        print(f"  [influxdb] {loaded:,}/{total:,} points written")
    client.close()
    return loaded


def _load_neo4j(df: pd.DataFrame, table_name: str) -> int:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError("neo4j is required for Neo4j loading") from exc

    info = get_connection_info("neo4j")
    uri = f"bolt://{info['host']}:{info['port']}"
    driver = GraphDatabase.driver(uri)
    with driver.session() as session:
        session.run(f"MATCH (n:{table_name}) DETACH DELETE n")

    total = len(df)
    batch = 2_000
    loaded = 0
    records = df.to_dict("records")
    with driver.session() as session:
        for start in range(0, total, batch):
            end = min(start + batch, total)
            chunk = records[start:end]
            for rec in chunk:
                for k, v in rec.items():
                    if pd.isna(v):
                        rec[k] = None
                    elif hasattr(v, "isoformat"):
                        rec[k] = v.isoformat()
            session.run(
                f"UNWIND $rows AS row CREATE (n:{table_name}) SET n = row",
                rows=chunk,
            )
            loaded = end
            print(f"  [neo4j] {loaded:,}/{total:,} nodes created")
    driver.close()
    return loaded


def _load_couchdb(df: pd.DataFrame, table_name: str) -> int:
    try:
        import requests as req
    except ImportError as exc:
        raise ImportError("requests is required for CouchDB loading") from exc

    info = get_connection_info("couchdb")
    base = info["uri"]
    db_url = f"{base}/{table_name}"

    # Delete and recreate the database
    req.delete(db_url)
    resp = req.put(db_url)
    if resp.status_code not in (201, 412):
        resp.raise_for_status()

    total = len(df)
    batch = 5_000
    loaded = 0
    records = df.to_dict("records")
    for start in range(0, total, batch):
        end = min(start + batch, total)
        docs = []
        for rec in records[start:end]:
            doc = {}
            for k, v in rec.items():
                if pd.isna(v):
                    doc[k] = None
                elif hasattr(v, "isoformat"):
                    doc[k] = v.isoformat()
                else:
                    doc[k] = v
            docs.append(doc)
        resp = req.post(f"{db_url}/_bulk_docs", json={"docs": docs})
        resp.raise_for_status()
        loaded = end
        print(f"  [couchdb] {loaded:,}/{total:,} docs inserted")
    return loaded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS: dict[str, str] = {
    "postgresql": "sql",
    "mysql": "sql",
    "mssql": "sql",
    "oracle": "sql",
    "sqlite": "sql",
    "mongodb": "mongodb",
    "redis": "redis",
    "elasticsearch": "elasticsearch",
    "influxdb": "influxdb",
    "neo4j": "neo4j",
    "couchdb": "couchdb",
}


def load_dataset(
    db_type: str,
    max_rows: int | None = None,
    table_name: str = "taxi_trips",
    df: pd.DataFrame | None = None,
) -> dict:
    """Load NYC Taxi data into a database. Returns stats.

    Args:
        df: Pre-loaded DataFrame to reuse. If None, reads from cache.
    """
    db_type = db_type.lower()
    if db_type not in _LOADERS:
        raise ValueError(
            f"Unsupported db_type '{db_type}'. Supported: {', '.join(sorted(_LOADERS))}"
        )

    cap = _MAX_ROWS_CAP.get(db_type)
    if cap is not None:
        max_rows = min(max_rows, cap) if max_rows is not None else cap

    if df is None:
        print(f"Loading dataset for {db_type} (max_rows={max_rows})...")
        df = get_dataset(max_rows=max_rows)
        df.columns = [c.lower() for c in df.columns]
    elif max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    print(f"  [{db_type}] Dataset: {len(df):,} rows, {len(df.columns)} columns")

    t0 = time.monotonic()
    loader_type = _LOADERS[db_type]
    if loader_type == "sql":
        rows = _load_sql(db_type, df, table_name)
    elif loader_type == "mongodb":
        rows = _load_mongodb(df, table_name)
    elif loader_type == "redis":
        rows = _load_redis(df, table_name)
    elif loader_type == "elasticsearch":
        rows = _load_elasticsearch(df, table_name)
    elif loader_type == "influxdb":
        rows = _load_influxdb(df, table_name)
    elif loader_type == "neo4j":
        rows = _load_neo4j(df, table_name)
    elif loader_type == "couchdb":
        rows = _load_couchdb(df, table_name)
    else:
        raise RuntimeError(f"No loader for type '{loader_type}'")
    elapsed = time.monotonic() - t0

    print(f"  Done: {rows:,} rows in {elapsed:.1f}s")
    return {
        "db_type": db_type,
        "rows_loaded": rows,
        "elapsed_seconds": round(elapsed, 2),
        "table": table_name,
    }


def cleanup_dataset(db_type: str, table_name: str = "taxi_trips") -> None:
    """Remove test data from database."""
    db_type = db_type.lower()
    if db_type not in _LOADERS:
        raise ValueError(f"Unsupported db_type '{db_type}'")

    loader_type = _LOADERS[db_type]

    if loader_type == "sql":
        import sqlalchemy

        info = get_connection_info(db_type)
        if db_type == "sqlite":
            uri = f"sqlite:///{info['path']}" if info.get("path") else None
            if uri is None:
                return
        elif db_type == "oracle":
            host = _env("oracle", "host", "localhost")
            port = _env("oracle", "port", "11521")
            uri = f"oracle+oracledb://{info['user']}:{info['password']}@{host}:{port}/?service_name=FREEPDB1"
        else:
            uri = info["uri"]
        engine = sqlalchemy.create_engine(uri)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()

    elif loader_type == "mongodb":
        from pymongo import MongoClient

        info = get_connection_info("mongodb")
        client = MongoClient(info["uri"])
        client[info["database"]].drop_collection(table_name)
        client.close()

    elif loader_type == "redis":
        import redis as redis_lib

        info = get_connection_info("redis")
        r = redis_lib.from_url(info["uri"])
        r.delete(f"{table_name}:meta")
        cursor, keys = r.scan(match=f"{table_name}:row:*", count=10_000)
        while keys:
            r.delete(*keys)
            if cursor == 0:
                break
            cursor, keys = r.scan(
                cursor=cursor,
                match=f"{table_name}:row:*",
                count=10_000,
            )

    elif loader_type == "elasticsearch":
        from elasticsearch import Elasticsearch

        info = get_connection_info("elasticsearch")
        es = Elasticsearch(info["uri"])
        if es.indices.exists(index=table_name):
            es.indices.delete(index=table_name)

    elif loader_type == "influxdb":
        from influxdb_client import InfluxDBClient

        info = get_connection_info("influxdb")
        client = InfluxDBClient(
            url=info["url"],
            token=info["token"],
            org=info["org"],
        )
        client.delete_api().delete(
            "1970-01-01T00:00:00Z",
            "2100-01-01T00:00:00Z",
            predicate="",
            bucket=info["bucket"],
            org=info["org"],
        )
        client.close()

    elif loader_type == "neo4j":
        from neo4j import GraphDatabase

        info = get_connection_info("neo4j")
        uri = f"bolt://{info['host']}:{info['port']}"
        driver = GraphDatabase.driver(uri)
        with driver.session() as session:
            session.run(f"MATCH (n:{table_name}) DETACH DELETE n")
        driver.close()

    elif loader_type == "couchdb":
        import requests as req

        info = get_connection_info("couchdb")
        req.delete(f"{info['uri']}/{table_name}")
