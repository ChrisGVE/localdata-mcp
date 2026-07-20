"""End-to-end tests for the chunk-request tools.

``request_data_chunk`` and ``request_multiple_chunks`` are registered MCP tools
that could never return data: the protocol's loader was a placeholder returning
``None`` for every chunk id, and no data source was ever passed to it. Both tools
answered "Chunk N not available" for every query, and the unit suite stayed green
because it exercised the protocol and the manager separately.

The assertions here compare returned rows against ids known from the fixture, so
a chunk that comes back plausible-but-wrong — the wrong offset, a truncated tail,
the first chunk repeated — fails rather than passes.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from localdata_mcp import DatabaseManager

# Path security restricts connections to the working directory.
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE = os.path.join(FIXTURES_DIR, "chunking_rows.csv")

SEED = 20260720
TOTAL_ROWS = 50_000


@pytest.fixture(scope="module", autouse=True)
def chunking_fixture() -> None:
    """A result too large to return whole, with a row id that identifies itself.

    Every row carries its own zero-based index in ``row_id``, which is what lets
    a test say "chunk 100 must be rows 21700-21916" and mean it.
    """
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.DataFrame(
        {
            "row_id": range(TOTAL_ROWS),
            "measure": np.round(rng.normal(size=TOTAL_ROWS), 4),
            "label": [
                f"row-{i}-with-padding-to-widen-the-result" for i in range(TOTAL_ROWS)
            ],
        }
    ).to_csv(FIXTURE, index=False)


@pytest.fixture
def buffered_query():
    """Run the large query and hand back the manager, its id, and the chunk plan."""
    manager = DatabaseManager()
    manager.connect_database("chunky", "csv", FIXTURE)
    response = json.loads(manager.execute_query("chunky", "SELECT * FROM data_table"))
    query_id = response["metadata"]["query_id"]
    availability = manager.query_buffers[query_id].response_metadata.chunk_availability
    return manager, query_id, availability


def _expected_row_ids(chunk_id: int, chunk_size: int) -> list:
    start = chunk_id * chunk_size
    return list(range(start, min(start + chunk_size, TOTAL_ROWS)))


class TestRequestDataChunk:
    def test_first_chunk_returns_the_first_rows(self, buffered_query):
        manager, query_id, availability = buffered_query

        chunk = json.loads(manager.request_data_chunk(query_id, 0))

        assert [row["row_id"] for row in chunk["data"]] == _expected_row_ids(
            0, availability.chunk_size
        )

    def test_a_middle_chunk_is_offset_correctly(self, buffered_query):
        """The failure this catches is a loader that always returns chunk 0."""
        manager, query_id, availability = buffered_query

        chunk = json.loads(manager.request_data_chunk(query_id, 100))

        assert [row["row_id"] for row in chunk["data"]] == _expected_row_ids(
            100, availability.chunk_size
        )

    def test_the_last_chunk_reaches_the_final_row(self, buffered_query):
        """Chunk availability must span the whole result, not the buffered head.

        It was sized from the first chunk alone, so it advertised 10 chunks --
        2,170 rows of 50,000 -- and refused every id past that.
        """
        manager, query_id, availability = buffered_query
        last = availability.total_chunks - 1

        chunk = json.loads(manager.request_data_chunk(query_id, last))
        row_ids = [row["row_id"] for row in chunk["data"]]

        assert row_ids[-1] == TOTAL_ROWS - 1
        assert row_ids == _expected_row_ids(last, availability.chunk_size)

    def test_chunk_count_covers_every_row(self, buffered_query):
        _, _, availability = buffered_query

        assert availability.chunk_size * availability.total_chunks >= TOTAL_ROWS
        assert availability.chunk_size * (availability.total_chunks - 1) < TOTAL_ROWS

    def test_a_chunk_past_the_end_is_refused(self, buffered_query):
        manager, query_id, availability = buffered_query

        result = manager.request_data_chunk(query_id, availability.total_chunks + 5)

        assert "not available" in result


class TestRequestMultipleChunks:
    def test_each_requested_chunk_comes_back_with_its_own_rows(self, buffered_query):
        """Returning the same chunk under several ids would still look like a dict."""
        manager, query_id, availability = buffered_query

        chunks = json.loads(manager.request_multiple_chunks(query_id, "0,2,7"))

        assert sorted(chunks) == ["0", "2", "7"]
        for chunk_id in (0, 2, 7):
            assert [
                row["row_id"] for row in chunks[str(chunk_id)]["data"]
            ] == _expected_row_ids(chunk_id, availability.chunk_size)

    def test_malformed_ids_are_rejected(self, buffered_query):
        manager, query_id, _ = buffered_query

        assert "Invalid chunk_ids format" in manager.request_multiple_chunks(
            query_id, "0,not-a-number"
        )


class TestQueryMetadata:
    def test_metadata_serializes(self, buffered_query):
        """It was built with the stdlib encoder and carried numpy scalars, so it
        returned a serialization error for every query it was ever asked about."""
        manager, query_id, _ = buffered_query

        payload = json.loads(manager.get_query_metadata(query_id))

        assert payload["query_info"]["query_id"] == query_id
        assert "data_quality_report" in payload
