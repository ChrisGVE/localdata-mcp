"""Deterministic test data generation."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List


class TestDataGenerator:
    """Generates reproducible test data with fixed seed."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_standard_rows(self, n: int) -> List[Dict[str, Any]]:
        """Standard table: id, name, email, amount, created_at, is_active, category, score, notes."""
        categories = [
            "electronics",
            "clothing",
            "food",
            "books",
            "toys",
            "sports",
            "home",
            "garden",
            "auto",
            "health",
        ]
        rows = []
        base_date = datetime(2020, 1, 1)
        for i in range(n):
            rows.append(
                {
                    "id": i + 1,
                    "name": f"User_{i + 1:06d}",
                    "email": f"user{i + 1}@example.com",
                    "amount": round(self.rng.uniform(1.0, 10000.0), 2),
                    "created_at": (base_date + timedelta(minutes=i)).isoformat(),
                    "is_active": self.rng.choice([True, False]),
                    "category": self.rng.choice(categories),
                    "score": round(self.rng.gauss(50, 15), 4),
                    "notes": f"Note for user {i + 1}"
                    if self.rng.random() > 0.3
                    else None,
                }
            )
        return rows

    def generate_timeseries_rows(
        self, n: int, sensors: int = 50
    ) -> List[Dict[str, Any]]:
        """Time series: timestamp, sensor_id, temperature, humidity, pressure, status."""
        rows = []
        base = datetime(2024, 1, 1)
        for i in range(n):
            rows.append(
                {
                    "timestamp": (base + timedelta(minutes=i)).isoformat(),
                    "sensor_id": (i % sensors) + 1,
                    "temperature": round(self.rng.gauss(22, 5), 2),
                    "humidity": round(self.rng.uniform(20, 90), 2),
                    "pressure": round(self.rng.gauss(1013, 10), 2),
                    "status": self.rng.choice(["ok", "warning", "error"]),
                }
            )
        return rows

    def generate_unicode_rows(self, n: int) -> List[Dict[str, Any]]:
        """Unicode stress test: CJK, emoji, Arabic, accented chars."""
        samples = [
            "Hello 世界",
            "日本語テスト",
            "한국어",
            "مرحبا",
            "Привет",
            "café résumé naïve",
            "🎉🚀💡🔥",
            "Ελληνικά",
            "ñoño",
            "𝕳𝖊𝖑𝖑𝖔",
        ]
        rows = []
        for i in range(n):
            rows.append(
                {
                    "id": i + 1,
                    "text": self.rng.choice(samples) + f" #{i + 1}",
                    "description": self.rng.choice(samples) * 3,
                }
            )
        return rows

    def generate_wide_rows(self, n: int, columns: int = 100) -> List[Dict[str, Any]]:
        """Wide table: 100 columns of mixed types."""
        rows = []
        for i in range(n):
            row: Dict[str, Any] = {"id": i + 1}
            for c in range(columns):
                if c % 3 == 0:
                    row[f"int_col_{c}"] = self.rng.randint(0, 1000000)
                elif c % 3 == 1:
                    row[f"float_col_{c}"] = round(self.rng.random() * 1000, 4)
                else:
                    row[f"text_col_{c}"] = f"val_{self.rng.randint(0, 100)}"
            rows.append(row)
        return rows

    def generate_blob_data(self, size_bytes: int = 1024) -> bytes:
        """Generate binary data with PNG header for MIME detection."""
        header = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
        return header + bytes(
            self.rng.getrandbits(8) for _ in range(size_bytes - len(header))
        )
