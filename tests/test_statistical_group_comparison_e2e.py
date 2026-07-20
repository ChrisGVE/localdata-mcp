"""End-to-end tests for group comparison: hypothesis testing and ANOVA post-hoc.

These tests drive the MCP tools the way a client does — connect a CSV, call the
``DatabaseManager`` method, assert on the *content* of the JSON — and they check
statements about the world rather than the shape of the envelope:

- a caller who names a grouping column gets a comparison **of those groups**,
  and it finds the difference the fixture actually contains (issue: automatic
  selection only ever profiled the columns);
- Tukey post-hoc reports the named pairwise comparisons with the mean
  differences that are really in the data (issue: extraction raised and the
  failure was swallowed, leaving ``post_hoc_results`` empty).

Fixtures are generated with a fixed seed so every numeric expectation below is
reproducible rather than lucky, and they live under ``tests/fixtures/`` because
path security restricts connections to the working directory.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from localdata_mcp import DatabaseManager

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

SEED = 20260721

# Two normal groups, 2.5 units (1.25 pooled SDs) apart.
NORMAL_GROUP_MEANS = {"A": 10.0, "B": 12.5}
NORMAL_GROUP_SD = 2.0
NORMAL_GROUP_SIZE = 60

# Three normal groups for ANOVA and its post-hoc comparisons.
THREE_GROUP_MEANS = {"ctrl": 20.0, "low": 23.5, "high": 27.0}
THREE_GROUP_SD = 3.0
THREE_GROUP_SIZE = 45


def _fp(filename: str) -> str:
    return os.path.join(FIXTURES_DIR, filename)


@pytest.fixture(scope="module", autouse=True)
def group_comparison_fixtures() -> None:
    """Write the CSV fixtures these tests analyze."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Normal, two groups — the assumptions a t-test needs are satisfied.
    pd.DataFrame(
        {
            "grp": np.repeat(list(NORMAL_GROUP_MEANS), NORMAL_GROUP_SIZE),
            "value": np.round(
                np.concatenate(
                    [
                        rng.normal(mean, NORMAL_GROUP_SD, NORMAL_GROUP_SIZE)
                        for mean in NORMAL_GROUP_MEANS.values()
                    ]
                ),
                4,
            ),
        }
    ).to_csv(_fp("gc_two_normal_groups.csv"), index=False)

    # Strongly skewed, two groups — normality fails, so the non-parametric
    # counterpart is the honest answer. Scales 1.0 and 3.0 differ clearly.
    pd.DataFrame(
        {
            "grp": np.repeat(["A", "B"], NORMAL_GROUP_SIZE),
            "value": np.round(
                np.concatenate(
                    [
                        rng.exponential(1.0, NORMAL_GROUP_SIZE),
                        rng.exponential(3.0, NORMAL_GROUP_SIZE),
                    ]
                ),
                4,
            ),
        }
    ).to_csv(_fp("gc_two_skewed_groups.csv"), index=False)

    # Three normal groups — a multi-group comparison and Tukey post-hoc.
    pd.DataFrame(
        {
            "treatment": np.repeat(list(THREE_GROUP_MEANS), THREE_GROUP_SIZE),
            "response": np.round(
                np.concatenate(
                    [
                        rng.normal(mean, THREE_GROUP_SD, THREE_GROUP_SIZE)
                        for mean in THREE_GROUP_MEANS.values()
                    ]
                ),
                4,
            ),
        }
    ).to_csv(_fp("gc_three_groups.csv"), index=False)


@pytest.fixture
def db() -> DatabaseManager:
    return DatabaseManager()


def _connect(db: DatabaseManager, name: str, fixture: str) -> None:
    db.connect_database(name, "csv", _fp(fixture))


def _test_names(result: Dict[str, Any]) -> List[str]:
    return [entry["test_name"] for entry in result["test_results"]]


def _named_test(result: Dict[str, Any], fragment: str) -> Dict[str, Any]:
    """Return the single test result whose name contains *fragment*."""
    matches = [
        entry
        for entry in result["test_results"]
        if fragment.lower() in entry["test_name"].lower()
    ]
    assert matches, f"no test named like '{fragment}' — got {_test_names(result)}"
    return matches[0]


ALL_ROWS = "SELECT * FROM data_table"


class TestAutomaticGroupComparison:
    """'auto' must answer the question the caller asked."""

    def test_two_normal_groups_get_a_t_test_that_finds_the_gap(
        self, db: DatabaseManager
    ) -> None:
        """Means 10.0 and 12.5 with SD 2.0 — a t-test must see the difference."""
        _connect(db, "gc_norm", "gc_two_normal_groups.csv")
        result = json.loads(
            db.analyze_hypothesis_test(
                "gc_norm", ALL_ROWS, column="value", group_column="grp"
            )
        )

        comparison = _named_test(result, "t-test")
        assert comparison["p_value"] < 0.001, (
            f"a 1.25-sigma gap between 60-sample groups must be significant, "
            f"got p={comparison['p_value']}"
        )
        info = comparison["additional_info"]
        assert {info["group1"], info["group2"]} == {"A", "B"}
        assert info["group1_mean"] == pytest.approx(NORMAL_GROUP_MEANS["A"], abs=0.6)
        assert info["group2_mean"] == pytest.approx(NORMAL_GROUP_MEANS["B"], abs=0.6)
        assert abs(info["cohens_d"]) > 0.8, "1.25 pooled SDs is a large effect"

    def test_normality_tests_are_still_reported_alongside(
        self, db: DatabaseManager
    ) -> None:
        """The column profile is useful context and must not be dropped."""
        _connect(db, "gc_norm_ctx", "gc_two_normal_groups.csv")
        result = json.loads(
            db.analyze_hypothesis_test(
                "gc_norm_ctx", ALL_ROWS, column="value", group_column="grp"
            )
        )

        names = " ".join(_test_names(result))
        assert "Shapiro-Wilk" in names
        assert "Kolmogorov-Smirnov" in names

    def test_skewed_groups_get_the_non_parametric_test(
        self, db: DatabaseManager
    ) -> None:
        """Exponential samples break normality, so Mann-Whitney U is correct."""
        _connect(db, "gc_skew", "gc_two_skewed_groups.csv")
        result = json.loads(
            db.analyze_hypothesis_test(
                "gc_skew", ALL_ROWS, column="value", group_column="grp"
            )
        )

        names = _test_names(result)
        assert not any(
            "t-test" in name for name in names
        ), f"a t-test is not defensible on skewed data, got {names}"
        comparison = _named_test(result, "Mann-Whitney")
        assert (
            comparison["p_value"] < 0.01
        ), f"scale 1.0 vs 3.0 must be detected, got p={comparison['p_value']}"

    def test_three_groups_get_a_multi_group_comparison(
        self, db: DatabaseManager
    ) -> None:
        """Means 20 / 23.5 / 27 — the comparison must cover all three groups."""
        _connect(db, "gc_three", "gc_three_groups.csv")
        result = json.loads(
            db.analyze_hypothesis_test(
                "gc_three",
                ALL_ROWS,
                column="response",
                group_column="treatment",
            )
        )

        names = _test_names(result)
        multi = [
            entry
            for entry in result["test_results"]
            if "ANOVA" in entry["test_name"] or "Kruskal" in entry["test_name"]
        ]
        assert multi, f"expected a three-group comparison, got {names}"
        comparison = multi[0]
        assert comparison["p_value"] < 0.001, (
            f"three separated treatments must be significant, "
            f"got p={comparison['p_value']}"
        )
        assert set(comparison["additional_info"]["groups"]) == set(THREE_GROUP_MEANS)

    def test_without_a_group_column_only_the_profile_is_reported(
        self, db: DatabaseManager
    ) -> None:
        """No grouping column named means no group comparison — unchanged."""
        _connect(db, "gc_nogroup", "gc_two_normal_groups.csv")
        result = json.loads(
            db.analyze_hypothesis_test("gc_nogroup", ALL_ROWS, column="value")
        )

        names = _test_names(result)
        assert names == ["Shapiro-Wilk (value)", "Kolmogorov-Smirnov (value)"], names


class TestAnovaPostHocComparisons:
    """A significant F-test must be followed by usable pairwise comparisons."""

    def test_tukey_reports_every_named_pair_with_true_mean_differences(
        self, db: DatabaseManager
    ) -> None:
        _connect(db, "ph", "gc_three_groups.csv")
        result = json.loads(
            db.analyze_anova(
                "ph", ALL_ROWS, dependent_var="response", group_var="treatment"
            )
        )

        post_hoc = result["post_hoc_results"]
        assert post_hoc, (
            "a significant three-group ANOVA must produce post-hoc comparisons, "
            f"got {post_hoc!r}"
        )
        (summary,) = post_hoc.values()
        assert summary["method"] == "Tukey HSD"

        comparisons = summary["comparisons"]
        pairs = {frozenset((entry["group1"], entry["group2"])) for entry in comparisons}
        assert pairs == {
            frozenset(("ctrl", "low")),
            frozenset(("ctrl", "high")),
            frozenset(("high", "low")),
        }, f"all three pairs must be compared, got {pairs}"

        # Every reported mean difference must equal the difference actually
        # present in the fixture, computed independently here.
        observed = (
            pd.read_csv(_fp("gc_three_groups.csv"))
            .groupby("treatment")["response"]
            .mean()
        )
        for entry in comparisons:
            expected = observed[entry["group2"]] - observed[entry["group1"]]
            assert entry["mean_diff"] == pytest.approx(expected, rel=1e-9), (
                f"{entry['group1']} vs {entry['group2']}: reported "
                f"{entry['mean_diff']}, data says {expected}"
            )
            assert 0.0 <= entry["p_value"] <= 1.0
            assert entry["significant"] is True, (
                f"{entry['group1']} vs {entry['group2']} are "
                f"{abs(expected):.2f} apart and must be flagged significant"
            )
            assert entry["lower_ci"] < entry["mean_diff"] < entry["upper_ci"]
