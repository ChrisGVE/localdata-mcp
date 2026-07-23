"""tests/v3/test_visualize_style.py — the styling/palette layer (E12.6).

The palette resolver and StyleSpec builder in isolation (extends
FR-503): a preset name resolves to seaborn's pure hex color list, a
custom color list is validated, and an unknown palette or colormap is a
structured NX-3 refusal — never a silent fallback. resolve_style folds
the config-backed VisualizeDefaults with per-call overrides, the
override wins, and an unknown override key is refused. The rc_overrides
carry the palette as an axes prop_cycle so the renderer applies it
through rc_context (never seaborn's global set_theme).
"""

from __future__ import annotations

import pytest

from localdata_mcp.nexus.chokepoint.guard import (
    GuardedExecutionError,
    VisualizeDefaults,
)
from localdata_mcp.nexus.config.models import ConfigModel
from localdata_mcp.visualize.style import (
    QUALITATIVE_PRESETS,
    Palette,
    StyleSpec,
    resolve_palette,
    resolve_style,
)


def _defaults() -> VisualizeDefaults:
    return VisualizeDefaults.from_config(ConfigModel().visualize)


class TestResolvePalette:
    @pytest.mark.parametrize("preset", QUALITATIVE_PRESETS)
    def test_preset_resolves_to_hex_colors(self, preset: str) -> None:
        palette = resolve_palette(preset, "viridis")
        assert isinstance(palette, Palette)
        assert palette.colors
        assert all(c.startswith("#") for c in palette.colors)
        assert palette.sequential_cmap == "viridis"

    def test_custom_color_list_is_accepted_and_normalised(self) -> None:
        palette = resolve_palette(["#1b9e77", "tab:blue", "red"], "magma")
        # every entry normalised to a hex string
        assert all(c.startswith("#") for c in palette.colors)
        assert len(palette.colors) == 3

    def test_unknown_preset_is_refused_naming_the_set(self) -> None:
        with pytest.raises(GuardedExecutionError) as excinfo:
            resolve_palette("rainbow", "viridis")
        assert "colorblind" in str(excinfo.value)

    def test_empty_custom_palette_is_refused(self) -> None:
        with pytest.raises(GuardedExecutionError):
            resolve_palette([], "viridis")

    def test_invalid_custom_color_is_refused(self) -> None:
        with pytest.raises(GuardedExecutionError):
            resolve_palette(["#1b9e77", "notacolor"], "viridis")

    def test_unknown_sequential_cmap_is_refused(self) -> None:
        with pytest.raises(GuardedExecutionError):
            resolve_palette("deep", "no_such_cmap")


class TestResolveStyle:
    def test_defaults_flow_through_from_config(self) -> None:
        style = resolve_style(_defaults())
        assert style.figsize == (6.4, 4.8)
        assert style.dpi == 100
        assert style.despine is True
        assert style.palette.sequential_cmap == "viridis"
        # the colorblind default palette
        assert style.palette.colors[0] == "#0173b2"

    def test_per_call_palette_overrides_the_default(self) -> None:
        style = resolve_style(_defaults(), palette="bright")
        default = resolve_style(_defaults())
        assert style.palette.colors != default.palette.colors

    def test_style_override_keys_win_over_config(self) -> None:
        style = resolve_style(
            _defaults(),
            style={
                "figure_width_inches": 8.0,
                "dpi": 150,
                "grid": False,
                "sequential_cmap": "magma",
                "fit_color": "#000000",
            },
        )
        assert style.figsize[0] == 8.0
        assert style.dpi == 150
        assert style.grid is False
        assert style.palette.sequential_cmap == "magma"
        assert style.fit_color == "#000000"

    def test_unknown_style_key_is_refused(self) -> None:
        with pytest.raises(GuardedExecutionError) as excinfo:
            resolve_style(_defaults(), style={"bogus": 1})
        assert "bogus" in str(excinfo.value)

    def test_non_positive_figure_dimension_is_refused(self) -> None:
        with pytest.raises(GuardedExecutionError):
            resolve_style(_defaults(), style={"figure_width_inches": 0})


class TestRcOverrides:
    def test_prop_cycle_carries_the_palette_colors(self) -> None:
        style = resolve_style(_defaults(), palette=["#111111", "#222222"])
        rc = style.rc_overrides()
        cycle_colors = rc["axes.prop_cycle"].by_key()["color"]
        assert cycle_colors == ["#111111", "#222222"]
        assert rc["axes.grid"] is True
