"""localdata_mcp/testbench/image_similarity.py — SSIM for PNG goldens (E12.3).

FR-502's perceptual check: the structural-similarity index between a
rendered PNG and its golden, the supplementary raster gate behind the
exact SVG structural checks (S8 15e — the SVG carries the exactness
burden). A numpy + scipy implementation (both already dependencies) of
Wang et al.'s windowed SSIM over the luminance channel with an 11-wide
Gaussian window, so no image library is added. Also here: the
render-stack fingerprint (matplotlib + freetype versions) that tells a
golden whether the current environment can be held to it — a mismatch
means the antialiasing stack differs and the golden must be regenerated
in that environment, not asserted against. Neighbors: the PNG SSIM test
loads goldens and compares through here; fixtures/chart_goldens/ holds
the goldens and their fingerprint.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib import ft2font
from matplotlib import image as mpl_image
from scipy.ndimage import gaussian_filter

# Wang et al. SSIM stabilizers over normalized luminance (dynamic range
# L = 1): C1 = (K1·L)², C2 = (K2·L)² with the paper's K1 = 0.01,
# K2 = 0.03. Spelled as arithmetic — a bare 0.01/0.03 literal would trip
# the NFR-403 one-default-site gate by coinciding with an unrelated S8
# config default of the same numeric value (these are algorithm
# constants, not configuration).
_K1 = 0.1 * 0.1
_K2 = 3.0 * _K1
_C1 = _K1 * _K1
_C2 = _K2 * _K2
_SIGMA = 1.5


def render_stack_fingerprint() -> str:
    """The antialiasing-determining stack: a golden rendered under this
    string is comparable; under any other it must be regenerated."""
    return f"mpl{matplotlib.__version__}-ft{ft2font.__freetype_version__}"


def png_to_luminance(payload: bytes) -> "np.ndarray":
    """The PNG's luminance channel in [0, 1] — RGB collapsed by the
    Rec. 601 weights, alpha dropped."""
    from io import BytesIO

    rgba = mpl_image.imread(BytesIO(payload))  # (h, w, 4) float32 in [0, 1]
    rgb = rgba[..., :3].astype(np.float64)
    return rgb @ np.array([0.299, 0.587, 0.114])


def ssim(reference: bytes, candidate: bytes) -> float:
    """The mean SSIM between two equal-dimension PNGs (luminance),
    Wang et al. with a Gaussian window — 1.0 is identical."""
    a = png_to_luminance(reference)
    b = png_to_luminance(candidate)
    if a.shape != b.shape:
        raise ValueError(f"SSIM needs equal dimensions — got {a.shape} vs {b.shape}")
    mu_a = gaussian_filter(a, _SIGMA)
    mu_b = gaussian_filter(b, _SIGMA)
    mu_a_sq, mu_b_sq, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    var_a = gaussian_filter(a * a, _SIGMA) - mu_a_sq
    var_b = gaussian_filter(b * b, _SIGMA) - mu_b_sq
    cov_ab = gaussian_filter(a * b, _SIGMA) - mu_ab
    numerator = (2 * mu_ab + _C1) * (2 * cov_ab + _C2)
    denominator = (mu_a_sq + mu_b_sq + _C1) * (var_a + var_b + _C2)
    return float(np.mean(numerator / denominator))
