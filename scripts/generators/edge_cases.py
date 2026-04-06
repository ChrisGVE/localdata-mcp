"""Generate edge-case test fixtures."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from generators._common import sub_dir


def _write_empty_csv(d: str) -> str:
    """Write a CSV with headers only (no data rows)."""
    path = os.path.join(d, "empty.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("id,name,value\n")
    return path


def _write_bom_csv(d: str) -> str:
    """Write a CSV with a UTF-8 byte-order mark."""
    path = os.path.join(d, "bom.csv")
    with open(path, "wb") as fh:
        fh.write(b"\xef\xbb\xbf")  # UTF-8 BOM
        fh.write(b"id,name,value\r\n")
        fh.write(b"1,alpha,10.5\r\n")
        fh.write(b"2,beta,20.3\r\n")
        fh.write(b"3,gamma,30.1\r\n")
    return path


def _write_unicode_csv(d: str) -> str:
    """Write a CSV containing various Unicode characters."""
    rows = [
        "id,name,city,note",
        '1,"Ren\u00e9 Descartes","Touraine","cogito ergo sum"',
        '2,"Sophia M\u00fcller","M\u00fcnchen","\u00dcberraschung"',
        '3,"\u5c71\u672c\u9686","Tokyo","\u3053\u3093\u306b\u3061\u306f"',
        '4,"Emir \u00d6zdemir","\u0130stanbul","merhaba d\u00fcnya"',
        '5,"\u041e\u043b\u044c\u0433\u0430 \u0418\u0432\u0430\u043d\u043e\u0432\u0430","\u041c\u043e\u0441\u043a\u0432\u0430","\u043f\u0440\u0438\u0432\u0435\u0442"',
        '6,"\u5f20\u4f1f","\u5317\u4eac","\u4f60\u597d\u4e16\u754c"',
        '7,"\u00c1ine N\u00ed Bhriain","Baile \u00c1tha Cliath","dia duit"',
        '8,"Bj\u00f6rn Johansson","G\u00f6teborg","hej v\u00e4rlden"',
        '9,"Carlos Mu\u00f1oz","Espa\u00f1a","hola mundo"',
        '10,"Hans Gr\u00fcber","\u00d6sterreich","servus"',
    ]
    path = os.path.join(d, "unicode.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    return path


def _write_wide_csv(d: str, n_cols: int = 100) -> str:
    """Write a CSV with many columns."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {f"col_{i:03d}": rng.normal(0, 1, 100).round(4) for i in range(n_cols)}
    )
    path = os.path.join(d, "wide.csv")
    df.to_csv(path, index=False)
    return path


def generate_edge_cases(output_dir: str) -> list[str]:
    """Generate edge-case CSV files in edge_cases/ subdirectory."""
    d = sub_dir(output_dir, "edge_cases")
    return [
        _write_empty_csv(d),
        _write_bom_csv(d),
        _write_unicode_csv(d),
        _write_wide_csv(d),
    ]
