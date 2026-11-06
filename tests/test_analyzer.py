import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smc_analyzer import SMCAnalyzer


def sample_candles(n=160):
    rng = np.random.default_rng(42)
    base = 100 + np.linspace(0, 4, n) + 4 * np.sin(np.arange(n) * 0.22)
    opens = base + rng.normal(0, 0.14, n)
    closes = base + rng.normal(0, 0.14, n)
    highs = np.maximum(opens, closes) + 0.25
    lows = np.minimum(opens, closes) - 0.25
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.full(n, 100.0),
    })


def test_outputs_are_aligned_with_input():
    data = sample_candles()
    analyzer = SMCAnalyzer()
    swings = analyzer.swing_highs_lows(data, swing_length=5)
    assert len(swings) == len(data)
    assert set(swings.columns) == {"HighLow", "Level"}
    structure = analyzer.bos_choch(data, swings)
    assert len(structure) == len(data)
    assert set(structure.columns) == {"BOS", "CHOCH", "Level", "BrokenIndex"}
    labels = analyzer.market_structure(data, swing_length=5)
    assert len(labels) == 6
    assert all(len(x) == len(data) for x in labels)
    expected = [
        (analyzer.ob(data, swings), {"OB", "Top", "Bottom", "OBVolume", "MitigatedIndex", "Percentage"}),
        (analyzer.fvg(data), {"FVG", "Top", "Bottom", "MitigatedIndex"}),
        (analyzer.liquidity(data, swings), {"Liquidity", "Level", "End", "Swept"}),
    ]
    for output, columns in expected:
        assert len(output) == len(data)
        assert set(output.columns) == columns


def test_short_sample_returns_empty_indicators():
    data = sample_candles(24)
    analyzer = SMCAnalyzer()
    swings = analyzer.swing_highs_lows(data, swing_length=5)
    assert analyzer.fvg(data)["FVG"].isna().all()
    assert analyzer.ob(data, swings)["OB"].isna().all()
    assert analyzer.liquidity(data, swings)["Liquidity"].isna().all()


def test_fvg_identifies_a_displacement_gap():
    data = sample_candles(100)
    data.loc[48, ["open", "high", "low", "close"]] = [100.2, 100.8, 99.9, 100.4]
    data.loc[49, ["open", "high", "low", "close"]] = [100.9, 105.0, 100.85, 104.7]
    data.loc[50, ["open", "high", "low", "close"]] = [105.8, 106.4, 105.5, 106.0]
    result = SMCAnalyzer().fvg(data, join_consecutive=False)
    assert result.loc[49, "FVG"] == 1
    assert result.loc[49, "Top"] == 105.5
    assert result.loc[49, "Bottom"] == 100.8
