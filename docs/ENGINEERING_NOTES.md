# Engineering notes

This repository contains selected OHLCV analysis functions from a larger, private trading system. It does not include the decision engine.

## Confirmed swings are delayed

The swing detector compares a candidate high or low with candles both before and after it. That means the label is placed at the pivot timestamp for historical inspection, but it cannot be known at that time in live operation. Historical tests must respect the right-hand confirmation window.

## Structural breaks and market structure

The analyzer builds BOS/CHOCH events from detected swing points. The market-structure method compares successive swing highs and lows to label higher highs, higher lows, lower highs and lower lows. SMC terminology and detection definitions are not standardized across trading systems.

## Zones and future interactions

Order-block and FVG methods can annotate a zone with a later mitigation index. Liquidity pools can similarly record a later sweep. Those indices are retrospective annotations, **not** values available when the zone first formed. In particular, the selected liquidity implementation uses the final close to scale its cluster tolerance, so it should not be presented as causally correct for point-in-time backtesting without further changes.

## Testing limits

The small included test suite covers aligned outputs, short histories and one synthetic bullish FVG. It does not validate every structural definition, live behavior, historical execution assumptions, or profitability. Those require broader datasets, causal tests, and explicitly specified assumptions.

## Private-system research

The full private system has a separate optimization report. [Read the public case study](https://github.com/NFTPSYCHO5524/NFTPSYCHO5524/blob/main/projects/smc-market-structure-engine/README.md) for historical context. Those reported backtest numbers cannot be attributed to this standalone analyzer, nor treated as independent live-performance results.

## Publication boundary

Signal generation, entry selection, proprietary scoring and thresholds, position sizing, execution, credentials, and production orchestration are intentionally absent.
