"""Selected SMC analysis methods from my private trading bot.

This standalone module omits signal generation, entry rules, confluence scoring,
risk management, exchange connections, credentials and production orchestration.
"""
from functools import wraps
from typing import Tuple
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame

class SMCAnalyzer:

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _cached(key_prefix):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                df_ids = []
                for a in args:
                    if isinstance(a, pd.DataFrame):
                        # Use length + last timestamp as key — fast, unique, no id() reuse risk
                        ts_val = str(a['timestamp'].iloc[-1]) if 'timestamp' in a.columns else str(a.index[-1])
                        df_ids.append(f"{len(a)}_{ts_val}")
                    else:
                        df_ids.append(str(id(a)))
                # Include kwargs that affect output (join_consecutive, close_mitigation etc)
                kwarg_hash = hash(tuple(sorted(
                    (k, v) for k, v in kwargs.items()
                    if isinstance(v, (bool, int, float, str))
                )))
                cache_key = f"{key_prefix}_{'_'.join(map(str, df_ids))}_{kwarg_hash}"
                if cache_key in self._cache:
                    return self._cache[cache_key]
                result = func(self, *args, **kwargs)
                self._cache[cache_key] = result
                return result
            return wrapper
        return decorator

    def inputvalidator(input_="ohlc"):
        def dfcheck(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                args = list(args)
                i = 0 if isinstance(args[0], pd.DataFrame) else 1

                args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

                inputs = {
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": kwargs.get("column", "close").lower(),
                    "v": "volume",
                }

                if inputs["c"] != "close":
                    kwargs["column"] = inputs["c"]

                for l in input_:
                    if inputs[l] not in args[i].columns:
                        raise LookupError(
                            'Must have a dataframe column named "{0}"'.format(inputs[l])
                        )

                return func(*args, **kwargs)

            return wrap

        return dfcheck

    @_cached("swings")
    @inputvalidator(input_="ohlc")
    def swing_highs_lows(
        self,
        ohlc: DataFrame,
        swing_length: int = 10,
        clean_consecutive: bool = True,
    ) -> DataFrame:
        """
        Pure SMC Swing Highs and Lows - Custom Fixed & Enhanced Version
        - Detects swing highs: highest high in left/right window (swing_length bars each side)
        - Detects swing lows: lowest low in left/right window
        - Optional consecutive cleaning: removes weaker consecutive swings (keeps stronger level)
        - Edge handling: cleans first/last swings to avoid false extremes
        - Returns: DataFrame with 'HighLow' (1 = swing high, -1 = swing low, NaN otherwise), 'Level' (price)
        """
        ohlc = ohlc.copy()
        ohlc_len = len(ohlc)
        if ohlc_len < swing_length * 2 + 1:
            empty = np.nan
            return pd.concat([
                pd.Series([empty] * ohlc_len, name="HighLow"),
                pd.Series([empty] * ohlc_len, name="Level")
            ], axis=1)

        high = ohlc['high'].values
        low = ohlc['low'].values

        left = swing_length
        right = swing_length
        total_window = left + right + 1

        # Pre-allocate
        high_low = np.full(ohlc_len, np.nan)
        level = np.full(ohlc_len, np.nan)

        # Detect initial swings (center higher/lower than left+right)
        for i in range(left, ohlc_len - right):
            window_high = high[i - left:i + right + 1]
            window_low = low[i - left:i + right + 1]

            if high[i] == window_high.max():
                high_low[i] = 1
                level[i] = high[i]
            if low[i] == window_low.min():
                high_low[i] = -1
                level[i] = low[i]

        if clean_consecutive:
            # Clean consecutive highs: keep only stronger (higher high)
            while True:
                changed = False
                for i in range(1, ohlc_len):
                    if high_low[i] == 1 and high_low[i-1] == 1:
                        if level[i] > level[i-1]:
                            high_low[i-1] = np.nan
                            level[i-1] = np.nan
                        else:
                            high_low[i] = np.nan
                            level[i] = np.nan
                        changed = True
                    elif high_low[i] == -1 and high_low[i-1] == -1:
                        if level[i] < level[i-1]:
                            high_low[i-1] = np.nan
                            level[i-1] = np.nan
                        else:
                            high_low[i] = np.nan
                            level[i] = np.nan
                        changed = True
                if not changed:
                    break

            # Edge fix: first/last swing often false – flip if needed (package behavior)
            valid_indices = np.where(~np.isnan(high_low))[0]
            if len(valid_indices) > 0:
                first = valid_indices[0]
                last = valid_indices[-1]
                if high_low[first] == 1:
                    high_low[first] = -1
                    level[first] = low[first]
                elif high_low[first] == -1:
                    high_low[first] = 1
                    level[first] = high[first]
                # NEW — only flip if last swing is too close to data edge to be confirmed
                _edge_threshold = swing_length   # within swing_length bars of end = unconfirmed
                if (ohlc_len - 1 - last) < _edge_threshold:
                    # Last swing is near the right edge — may be unreliable, apply flip
                    if high_low[last] == -1:
                        high_low[last] = 1
                        level[last] = high[last]
                    elif high_low[last] == 1:
                        high_low[last] = -1
                        level[last] = low[last]
                # else: last swing is confirmed by enough bars on the right — leave as detected

        # Final cleanup: NaN where no swing
        high_low = np.where(np.isnan(high_low), np.nan, high_low)
        level = np.where(~np.isnan(high_low), level, np.nan)

        return pd.concat([
            pd.Series(high_low, name="HighLow"),
            pd.Series(level, name="Level")
        ], axis=1)

    @_cached("bos_choch")
    @inputvalidator(input_="ohlc")
    def bos_choch(
        self,
        ohlc: DataFrame,
        swing_highs_lows: DataFrame,
        close_break: bool = True,
        clean_overlaps: bool = True,
    ) -> DataFrame:
        """
        Pure SMC BOS & CHOCH - FIXED LABELING
        - Now correctly labels BOS/CHOCH at the ACTUAL breaking candle (not the swing)
        - BrokenIndex is now the primary event index
        - Fully SMC compliant and more accurate for downstream logic
        """
        ohlc = ohlc.copy()
        swings = swing_highs_lows.copy()
        ohlc_len = len(ohlc)

        if ohlc_len < 50:
            empty = np.nan
            return pd.concat([
                pd.Series([empty]*ohlc_len, name="BOS"),
                pd.Series([empty]*ohlc_len, name="CHOCH"),
                pd.Series([empty]*ohlc_len, name="Level"),
                pd.Series([empty]*ohlc_len, name="BrokenIndex")
            ], axis=1)

        close = ohlc['close'].values
        high = ohlc['high'].values
        low = ohlc['low'].values
        swing_hl = swings['HighLow'].values
        swing_level = swings['Level'].values

        bos = np.full(ohlc_len, np.nan)
        choch = np.full(ohlc_len, np.nan)
        level = np.full(ohlc_len, np.nan)
        broken_idx = np.full(ohlc_len, np.nan)

        valid_swings = np.where(~np.isnan(swing_hl))[0]
        if len(valid_swings) < 4:
            return pd.concat([pd.Series(bos, name="BOS"), pd.Series(choch, name="CHOCH"),
                            pd.Series(level, name="Level"), pd.Series(broken_idx, name="BrokenIndex")], axis=1)

        swing_list = []
        for i in valid_swings:
            swing_list.append({
                'idx': i,
                'type': int(swing_hl[i]),
                'price': swing_level[i]
            })

        # Detect patterns and label at the actual breaking candle
        for pos in range(3, len(swing_list)):
            s4 = swing_list[pos]
            s3 = swing_list[pos-1]
            s2 = swing_list[pos-2]
            s1 = swing_list[pos-3]

            # Bullish BOS block
            if (s1['type'] == -1 and s3['type'] == -1 and s1['price'] > s3['price'] and
                s2['type'] == 1 and s4['type'] == 1 and s2['price'] < s4['price']):
                break_level = s2['price']   # ← prior swing HIGH being broken
                for k in range(s3['idx'] + 1, ohlc_len):   # ← search from AFTER s3
                    if (close[k] > break_level if close_break else high[k] > break_level):
                        bos[k] = 1
                        level[k] = break_level
                        broken_idx[k] = k
                        break

            # Bearish BOS block
            elif (s1['type'] == 1 and s3['type'] == 1 and s1['price'] < s3['price'] and
                s2['type'] == -1 and s4['type'] == -1 and s2['price'] > s4['price']):
                break_level = s2['price']   # ← prior swing LOW being broken
                for k in range(s3['idx'] + 1, ohlc_len):   # ← search from AFTER s3
                    if (close[k] < break_level if close_break else low[k] < break_level):
                        bos[k] = -1
                        level[k] = break_level
                        broken_idx[k] = k
                        break

            # Bullish CHOCH
            if (s1['type'] == 1 and s3['type'] == 1 and s1['price'] > s3['price'] and
                s2['type'] == -1 and s4['type'] == 1 and s4['price'] > s2['price']):
                break_level = s3['price']
                for k in range(s2['idx'] + 1, ohlc_len):
                    if (close[k] > break_level if close_break else high[k] > break_level):
                        choch[k] = 1
                        level[k] = break_level
                        broken_idx[k] = k
                        break

            # Bearish CHOCH
            elif (s1['type'] == -1 and s3['type'] == -1 and s1['price'] < s3['price'] and
                  s2['type'] == 1 and s4['type'] == -1 and s4['price'] < s2['price']):
                break_level = s3['price']
                for k in range(s2['idx'] + 1, ohlc_len):
                    if (close[k] < break_level if close_break else low[k] < break_level):
                        choch[k] = -1
                        level[k] = break_level
                        broken_idx[k] = k
                        break

        if clean_overlaps:
            active = np.where(~np.isnan(bos) | ~np.isnan(choch))[0]
            for i in range(len(active)):
                for j in range(i):
                    if broken_idx[active[j]] >= broken_idx[active[i]] > active[j]:
                        bos[active[j]] = np.nan
                        choch[active[j]] = np.nan
                        level[active[j]] = np.nan
                        broken_idx[active[j]] = np.nan

        return pd.concat([
            pd.Series(bos, name="BOS"),
            pd.Series(choch, name="CHOCH"),
            pd.Series(level, name="Level"),
            pd.Series(broken_idx, name="BrokenIndex")
        ], axis=1)

    @inputvalidator(input_="ohlc")
    def market_structure(self, df: pd.DataFrame, swing_length: int = 10, close_break: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Market Structure - Updated for correct BOS/CHOCH labeling
        Now uses the actual breaking candle (much more accurate timing)
        """
        if len(df) < 50:
            empty = pd.Series(False, index=df.index)
            return empty, empty, empty, empty, empty, empty

        ohlc = df[['open', 'high', 'low', 'close']].copy()
        ohlc.columns = [col.lower() for col in ohlc.columns]

        swings = self.swing_highs_lows(df, swing_length=swing_length)
        structure = self.bos_choch(df, swings, close_break=close_break)

        # Now correctly uses the breaking candle index
        bos = structure['BOS'].notna().fillna(False)
        choch = structure['CHOCH'].notna().fillna(False)

        # Swing-based HH/HL/LH/LL (unchanged - still accurate)
        swing_highs_clean = swings['Level'].where(swings['HighLow'] == 1).dropna()
        swing_lows_clean  = swings['Level'].where(swings['HighLow'] == -1).dropna()

        hh_clean = swing_highs_clean > swing_highs_clean.shift(1)
        lh_clean = swing_highs_clean < swing_highs_clean.shift(1)
        hl_clean = swing_lows_clean  > swing_lows_clean.shift(1)
        ll_clean = swing_lows_clean  < swing_lows_clean.shift(1)

        # Map back to full index
        hh = hh_clean.reindex(df.index, fill_value=False)
        lh = lh_clean.reindex(df.index, fill_value=False)
        hl = hl_clean.reindex(df.index, fill_value=False)
        ll = ll_clean.reindex(df.index, fill_value=False)

        logging.debug(f"Market Structure -> BOS={bos.sum()}, CHOCH={choch.sum()}, HH={hh.sum()}, HL={hl.sum()}, LH={lh.sum()}, LL={ll.sum()}")

        return bos, choch, hh, hl, lh, ll

    @_cached("ob")
    @inputvalidator(input_="ohlc")
    def ob(
        self,
        ohlc: DataFrame,
        swing_highs_lows: DataFrame,
        close_mitigation: bool = False,
    ) -> DataFrame:
        """
        Pure SMC Order Blocks - Fixed & Compliant
        - Detects bullish OB: last bearish/down-close candle before bullish displacement/BOS (demand zone)
        - Bearish OB: last bullish/up-close candle before bearish displacement/BOS (supply zone)
        - Mitigation: future-only (wick or close into zone, no self/negative indices)
        - Volume/Percentage: preserved for strength (forex = 0 handled)
        - No package bugs: clean future scan, accurate first mitigation index
        Returns same columns as original for drop-in compatibility
        """
        ohlc = ohlc.copy()
        swings = swing_highs_lows.copy()

        ohlc_len = len(ohlc)
        if ohlc_len < 50:
            empty = np.nan
            return pd.concat([
                pd.Series([empty]*ohlc_len, name="OB"),
                pd.Series([empty]*ohlc_len, name="Top"),
                pd.Series([empty]*ohlc_len, name="Bottom"),
                pd.Series([empty]*ohlc_len, name="OBVolume"),
                pd.Series([empty]*ohlc_len, name="MitigatedIndex"),
                pd.Series([empty]*ohlc_len, name="Percentage"),
            ], axis=1)

        high = ohlc['high'].values
        low = ohlc['low'].values
        open_ = ohlc['open'].values
        close = ohlc['close'].values
        volume = ohlc['volume'].values if 'volume' in ohlc.columns else np.zeros(ohlc_len)

        swing_hl = swings['HighLow'].values

        ob = np.full(ohlc_len, np.nan)
        top = np.full(ohlc_len, np.nan)
        bottom = np.full(ohlc_len, np.nan)
        ob_vol = np.full(ohlc_len, np.nan)
        mitigated_idx = np.full(ohlc_len, np.nan)
        percentage = np.full(ohlc_len, np.nan)

        swing_high_idx = np.where(swing_hl == 1)[0]
        swing_low_idx = np.where(swing_hl == -1)[0]

        # Bullish OBs
        for i in range(1, ohlc_len):
            pos = np.searchsorted(swing_high_idx, i)
            prior_high_idx = swing_high_idx[pos - 1] if pos > 0 else None
            if prior_high_idx is not None and close[i] > high[prior_high_idx]:
                leg_start = prior_high_idx + 1
                leg_end = i
                if leg_start >= leg_end:
                    continue

                # SMC: bullish OB = last DOWN-CLOSE (bearish) candle in the leg
                # Scan backwards from just before the break candle
                ob_idx = None
                for k in range(leg_end - 1, leg_start - 1, -1):
                    if close[k] < open_[k]:   # must be a down-close candle
                        ob_idx = k
                        break

                if ob_idx is None:
                    continue   # no bearish candle in leg — no valid OB

                ob[ob_idx] = 1
                top[ob_idx] = high[ob_idx]
                bottom[ob_idx] = low[ob_idx]

                vol_sum = volume[i] + (volume[i-1] if i >= 1 else 0) + (volume[i-2] if i >= 2 else 0)
                ob_vol[ob_idx] = vol_sum
                high_vol = volume[i] + (volume[i-1] if i >= 1 else 0)
                low_vol = (volume[i-2] if i >= 2 else 0)
                max_v = max(high_vol, low_vol) if max(high_vol, low_vol) > 0 else 1
                percentage[ob_idx] = min(high_vol, low_vol) / max_v * 100

        # Bearish OBs
        for i in range(1, ohlc_len):
            pos = np.searchsorted(swing_low_idx, i)
            prior_low_idx = swing_low_idx[pos - 1] if pos > 0 else None
            if prior_low_idx is not None and close[i] < low[prior_low_idx]:
                leg_start = prior_low_idx + 1
                leg_end = i
                if leg_start >= leg_end:
                    continue

                # SMC: bearish OB = last UP-CLOSE (bullish) candle in the leg
                # Scan backwards from just before the break candle
                ob_idx = None
                for k in range(leg_end - 1, leg_start - 1, -1):
                    if close[k] > open_[k]:   # must be an up-close candle
                        ob_idx = k
                        break

                if ob_idx is None:
                    continue   # no bullish candle in leg — no valid OB

                ob[ob_idx] = -1
                top[ob_idx] = high[ob_idx]
                bottom[ob_idx] = low[ob_idx]

                vol_sum = volume[i] + (volume[i-1] if i >= 1 else 0) + (volume[i-2] if i >= 2 else 0)
                ob_vol[ob_idx] = vol_sum
                high_vol = (volume[i-2] if i >= 2 else 0)
                low_vol = volume[i] + (volume[i-1] if i >= 1 else 0)
                max_v = max(high_vol, low_vol) if max(high_vol, low_vol) > 0 else 1
                percentage[ob_idx] = min(high_vol, low_vol) / max_v * 100

        # Future-only mitigation
        for idx in range(ohlc_len):
            if np.isnan(ob[idx]):
                continue
            zone_top = top[idx]
            zone_bottom = bottom[idx]
            dir_sign = int(ob[idx])

            future_slice = slice(idx + 1, None)
            future_high = high[future_slice]
            future_low = low[future_slice]
            future_open = open_[future_slice]
            future_close = close[future_slice]

            if dir_sign == 1:
                mitigated = (np.minimum(future_open, future_close) <= zone_bottom) if close_mitigation else (future_low <= zone_bottom)
            else:
                mitigated = (np.maximum(future_open, future_close) >= zone_top) if close_mitigation else (future_high >= zone_top)

            if np.any(mitigated):
                first_hit = np.argmax(mitigated)
                mitigated_idx[idx] = idx + 1 + first_hit

        return pd.concat([
            pd.Series(ob, name="OB"),
            pd.Series(top, name="Top"),
            pd.Series(bottom, name="Bottom"),
            pd.Series(ob_vol, name="OBVolume"),
            pd.Series(mitigated_idx, name="MitigatedIndex"),
            pd.Series(percentage, name="Percentage"),
        ], axis=1)

    @_cached("liquidity")
    @inputvalidator(input_="ohlc")
    def liquidity(
        self,
        ohlc: DataFrame,
        swing_highs_lows: DataFrame,
        range_percent: float = 0.002,
        min_cluster_size: int = 2,
        clean_overlaps: bool = True,
    ) -> DataFrame:
        """
        Pure SMC Liquidity Pools - Updated for correct BOS/CHOCH labeling
        Now respects actual break candles for swept detection.
        """
        ohlc = ohlc.copy()
        swings = swing_highs_lows.copy()
        ohlc_len = len(ohlc)

        if ohlc_len < 50:
            empty = np.nan
            return pd.concat([
                pd.Series([empty]*ohlc_len, name="Liquidity"),
                pd.Series([empty]*ohlc_len, name="Level"),
                pd.Series([empty]*ohlc_len, name="End"),
                pd.Series([empty]*ohlc_len, name="Swept")
            ], axis=1)

        high = ohlc['high'].values
        low = ohlc['low'].values
        close = ohlc['close'].values
        swing_hl = swings['HighLow'].values
        swing_level = swings['Level'].values

        liquidity = np.full(ohlc_len, np.nan)
        level_arr = np.full(ohlc_len, np.nan)
        end_arr = np.full(ohlc_len, np.nan)
        swept_arr = np.full(ohlc_len, np.nan)

        current_price = close[-1] if ohlc_len > 0 else 0
        dynamic_range = current_price * range_percent

        valid_highs = [(i, swing_level[i]) for i in range(ohlc_len) if swing_hl[i] == 1 and not np.isnan(swing_level[i])]
        valid_lows = [(i, swing_level[i]) for i in range(ohlc_len) if swing_hl[i] == -1 and not np.isnan(swing_level[i])]

        def cluster_swings(swings_list, dir_sign):
            clusters = []
            used = set()
            for start_idx, start_price in swings_list:
                if start_idx in used: continue
                cluster_indices = [start_idx]
                cluster_prices = [start_price]
                cluster_end = start_idx
                for next_idx, next_price in swings_list:
                    if next_idx <= start_idx or next_idx in used: continue
                    if abs(next_price - start_price) <= dynamic_range:
                        cluster_indices.append(next_idx)
                        cluster_prices.append(next_price)
                        cluster_end = next_idx
                        used.add(next_idx)
                if len(cluster_prices) >= min_cluster_size:
                    avg_level = sum(cluster_prices) / len(cluster_prices)
                    clusters.append((cluster_indices[0], avg_level, cluster_end))
                    used.add(start_idx)
            return clusters

        # High pools (sell-side liquidity)
        high_clusters = cluster_swings(valid_highs, 1)
        for start_idx, avg_level, end_idx in high_clusters:
            liquidity[start_idx] = 1
            level_arr[start_idx] = avg_level
            end_arr[start_idx] = end_idx

        # Low pools (buy-side liquidity)
        low_clusters = cluster_swings(valid_lows, -1)
        for start_idx, avg_level, end_idx in low_clusters:
            liquidity[start_idx] = -1
            level_arr[start_idx] = avg_level
            end_arr[start_idx] = end_idx

        # Swept detection using actual break candles
        for i in np.where(~np.isnan(liquidity))[0]:
            pool_level = level_arr[i]
            dir_sign = liquidity[i]
            future_slice = slice(i + 1, None)
            future_high = high[future_slice]
            future_low = low[future_slice]
            if dir_sign == 1:   # High pool swept when price goes above
                swept = future_high >= pool_level
            else:               # Low pool swept when price goes below
                swept = future_low <= pool_level
            if np.any(swept):
                first_sweep = np.argmax(swept)
                swept_arr[i] = i + 1 + first_sweep

        logging.debug(f"Liquidity pools -> High: { (liquidity==1).sum() }, Low: { (liquidity==-1).sum() }")

        return pd.concat([
            pd.Series(liquidity, name="Liquidity"),
            pd.Series(level_arr, name="Level"),
            pd.Series(end_arr, name="End"),
            pd.Series(swept_arr, name="Swept")
        ], axis=1)

    @_cached("fvg")
    @inputvalidator(input_="ohlc")
    def fvg(
        self,
        ohlc: DataFrame,
        join_consecutive: bool = True,
        close_mitigation: bool = True,
    ) -> DataFrame:
        """
        Pure SMC Fair Value Gap (FVG) - Custom Enhanced Version
        - Bullish FVG: strong bullish displacement leaves inefficiency (prior high < next low)
        - Bearish FVG: strong bearish displacement leaves inefficiency (prior low > next high)
        - Strict 3-candle pattern with body/displacement confirmation for purity
        - Optional join_consecutive: merge adjacent gaps into larger inefficiency
        - Future-only mitigation: first candle that fills (wick or close into gap)
        - Returns: DataFrame with 'FVG' (1 bull, -1 bear, NaN none), 'Top', 'Bottom', 'MitigatedIndex'
        Drop-in compatible with package smc.fvg for easy switch
        """
        ohlc = ohlc.copy()
        ohlc_len = len(ohlc)
        if ohlc_len < 50:
            empty = np.nan
            return pd.concat([
                pd.Series([empty]*ohlc_len, name="FVG"),
                pd.Series([empty]*ohlc_len, name="Top"),
                pd.Series([empty]*ohlc_len, name="Bottom"),
                pd.Series([empty]*ohlc_len, name="MitigatedIndex")
            ], axis=1)

        high = ohlc['high'].values
        low = ohlc['low'].values
        open_ = ohlc['open'].values   # Underscore to avoid shadowing built-in open
        close = ohlc['close'].values

        # Pre-allocate
        fvg = np.full(ohlc_len, np.nan)
        top = np.full(ohlc_len, np.nan)
        bottom = np.full(ohlc_len, np.nan)
        mitigated_idx = np.full(ohlc_len, np.nan)

        # Detect raw FVGs (3-candle imbalance + displacement confirmation)
        for i in range(1, ohlc_len - 1):
            prev = i - 1
            curr = i
            next_ = i + 1

            # Bullish: prior high < next low (gap up) + bullish displacement on middle candle
            if high[prev] < low[next_] and close[curr] > open_[curr]:
                body_size = close[curr] - open_[curr]
                range_size = high[curr] - low[curr]
                strong_body = body_size / range_size > 0.6 if range_size > 0 else True
                if strong_body:
                    fvg[curr] = 1
                    top[curr] = low[next_]      # Upper bound (next low)
                    bottom[curr] = high[prev]   # Lower bound (prior high)

            # Bearish: prior low > next high (gap down) + bearish displacement
            elif low[prev] > high[next_] and close[curr] < open_[curr]:
                body_size = open_[curr] - close[curr]
                range_size = high[curr] - low[curr]
                strong_body = body_size / range_size > 0.6 if range_size > 0 else True
                if strong_body:
                    fvg[curr] = -1
                    top[curr] = low[prev]       # Upper bound (prior low)
                    bottom[curr] = high[next_]  # Lower bound (next high)

        if join_consecutive:
            # Merge adjacent same-direction gaps (larger inefficiency)
            i = 1
            while i < ohlc_len - 1:
                if np.isnan(fvg[i]) or np.isnan(fvg[i+1]):
                    i += 1
                    continue
                if fvg[i] == fvg[i+1]:
                    top[i+1] = max(top[i], top[i+1])
                    bottom[i+1] = min(bottom[i], bottom[i+1])
                    fvg[i] = np.nan
                    top[i] = np.nan
                    bottom[i] = np.nan
                i += 1

        # Future-only mitigation scan
        for i in np.where(~np.isnan(fvg))[0]:
            gap_top = top[i]
            gap_bottom = bottom[i]
            dir_sign = int(fvg[i])

            future_slice = slice(i + 1, None)
            future_high = high[future_slice]
            future_low = low[future_slice]
            future_open = open_[future_slice]
            future_close = close[future_slice]

            if dir_sign == 1:  # Bullish FVG filled from above
                if close_mitigation:
                    filled = np.minimum(future_open, future_close) <= gap_top
                else:
                    filled = future_low <= gap_top
            else:  # Bearish FVG filled from below
                if close_mitigation:
                    filled = np.maximum(future_open, future_close) >= gap_bottom
                else:
                    filled = future_high >= gap_bottom

            if np.any(filled):
                first_fill = np.argmax(filled)
                mitigated_idx[i] = i + 1 + first_fill

        return pd.concat([
            pd.Series(fvg, name="FVG"),
            pd.Series(top, name="Top"),
            pd.Series(bottom, name="Bottom"),
            pd.Series(mitigated_idx, name="MitigatedIndex")
        ], axis=1)
