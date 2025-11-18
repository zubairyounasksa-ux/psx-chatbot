from fuzzywuzzy import fuzz
import re


# ===================== Detection helpers =====================

def _detect_ticker(query, companies):
    """Find ticker from query words, or default if only one company."""
    words_upper = str(query).upper().split()
    for t in companies:
        if t.upper() in words_upper:
            return t

    # If only one company loaded, use it as default
    if len(companies) == 1:
        return next(iter(companies))

    return None


def _detect_period(query, periods):
    """Detect year/period appearing in query (preferred) or fallback via fuzzy."""
    if not periods:
        return None

    q = str(query)

    # 1) explicit 4-digit year that is in periods
    full_years = re.findall(r"\b((?:19|20)\d{2})\b", q)
    for y in full_years:
        if y in periods:
            return y

    # 2) phrase / fuzzy match
    best = None
    best_score = 0
    q_lower = q.lower()
    for p in periods:
        p_str = str(p)
        if p_str.lower() in q_lower:
            return p_str
        score = fuzz.partial_ratio(q_lower, p_str.lower())
        if score > best_score:
            best = p_str
            best_score = score

    return best if best_score >= 40 else None


def _extract_keywords(query, ticker):
    """
    Extract meaningful alphabetic tokens from the query,
    removing ticker and common stopwords.
    """
    q_lower = str(query).lower()
    words = re.findall(r"[a-zA-Z]+", q_lower)

    stopwords = {
        "of", "the", "in", "for", "and", "to", "from", "at",
        "on", "is", "are", "was", "were", "this", "that",
        "year", "years", "value", "what", "tell", "me"
    }

    ticker_lower = str(ticker).lower() if ticker else ""
    keywords = []
    for w in words:
        if len(w) < 3:
            continue
        if w in stopwords:
            continue
        if ticker_lower and w == ticker_lower:
            continue
        keywords.append(w)

    return keywords


def _find_precise_metric(query, metrics):
    """
    Try to find a precise metric whose full phrase appears in the query.
    Example: query 'profit after taxation of GAL in 2022'
             metric 'Profit after Taxation'  -> precise
    """
    q_lower = str(query).lower()
    phrase_matches = []

    for m in metrics:
        m_str = str(m).strip()
        if not m_str:
            continue
        m_low = m_str.lower()
        if m_low in q_lower and len(m_str.split()) > 1:
            phrase_matches.append(m_str)

    if phrase_matches:
        # Prefer most specific phrase: more words, then more characters
        return max(phrase_matches, key=lambda s: (len(s.split()), len(s)))

    return None


def _to_number(val):
    """Best-effort conversion to float, handling commas etc."""
    try:
        s = str(val).replace(",", "").replace(" ", "")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


# ===================== Data lookup helpers =====================

def _lookup_metric_with_context(comp, metric, period):
    """
    Look up a metric value and also previous + latest year values.

    Returns dict:
      {
        "period": used_period,
        "value": current_value,
        "prev_period": ... or None,
        "prev_value": ... or None,
        "latest_period": ... or None,
        "latest_value": ... or None,
      }
    or None if metric not found.
    """
    if not metric:
        return None

    for df in comp.get("files", []):
        try:
            first_col = df.iloc[:, 0].astype(str)
        except Exception:
            continue

        row = df[first_col == str(metric)]
        if row.empty:
            continue

        # valid period columns (ignore Trend)
        valid_cols = [
            c for c in df.columns[1:]
            if str(c).strip() and str(c).lower() != "trend"
        ]
        if not valid_cols:
            continue

        # Which period to use:
        # - if user gave a year and it's present, use that
        # - otherwise use the latest year (rightmost column)
        if period and period in valid_cols:
            used_period = period
        else:
            used_period = valid_cols[-1]  # default = latest year

        if used_period not in df.columns:
            continue

        cur_val = row[used_period].values[0]

        # previous year (immediately before used_period)
        idx = valid_cols.index(used_period)
        prev_col = valid_cols[idx - 1] if idx > 0 else None
        prev_val = row[prev_col].values[0] if prev_col else None

        # latest year in the sheet (e.g., 2025)
        latest_col = valid_cols[-1]
        latest_val = row[latest_col].values[0]

        return {
            "period": used_period,
            "value": cur_val,
            "prev_period": prev_col,
            "prev_value": prev_val,
            "latest_period": latest_col,
            "latest_value": latest_val,
            "valid_cols": valid_cols,
            "row": row,
        }

    return None


def _get_metric_series(comp, metric):
    """
    Return full time series for a metric as {period -> value}, using first matching df.
    """
    if not metric:
        return {}

    for df in comp.get("files", []):
        try:
            first_col = df.iloc[:, 0].astype(str)
        except Exception:
            continue

        row = df[first_col == str(metric)]
        if row.empty:
            continue

        valid_cols = [
            c for c in df.columns[1:]
            if str(c).strip() and str(c).lower() != "trend"
        ]
        series = {}
        for c in valid_cols:
            series[str(c)] = row[c].values[0]
        return series

    return {}


def _lookup_multiple_metrics_contexts(comp, metrics, period):
    """
    For a list of metrics, return list of (metric, context_dict) where context_dict
    is from _lookup_metric_with_context.
    """
    out = []
    for m in metrics:
        ctx = _lookup_metric_with_context(comp, m, period)
        if ctx is not None:
            out.append((m, ctx))
    return out


# ===================== Formatting helpers =====================

def _format_single_metric_answer(tick, metric, ctx, comp):
    """
    Human-friendly answer with analysis:
    - value in requested year,
    - comparison vs previous year,
    - comparison vs latest year (e.g., 2025),
    - multi-year trend view (CAGR and where this year sits in history).
    """
    period = ctx["period"]
    value = ctx["value"]
    prev_period = ctx["prev_period"]
    prev_value = ctx["prev_value"]
    latest_period = ctx["latest_period"]
    latest_value = ctx["latest_value"]

    val_num = _to_number(value)
    prev_num = _to_number(prev_value) if prev_period is not None else None
    latest_num = _to_number(latest_value) if latest_period is not None else None

    lines = [f"{tick} – {metric} in {period}: {value}"]

    # --- vs previous year ---
    if prev_period is not None and prev_value is not None:
        if val_num is not None and prev_num not in (None, 0):
            diff = val_num - prev_num
            pct = diff / abs(prev_num) * 100
            if diff > 0:
                direction = "increased"
            elif diff < 0:
                direction = "decreased"
            else:
                direction = "remained almost flat"

            lines.append(
                f"Compared to {prev_period}, it {direction} by "
                f"{abs(diff):,.2f} ({abs(pct):.1f}%). Previous value: {prev_value}."
            )
        else:
            lines.append(f"Previous period {prev_period}: {prev_value}.")

    # --- vs latest year (e.g., 2025) ---
    if latest_period is not None and latest_period != period and latest_value is not None:
        if latest_num is not None and val_num not in (None, 0):
            diff_l = latest_num - val_num
            pct_l = diff_l / abs(val_num) * 100
            if diff_l > 0:
                dir_l = "higher"
            elif diff_l < 0:
                dir_l = "lower"
            else:
                dir_l = "at the same level"

            lines.append(
                f"Relative to the latest reported year {latest_period}, "
                f"the value in {period} is {dir_l} by "
                f"{abs(diff_l):,.2f} ({abs(pct_l):.1f}%). Latest value: {latest_value}."
            )
        else:
            lines.append(
                f"Latest reported year {latest_period} has value {latest_value}."
            )

    # --- Expert-style multi-year view using the full series ---
    series_raw = _get_metric_series(comp, metric)
    series_nums = {
        p: _to_number(v)
        for p, v in series_raw.items()
        if _to_number(v) is not None
    }

    if series_nums and len(series_nums) >= 3 and val_num is not None:
        # Sort by period (as number if possible)
        def _key(p):
            try:
                return int(p)
            except Exception:
                return p

        sorted_items = sorted(series_nums.items(), key=lambda x: _key(x[0]))
        periods_sorted = [p for p, _ in sorted_items]
        values_sorted = [v for _, v in sorted_items]

        first_period, first_val = periods_sorted[0], values_sorted[0]
        last_period, last_val = periods_sorted[-1], values_sorted[-1]

        if first_val not in (None, 0):
            n_years = len(values_sorted) - 1
            if n_years > 0:
                cagr = (last_val / abs(first_val)) ** (1 / n_years) - 1
            else:
                cagr = 0
        else:
            cagr = None

        min_val = min(values_sorted)
        max_val = max(values_sorted)

        # Where does the current value sit vs history?
        position_comment = None
        if max_val == min_val:
            position_comment = "is broadly in line with its historical range."
        else:
            if val_num >= 0.9 * max_val:
                position_comment = "is near its historical peak."
            elif val_num <= 1.1 * min_val:
                position_comment = "is close to the low end of its historical range."
            else:
                position_comment = "sits roughly in the middle of its historical range."

        expert_lines = []
        expert_lines.append(
            f"Over the available history ({first_period}–{last_period}), "
            f"this metric has generally trended {'upwards' if last_val >= first_val else 'downwards'}."
        )
        if cagr is not None:
            expert_lines.append(
                f"The implied compound annual growth rate (CAGR) over this period "
                f"is approximately {cagr * 100:,.1f}%."
            )
        if position_comment:
            expert_lines.append(
                f"The {period} figure {position_comment}"
            )

        lines.append("")
        lines.append("Expert view:")
        lines.append(" ".join(expert_lines))

    return "\n".join(lines)


def _format_multi_metric_answer(tick, keywords, metric_contexts, period):
    """
    Bullet summary when many metrics match a keyword (e.g. 'assets', 'profit'),
    with directional commentary vs previous and latest year.
    """
    kw_text = ", ".join(sorted(set(keywords)))
    period_text = f" in {period}" if period else ""

    lines = [f"{tick} – metrics related to {kw_text}{period_text}:"]
    for metric, ctx in sorted(metric_contexts, key=lambda x: str(x[0])):
        period_m = ctx["period"]
        value = ctx["value"]
        prev_period = ctx["prev_period"]
        prev_value = ctx["prev_value"]
        latest_period = ctx["latest_period"]
        latest_value = ctx["latest_value"]

        val_num = _to_number(value)
        prev_num = _to_number(prev_value) if prev_period is not None else None
        latest_num = _to_number(latest_value) if latest_period is not None else None

        line = f"- {metric} in {period_m}: {value}"

        # vs previous
        if prev_period is not None and val_num is not None and prev_num not in (None, 0):
            diff = val_num - prev_num
            pct = diff / abs(prev_num) * 100
            if diff > 0:
                arrow = "↑"
            elif diff < 0:
                arrow = "↓"
            else:
                arrow = "→"
            line += f" ({arrow} vs {prev_period}: {abs(pct):.1f}%)."
        elif prev_period is not None and prev_value is not None:
            line += f" (previous {prev_period}: {prev_value})."

        # vs latest
        if latest_period is not None and latest_period != period_m and latest_value is not None and val_num not in (None, 0):
            if latest_num not in (None, 0):
                diff_l = latest_num - val_num
                pct_l = diff_l / abs(val_num) * 100
                if diff_l > 0:
                    arrow2 = "↑"
                elif diff_l < 0:
                    arrow2 = "↓"
                else:
                    arrow2 = "→"
                line += f" Latest {latest_period}: {latest_value} ({arrow2} {abs(pct_l):.1f}% vs {period_m})."

        lines.append(line)

    return "\n".join(lines)


# ===================== Main entrypoint =====================

def answer_query(query, companies):
    """
    Main Q&A function.

    Behaviour:
    - Detect ticker and period.
    - If query contains a precise metric phrase (e.g. 'Total Assets', 'Profit after Taxation'),
      return that single metric with:
        * value,
        * change vs previous year,
        * change vs latest year (e.g., 2025),
        * multi-year expert commentary.
    - Otherwise, extract keywords from the query (e.g. 'assets', 'profit', 'cash')
      and return ALL metrics whose names contain those keywords, each with trend vs
      previous year and latest year.
    - If still nothing useful, fall back to fuzzy best-match metric with trend.
    """
    if not query:
        return "Empty query."

    if not companies:
        return "No companies loaded."

    # --------- Ticker & period ---------
    tick = _detect_ticker(query, companies)
    if not tick or tick not in companies:
        return "No company ticker detected in query."

    comp = companies[tick]
    periods = comp.get("periods", [])
    metrics = comp.get("metrics", [])

    period = _detect_period(query, periods)

    # --------- First: precise metric phrase ---------
    precise_metric = _find_precise_metric(query, metrics)
    if precise_metric:
        ctx = _lookup_metric_with_context(comp, precise_metric, period)
        if ctx is not None:
            return _format_single_metric_answer(tick, precise_metric, ctx, comp)

    # --------- Second: keyword-based multi-metric ---------
    keywords = _extract_keywords(query, tick)
    keyword_metrics = set()

    for m in metrics:
        m_low = str(m).lower()
        for kw in keywords:
            kw_low = kw.lower()
            # small plural/singular handling: asset <-> assets
            if kw_low.endswith("s") and kw_low[:-1] in m_low:
                match = True
            elif (kw_low + "s") in m_low:
                match = True
            else:
                match = kw_low in m_low

            if match:
                keyword_metrics.add(m)
                break

    if keyword_metrics:
        metric_contexts = _lookup_multiple_metrics_contexts(comp, keyword_metrics, period)
        if metric_contexts:
            # If only one metric matched, behave like single metric
            if len(metric_contexts) == 1:
                m, ctx = metric_contexts[0]
                return _format_single_metric_answer(tick, m, ctx, comp)
            # Otherwise bullets
            return _format_multi_metric_answer(tick, keywords, metric_contexts, period)

    # --------- Final: fuzzy single-metric fallback ---------
    best = None
    best_score = 0
    q_lower = str(query).lower()
    for m in metrics:
        m_low = str(m).lower()
        score = fuzz.partial_ratio(q_lower, m_low)
        if score > best_score:
            best = m
            best_score = score

    if not best or best_score < 45:
        return "Metric not found."

    ctx = _lookup_metric_with_context(comp, best, period)
    if ctx is None:
        return "Value not found."

    return _format_single_metric_answer(tick, best, ctx, comp)
