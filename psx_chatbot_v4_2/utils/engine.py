from fuzzywuzzy import fuzz
import re


# ===================== Low-level helpers =====================

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

def _lookup_metric_with_neighbors(comp, metric, period):
    """
    Look up a metric value and also previous/next period values.

    Returns dict:
      {
        "period": used_period,
        "value": current_value,
        "prev_period": ... or None,
        "prev_value": ... or None,
        "next_period": ... or None,
        "next_value": ... or None,
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

        # Decide which period column to use
        if period and period in valid_cols:
            used_period = period
        else:
            # fallback: first available column
            used_period = valid_cols[0]

        if used_period not in df.columns:
            continue

        cur_val = row[used_period].values[0]

        # find neighbours based on column order
        idx = valid_cols.index(used_period)
        prev_col = valid_cols[idx - 1] if idx > 0 else None
        next_col = valid_cols[idx + 1] if idx + 1 < len(valid_cols) else None

        prev_val = row[prev_col].values[0] if prev_col else None
        next_val = row[next_col].values[0] if next_col else None

        return {
            "period": used_period,
            "value": cur_val,
            "prev_period": prev_col,
            "prev_value": prev_val,
            "next_period": next_col,
            "next_value": next_val,
        }

    return None


def _lookup_multiple_metrics_contexts(comp, metrics, period):
    """
    For a list of metrics, return list of (metric, context_dict) where context_dict
    is from _lookup_metric_with_neighbors.
    """
    out = []
    for m in metrics:
        ctx = _lookup_metric_with_neighbors(comp, m, period)
        if ctx is not None:
            out.append((m, ctx))
    return out


# ===================== Formatting helpers =====================

def _format_single_metric_answer(tick, metric, ctx):
    """Human-friendly answer with trend commentary for one metric."""
    period = ctx["period"]
    value = ctx["value"]
    prev_period = ctx["prev_period"]
    prev_value = ctx["prev_value"]
    next_period = ctx["next_period"]
    next_value = ctx["next_value"]

    val_num = _to_number(value)
    prev_num = _to_number(prev_value) if prev_period is not None else None
    next_num = _to_number(next_value) if next_period is not None else None

    lines = [f"{tick} – {metric} in {period}: {value}"]

    # Compare with previous
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

    # Compare with next
    if next_period is not None and next_value is not None and val_num is not None and next_num is not None:
        diff_n = next_num - val_num
        pct_n = diff_n / abs(val_num) * 100 if val_num != 0 else None
        if diff_n > 0:
            direction_n = "further increased"
        elif diff_n < 0:
            direction_n = "declined"
        else:
            direction_n = "stayed at the same level"
        if pct_n is not None:
            lines.append(
                f"In {next_period}, it {direction_n} to {next_value} "
                f"(change of {abs(diff_n):,.2f}, {abs(pct_n):.1f}% vs {period})."
            )
        else:
            lines.append(
                f"In {next_period}, reported value is {next_value}."
            )

    return "\n".join(lines)


def _format_multi_metric_answer(tick, keywords, metric_contexts, period):
    """
    Short bullet summary when many metrics match a keyword (e.g. 'assets').
    """
    kw_text = ", ".join(sorted(set(keywords)))
    period_text = f" in {period}" if period else ""

    lines = [f"{tick} – metrics related to {kw_text}{period_text}:"]
    for metric, ctx in sorted(metric_contexts, key=lambda x: str(x[0])):
        period_m = ctx["period"]
        value = ctx["value"]
        prev_period = ctx["prev_period"]
        prev_value = ctx["prev_value"]

        val_num = _to_number(value)
        prev_num = _to_number(prev_value) if prev_period is not None else None

        line = f"- {metric} in {period_m}: {value}"
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

        lines.append(line)

    return "\n".join(lines)


# ===================== Main entrypoint =====================

def answer_query(query, companies):
    """
    Main Q&A function.

    Behaviour:
    - Detect ticker and period.
    - If query contains a precise metric phrase (e.g. 'Total Assets', 'Profit after Taxation'),
      return that single metric with comparison to previous / next year.
    - Otherwise, extract keywords from the query (e.g. 'assets', 'profit', 'cash')
      and return ALL metrics whose names contain those keywords, each with trend vs previous year.
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
        ctx = _lookup_metric_with_neighbors(comp, precise_metric, period)
        if ctx is not None:
            return _format_single_metric_answer(tick, precise_metric, ctx)

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
                return _format_single_metric_answer(tick, m, ctx)
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

    ctx = _lookup_metric_with_neighbors(comp, best, period)
    if ctx is None:
        return "Value not found."

    return _format_single_metric_answer(tick, best, ctx)
