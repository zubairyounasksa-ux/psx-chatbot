from fuzzywuzzy import fuzz
import re


# ----------------- Helper functions -----------------


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

    # 1) Try to find explicit 4-digit year in query and check if it exists in periods
    years_in_query = re.findall(r"\b(19|20)\d{2}\b", q)
    if years_in_query:
        # years_in_query is list of tuples ('20', '23'); rebuild full year from text
        # easier: search again but capture full match
        full_years = re.findall(r"\b((?:19|20)\d{2})\b", q)
        for y in full_years:
            if y in periods:
                return y

    # 2) Fuzzy fallback
    best = None
    best_score = 0
    q_lower = q.lower()
    for p in periods:
        p_str = str(p)
        if p_str in q_lower:
            # direct phrase present
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
        # Prefer most specific: more words, then more characters
        return max(phrase_matches, key=lambda s: (len(s.split()), len(s)))

    return None


def _lookup_single_metric_value(comp, metric, period):
    """Look up one metric value for a company and (optional) period."""
    if not metric:
        return None

    for df in comp.get("files", []):
        try:
            first_col = df.iloc[:, 0].astype(str)
        except Exception:
            continue

        row = df[first_col == str(metric)]
        if not row.empty:
            if period and period in df.columns:
                return row[period].values[0]
            # fallback to first numeric column
            if df.shape[1] > 1:
                return row.iloc[0, 1]

    return None


def _lookup_multiple_metrics_values(comp, metrics, period):
    """Look up several metrics and return list of (metric, value)."""
    out = []
    for m in metrics:
        val = _lookup_single_metric_value(comp, m, period)
        if val is not None:
            out.append((m, val))
    return out


# ----------------- Main entrypoint -----------------


def answer_query(query, companies):
    """
    Main Q&A function.

    Behaviour:
    - Detect ticker and period.
    - If query contains a precise metric phrase (e.g. 'Total Assets', 'Profit after Taxation'),
      return that single metric.
    - Otherwise, extract keywords from the query (e.g. 'assets', 'profit', 'cash')
      and return ALL metrics whose names contain those keywords.
    - If still nothing useful, fall back to fuzzy single-metric match.
    """
    if not query:
        return "Empty query."

    if not companies:
        return "No companies loaded."

    # ----------------- Ticker & period -----------------
    tick = _detect_ticker(query, companies)
    if not tick or tick not in companies:
        return "No company ticker detected in query."

    comp = companies[tick]
    periods = comp.get("periods", [])
    metrics = comp.get("metrics", [])

    period = _detect_period(query, periods)

    # ----------------- Precise metric phrase -----------------
    precise_metric = _find_precise_metric(query, metrics)
    if precise_metric:
        val = _lookup_single_metric_value(comp, precise_metric, period)
        if val is not None:
            if period:
                return f"{tick} – {precise_metric} in {period}: {val}"
            return f"{tick} – {precise_metric}: {val}"

    # ----------------- Keyword-based multi-metric logic -----------------
    keywords = _extract_keywords(query, tick)

    keyword_metrics = set()
    for m in metrics:
        m_low = str(m).lower()
        for kw in keywords:
            # simple plural/singular handling: asset <-> assets, liability <-> liabilities
            kw_low = kw.lower()
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
        results = _lookup_multiple_metrics_values(comp, keyword_metrics, period)
        if results:
            # If only one metric matched, behave like single result
            if len(results) == 1:
                m, v = results[0]
                if period:
                    return f"{tick} – {m} in {period}: {v}"
                return f"{tick} – {m}: {v}"

            # Otherwise list all related metrics
            period_text = f" in {period}" if period else ""
            key_text = ", ".join(sorted(set(keywords)))
            lines = [f"{tick} – metrics related to {key_text}{period_text}:"]
            for m, v in sorted(results, key=lambda x: str(x[0])):
                lines.append(f"- {m}: {v}")
            return "\n".join(lines)

    # ----------------- Final fuzzy fallback -----------------
    # If everything above failed, pick the best fuzzy match metric
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

    val = _lookup_single_metric_value(comp, best, period)
    if val is None:
        return "Value not found."

    if period:
        return f"{tick} – {best} in {period}: {val}"
    return f"{tick} – {best}: {val}"
