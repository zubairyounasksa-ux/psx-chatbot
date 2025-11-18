from fuzzywuzzy import fuzz
import re


def best_match(query, options, threshold=50):
    """
    Find the best matching string in `options` for the user `query`.

    Strategy:
    1) First, look for exact phrase matches (case-insensitive) where the full
       option text appears inside the query. Among these, prefer the most
       specific (more words, then more characters).
    2) If no phrase matches are found, fall back to fuzzy matching
       (fuzz.partial_ratio) and apply a score threshold.
    """
    if not query or not options:
        return None

    q = str(query).lower()

    # Normalise all options as strings
    norm_options = [str(opt) for opt in options]

    # --- 1) Exact phrase matches ---
    phrase_matches = []
    for opt_str in norm_options:
        o = opt_str.lower().strip()
        if not o:
            continue
        if o in q:
            phrase_matches.append(opt_str)

    if phrase_matches:
        # Prefer the most specific phrase: more words, then more characters
        best_phrase = max(
            phrase_matches,
            key=lambda s: (len(s.split()), len(s))
        )
        return best_phrase

    # --- 2) Fuzzy fallback ---
    best = None
    best_score = 0
    for opt_str in norm_options:
        o = opt_str.lower().strip()
        if not o:
            continue
        score = fuzz.partial_ratio(q, o)
        if score > best_score:
            best = opt_str
            best_score = score

    return best if best_score >= threshold else None


def _lookup_single_metric_value(comp, metric, period):
    """
    Helper to look up one metric for a given company and (optional) period.
    Returns the value or None.
    """
    if not metric:
        return None

    for df in comp.get("files", []):
        try:
            first_col = df.iloc[:, 0].astype(str)
        except Exception:
            continue

        row = df[first_col == str(metric)]
        if not row.empty:
            # If a specific period column was matched and exists in this df
            if period and period in df.columns:
                return row[period].values[0]

            # Fallback: use first numeric column after heading
            if df.shape[1] > 1:
                return row.iloc[0, 1]

    return None


def _lookup_multiple_metrics_values(comp, metrics, period):
    """
    Helper to look up several metrics and return list of (metric, value).
    """
    results = []
    for m in metrics:
        val = _lookup_single_metric_value(comp, m, period)
        if val is not None:
            results.append((m, val))
    return results


def _extract_keywords(query, ticker):
    """
    Extract significant alphabetic tokens from the query,
    removing the ticker and common stopwords.
    """
    q_lower = str(query).lower()
    words = re.findall(r"[a-zA-Z]+", q_lower)

    stopwords = {
        "of", "the", "in", "for", "and", "to", "from", "at",
        "on", "is", "are", "was", "were", "this", "that",
        "year", "years", "value"
    }

    ticker_lower = str(ticker).lower() if ticker else ""
    keywords = []
    for w in words:
        if len(w) < 3:
            continue
        if w in stopwords:
            continue
        if ticker_lower and w == ticker_lower.lower():
            continue
        keywords.append(w)

    return keywords


def answer_query(query, companies):
    """
    Main Q&A function.

    - Detects ticker from query words (e.g., 'GAL', 'SAZEW').
    - If query is vague (e.g., 'assets', 'profit', 'cash'), return all metrics
      containing those keywords for the selected period.
    - If query contains a precise metric phrase (e.g., 'profit after taxation'),
      return just that metric.
    """
    if not query:
        return "Empty query."

    if not companies:
        return "No companies loaded."

    # --- Detect ticker from query words ---
    words_upper = str(query).upper().split()
    tick = None

    for t in companies:
        if t.upper() in words_upper:
            tick = t
            break

    # Convenience: if only one company loaded and no ticker written
    if not tick and len(companies) == 1:
        tick = next(iter(companies))

    if not tick or tick not in companies:
        return "No company ticker detected in query."

    comp = companies[tick]
    q_lower = str(query).lower()

    # --- Detect period (year etc.) once ---
    period = best_match(query, comp.get("periods", []), threshold=40)

    metrics = comp.get("metrics", [])

    # --- Try to find a precise metric phrase first ---
    precise_metric = best_match(query, metrics, threshold=50)
    is_precise = (
        precise_metric is not None
        and precise_metric.lower().strip() in q_lower
        and len(precise_metric.split()) > 1  # multi-word metric like "Total Assets"
    )

    if is_precise:
        val = _lookup_single_metric_value(comp, precise_metric, period)
        if val is not None:
            if period:
                return f"{tick} – {precise_metric} in {period}: {val}"
            return f"{tick} – {precise_metric}: {val}"

    # -------------------------------------------------
    # Generic keyword logic: show all matching metrics
    # -------------------------------------------------
    keywords = _extract_keywords(query, tick)

    keyword_metrics = set()
    for m in metrics:
        m_low = str(m).lower()
        for kw in keywords:
            if kw in m_low:
                keyword_metrics.add(m)
                break

    if keyword_metrics:
        # If we only have one good match, behave like a normal single answer
        if len(keyword_metrics) == 1:
            metric = next(iter(keyword_metrics))
            val = _lookup_single_metric_value(comp, metric, period)
            if val is not None:
                if period:
                    return f"{tick} – {metric} in {period}: {val}"
                return f"{tick} – {metric}: {val}"
        else:
            # Multiple matches -> show a small list
            results = _lookup_multiple_metrics_values(comp, keyword_metrics, period)
            if results:
                period_text = f" in {period}" if period else ""
                lines = [f"{tick} – metrics related to {', '.join(keywords)}{period_text}:"]
                for m, v in sorted(results, key=lambda x: str(x[0])):
                    lines.append(f"- {m}: {v}")
                return "\n".join(lines)

    # -------------------------------------------------
    # Final fallback: generic fuzzy best_match
    # -------------------------------------------------
    metric = best_match(query, metrics, threshold=45)
    if not metric:
        return "Metric not found."

    val = _lookup_single_metric_value(comp, metric, period)
    if val is None:
        return "Value not found."

    if period:
        return f"{tick} – {metric} in {period}: {val}"
    return f"{tick} – {metric}: {val}"
