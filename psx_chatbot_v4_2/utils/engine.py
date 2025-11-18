from fuzzywuzzy import fuzz


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
    norm_options = []
    for opt in options:
        opt_str = str(opt)
        norm_options.append(opt_str)

    # --- 1) Exact phrase matches ---
    phrase_matches = []
    for opt_str in norm_options:
        o = opt_str.lower().strip()
        if not o:
            continue
        # If the full metric/period text appears in the query, treat as phrase match
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
    Returns (value or None).
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


def answer_query(query, companies):
    """
    Main Q&A function.

    - Detects ticker from query words (e.g., 'GAL', 'SAZEW').
    - If query is generic 'profit', returns all profit-related metrics.
    - Otherwise uses best_match() to find the most relevant metric and period.
    """
    if not query:
        return "Empty query."

    if not companies:
        return "No companies loaded."

    # --- Detect ticker from query words ---
    words = str(query).upper().split()
    tick = None

    for t in companies:
        if t.upper() in words:
            tick = t
            break

    # Convenience: if only one company loaded and no ticker written
    if not tick and len(companies) == 1:
        tick = next(iter(companies))

    if not tick or tick not in companies:
        return "No company ticker detected in query."

    comp = companies[tick]
    q_lower = query.lower()

    # --- Detect period (year etc.) once ---
    period = best_match(query, comp.get("periods", []), threshold=40)

    # -------------------------------------------------
    # SPECIAL CASE: generic "profit" query
    # -------------------------------------------------
    # If user says just "profit" without specifying gross/before/after/net/tax,
    # show ALL profit-related metrics for that period.
    generic_profit = (
        "profit" in q_lower
        and not any(w in q_lower for w in ["gross", "before", "after", "tax", "taxation", "operating", "net"])
    )

    if generic_profit:
        all_metrics = comp.get("metrics", [])
        profit_metrics = [m for m in all_metrics if "profit" in str(m).lower()]

        if profit_metrics:
            results = _lookup_multiple_metrics_values(comp, profit_metrics, period)
            if results:
                period_text = f" in {period}" if period else ""
                lines = [f"{tick} – metrics related to 'profit'{period_text}:"]
                for m, v in results:
                    lines.append(f"- {m}: {v}")
                return "\n".join(lines)
            # If somehow no values found, fall back to normal logic

    # -------------------------------------------------
    # NORMAL CASE: precise metric
    # -------------------------------------------------
    metric = best_match(query, comp.get("metrics", []), threshold=50)
    if not metric:
        return "Metric not found."

    val = _lookup_single_metric_value(comp, metric, period)
    if val is None:
        return "Value not found."

    if period:
        return f"{tick} – {metric} in {period}: {val}"
    return f"{tick} – {metric}: {val}"
