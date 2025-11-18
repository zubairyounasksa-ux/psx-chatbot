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


def answer_query(query, companies):
    """
    Main Q&A function.

    - Detects ticker from query words (e.g., 'GAL', 'SAZEW').
    - Uses best_match() to find the most relevant metric and period.
    - Searches across all CSV DataFrames for that company and returns
      the value.
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

    # Optional convenience: if only one company loaded and no ticker typed
    if not tick and len(companies) == 1:
        tick = next(iter(companies))  # use the only loaded ticker

    if not tick or tick not in companies:
        return "No company ticker detected in query."

    comp = companies[tick]

    # --- Find metric ---
    metric = best_match(query, comp.get("metrics", []), threshold=50)
    if not metric:
        return "Metric not found."

    # --- Find period (year etc.) ---
    period = best_match(query, comp.get("periods", []), threshold=40)

    # --- Search in all DataFrames for this company ---
    for df in comp.get("files", []):
        try:
            first_col = df.iloc[:, 0].astype(str)
        except Exception:
            continue

        row = df[first_col == str(metric)]
        if not row.empty:
            # If a specific period column was matched and exists in this df
            if period and period in df.columns:
                val = row[period].values[0]
                return f"{tick} – {metric} in {period}: {val}"

            # Fallback: use first numeric column after heading
            if df.shape[1] > 1:
                val = row.iloc[0, 1]
                return f"{tick} – {metric}: {val}"

    return "Value not found."
