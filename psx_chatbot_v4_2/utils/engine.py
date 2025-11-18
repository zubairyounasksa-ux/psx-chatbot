from fuzzywuzzy import fuzz
import re
import requests
import pandas as pd


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
        "year", "years", "value", "what", "tell", "me",
        "overview", "snapshot", "summary", "company", "view"
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


# ===================== Role mapping for cross-metric analysis =====================

ROLE_SYNONYMS = {
    "sales": [
        "sales", "revenue", "net sales", "turnover"
    ],
    "gross_profit": [
        "gross profit"
    ],
    "pbt": [
        "profit before taxation", "profit before tax", "pbt"
    ],
    "pat": [
        "profit after taxation", "profit after tax", "net profit", "profit after tax (pat)"
    ],
    "equity": [
        "shareholder equity", "shareholders equity", "shareholders' equity",
        "total equity", "owners equity", "owners' equity", "share capital and reserves",
        "share capital & reserves"
    ],
    "total_assets": [
        "total assets"
    ],
    "total_liabilities": [
        "total liabilities", "total liability"
    ],
    "cfo": [
        "cash from operation", "cash from operations",
        "cash flow from operations", "net cash from operating activities"
    ],
}


def _build_role_index(comp):
    """
    For each 'role' (sales, pat, equity, etc.) find the best matching metric name
    in this company's headings using fuzzy matching.
    Returns: { role: metric_name }
    """
    metrics = comp.get("metrics", [])
    index = {}

    for role, syns in ROLE_SYNONYMS.items():
        best_metric = None
        best_score = 0
        for m in metrics:
            m_low = str(m).lower()
            for syn in syns:
                score = fuzz.partial_ratio(m_low, syn.lower())
                if score > best_score:
                    best_score = score
                    best_metric = m
        # Only keep if reasonably confident
        if best_metric is not None and best_score >= 70:
            index[role] = best_metric

    return index


def _get_role_value(comp, role_index, role, period):
    """
    Get (display_value, numeric_value) for a given role in the requested period.
    Uses the same context lookup to align periods.
    """
    metric = role_index.get(role)
    if not metric:
        return None, None

    ctx = _lookup_metric_with_context(comp, metric, period)
    if not ctx:
        return None, None

    val = ctx["value"]
    num = _to_number(val)
    return val, num


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
        "valid_cols": [...],
        "row": pandas.DataFrame row,
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


def _default_period_from_company(comp):
    """
    If user does not specify a year in overview, pick the latest available period.
    """
    periods = comp.get("periods", [])
    if not periods:
        return None

    # Try numeric sort if possible
    def _key(p):
        try:
            return int(p)
        except Exception:
            return p

    return sorted(periods, key=_key)[-1]


# ===================== Cross-metric expert analysis helpers =====================

def _interpret_margin(name, margin):
    """Simple heuristic interpretation for a % margin."""
    if margin is None:
        return ""
    if margin >= 25:
        return f"{name} is exceptionally strong."
    elif margin >= 15:
        return f"{name} is strong and healthy."
    elif margin >= 8:
        return f"{name} is moderate and acceptable."
    elif margin >= 3:
        return f"{name} is thin; profitability is sensitive to cost pressures."
    else:
        return f"{name} is very weak and indicates limited pricing power or high costs."


def _interpret_roe(roe):
    if roe is None:
        return ""
    if roe >= 25:
        return "ROE is excellent, reflecting very strong value creation for shareholders."
    elif roe >= 15:
        return "ROE is solid and attractive in most markets."
    elif roe >= 10:
        return "ROE is reasonable but not outstanding."
    else:
        return "ROE is weak and suggests limited return on equity capital."


def _interpret_roa(roa):
    if roa is None:
        return ""
    if roa >= 10:
        return "ROA is outstanding given the asset base."
    elif roa >= 5:
        return "ROA is healthy and indicates efficient use of assets."
    elif roa >= 2:
        return "ROA is modest but acceptable for some capital-intensive sectors."
    else:
        return "ROA is low, suggesting heavy assets relative to profit."


def _interpret_leverage(leverage):
    if leverage is None:
        return ""
    if leverage <= 0.5:
        return "Balance sheet is very conservative with low leverage."
    elif leverage <= 1.5:
        return "Leverage is moderate and generally comfortable."
    elif leverage <= 2.5:
        return "Leverage is on the higher side and needs monitoring."
    else:
        return "Leverage is aggressive; the company is highly dependent on debt."


def _interpret_cash_conversion(conv):
    if conv is None:
        return ""
    if conv >= 1.2:
        return "Cash conversion is excellent; cash generation exceeds accounting profit."
    elif conv >= 0.9:
        return "Cash conversion is healthy; most profit is converting into cash."
    elif conv >= 0.6:
        return "Cash conversion is moderate; watch working capital and cash leakage."
    else:
        return "Cash conversion is weak; there may be collection, working capital, or quality-of-earnings issues."


def _build_cross_metric_snapshot(tick, period, comp, role_index):
    """
    Build cross-metric ratios (margins, ROE, ROA, leverage, cash conversion)
    for the given period using role_index.
    Returns list of text lines, or empty list if nothing computed.
    """
    lines = []

    # Base values
    sales_val, sales = _get_role_value(comp, role_index, "sales", period)
    pat_val, pat = _get_role_value(comp, role_index, "pat", period)
    gp_val, gp = _get_role_value(comp, role_index, "gross_profit", period)
    pbt_val, pbt = _get_role_value(comp, role_index, "pbt", period)
    eq_val, eq = _get_role_value(comp, role_index, "equity", period)
    ta_val, ta = _get_role_value(comp, role_index, "total_assets", period)
    tl_val, tl = _get_role_value(comp, role_index, "total_liabilities", period)
    cfo_val, cfo = _get_role_value(comp, role_index, "cfo", period)

    # Margins
    if sales not in (None, 0):
        # Gross margin
        if gp is not None:
            gross_margin = gp / sales * 100
            lines.append(
                f"- Gross margin ≈ {gross_margin:,.1f}% "
                f"(Gross Profit {gp_val} / Sales {sales_val}). "
                f"{_interpret_margin('Gross margin', gross_margin)}"
            )

        # PBT margin
        if pbt is not None:
            pbt_margin = pbt / sales * 100
            lines.append(
                f"- PBT margin ≈ {pbt_margin:,.1f}% "
                f"(PBT {pbt_val} / Sales {sales_val}). "
                f"{_interpret_margin('PBT margin', pbt_margin)}"
            )

        # Net margin (PAT)
        if pat is not None:
            net_margin = pat / sales * 100
            lines.append(
                f"- Net margin ≈ {net_margin:,.1f}% "
                f"(Profit after Taxation {pat_val} / Sales {sales_val}). "
                f"{_interpret_margin('Net margin', net_margin)}"
            )

    # ROE
    if pat is not None and eq not in (None, 0):
        roe = pat / eq * 100
        lines.append(
            f"- Return on Equity (ROE) ≈ {roe:,.1f}% "
            f"(Profit after Taxation {pat_val} / Equity {eq_val}). "
            f"{_interpret_roe(roe)}"
        )

    # ROA
    if pat is not None and ta not in (None, 0):
        roa = pat / ta * 100
        lines.append(
            f"- Return on Assets (ROA) ≈ {roa:,.1f}% "
            f"(Profit after Taxation {pat_val} / Total Assets {ta_val}). "
            f"{_interpret_roa(roa)}"
        )

    # Leverage
    if tl is not None and eq not in (None, 0):
        leverage = tl / eq
        lines.append(
            f"- Leverage (Liabilities / Equity) ≈ {leverage:,.2f}x "
            f"(Total Liabilities {tl_val} / Equity {eq_val}). "
            f"{_interpret_leverage(leverage)}"
        )

    # Cash conversion
    if cfo is not None and pat not in (None, 0):
        conv = cfo / pat
        lines.append(
            f"- Cash conversion (CFO / PAT) ≈ {conv:,.2f}x "
            f"(Cash from Operation {cfo_val} / Profit after Taxation {pat_val}). "
            f"{_interpret_cash_conversion(conv)}"
        )

    if lines:
        header = f"Cross-metric snapshot for {tick} in {period}:"
        return [header] + lines

    return []


# ===================== PSX DPS live scraper =====================

def _fetch_psx_overview(ticker):
    """
    Fetch latest price, free float %, EPS by year, and latest payout
    from https://dps.psx.com.pk/company/{ticker}.

    Uses:
      - regex on raw HTML for price & free float
      - pandas.read_html for Financials and Payouts tables, with
        very loose matching on headers / labels.

    Returns dict:
      {
        "price": "560.45",
        "free_float_pct": "40.00%",
        "eps_by_year": {"2025": "107.58", "2024": "18.34", ...},
        "latest_payout": {"date": "...", "details": "...", "book_closure": "..."},
        "source_url": url,
      }
    or None on failure.
    """
    url = f"https://dps.psx.com.pk/company/{ticker}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception:
        return None

    html = resp.text

    # ---------- 1) Latest price ----------
    price = None
    m_price = re.search(r"Rs\.\s*([\d,]+\.\d+)", html)
    if m_price:
        price = m_price.group(1)

    # ---------- 2) Free float % ----------
    free_float_pct = None
    try:
        ff_matches = re.findall(
            r"Free\s*Float[^%]*?([\d.,]+\s*%)",
            html,
            flags=re.I | re.S,
        )
        if ff_matches:
            free_float_pct = ff_matches[-1].strip()
    except Exception:
        pass

    # ---------- 3) Read all tables ----------
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []

    eps_by_year: dict[str, str] = {}
    latest_payout = None

    # ---------- 3a) Find EPS by year from any "financials" table ----------
    for df in tables:
        if df.shape[0] < 1 or df.shape[1] < 2:
            continue

        # First column usually contains labels like "Sales", "Profit after Taxation", "EPS"
        first_col = df.iloc[:, 0].astype(str).str.strip().str.lower()

        # Find any row where label contains "eps"
        eps_row_idx = None
        for i, label in enumerate(first_col):
            if "eps" in label:
                eps_row_idx = i
                break

        if eps_row_idx is None:
            continue  # this table is not the one we want

        eps_row = df.iloc[eps_row_idx]

        # Remaining columns: headers may be "2025", "2024", "2023", "2022" or like "2025*"
        for j, col in enumerate(df.columns[1:], start=1):
            col_name = str(col)
            m_year = re.search(r"(19|20)\d{2}", col_name)
            if not m_year:
                continue
            year = m_year.group(0)
            eps_val = eps_row.iloc[j]
            # skip blanks / NaN
            if pd.isna(eps_val):
                continue
            eps_by_year[year] = str(eps_val)

        if eps_by_year:
            break  # stop once we have one EPS row

    # ---------- 3b) Find latest payout from any "payouts" table ----------
    for df in tables:
        if df.shape[0] < 1 or df.shape[1] < 2:
            continue

        cols_lower = [str(c).strip().lower() for c in df.columns]

        # We want a table that has some date column and some book-closure column
        has_date = any("date" in c for c in cols_lower)
        has_closure = any("closure" in c or ("book" in c and "close" in c) for c in cols_lower)
        if not (has_date and has_closure):
            continue

        row0 = df.iloc[0]  # assume first row is latest payout

        def get_col(*keywords):
            """
            Return cell for the first column whose header contains ALL keywords.
            """
            for idx, cname in enumerate(cols_lower):
                if all(k in cname for k in keywords):
                    return str(row0.iloc[idx])
            return ""

        date_val = get_col("date")
        # Some DPS tables have both 'Financial Results' and 'Details' or 'Entitlement'
        fin_res_val = get_col("financial", "result")
        details_val = get_col("detail") or get_col("entitlement") or get_col("dividend")
        book_val = get_col("book", "closure") or get_col("closure")

        details_parts = []
        if fin_res_val:
            details_parts.append(fin_res_val)
        if details_val:
            details_parts.append(details_val)
        details_combined = " ".join(details_parts).strip() or None

        latest_payout = {
            "date": date_val.strip() or None,
            "details": details_combined,
            "book_closure": book_val.strip() or None,
        }
        break  # stop after first matching payouts table

    return {
        "price": price,
        "free_float_pct": free_float_pct,
        "eps_by_year": eps_by_year,
        "latest_payout": latest_payout,
        "source_url": url,
    }


# ===================== Company snapshot (overview) =====================

def _build_company_snapshot_text(tick, period, comp, role_index):
    """
    Build a compact analyst-style overview for a company in a given period.
    Includes: sales growth, PAT, margins, ROE, leverage, cash conversion,
    plus live PSX DPS snapshot (price, free float, EPS by year, latest payout).
    """
    if period is None:
        period = _default_period_from_company(comp)

    if period is None:
        return "No period information available for overview."

    # Get base role values and contexts
    sales_metric = role_index.get("sales")
    pat_metric = role_index.get("pat")

    sales_ctx = _lookup_metric_with_context(comp, sales_metric, period) if sales_metric else None
    pat_ctx = _lookup_metric_with_context(comp, pat_metric, period) if pat_metric else None

    sales_val = sales_num = None
    if sales_ctx:
        sales_val = sales_ctx["value"]
        sales_num = _to_number(sales_val)

    pat_val = pat_num = None
    if pat_ctx:
        pat_val = pat_ctx["value"]
        pat_num = _to_number(pat_val)

    # Header
    lines = [f"{tick} – {period} Financial Overview"]

    # ---------- 1) Top-line & earnings ----------
    top_line_sentences = []

    # Sales YoY and CAGR
    if sales_ctx and sales_num is not None:
        prev_p = sales_ctx["prev_period"]
        prev_v = sales_ctx["prev_value"]
        prev_n = _to_number(prev_v) if prev_p is not None else None

        if prev_p is not None and prev_n not in (None, 0):
            diff = sales_num - prev_n
            pct = diff / abs(prev_n) * 100
            direction = "grew" if diff > 0 else "declined" if diff < 0 else "was flat"
            top_line_sentences.append(
                f"Sales in {period} were {sales_val}, which {direction} by {abs(pct):.1f}% versus {prev_p}."
            )
        else:
            top_line_sentences.append(
                f"Sales in {period} were {sales_val}."
            )

        # Multi-year sales CAGR
        series_sales = _get_metric_series(comp, sales_metric)
        series_nums = {
            p: _to_number(v)
            for p, v in series_sales.items()
            if _to_number(v) is not None
        }
        if series_nums and len(series_nums) >= 2:
            def _key(p):
                try:
                    return int(p)
                except Exception:
                    return p

            sorted_items = sorted(series_nums.items(), key=lambda x: _key(x[0]))
            first_p, first_v = sorted_items[0]
            last_p, last_v = sorted_items[-1]
            if first_v not in (None, 0):
                n_years = len(sorted_items) - 1
                if n_years > 0:
                    cagr = (last_v / abs(first_v)) ** (1 / n_years) - 1
                    top_line_sentences.append(
                        f"Over {first_p}–{last_p}, sales have grown at an approximate CAGR of {cagr*100:.1f}%."
                    )

    # PAT YoY
    if pat_ctx and pat_num is not None:
        prev_p = pat_ctx["prev_period"]
        prev_v = pat_ctx["prev_value"]
        prev_n = _to_number(prev_v) if prev_p is not None else None

        if prev_p is not None and prev_n not in (None, 0):
            diff = pat_num - prev_n
            pct = diff / abs(prev_n) * 100
            direction = "increased" if diff > 0 else "decreased" if diff < 0 else "was broadly unchanged"
            top_line_sentences.append(
                f"Profit after taxation in {period} was {pat_val}, which {direction} by {abs(pct):.1f}% versus {prev_p}."
            )
        else:
            top_line_sentences.append(
                f"Profit after taxation in {period} stood at {pat_val}."
            )

    if top_line_sentences:
        lines.append("")
        lines.append("Headline performance:")
        lines.append(" ".join(top_line_sentences))

    # ---------- 2) Profitability, returns, leverage, cash ----------
    cross_lines = _build_cross_metric_snapshot(tick, period, comp, role_index)
    if cross_lines:
        lines.append("")
        lines.extend(cross_lines)

    # ---------- 3) Live PSX DPS snapshot ----------
    psx_data = _fetch_psx_overview(tick)
    if psx_data:
        price = psx_data.get("price")
        free_float_pct = psx_data.get("free_float_pct")
        eps_by_year = psx_data.get("eps_by_year") or {}
        latest_payout = psx_data.get("latest_payout")

        psx_lines = []
        psx_lines.append(f"Market snapshot (PSX DPS for {tick}):")

        if price:
            psx_lines.append(f"- Latest price: Rs. {price}")
        if free_float_pct:
            psx_lines.append(f"- Free float: {free_float_pct}")

        if eps_by_year:
            # sort years descending
            year_items = sorted(eps_by_year.items(), key=lambda x: x[0], reverse=True)
            eps_str = "; ".join(f"{yr}: {val}" for yr, val in year_items)
            psx_lines.append(f"- EPS by year (Annual, as per PSX DPS): {eps_str}")

        if latest_payout:
            d = str(latest_payout.get("date", "")).strip()
            det = str(latest_payout.get("details", "")).strip()
            bc = str(latest_payout.get("book_closure", "")).strip()
            parts = []
            if d:
                parts.append(d)
            if det:
                parts.append(det)
            if bc:
                parts.append(f"Book closure: {bc}")
            if parts:
                psx_lines.append("- Latest payout: " + " – ".join(parts))

        if len(psx_lines) > 1:
            lines.append("")
            lines.extend(psx_lines)

    return "\n".join(lines)


# ===================== Formatting helpers for metric-based answers =====================

def _format_single_metric_answer(tick, metric, ctx, comp, role_index):
    """
    Human-friendly answer with analysis:
    - value in requested year,
    - comparison vs previous year,
    - comparison vs latest year (e.g., 2025),
    - multi-year trend view (CAGR and where this year sits in history),
    - cross-metric ratios (margins, ROE, ROA, leverage, cash conversion).
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
        last_period, last_val = periods_sorted[-1]

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
        expert_lines.append(
            f"The {period} figure {position_comment}"
        )

        lines.append("")
        lines.append("Expert view:")
        lines.append(" ".join(expert_lines))

    # --- Cross-metric ratios (true expert snapshot) ---
    cross_lines = _build_cross_metric_snapshot(tick, period, comp, role_index)
    if cross_lines:
        lines.append("")
        lines.extend(cross_lines)

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
    - If query looks like 'overview/snapshot/summary of TICKER YEAR':
        => Return a company-level mini-report:
           * sales & PAT with YoY and multi-year sales CAGR,
           * margins, ROE, ROA, leverage, cash-conversion (cross-metric view),
           * plus PSX DPS live snapshot (price, free float %, EPS by year, latest payout).
    - Otherwise:
        - If query contains a precise metric phrase (e.g. 'Total Assets', 'Profit after Taxation'),
          return that single metric with:
            * value,
            * change vs previous year,
            * change vs latest year,
            * multi-year expert commentary,
            * cross-metric ratios.
        - Else, extract keywords (e.g. 'assets', 'profit', 'cash') and return all matching
          metrics with trend vs previous & latest years.
        - Else, fall back to fuzzy best-match metric with trend and expert view.
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
    role_index = _build_role_index(comp)

    q_lower = str(query).lower()

    # --------- Special mode: company snapshot / overview ---------
    if any(k in q_lower for k in ["overview", "snapshot", "summary", "company view"]):
        return _build_company_snapshot_text(tick, period, comp, role_index)

    # --------- First: precise metric phrase ---------
    precise_metric = _find_precise_metric(query, metrics)
    if precise_metric:
        ctx = _lookup_metric_with_context(comp, precise_metric, period)
        if ctx is not None:
            return _format_single_metric_answer(tick, precise_metric, ctx, comp, role_index)

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
            # If only one metric matched, behave like single metric (with full expert view)
            if len(metric_contexts) == 1:
                m, ctx = metric_contexts[0]
                return _format_single_metric_answer(tick, m, ctx, comp, role_index)
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

    return _format_single_metric_answer(tick, best, ctx, comp, role_index)

