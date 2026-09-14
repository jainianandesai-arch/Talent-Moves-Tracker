"""
live_sources.py
----------------
Fetch functions for the free, no-key public APIs wired into Talent Moves
Tracker: StatCan WDS (Canada labour data), BLS (US labour data), and SEC
EDGAR (US peer 8-K/proxy filings). Each function was verified against the
live endpoint before being wired in here — see the PR/commit notes for the
raw responses captured during that check.

Every function returns a dict with:
    - "ok": bool
    - "retrieved_at": ISO8601 UTC timestamp of the actual HTTP call
    - "source": human-readable source name
    - "source_url": the exact endpoint hit
    - "data": the parsed payload (shape varies by function) if ok else None
    - "error": error string if not ok

Nothing here is a "Disclosed fact" placeholder — every value returned was
read from a live HTTP response at call time, tagged with source + timestamp
per the governance rules (untagged/blended figures are not permitted
downstream).
"""

from __future__ import annotations

import datetime as _dt
import json
import xml.etree.ElementTree as _ET
from urllib import request as _request
from urllib.error import URLError, HTTPError

USER_AGENT = "TalentMovesTracker/1.0 (People Analytics research tool; contact: jainianandesai@gmail.com)"

STATCAN_WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"
BLS_API_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
SEC_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"

# SEC CIKs confirmed live against SEC's own company_tickers.json — see governance
# note above. John Hancock has NO standalone SEC filer (wholly owned Manulife
# subsidiary); its exec-appointment filings surface only under Manulife's CIK.
SEC_CIK_BY_PEER = {
    "MetLife": "0001099219",
    "Prudential Financial": "0001137774",
    "Lincoln Financial": "0000059558",
    "Principal Financial": "0001126328",
    "Manulife Financial (parent of John Hancock)": "0001086888",
}

# StatCan LFS vectors (cube 14100287 — Labour force characteristics, monthly,
# seasonally adjusted). Confirmed live: releaseTime 2026-09-04, current through
# Aug 2026 at time of wiring.
STATCAN_VECTORS = {
    "Canada unemployment rate (%)": 2062815,
}

# BLS series IDs. CES5552400001 = All employees, Insurance carriers, thousands,
# seasonally adjusted. Confirmed live and current through Aug 2026 (preliminary).
BLS_SERIES = {
    "US insurance carriers employment (thousands)": "CES5552400001",
    "US unemployment rate (%)": "LNS14000000",
}


def _http_get_json(url: str, timeout: int = 15) -> dict:
    req = _request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with _request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_text(url: str, timeout: int = 15) -> str:
    req = _request.Request(url, headers={"User-Agent": USER_AGENT})
    with _request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _http_post_json(url: str, payload, timeout: int = 15) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = _request.Request(
        url, data=body, headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"}, method="POST"
    )
    with _request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_statcan_lfs(latest_n: int = 3) -> dict:
    """Pull the latest N months for each StatCan LFS vector in STATCAN_VECTORS."""
    retrieved_at = _now_iso()
    try:
        payload = [{"vectorId": vid, "latestN": latest_n} for vid in STATCAN_VECTORS.values()]
        raw = _http_post_json(f"{STATCAN_WDS_BASE}/getDataFromVectorsAndLatestNPeriods", payload)
        series = {}
        for label, vid in STATCAN_VECTORS.items():
            match = next((r for r in raw if r.get("object", {}).get("vectorId") == vid), None)
            if match and match.get("status") == "SUCCESS":
                points = match["object"]["vectorDataPoint"]
                series[label] = [{"period": p["refPer"], "value": p["value"]} for p in points]
            else:
                series[label] = None
        return {
            "ok": True,
            "retrieved_at": retrieved_at,
            "source": "Statistics Canada — Labour Force Survey (WDS API)",
            "source_url": f"{STATCAN_WDS_BASE}/getDataFromVectorsAndLatestNPeriods",
            "data": series,
            "error": None,
        }
    except (URLError, HTTPError, TimeoutError, ValueError, KeyError) as e:
        return {
            "ok": False,
            "retrieved_at": retrieved_at,
            "source": "Statistics Canada — Labour Force Survey (WDS API)",
            "source_url": f"{STATCAN_WDS_BASE}/getDataFromVectorsAndLatestNPeriods",
            "data": None,
            "error": str(e),
        }


def fetch_bls_series(latest_n: int = 6) -> dict:
    """Pull the latest observations for each BLS series in BLS_SERIES."""
    retrieved_at = _now_iso()
    try:
        raw = _http_post_json(BLS_API_BASE, {"seriesid": list(BLS_SERIES.values())})
        if raw.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(f"BLS API returned status={raw.get('status')}: {raw.get('message')}")
        by_id = {s["seriesID"]: s["data"][:latest_n] for s in raw["Results"]["series"]}
        series = {}
        for label, sid in BLS_SERIES.items():
            points = by_id.get(sid, [])
            series[label] = [
                {"period": f"{p['year']}-{p['period']} ({p['periodName']})", "value": p["value"]} for p in points
            ]
        return {
            "ok": True,
            "retrieved_at": retrieved_at,
            "source": "U.S. Bureau of Labor Statistics (public API)",
            "source_url": BLS_API_BASE,
            "data": series,
            "error": None,
        }
    except (URLError, HTTPError, TimeoutError, ValueError, KeyError) as e:
        return {
            "ok": False,
            "retrieved_at": retrieved_at,
            "source": "U.S. Bureau of Labor Statistics (public API)",
            "source_url": BLS_API_BASE,
            "data": None,
            "error": str(e),
        }


def fetch_sec_recent_filings(peer_name: str, forms: tuple[str, ...] = ("8-K", "DEF 14A", "10-K"), limit: int = 8) -> dict:
    """Pull the most recent filings of the given forms for one US peer from SEC EDGAR.

    8-K filings are where executive-appointment/departure disclosures (Item 5.02)
    show up; DEF 14A (proxy) carries named-executive-officer detail; 10-K is the
    annual report. This does not filter to Item 5.02 specifically (that requires
    fetching and parsing each 8-K body, out of scope for a list-level pull) — the
    UI should say "browse these filings for leadership-change 8-Ks" rather than
    claiming every listed 8-K is a personnel move.
    """
    retrieved_at = _now_iso()
    cik = SEC_CIK_BY_PEER.get(peer_name)
    if not cik:
        return {
            "ok": False,
            "retrieved_at": retrieved_at,
            "source": "SEC EDGAR",
            "source_url": None,
            "data": None,
            "error": f"No known SEC CIK mapped for '{peer_name}' (may not be a standalone SEC filer).",
        }
    url = f"{SEC_SUBMISSIONS_BASE}/CIK{cik}.json"
    try:
        raw = _http_get_json(url)
        recent = raw["filings"]["recent"]
        rows = []
        for i, form in enumerate(recent["form"]):
            if form in forms and len(rows) < limit:
                accession = recent["accessionNumber"][i].replace("-", "")
                doc = recent["primaryDocument"][i]
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
                rows.append(
                    {
                        "form": form,
                        "filed": recent["filingDate"][i],
                        "url": filing_url,
                    }
                )
        return {
            "ok": True,
            "retrieved_at": retrieved_at,
            "source": f"SEC EDGAR — {raw.get('name', peer_name)} (CIK {cik})",
            "source_url": url,
            "data": rows,
            "error": None,
        }
    except (URLError, HTTPError, TimeoutError, ValueError, KeyError) as e:
        return {
            "ok": False,
            "retrieved_at": retrieved_at,
            "source": f"SEC EDGAR — {peer_name}",
            "source_url": url,
            "data": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Sources tested and confirmed NOT viable as company-specific live feeds —
# documented here (not silently dropped) so nobody re-attempts the same dead
# end without re-reading this:
#
# - Job Bank Canada open data (monthly all-postings CSV): real, no-key, ~50MB/
#   month, but has NO employer-name column — only NOC code/title/location/
#   salary. Usable for NATIONAL labour-market stats (e.g. total actuarial
#   postings by NOC), NOT for "how many postings does Manulife have."
# - Adzuna API: requires a registered app_id + app_key (free tier still needs
#   signup). Confirmed via a live AUTH_FAIL response — not wired in until a
#   real key is supplied by the user.
# - SEDAR+ (Canadian filings), Ontario Mass Termination Notices, WARN Act
#   notices: no public machine-readable API found; these stay manual-paste.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trade-press RSS feeds — confirmed real, robots.txt-permitted, no key needed.
# executivemoves.com (no hyphen) is a parked domain-for-sale page, NOT the real
# site — confirmed live and rejected. The real site uses a hyphen.
# ---------------------------------------------------------------------------

TRADE_PRESS_FEEDS = {
    "Executive Moves (Insurance category)": "https://executive-moves.com/category/industry/insurance/feed/",
    "Insurance Edge": "https://insurance-edge.net/feed/",
}


def fetch_trade_press_mentions(peer_aliases: dict[str, list[str]] | None = None, limit_per_feed: int = 30) -> dict:
    """Fetch both trade-press RSS feeds and return EVERY item — industry-wide,
    not filtered to any peer set. Each item is tagged with `matched_peers`
    (which locked peers, if any, it mentions — empty list if none) so callers
    can decide what counts as peer-relevant without the feed silently
    dropping every other company's real, current executive move.

    peer_aliases is optional: {canonical_peer_name: [alias1, alias2, ...]}.
    Pass None/{} to skip peer-tagging and just get every item.
    """
    retrieved_at = _now_iso()
    all_items = []
    feed_errors = {}

    flat_aliases = [
        (canonical, alias.lower()) for canonical, aliases in (peer_aliases or {}).items() for alias in aliases
    ]

    for feed_name, feed_url in TRADE_PRESS_FEEDS.items():
        try:
            raw_xml = _http_get_text(feed_url)
            root = _ET.fromstring(raw_xml)
            items = root.findall(".//item")[:limit_per_feed]
            for item in items:
                title_el = item.find("title")
                desc_el = item.find("description")
                link_el = item.find("link")
                pubdate_el = item.find("pubDate")

                title = title_el.text if title_el is not None else ""
                description = desc_el.text if desc_el is not None else ""
                link = link_el.text if link_el is not None else ""
                pubdate = pubdate_el.text if pubdate_el is not None else ""

                haystack = f"{title} {description}".lower()
                matched_peers = sorted({canonical for canonical, alias in flat_aliases if alias in haystack})

                all_items.append(
                    {
                        "feed": feed_name,
                        "title": title,
                        "link": link,
                        "pub_date": pubdate,
                        "matched_peers": matched_peers,
                    }
                )
        except (URLError, HTTPError, TimeoutError, _ET.ParseError) as e:
            feed_errors[feed_name] = str(e)

    return {
        "ok": not feed_errors or len(feed_errors) < len(TRADE_PRESS_FEEDS),
        "retrieved_at": retrieved_at,
        "source": "Trade-press RSS feeds (" + ", ".join(TRADE_PRESS_FEEDS.keys()) + ")",
        "source_url": None,
        "data": all_items,
        "feed_errors": feed_errors,
        "error": "; ".join(f"{k}: {v}" for k, v in feed_errors.items()) if feed_errors else None,
    }


JOB_BANK_NOTE = (
    "Job Bank Canada's open-data postings file has no employer-name field — it can only "
    "answer national/NOC-level questions (e.g. 'how many actuarial postings exist in Canada "
    "this month'), not company-specific ones. Company-level posting counts are not available "
    "in this tool — career pages are bot-blocked and no other free API surfaces them."
)

ADZUNA_NOTE = (
    "Adzuna requires a registered app_id/app_key (free tier, but needs signup) that this "
    "session does not have. Not wired in — provide credentials to enable it."
)
