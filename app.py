"""
Talent Moves Tracker
---------------------
Competitive talent-intelligence tool for Manulife People Analytics.

Fully live, no manual data entry anywhere: StatCan WDS, BLS, and SEC EDGAR
public APIs, plus two trade-press RSS feeds (Executive Moves, Insurance Edge)
filtered to the 10 locked peer companies and parsed automatically into the
move table. Sections with no viable free/no-key live source (Workforce &
Talent Strategy; company-specific job-posting counts) are left honestly
blank rather than faked or backed by a paste box — see the Methodology tab
for the full list of sources tested and rejected.

Every section of the UI is labeled with which live source it's on. Nothing
here blends a "Disclosed fact" (from a named filing/article) with an
"Inferred signal" (a pattern we noticed) without saying which is which.

See live_sources.py for the API fetch functions and a record of which
candidate sources were tested and rejected (no employer field, no free key,
bot-blocked), so nobody re-attempts the same dead end blind.
"""

import datetime as dt
import os
import re
from dataclasses import dataclass

import anthropic
import pandas as pd
import streamlit as st

import live_sources

CLAUDE_MODEL = "claude-sonnet-4-5"
CLAUDE_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}

ASK_SYSTEM_PROMPT = """You are a research assistant for Manulife's People Analytics team, answering \
ad-hoc questions about competitive talent movement among a locked set of insurance peers (Canada: Sun \
Life, Canada Life, RBC Insurance, iA Financial Group, Empire Life; US: John Hancock, MetLife, Prudential \
Financial, Lincoln Financial, Principal Financial).

You will be given LIVE DATA CONTEXT below, pulled moments ago from real sources (SEC EDGAR filings, \
StatCan/BLS labour data, and trade-press RSS headlines). Rules:
- Prefer answering from the given context and cite it (name the source, e.g. "per SEC EDGAR" or "per \
Executive Moves").
- Only use the web_search tool if the context doesn't cover the question, or the user is asking about \
something clearly outside this session's live pull (e.g. a different company, older news, or a broader \
industry question).
- Never fabricate a specific person, title, or date. If you don't have it and can't find it, say so \
plainly rather than guessing.
- Distinguish clearly between what's a "Disclosed fact" (from a named filing/article) and any inference \
you're making on top of it.
- Keep answers concise and structured — this is an analyst tool, not a chat companion.
- This tool does not track or infer anything about non-executive/individual employees; keep answers \
scoped to publicly disclosed leadership moves and labour-market data.
"""

st.set_page_config(
    page_title="Talent Moves Tracker",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

HOME_COMPANY = "Manulife"

# ---------------------------------------------------------------------------
# LOCKED peer set — do not change without asking (per governance decision).
# Two separate markets; reports never merge them into one list.
# ---------------------------------------------------------------------------

LOCKED_PEER_SETS: dict[str, list[str]] = {
    "Canada": ["Sun Life", "Canada Life", "RBC Insurance", "iA Financial Group", "Empire Life"],
    "US": ["John Hancock", "MetLife", "Prudential Financial", "Lincoln Financial", "Principal Financial"],
}

# Aliases used only for text-matching in pasted content (extract_companies),
# not for changing the locked peer list itself.
COMPANY_ALIASES: dict[str, list[str]] = {
    "Sun Life": ["Sun Life", "Sun Life Financial"],
    "Canada Life": ["Canada Life", "Great-West Lifeco", "Great-West Life", "GWL"],
    "RBC Insurance": ["RBC Insurance", "RBC"],
    "iA Financial Group": ["iA Financial Group", "iA Financial", "Industrial Alliance"],
    "Empire Life": ["Empire Life"],
    "John Hancock": ["John Hancock"],
    "MetLife": ["MetLife"],
    "Prudential Financial": ["Prudential Financial", "Prudential"],
    "Lincoln Financial": ["Lincoln Financial", "Lincoln National"],
    "Principal Financial": ["Principal Financial", "Principal Financial Group"],
    HOME_COMPANY: ["Manulife", "Manulife Financial"],
}

ALL_KNOWN_COMPANIES = sorted(
    {alias for aliases in COMPANY_ALIASES.values() for alias in aliases}, key=len, reverse=True
)


def canonical_company(matched_alias: str) -> str:
    for canonical, aliases in COMPANY_ALIASES.items():
        if matched_alias.lower() in {a.lower() for a in aliases}:
            return canonical
    return matched_alias


# ---------------------------------------------------------------------------
# Vocabulary for title-seniority and function classification
# ---------------------------------------------------------------------------

SENIORITY_RANK = [
    (100, [r"\bchief executive officer\b", r"\bceo\b", r"\bpresident and ceo\b"]),
    (95, [r"\bpresident\b(?!.{0,20}vice)"]),
    (90, [r"\bvice[\s-]?chair\b", r"\bexecutive vice[\s-]?chair\b"]),
    (85, [r"\bchief\s+\w[\w\s&/]*\s+officer\b", r"\bc[a-z]o\b"]),
    (80, [r"\bgroup head\b", r"\bhead of group\b"]),
    (75, [r"\bexecutive vice president\b", r"\bevp\b"]),
    (70, [r"\bglobal head\b"]),
    (65, [r"\bsenior vice president\b", r"\bsvp\b"]),
    (60, [r"\bmanaging director\b", r"\bmd\b"]),
    (55, [r"\bvice president\b", r"\bvp\b"]),
    (50, [r"\bhead of\b"]),
    (45, [r"\bsenior director\b"]),
    (40, [r"\bdirector\b"]),
    (35, [r"\bsenior manager\b"]),
    (30, [r"\bmanager\b"]),
    (20, [r"\bassociate\b"]),
    (10, [r"\banalyst\b"]),
]

SCOPE_EXPANSION_PATTERNS = [
    r"\bglobal\b", r"\bworldwide\b", r"\benterprise[\s-]?wide\b", r"\ball[\s-]?markets\b",
    r"\bnorth america\b", r"\binternational\b", r"\bgroup[\s-]?wide\b",
    r"\band\b.{0,40}\b(chief|head|officer)\b",
]
SCOPE_NARROWING_HINTS = [r"\binterim\b", r"\bacting\b"]

# Provisional role-family tags — NOT a fixed taxonomy per governance rule 4.
# This keyword map is a placeholder used only until enough real postings/moves
# accumulate to justify grouping by NOC (Canada) / O*NET-SOC (US) codes on
# Manulife's own current postings. Treat "Function" values below as rough,
# interim labels, not a settled schema.
FUNCTION_KEYWORDS = {
    "Actuarial": ["actuary", "actuarial", "fsa", "fcia", "asa"],
    "Risk": [
        "risk", "credit risk", "market risk", "operational risk", "compliance",
        "chief risk officer", "cro", "internal audit", "audit", "regulatory affairs",
        "aml", "anti-money laundering", "financial crime",
    ],
    "Finance": [
        "finance", "cfo", "chief financial officer", "treasury", "treasurer",
        "controller", "investor relations", "accounting", "financial planning",
        "fp&a", "tax",
    ],
    "HR/People": [
        "human resources", "hr", "chief human resources officer", "chro",
        "people", "talent", "chief people officer", "culture", "diversity",
        "total rewards", "compensation and benefits",
    ],
    "Technology": [
        "technology", "chief technology officer", "cto", "chief information officer",
        "cio", "engineering", "digital", "data", "chief data officer", "cdo",
        "artificial intelligence", "ai", "cybersecurity", "chief information security officer",
        "ciso", "infrastructure", "software",
    ],
    "Operations": [
        "operations", "chief operating officer", "coo", "supply chain",
        "operational excellence", "service delivery", "branch operations",
        "shared services",
    ],
    "Legal": [
        "legal", "general counsel", "chief legal officer", "clo", "corporate secretary",
        "litigation", "regulatory counsel",
    ],
    "Marketing": [
        "marketing", "chief marketing officer", "cmo", "communications",
        "chief communications officer", "brand", "public relations", "pr ",
    ],
}

# ---------------------------------------------------------------------------
# Direction vocabulary
# ---------------------------------------------------------------------------

INBOUND_VERBS = [
    "joins", "joined", "joining", "has joined", "will join", "hires", "hired",
    "appoints", "appointed", "names", "named", "welcomes", "welcomed",
    "brings on", "brought on", "recruits", "recruited", "onboards", "onboarded",
    "lands at", "landed at", "signs on", "signed on", "comes aboard",
]
OUTBOUND_VERBS = [
    "departs", "departed", "departing", "leaves", "left", "leaving",
    "steps down", "stepped down", "resigns", "resigned", "resigning",
    "exits", "exited", "exiting", "retires", "retired", "retiring",
    "to leave", "parts ways", "parted ways", "is leaving", "has left",
    "moves on from", "moved on from", "departs from", "out at",
]
INTERNAL_VERBS = [
    "promotes", "promoted", "promoting", "elevates", "elevated",
    "moves to", "moved to", "moves into", "moved into", "shifts to", "shifted to",
    "transitions to", "transitioned to", "expands role", "adds", "added the role",
    "takes on", "took on", "steps into", "stepped into", "internal promotion",
    "succeeds", "succeeded",
]

DATE_PATTERN = re.compile(
    r"("
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:effective\s+)?(?:Q[1-4]\s+\d{4})"
    r")",
    re.IGNORECASE,
)

SOURCE_PATTERN = re.compile(
    r"\(([^()]*(?:according to|per|source|reported by|via)[^()]*)\)"
    r"|(?:according to|per|source:|reported by|via)\s+([A-Z][\w&.\s]{2,60})",
    re.IGNORECASE,
)

SAMPLE_TEXT = """
Sun Life named Priya Anand as Chief Actuary, effective March 1, 2026, according to a company press release.
John Hancock's Head of Technology, Marcus Wei, has left the firm after six years; he departed on Feb 14, 2026 (Insurance Business).
Canada Life promoted Fatima Khan from Director, Compliance to Head of Compliance, expanding her mandate to cover the US and Canadian businesses, effective Jan 15, 2026 (People Matters).
Prudential Financial named Robert Ellis as its new Chief Financial Officer, succeeding the retiring Anne Ostrander, according to a company press release dated 2026-01-10.
RBC Insurance's Head of Marketing, Sarah Cole, is leaving to join a fintech startup, according to MarketScreener.
Principal Financial appointed David Nguyen as Executive Vice President, Global Technology, per a statement issued April 2026.
Empire Life's Chief Operating Officer, Michael Turner, steps down after a decade in the role; the company has not named a successor (InsuranceNewsNet).
iA Financial Group elevated Marc Bouchard to Senior Director, Risk Management from Manager, Credit Risk, internal promotion announced Q2 2026 (Executive Moves).
Lincoln Financial's General Counsel, Laura Bennett, resigned effective June 30, 2026, to pursue an opportunity outside the firm.
MetLife hired Kevin Osei as Managing Director, Data & Analytics after five years at a Canadian pension fund, effective May 2026 (company announcement).
"""

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def split_into_candidate_lines(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    lines: list[str] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", raw_line)
        for s in sentences:
            s = s.strip(" -•\t")
            if len(s) > 15:
                lines.append(s)
    return lines


def _find_verb_hit(text_lower: str, verbs: list[str]) -> str | None:
    for v in verbs:
        if v in text_lower:
            return v
    return None


def classify_direction(line: str) -> tuple[str, str | None]:
    text_lower = line.lower()
    inbound_hit = _find_verb_hit(text_lower, INBOUND_VERBS)
    outbound_hit = _find_verb_hit(text_lower, OUTBOUND_VERBS)
    internal_hit = _find_verb_hit(text_lower, INTERNAL_VERBS)

    if outbound_hit and re.search(r"\bto\s+(join|become|take|lead)\b", text_lower):
        return "Inbound", outbound_hit + " ... to join"

    if inbound_hit and outbound_hit:
        inbound_pos = text_lower.find(inbound_hit)
        outbound_pos = text_lower.find(outbound_hit)
        return ("Inbound", inbound_hit) if inbound_pos <= outbound_pos else ("Outbound", outbound_hit)

    if internal_hit:
        return "Internal", internal_hit
    if inbound_hit:
        return "Inbound", inbound_hit
    if outbound_hit:
        return "Outbound", outbound_hit

    if re.search(r"\bfrom\b.{3,60}\bto\b", text_lower) and re.search(r"\b(role|title|position)\b", text_lower):
        return "Internal", "from...to (role change)"

    if re.search(r"\bbecomes\b", text_lower):
        if re.search(r"\bat\b|\bwithin\b|\binternal(ly)?\b", text_lower):
            return "Internal", "becomes (internal)"
        return "Inbound", "becomes"

    companies_found = extract_companies(line)
    if len(companies_found) >= 1 and re.search(r"\bnew\b.{0,15}\b(role|position|title)\b", text_lower):
        return "Inbound", "new role (fallback)"

    return "Unclassified", None


def extract_companies(line: str) -> list[str]:
    found = []
    for company in ALL_KNOWN_COMPANIES:
        pattern = r"\b" + re.escape(company) + r"\b"
        if re.search(pattern, line, re.IGNORECASE):
            found.append(canonical_company(company))
    deduped: list[str] = []
    for c in found:
        if c not in deduped:
            deduped.append(c)
    return deduped


# Verbs where the COMPANY is the grammatical subject ("<Company> appoints <Person>").
COMPANY_SUBJECT_VERBS = (
    r"(?i:appoints?|appointed|names?|named|promotes?|promoted|elevates?|elevated|"
    r"welcomes?|welcomed|hires?|hired|recruits?|recruited)"
)
# Verbs where the PERSON is the grammatical subject ("<Person> joins <Company>").
PERSON_SUBJECT_VERBS = r"(?i:joins?|joined)"


def extract_headline_company(line: str) -> str:
    """Extract the organization name from a headline-style move announcement,
    for ANY company — not limited to the 10 locked peers. Checks the
    person-subject shape first ("<Person> joins <Company>") since otherwise
    the company-subject pattern would wrongly treat the person's name as the
    company (both patterns can match text that ends in "joins").
    """
    m2 = re.search(r"\bjoins?\s+([A-Z][\w&.,'’\-\s]{1,60}?)\s+(?:as\b|,|\.|$)", line)
    if m2:
        return m2.group(1).strip()

    m = re.match(r"^([A-Z][\w&.,'’\-\s]{1,60}?)(?:'s)?\s+" + COMPANY_SUBJECT_VERBS + r"\b", line)
    if m:
        return m.group(1).strip().rstrip("'").rstrip(",")

    return ""


ROLE_WORDS = {
    "head", "chief", "president", "director", "officer", "manager", "vice",
    "senior", "executive", "global", "group", "bank", "financial", "corp",
    "corporation", "inc", "the", "board", "committee", "general", "counsel",
    "life", "insurance", "hancock",
}


def _trim_trailing_role_words(candidate: str) -> str:
    """Cut a candidate off at the first title/role word a greedy capture pulled
    in, e.g. "John Smith Chief Risk" -> "John Smith" (the title starts at
    "Chief", which is not necessarily the last word captured)."""
    words = candidate.split()
    for i, w in enumerate(words):
        if w.rstrip(",").lower().rstrip("'s") in ROLE_WORDS:
            return " ".join(words[:i])
    return candidate


def _looks_like_person_name(candidate: str) -> bool:
    if not candidate:
        return False
    words = candidate.split()
    if len(words) < 2:
        return False
    if any(w.rstrip(",").lower().rstrip("'s") in ROLE_WORDS for w in words):
        return False
    if any(w.endswith("'s") for w in words):
        return False
    for company in ALL_KNOWN_COMPANIES:
        if company.lower() in candidate.lower():
            return False
    return True


def extract_person(line: str) -> str:
    match = re.search(
        r"\b(?i:appoints?|appointed|names?|named|promotes?|promoted|elevates?|elevated|"
        r"welcomes?|welcomed|hires?|hired|recruits?|recruited)\s+"
        r"([A-Z][a-zA-Z.\'-]+(?:\s+[A-Z][a-zA-Z.\'-]+){1,3})",
        line,
    )
    if match:
        candidate = _trim_trailing_role_words(match.group(1).strip())
        if _looks_like_person_name(candidate):
            return candidate

    for match in re.finditer(r",\s*([A-Z][a-zA-Z.\'-]+(?:\s+[A-Z][a-zA-Z.\'-]+){1,2})\s*,", line):
        candidate = _trim_trailing_role_words(match.group(1).strip())
        if _looks_like_person_name(candidate):
            return candidate

    match = re.match(r"^([A-Z][a-zA-Z.\'-]+(?:\s+[A-Z][a-zA-Z.\'-]+){1,3})", line.strip())
    if match:
        candidate = match.group(1).strip()
        for company in ALL_KNOWN_COMPANIES:
            candidate = re.sub(r"\s+" + re.escape(company) + r"$", "", candidate, flags=re.IGNORECASE)
        candidate = _trim_trailing_role_words(candidate.strip())
        if _looks_like_person_name(candidate):
            return candidate

    return ""


def extract_date(line: str) -> str:
    match = DATE_PATTERN.search(line)
    return match.group(1) if match else ""


def extract_source(line: str) -> str:
    match = SOURCE_PATTERN.search(line)
    if match:
        return (match.group(1) or match.group(2) or "").strip()
    return ""


def _extract_titles(line: str) -> tuple[str, str]:
    m = re.search(
        r"from\s+([A-Z][\w,&/\s-]{2,60}?)\s+to\s+([A-Z][\w,&/\s-]{2,60}?)(?:[,.;]|\s+(?:at|effective|after|since|starting)\b|$)",
        line,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    new_title = ""
    prior_title = ""

    m_new = re.search(
        r"\bas\s+(?:its\s+|the\s+|a\s+|new\s+)*([A-Z][\w,&/\s-]{2,60}?)(?:[,.;]|\s+(?:effective|after|since|starting|succeeding)\b|$)",
        line,
    )
    if m_new:
        new_title = m_new.group(1).strip()

    m_prior = re.search(
        r"(?:previously|formerly)\s+(?:its\s+|the\s+|a\s+)?([A-Z][\w,&/\s-]{2,60}?)(?:\s+at\b|\s+of\b|[,.;]|$)",
        line,
        re.IGNORECASE,
    )
    if m_prior:
        prior_title = m_prior.group(1).strip()

    if not new_title:
        m_generic = re.search(r",\s*([A-Z][\w,&/\s-]{2,60}?)\s+(?:of|at)\s+[A-Z]", line)
        if m_generic:
            title_guess = m_generic.group(1).strip()
            if _find_verb_hit(line.lower(), OUTBOUND_VERBS):
                prior_title = prior_title or title_guess
            else:
                new_title = new_title or title_guess

    return prior_title, new_title


def _seniority_score(title: str) -> int:
    title_lower = title.lower()
    for rank, patterns in SENIORITY_RANK:
        for p in patterns:
            if re.search(p, title_lower):
                return rank
    return 0


def classify_title_delta(prior_title: str, new_title: str) -> str:
    if not new_title:
        return ""
    if not prior_title:
        if any(re.search(p, new_title.lower()) for p in SCOPE_EXPANSION_PATTERNS):
            return "Expanded Remit"
        return "Lateral"

    prior_rank = _seniority_score(prior_title)
    new_rank = _seniority_score(new_title)

    if new_rank > prior_rank:
        return "Promotion"

    scope_expanded = any(re.search(p, new_title.lower()) for p in SCOPE_EXPANSION_PATTERNS)
    scope_narrowed = any(re.search(p, new_title.lower()) for p in SCOPE_NARROWING_HINTS)

    if new_rank == prior_rank:
        if scope_expanded and not scope_narrowed:
            return "Expanded Remit"
        return "Lateral"

    if scope_expanded:
        return "Expanded Remit"
    return "Lateral"


def classify_function(title: str, full_line: str = "") -> str:
    search_text = f"{title} {full_line}".lower()
    scores = {}
    for func, keywords in FUNCTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in search_text)
        if count:
            scores[func] = count
    if not scores:
        return "Other"
    return max(scores, key=scores.get)


@dataclass
class Move:
    person: str = ""
    direction: str = "Unclassified"
    prior_title: str = ""
    new_title: str = ""
    prior_company: str = ""
    new_company: str = ""
    date: str = ""
    source: str = ""
    title_delta: str = ""
    function: str = ""
    raw_text: str = ""
    input_method: str = "Manual paste"


MOVE_COLUMNS = [
    "Person", "Direction", "Prior Title", "New Title", "Prior Company",
    "New Company", "Date", "Source", "Title Delta", "Function", "Raw Text", "Input Method",
]


def _build_move(line: str, input_method: str, default_source: str) -> Move:
    """Build a single Move from one already-isolated line/headline. Does NOT
    split the text further — callers decide what counts as one unit (a
    multi-sentence paste needs sentence-splitting; a single RSS headline is
    already one unit and must NOT be re-split, since headlines routinely
    contain abbreviations like "Robert E. Barlow" that a sentence-boundary
    splitter would misread as two sentences)."""
    direction, _ = classify_direction(line)
    prior_title, new_title = _extract_titles(line)
    companies = extract_companies(line)
    person = extract_person(line)
    move_date = extract_date(line)
    source = extract_source(line) or default_source

    prior_company, new_company = "", ""
    if companies:
        if direction == "Outbound":
            prior_company = companies[0]
        elif direction == "Inbound":
            new_company = companies[0]
        else:
            if len(companies) >= 2:
                prior_company, new_company = companies[0], companies[1]
            else:
                prior_company = new_company = companies[0]

    title_delta = classify_title_delta(prior_title, new_title)
    function = classify_function(new_title or prior_title, line)

    return Move(
        person=person, direction=direction, prior_title=prior_title, new_title=new_title,
        prior_company=prior_company, new_company=new_company, date=move_date, source=source,
        title_delta=title_delta, function=function, raw_text=line, input_method=input_method,
    )


def _moves_to_df(moves: list["Move"]) -> pd.DataFrame:
    if not moves:
        return pd.DataFrame(columns=MOVE_COLUMNS)
    return pd.DataFrame(
        [
            {
                "Person": m.person, "Direction": m.direction, "Prior Title": m.prior_title,
                "New Title": m.new_title, "Prior Company": m.prior_company, "New Company": m.new_company,
                "Date": m.date, "Source": m.source, "Title Delta": m.title_delta,
                "Function": m.function, "Raw Text": m.raw_text, "Input Method": m.input_method,
            }
            for m in moves
        ]
    )


def parse_single_headline(headline: str, input_method: str = "Manual paste", default_source: str = "") -> pd.DataFrame:
    """Parse exactly one headline/title into exactly one move row, with no
    sentence-splitting — use this for RSS headlines and similar single-unit
    text. See parse_moves for multi-sentence pasted paragraphs."""
    return _moves_to_df([_build_move(headline.strip(), input_method, default_source)])


def parse_moves(text: str, input_method: str = "Manual paste", default_source: str = "") -> pd.DataFrame:
    """Parse raw (possibly multi-sentence) text into structured moves.

    input_method tags every resulting row (e.g. "Manual paste" or
    "Auto (Executive Moves)") so the review table always shows where a row
    came from — auto-fetched rows still need a human glance before they're
    treated as fact, per governance, but they no longer require re-typing.
    default_source is used when a line has no inline source attribution of
    its own (e.g. an RSS headline whose "source" is simply the feed itself).
    """
    lines = split_into_candidate_lines(text)
    moves = [_build_move(line, input_method, default_source) for line in lines]
    return _moves_to_df(moves)


def fetch_auto_trade_press_moves() -> tuple[pd.DataFrame, dict]:
    """Fetch both trade-press RSS feeds live and parse EVERY item into a
    structured move row automatically — industry-wide, any company, not
    restricted to the 10 locked peers or to a country. No paste step.

    Company is extracted generically (extract_headline_company) so a real,
    current move at any insurance-industry company shows up, not just the
    rare item that happens to name one of our 10 peers. A "Matched Locked
    Peer" column still flags rows relevant to our peer set specifically, for
    the Live Report tab to filter on — but nothing is hidden from this table.

    Every row is tagged Input Method = "Auto (<feed name>)" and Source = the
    article link. Returns (moves_df, raw_fetch_result).
    """
    result = cached_fetch_trade_press()
    frames = []
    for item in result["data"]:
        title = item["title"]
        row_df = parse_single_headline(title, input_method=f"Auto ({item['feed']})", default_source=item["link"])
        if row_df.empty or row_df.at[0, "Direction"] == "Unclassified":
            # Unclassified means no hiring/departure/promotion verb was found at
            # all — almost always a non-personnel story (deals, research, events)
            # rather than a genuine move with low-confidence classification, so
            # it's dropped here rather than shown as a move.
            continue

        company = extract_headline_company(title)
        direction = row_df.at[0, "Direction"]
        if company:
            if direction == "Outbound":
                row_df.at[0, "Prior Company"] = company
            else:
                row_df.at[0, "New Company"] = company

        if not row_df.at[0, "Date"]:
            row_df.at[0, "Date"] = item["pub_date"]
        row_df["Matched Locked Peer"] = ", ".join(item["matched_peers"])
        row_df["Feed"] = item["feed"]
        frames.append(row_df)

    out_columns = MOVE_COLUMNS + ["Matched Locked Peer", "Feed"]
    if not frames:
        return pd.DataFrame(columns=out_columns), result
    return pd.concat(frames, ignore_index=True), result


# ---------------------------------------------------------------------------
# So What (rule-based counts/proportions) — used inside Talent Flow Detail
# ---------------------------------------------------------------------------


def build_sourcing_section(df: pd.DataFrame) -> str:
    outbound = df[df["Direction"] == "Outbound"]
    if outbound.empty:
        return "_No outbound moves recorded — no peer-sourcing signal yet._"

    lines = [f"**{len(outbound)}** outbound move(s) recorded, i.e. talent this peer set is releasing.", ""]
    by_company = outbound["Prior Company"].replace("", "Unspecified").value_counts()
    lines.append("**By company releasing talent:**")
    for company, count in by_company.items():
        funcs = outbound[outbound["Prior Company"].replace("", "Unspecified") == company]["Function"]
        func_summary = ", ".join(f"{f} ({c})" for f, c in funcs.value_counts().items())
        lines.append(f"- {company}: {count} move(s) — {func_summary}")

    lines.append("")
    lines.append("**By function/level:**")
    by_func = outbound["Function"].value_counts()
    for func, count in by_func.items():
        titles = outbound[outbound["Function"] == func]["Prior Title"].replace("", "(title n/a)")
        lines.append(f"- {func}: {count} move(s) — e.g. {', '.join(titles.head(3))}")

    return "\n".join(lines)


def build_retention_section(df: pd.DataFrame) -> str:
    internal = df[df["Direction"] == "Internal"]
    if df.empty:
        return "_No moves recorded yet._"

    total_by_func = df["Function"].value_counts()
    internal_by_func = internal["Function"].value_counts()

    lines = ["**Internal-promotion share by function** (bench-strength signal):", ""]
    flagged = []
    for func in sorted(set(total_by_func.index) | set(internal_by_func.index)):
        total = total_by_func.get(func, 0)
        internal_count = internal_by_func.get(func, 0)
        share = (internal_count / total * 100) if total else 0
        lines.append(f"- {func}: {internal_count}/{total} moves internal ({share:.0f}%)")
        if total >= 2 and share >= 60:
            flagged.append(func)

    lines.append("")
    if flagged:
        lines.append(
            f"⚠️ **Bench-strength-building risk flag:** {', '.join(flagged)} show high internal-promotion "
            "share (≥60% with 2+ tracked moves)."
        )
    else:
        lines.append("No function currently shows a disproportionately high internal-promotion share.")

    return "\n".join(lines)


def build_benchmarking_section(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No moves recorded yet._"

    lines = ["**Inbound / Internal / Outbound mix:**", ""]
    mix = df["Direction"].value_counts()
    total = len(df)
    for direction in ["Inbound", "Internal", "Outbound", "Unclassified"]:
        count = mix.get(direction, 0)
        pct = (count / total * 100) if total else 0
        lines.append(f"- {direction}: {count} ({pct:.0f}%)")

    lines.append("")
    lines.append("**Title Delta distribution:**")
    delta_mix = df["Title Delta"].replace("", "Unclassified").value_counts()
    for delta, count in delta_mix.items():
        pct = (count / total * 100) if total else 0
        lines.append(f"- {delta}: {count} ({pct:.0f}%)")

    lines.append("")
    lines.append("**External-hiring skew by function:**")
    flagged = []
    for func in df["Function"].unique():
        func_df = df[df["Function"] == func]
        inbound_share = (func_df["Direction"] == "Inbound").mean() * 100
        lines.append(f"- {func}: {inbound_share:.0f}% of tracked moves are inbound (external hires)")
        if len(func_df) >= 2 and inbound_share >= 70:
            flagged.append(func)

    if flagged:
        lines.append("")
        lines.append(f"⚠️ **Disproportionate external hiring:** {', '.join(flagged)} — 70%+ external.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live snapshot report (flag + one-liner per peer, Bottom Line synthesis)
# ---------------------------------------------------------------------------

FLAG_RED, FLAG_GREEN, FLAG_GRAY = "🔴", "🟢", "⚪"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_statcan():
    return live_sources.fetch_statcan_lfs()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_bls():
    return live_sources.fetch_bls_series()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_sec(peer_name: str):
    return live_sources.fetch_sec_recent_filings(peer_name)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch_trade_press():
    return live_sources.fetch_trade_press_mentions(COMPANY_ALIASES)


def compute_company_flag(peer: str, moves_df: pd.DataFrame | None, sec_result: dict | None) -> tuple[str, str]:
    """Flag + one-liner for one peer.

    Disclosed-fact basis only: SEC filing recency (US peers) and/or manually
    reviewed Talent Flow Detail moves for this pull. Never infers "concerning"
    sentiment from an unread filing list alone — a bare 8-K listing gets a
    neutral flag with a pointer to go read it, not an assumed direction.
    """
    notes = []
    flag = FLAG_GRAY

    if moves_df is not None and not moves_df.empty and "Matched Locked Peer" in moves_df.columns:
        peer_moves = moves_df[moves_df["Matched Locked Peer"].fillna("").str.contains(peer, regex=False)]
        if not peer_moves.empty:
            outbound_n = (peer_moves["Direction"] == "Outbound").sum()
            inbound_n = (peer_moves["Direction"] == "Inbound").sum()
            internal_n = (peer_moves["Direction"] == "Internal").sum()
            if outbound_n > 0:
                flag = FLAG_RED
                notes.append(
                    f"{outbound_n} outbound move(s) found in live trade-press feeds this pull (Disclosed fact)."
                )
            elif inbound_n or internal_n:
                flag = FLAG_GREEN
                notes.append(
                    f"{inbound_n} inbound / {internal_n} internal move(s) found in live trade-press feeds this pull (Disclosed fact)."
                )

    if sec_result is not None:
        if sec_result["ok"] and sec_result["data"]:
            most_recent = sec_result["data"][0]
            filed = dt.date.fromisoformat(most_recent["filed"])
            days_since = (dt.date.today() - filed).days
            if days_since <= 45:
                if flag == FLAG_GRAY:
                    flag = FLAG_GREEN
                notes.append(
                    f"Recent {most_recent['form']} filed {most_recent['filed']} ({days_since}d ago) — "
                    f"review for Item 5.02 leadership disclosures. (Disclosed fact, SEC EDGAR, "
                    f"retrieved {sec_result['retrieved_at']})"
                )
            else:
                notes.append(
                    f"Most recent tracked SEC filing: {most_recent['form']} on {most_recent['filed']} "
                    f"— no near-term signal. (Disclosed fact, SEC EDGAR)"
                )
        elif sec_result["ok"]:
            notes.append("No recent 8-K/10-K/DEF 14A filings found on SEC EDGAR. (Disclosed fact)")
        else:
            notes.append(f"SEC EDGAR lookup unavailable for this peer: {sec_result['error']}")

    if not notes:
        notes.append("No live signal this pull — no matching trade-press headline or recent SEC filing found.")

    return flag, " ".join(notes)


def build_bottom_line(market: str, flags: dict[str, str], macro_result: dict) -> str:
    red_peers = [p for p, f in flags.items() if f == FLAG_RED]
    green_peers = [p for p, f in flags.items() if f == FLAG_GREEN]

    if red_peers:
        headline = f"{', '.join(red_peers)} show outbound talent signals this pull — worth a closer read."
    elif green_peers:
        headline = f"{', '.join(green_peers)} show notable activity this pull (see each line for whether that's a logged move or a filing to review); no outbound signal logged."
    else:
        headline = "No peer shows a live signal this pull — this is a quiet snapshot, not a data gap."

    macro_note = ""
    if macro_result.get("ok"):
        macro_note = " Sector-wide labour context is available in the Labour-Market Research tab (live, timestamped)."

    return f"{headline}{macro_note}"


# ---------------------------------------------------------------------------
# Governance / Methodology text
# ---------------------------------------------------------------------------

METHODOLOGY_TEXT = f"""
### Governance principles
- **Only public, first-party, or government data**: company filings, press releases,
  career pages, government labour data.
- Named individuals are in-scope **only** for leadership/executive appointments already
  disclosed in proxy circulars, 10-Ks/annual reports, or company newsroom releases.
  This tool never scrapes, infers, or displays individual-level employee data.
- Every figure is tagged **"Disclosed fact"** (directly stated in a named source) or
  **"Inferred signal"** (a pattern this tool noticed, e.g. a title-seniority comparison) —
  never blended without that tag.
- Every data point carries a **source name, link, and retrieval timestamp**. No exceptions.
- **Permanently excluded**: LinkedIn scraping/API, individual profile monitoring,
  forum/rumor sources (Blind, TheLayoff.com, individual Glassdoor reviews) as citable facts.

### Peer set (locked)
- **Canada (5):** {", ".join(LOCKED_PEER_SETS["Canada"])}
- **US (5):** {", ".join(LOCKED_PEER_SETS["US"])}

Canada and US reports are always shown separately — never merged into one cross-market list.

### Live data sources (tested and wired this session)
- **Statistics Canada — Labour Force Survey** (WDS API, no key): live, cached hourly.
- **U.S. Bureau of Labor Statistics** (public API, no key): live, cached hourly.
- **SEC EDGAR** (submissions API, no key): recent 8-K / 10-K / DEF 14A filings for
  MetLife, Prudential Financial, Lincoln Financial, and Principal Financial. **John
  Hancock has no standalone SEC filer** — it is a wholly owned Manulife subsidiary, so
  its own leadership-change 8-Ks (if any) would appear only under Manulife's own CIK,
  which this tool does not track (Manulife is the home company, not a tracked peer).
  A filing's presence is a Disclosed fact; whether it *contains* a leadership change is
  **not** auto-detected — the flag points you to read the filing, it does not claim to
  already know what's in it.

### Trade-press RSS feeds (live, filtered, fully automatic)
- **Executive Moves** (`executive-moves.com` — note the hyphen; `executivemoves.com` without
  one is an unrelated parked domain-for-sale page) and **Insurance Edge** (`insurance-edge.net`)
  both publish real, robots.txt-permitted RSS feeds. This tool fetches them live on every load,
  keeps only items whose title/description mentions one of the 10 locked peer companies, and
  parses each matched headline directly into the move table — **no paste, no manual entry**.
  Most pulls will legitimately return zero matches since these are general global insurance
  feeds, not peer-specific — an empty table most weeks is the correct, honest result of that,
  not a broken feature.

### Sources tested and found NOT viable as live feeds (documented so no one re-tries blind)
- **Job Bank Canada** open-data postings file: real, no-key, ~50K postings/month —
  but has **no employer-name field**. Usable only for national/NOC-level stats
  (e.g. total actuarial postings in Canada), not company-specific counts. Not wired in.
- **Adzuna API**: requires a registered app_id/app_key (confirmed via a live
  AUTH_FAIL response) — not wired in without real credentials.
- **Company career pages** (e.g. Manulife's own): confirmed bot-blocked
  (Akamai "Access Denied" on a direct request) — this tool does not attempt to
  bypass bot detection. Job-posting counts from career pages are not available here.
- **SEDAR+, Ontario Mass Termination Notices, US WARN Act notices**: no public
  machine-readable API found. Not wired in.
- **Workforce & Talent Strategy** (AI upskilling programs, wellness benefits, RTO
  mandates, etc.): no structured public API surfaces this as fetchable data — left
  blank in this version rather than faked or manually typed.

### What counts as a valid tracked move
- **Person** — a named individual (not "a spokesperson" or anonymous reference)
- **Prior or New role context** — at least one side of the move
- **Explicit source attribution** — the originating feed article, always linked

### Permanently excluded from this tool
- LinkedIn scraping/API or bulk profile monitoring of any kind.
- Manual data entry of any kind — every row in this tool comes from a live,
  no-key public source, fetched and parsed automatically.
- Non-leadership/non-executive personnel changes.

### Role/function grouping — provisional
Function tags shown throughout (Actuarial, Risk, Finance, HR/People, Technology,
Operations, Legal, Marketing, Other) are a **placeholder keyword map**, not a settled
taxonomy. Per governance decision, the real grouping should be based on Manulife's own
current open postings, mapped to **NOC** (Canada) / **O\\*NET-SOC** (US) reference codes —
that gets decided once a few weeks of real output exist, not hardcoded now.

### Snapshot, not a diff
Each report run is a **snapshot** — 🔴 concerning signal, 🟢 notable move, ⚪ nothing
new this pull — with one synthesis sentence per peer. There is no week-over-week delta
tracking in this version.
"""


def build_markdown_export(
    market: str,
    flags: dict[str, tuple[str, str]],
    bottom_line: str,
    labour_market_md: str,
    workforce_strategy_text: str,
    df: pd.DataFrame,
    sourcing_md: str,
    retention_md: str,
    benchmarking_md: str,
) -> str:
    lines = [f"# Talent Moves Tracker — {market} Report", ""]
    lines.append(f"_Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")

    lines.append("## 1. Talent Flow Tracker — Live Snapshot")
    lines.append("")
    for peer, (flag, note) in flags.items():
        lines.append(f"- {flag} **{peer}** — {note}")
    lines.append("")
    lines.append(f"**Bottom Line:** {bottom_line}")
    lines.append("")

    lines.append("## 2. Workforce & Talent Strategy")
    lines.append("")
    lines.append(workforce_strategy_text.strip() if workforce_strategy_text.strip() else "_No notes entered this pull._")
    lines.append("")

    lines.append("## 3. Labour-Market Research")
    lines.append("")
    lines.append(labour_market_md)
    lines.append("")

    lines.append("## 4. So What")
    lines.append("")
    lines.append("### Sourcing")
    lines.append(sourcing_md)
    lines.append("")
    lines.append("### Retention")
    lines.append(retention_md)
    lines.append("")
    lines.append("### Benchmarking")
    lines.append(benchmarking_md)
    lines.append("")

    lines.append("## Talent Flow Detail (full move table)")
    lines.append("")
    if df.empty:
        lines.append("_No moves recorded._")
    else:
        export_cols = [
            "Person", "Direction", "Prior Title", "New Title", "Prior Company",
            "New Company", "Date", "Title Delta", "Function", "Source",
        ]
        lines.append("| " + " | ".join(export_cols) + " |")
        lines.append("|" + "---|" * len(export_cols))
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row.get(c, "")).replace("|", "/") for c in export_cols) + " |")
    lines.append("")

    lines.append("## Methodology")
    lines.append(METHODOLOGY_TEXT)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

DIRECTION_OPTIONS = ["Inbound", "Internal", "Outbound", "Unclassified"]
TITLE_DELTA_OPTIONS = ["Promotion", "Lateral", "Expanded Remit"]
FUNCTION_OPTIONS = list(FUNCTION_KEYWORDS.keys()) + ["Other"]


def render_sidebar() -> tuple[str, list[str]]:
    with st.sidebar:
        st.header("Settings")
        market = st.radio("Market", list(LOCKED_PEER_SETS.keys()), index=0)
        st.caption("Canada and US are separate reports — peers are never compared cross-market.")

        all_peers = LOCKED_PEER_SETS[market]
        selected_peers = st.multiselect("Peer companies", all_peers, default=all_peers)

        st.divider()
        if st.button("🔄 Refresh live data", use_container_width=True):
            cached_fetch_statcan.clear()
            cached_fetch_bls.clear()
            cached_fetch_sec.clear()
            cached_fetch_trade_press.clear()
            st.rerun()
        st.caption(
            "Every source below is fetched live — no paste, no manual entry. Cached up to 1 hour "
            "to respect free-tier rate limits; use Refresh to force a new pull."
        )

    return market, selected_peers


def render_live_report_tab(market: str, peers: list[str], moves_df: pd.DataFrame | None) -> tuple[dict, str, str]:
    st.subheader(f"{market} — Live Snapshot")
    st.caption(
        "🔴 concerning signal · 🟢 notable move · ⚪ nothing new this pull. Fully live — no paste, no "
        "manual entry. Each line names its source and whether it's a Disclosed fact or an Inferred "
        "signal — never blended."
    )

    flags: dict[str, tuple[str, str]] = {}

    sec_results = {}
    if market == "US":
        for peer in peers:
            sec_results[peer] = cached_fetch_sec(peer)

    for peer in peers:
        sec_result = sec_results.get(peer) if market == "US" else None
        flag, note = compute_company_flag(peer, moves_df, sec_result)
        flags[peer] = (flag, note)
        st.markdown(f"{flag} **{peer}** — {note}")

    flags_simple = {p: f for p, (f, _) in flags.items()}
    macro_result = cached_fetch_statcan() if market == "Canada" else cached_fetch_bls()
    bottom_line = build_bottom_line(market, flags_simple, macro_result)

    st.divider()
    st.markdown(f"**Bottom Line:** {bottom_line}")

    if market == "US":
        with st.expander("SEC EDGAR detail (raw filings checked)"):
            for peer, result in sec_results.items():
                st.markdown(f"**{peer}**")
                if result["ok"]:
                    st.caption(f"Source: {result['source']} · Retrieved {result['retrieved_at']}")
                    if result["data"]:
                        for row in result["data"]:
                            st.markdown(f"- [{row['form']} — {row['filed']}]({row['url']})")
                    else:
                        st.caption("No matching filings found.")
                else:
                    st.warning(result["error"])

    return flags, bottom_line, market


def _render_labour_market_country(country: str) -> str:
    if country == "Canada":
        result = cached_fetch_statcan()
        source_label = "Statistics Canada — Labour Force Survey (WDS API)"
    else:
        result = cached_fetch_bls()
        source_label = "U.S. Bureau of Labor Statistics (public API)"

    if not result["ok"]:
        st.error(f"Could not reach {source_label}: {result['error']}")
        return f"_Live fetch failed: {result['error']}_"

    st.caption(f"Source: {source_label} · Retrieved {result['retrieved_at']} (Disclosed fact — direct API read)")

    md_lines = [f"Source: {source_label} · Retrieved {result['retrieved_at']}", ""]
    for label, points in result["data"].items():
        st.markdown(f"**{label}**")
        if points:
            table = pd.DataFrame(points)
            st.dataframe(table, use_container_width=True, hide_index=True)
            md_lines.append(f"**{label}**")
            for p in points:
                md_lines.append(f"- {p['period']}: {p['value']}")
            md_lines.append("")
        else:
            st.caption("No data points returned.")

    if country == "Canada":
        st.info(live_sources.JOB_BANK_NOTE)
        st.caption(live_sources.ADZUNA_NOTE)
        md_lines.append(f"_Note: {live_sources.JOB_BANK_NOTE}_")

    return "\n".join(md_lines)


def render_labour_market_tab(market: str) -> str:
    st.subheader("Labour-Market Research (live)")
    st.caption("Independent of the peer market selected in the sidebar — pick a country to view here.")

    ca_tab, us_tab = st.tabs(["🇨🇦 Canada", "🇺🇸 US"])
    with ca_tab:
        canada_md = _render_labour_market_country("Canada")
    with us_tab:
        us_md = _render_labour_market_country("US")

    return canada_md if market == "Canada" else us_md


WORKFORCE_STRATEGY_NOTE = (
    "No public, no-key API surfaces peer workforce-strategy announcements (AI upskilling "
    "programs, wellness benefits, RTO mandates, etc.) as structured, fetchable data — this would "
    "require either a paid news-monitoring API or a manual step, and per current direction this "
    "tool takes no manual input. Left blank rather than faked. If a specific structured source "
    "for this becomes available, it can be wired in the same way as the other live sources."
)


def render_workforce_strategy_tab() -> str:
    st.subheader("Workforce & Talent Strategy")
    st.info(WORKFORCE_STRATEGY_NOTE)
    return WORKFORCE_STRATEGY_NOTE


def render_talent_flow_detail_tab() -> pd.DataFrame:
    st.subheader("Talent Flow Detail — live, industry-wide")
    st.caption(
        "Fully automatic: pulls every item from the Executive Moves (Insurance) and Insurance Edge "
        "RSS feeds — both real, robots.txt-permitted, no key needed — and parses each headline into "
        "a row below. Not restricted to the 10 locked peers or to a country: this shows every "
        "executive move either feed reports across the insurance industry. The **Matched Locked "
        "Peer** column flags rows relevant to our specific peer set — used by the Live Report tab — "
        "but nothing here is hidden or filtered out. No paste, no manual entry."
    )

    auto_df, fetch_result = fetch_auto_trade_press_moves()
    st.caption(f"Retrieved {fetch_result['retrieved_at']} · Source: {fetch_result['source']}")
    if fetch_result["error"]:
        st.warning(f"Some feeds failed: {fetch_result['error']}")

    if auto_df.empty:
        st.info("Both feeds returned no parseable move headlines this pull.")
        return auto_df

    matched_count = (auto_df["Matched Locked Peer"] != "").sum()
    st.caption(f"{len(auto_df)} move(s) fetched this pull · {matched_count} match a locked peer.")

    st.subheader("Review")
    st.caption(
        "Auto-extracted from live headlines — correct any misclassified cell directly (this is "
        "review/correction of a live pull, not manual data entry)."
    )
    edited_df = st.data_editor(
        auto_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Direction": st.column_config.SelectboxColumn("Direction", options=DIRECTION_OPTIONS, required=False),
            "Title Delta": st.column_config.SelectboxColumn("Title Delta", options=TITLE_DELTA_OPTIONS, required=False),
            "Function": st.column_config.SelectboxColumn("Function", options=FUNCTION_OPTIONS, required=False),
        },
        key="move_editor",
    )

    st.subheader("So What")
    sourcing_tab, retention_tab, benchmarking_tab = st.tabs(["Sourcing", "Retention", "Benchmarking"])
    sourcing_md = build_sourcing_section(edited_df)
    retention_md = build_retention_section(edited_df)
    benchmarking_md = build_benchmarking_section(edited_df)
    with sourcing_tab:
        st.markdown(sourcing_md)
    with retention_tab:
        st.markdown(retention_md)
    with benchmarking_tab:
        st.markdown(benchmarking_md)
    st.session_state["_sourcing_md"] = sourcing_md
    st.session_state["_retention_md"] = retention_md
    st.session_state["_benchmarking_md"] = benchmarking_md

    return edited_df


def get_claude_client() -> anthropic.Anthropic | None:
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def build_ask_context(market: str, moves_df: pd.DataFrame | None) -> str:
    """Assemble a compact text snapshot of everything this session has already
    fetched live, so the model answers from real, retrieved data first and
    only reaches for web_search when the context genuinely doesn't cover it."""
    parts = [f"Current market selected in the app: {market}"]

    if moves_df is not None and not moves_df.empty:
        parts.append("\nLIVE TRADE-PRESS MOVES FETCHED THIS SESSION (industry-wide, from Executive Moves / Insurance Edge RSS):")
        cols = ["Person", "Direction", "Prior Company", "New Company", "New Title", "Function", "Matched Locked Peer", "Date", "Source"]
        for _, row in moves_df.iterrows():
            parts.append(" - " + " | ".join(f"{c}: {row.get(c, '')}" for c in cols if row.get(c, "")))
    else:
        parts.append("\nNo trade-press moves were fetched this session (feeds returned nothing parseable).")

    if market == "US":
        parts.append("\nSEC EDGAR — recent filings for US peers:")
        for peer in LOCKED_PEER_SETS["US"]:
            result = cached_fetch_sec(peer)
            if result["ok"] and result["data"]:
                latest = result["data"][0]
                parts.append(f" - {peer}: most recent {latest['form']} filed {latest['filed']} ({latest['url']})")
            elif result["ok"]:
                parts.append(f" - {peer}: no recent 8-K/10-K/DEF 14A on file")
            else:
                parts.append(f" - {peer}: SEC lookup unavailable ({result['error']})")
        macro = cached_fetch_bls()
    else:
        macro = cached_fetch_statcan()

    if macro["ok"]:
        parts.append(f"\nLabour-market data (retrieved {macro['retrieved_at']}):")
        for label, points in macro["data"].items():
            if points:
                latest = points[0]
                parts.append(f" - {label}: {latest['value']} ({latest['period']})")

    return "\n".join(parts)


def render_ask_tab(market: str, moves_df: pd.DataFrame | None) -> None:
    st.subheader("Ask")
    st.caption(
        "Natural-language questions about talent movement, answered by Claude. Grounded first in the "
        "live data already fetched this session (shown in the other tabs); falls back to a live web "
        "search only when that doesn't cover the question. Uses the Anthropic API — each question has "
        "a real cost."
    )

    client = get_claude_client()
    if client is None:
        st.warning(
            "No Anthropic API key configured. Add `ANTHROPIC_API_KEY` in Streamlit Cloud under "
            "**Settings → Secrets** (or a local `.streamlit/secrets.toml`) to enable this tab."
        )
        return

    if "ask_history" not in st.session_state:
        st.session_state["ask_history"] = []

    for msg in st.session_state["ask_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("e.g. \"What's the latest at MetLife?\" or \"Any actuarial leadership moves this month?\"")
    if not question:
        return

    st.session_state["ask_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    context = build_ask_context(market, moves_df)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1200,
                system=ASK_SYSTEM_PROMPT,
                tools=[CLAUDE_WEB_SEARCH_TOOL],
                messages=[
                    {"role": "user", "content": f"LIVE DATA CONTEXT:\n{context}\n\nQUESTION: {question}"}
                ],
            )
        except anthropic.APIError as e:
            placeholder.error(f"API error: {e}")
            st.session_state["ask_history"].pop()
            return

        answer_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        placeholder.markdown(answer_text)

        usage = response.usage
        st.caption(f"{usage.input_tokens:,} input / {usage.output_tokens:,} output tokens this query")

    st.session_state["ask_history"].append({"role": "assistant", "content": answer_text})


def render_methodology_tab() -> None:
    st.subheader("Methodology & Governance")
    with st.expander("Full governance & methodology note", expanded=True):
        st.markdown(METHODOLOGY_TEXT)


def main() -> None:
    st.title("🧭 Talent Moves Tracker")
    st.caption("Competitive talent intelligence for Manulife People Analytics — fully live sources, never blended without a tag.")

    market, peers = render_sidebar()

    live_tab, detail_tab, labour_tab, strategy_tab, ask_tab, methodology_tab = st.tabs(
        ["📡 Live Report", "📋 Talent Flow Detail", "📊 Labour-Market Research", "🧭 Workforce & Strategy", "💬 Ask", "📖 Methodology"]
    )

    moves_df = None

    with detail_tab:
        moves_df = render_talent_flow_detail_tab()

    with live_tab:
        flags, bottom_line, _ = render_live_report_tab(market, peers, moves_df)

    with labour_tab:
        labour_market_md = render_labour_market_tab(market)

    with strategy_tab:
        workforce_strategy_text = render_workforce_strategy_tab()

    with ask_tab:
        render_ask_tab(market, moves_df)

    with methodology_tab:
        render_methodology_tab()

    st.sidebar.divider()
    st.sidebar.subheader("Export")
    export_df = moves_df if moves_df is not None and not moves_df.empty else pd.DataFrame(columns=MOVE_COLUMNS)
    markdown_report = build_markdown_export(
        market=market,
        flags=flags,
        bottom_line=bottom_line,
        labour_market_md=labour_market_md,
        workforce_strategy_text=workforce_strategy_text,
        df=export_df,
        sourcing_md=st.session_state.get("_sourcing_md", "_No moves parsed yet._"),
        retention_md=st.session_state.get("_retention_md", "_No moves parsed yet._"),
        benchmarking_md=st.session_state.get("_benchmarking_md", "_No moves parsed yet._"),
    )
    st.sidebar.download_button(
        f"Download {market} report (.md)",
        data=markdown_report,
        file_name=f"talent_moves_{market.lower()}_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
