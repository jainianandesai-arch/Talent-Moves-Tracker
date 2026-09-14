"""
Talent Moves Tracker
---------------------
A rule-based (no LLM/API dependency) Streamlit tool that extracts personnel/executive
moves from pasted source text (press releases, news snippets, LinkedIn-style
announcements copied by hand, etc.), classifies each move, and produces a
human-reviewable table plus a "So What" analysis, exportable to Markdown.

Single-file app: parsing, classification, UI, and export all live here.
"""

import re
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Talent Moves Tracker",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Peer sets (used for Direction classification + Benchmarking)
# ---------------------------------------------------------------------------

PEER_SETS = {
    "Canada": [
        "RBC", "Royal Bank of Canada", "TD", "TD Bank", "Toronto-Dominion",
        "Scotiabank", "Bank of Nova Scotia", "BMO", "Bank of Montreal",
        "CIBC", "Canadian Imperial Bank of Commerce", "National Bank",
        "National Bank of Canada", "Manulife", "Sun Life", "Sun Life Financial",
        "Great-West Life", "Great-West Lifeco", "iA Financial", "Desjardins",
        "CPPIB", "CPP Investments", "OTPP", "Ontario Teachers' Pension Plan",
        "OMERS", "CDPQ", "Caisse de dépôt", "PSP Investments", "Brookfield",
        "Fairfax", "Fairfax Financial", "Power Corporation", "IGM Financial",
    ],
    "US": [
        "JPMorgan", "JPMorgan Chase", "J.P. Morgan", "Bank of America", "BofA",
        "Citigroup", "Citi", "Wells Fargo", "Goldman Sachs", "Morgan Stanley",
        "US Bank", "U.S. Bank", "PNC", "PNC Financial", "Truist",
        "Capital One", "American Express", "Amex", "BlackRock", "State Street",
        "Charles Schwab", "Fidelity", "Vanguard", "MetLife", "Prudential",
        "Prudential Financial", "AIG", "Berkshire Hathaway",
    ],
}

# Flattened company/organization list used by extract_companies / extract_person.
ALL_KNOWN_COMPANIES = sorted(set(PEER_SETS["Canada"] + PEER_SETS["US"]), key=len, reverse=True)

# ---------------------------------------------------------------------------
# Vocabulary for title-seniority and function classification
# ---------------------------------------------------------------------------

# Seniority ranks — higher number = more senior. Used for Title Delta (Promotion vs Lateral).
SENIORITY_RANK = [
    (100, [r"\bchief executive officer\b", r"\bceo\b", r"\bpresident and ceo\b"]),
    (95, [r"\bpresident\b(?!.{0,20}vice)"]),
    (90, [r"\bvice[\s-]?chair\b", r"\bexecutive vice[\s-]?chair\b"]),
    (85, [r"\bchief\s+\w[\w\s&/]*\s+officer\b", r"\bc[a-z]o\b"]),  # Chief X Officer / CFO/CRO/etc.
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

# Scope-breadth signals (used to distinguish "Expanded Remit" from a flat Lateral
# when seniority rank is roughly equal but the new title covers more ground).
SCOPE_EXPANSION_PATTERNS = [
    r"\bglobal\b", r"\bworldwide\b", r"\benterprise[\s-]?wide\b", r"\ball[\s-]?markets\b",
    r"\bnorth america\b", r"\binternational\b", r"\bgroup[\s-]?wide\b",
    r"\band\b.{0,40}\b(chief|head|officer)\b",  # "X and Y" combined mandate
]
SCOPE_NARROWING_HINTS = [r"\binterim\b", r"\bacting\b"]

FUNCTION_KEYWORDS = {
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
# Direction classification vocabulary
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

# ---------------------------------------------------------------------------
# Sample text for self-testing
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
Jane Smith joins RBC as Chief Risk Officer, effective March 1, 2025, according to an internal memo.
John Doe, previously Senior Vice President of Technology at TD, has left the bank after eight years; he departed on Feb 14, 2025 (per LinkedIn).
BMO promoted Maria Chen from Director, Compliance to Head of Compliance, expanding her mandate to cover the US and Canadian businesses, effective Jan 15, 2025.
Scotiabank named David Lee as its new Chief Financial Officer, succeeding the retiring Anne White, according to a company press release dated 2025-01-10.
CIBC's Head of Marketing, Priya Nair, is leaving to join a fintech startup, sources say.
Wells Fargo appointed Robert King as Executive Vice President, Global Technology, per a statement issued April 2025.
Sarah Johnson steps down as Chief Operating Officer of Manulife after a decade in the role; the company has not named a successor (Reuters).
National Bank elevated Marc Tremblay to Senior Director, Risk Management from Manager, Credit Risk, internal promotion announced Q2 2025.
Citigroup's General Counsel, Laura Bennett, resigned effective June 30, 2025, to pursue an opportunity outside the firm.
Kevin Wu joined Goldman Sachs as Managing Director, Data & Analytics after five years at a Canadian pension fund, effective May 2025 (company announcement).
"""

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def split_into_candidate_lines(text: str) -> list[str]:
    """Split raw pasted text into candidate move statements.

    Splits on newlines first, then further splits any long line on sentence
    boundaries so that multiple moves reported in one paragraph (or one
    sentence per move) both work.
    """
    if not text or not text.strip():
        return []

    lines: list[str] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        # Split on sentence-ending punctuation followed by a capital letter,
        # but avoid splitting on abbreviations like "Inc." or "U.S."
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", raw_line)
        for s in sentences:
            s = s.strip(" -•\t")
            if len(s) > 15:  # skip stray fragments/headers
                lines.append(s)
    return lines


def _find_verb_hit(text_lower: str, verbs: list[str]) -> str | None:
    for v in verbs:
        if v in text_lower:
            return v
    return None


def classify_direction(line: str) -> tuple[str, str | None]:
    """Classify a candidate line as Inbound / Internal / Outbound / Unclassified.

    Returns (direction, matched_verb). Uses a scored/prioritized approach so
    multi-clause sentences (e.g. "X leaves A to join B", "promoted from Y to Z")
    still resolve correctly instead of defaulting to Unclassified.
    """
    text_lower = line.lower()

    inbound_hit = _find_verb_hit(text_lower, INBOUND_VERBS)
    outbound_hit = _find_verb_hit(text_lower, OUTBOUND_VERBS)
    internal_hit = _find_verb_hit(text_lower, INTERNAL_VERBS)

    # Compound sentence: "X leaves/departs A to join/joins B" -> Outbound is the
    # newsworthy direction from the reporting company's point of view UNLESS the
    # sentence structure signals the move is framed as joining a new employer
    # (i.e. "leaving to join" reads as that person's Inbound move to company B).
    if outbound_hit and re.search(r"\bto\s+(join|become|take|lead)\b", text_lower):
        return "Inbound", outbound_hit + " ... to join"

    if inbound_hit and outbound_hit:
        # e.g. "after leaving X, joins Y" — prioritize the inbound clause since
        # that's the actionable/most recent state.
        inbound_pos = text_lower.find(inbound_hit)
        outbound_pos = text_lower.find(outbound_hit)
        return ("Inbound", inbound_hit) if inbound_pos <= outbound_pos else ("Outbound", outbound_hit)

    if internal_hit:
        return "Internal", internal_hit

    if inbound_hit:
        return "Inbound", inbound_hit

    if outbound_hit:
        return "Outbound", outbound_hit

    # --- Fallback heuristics for messy/multi-clause sentences with no direct verb hit ---

    # "from X to Y" pattern strongly suggests an internal move/promotion, even
    # without one of our INTERNAL_VERBS present (e.g. "X: Director to VP").
    if re.search(r"\bfrom\b.{3,60}\bto\b", text_lower) and re.search(r"\b(role|title|position)\b", text_lower):
        return "Internal", "from...to (role change)"

    # "named/appointed as successor to" / "succeeds" style without a verb hit already
    # covered above, but catch "X becomes Y" as an internal/inbound signal depending
    # on whether a prior employer for this same company is mentioned.
    if re.search(r"\bbecomes\b", text_lower):
        if re.search(r"\bat\b|\bwithin\b|\binternal(ly)?\b", text_lower):
            return "Internal", "becomes (internal)"
        return "Inbound", "becomes"

    # Two organizations mentioned with a clear switch preposition ("from ... at").
    companies_found = extract_companies(line)
    if len(companies_found) >= 1 and re.search(r"\bnew\b.{0,15}\b(role|position|title)\b", text_lower):
        return "Inbound", "new role (fallback)"

    return "Unclassified", None


def extract_companies(line: str) -> list[str]:
    """Find known peer-set company names mentioned in the line, longest match first."""
    found = []
    remaining = line
    for company in ALL_KNOWN_COMPANIES:
        pattern = r"\b" + re.escape(company) + r"\b"
        if re.search(pattern, remaining, re.IGNORECASE):
            found.append(company)
    # De-duplicate near-aliases (e.g. both "TD" and "TD Bank" matching) by keeping
    # only the longest match per rough prefix cluster.
    deduped: list[str] = []
    for c in sorted(found, key=len, reverse=True):
        if not any(c.lower() in d.lower() for d in deduped):
            deduped.append(c)
    return deduped


ROLE_WORDS = {
    "head", "chief", "president", "director", "officer", "manager", "vice",
    "senior", "executive", "global", "group", "bank", "financial", "corp",
    "corporation", "inc", "the", "board", "committee", "general", "counsel",
}


def _looks_like_person_name(candidate: str) -> bool:
    """Reject candidates that are clearly a company name, a title/role phrase,
    or a possessive company reference rather than an actual person."""
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
    """Best-effort extraction of the person's name.

    Tries, in priority order: (1) a name set off by commas in an appositive
    clause (e.g. "CIBC's Head of Marketing, Priya Nair, is leaving"), (2) a
    name following a hiring/promotion verb (handles past tense too), (3) a
    capitalized sequence at the very start of the sentence — but only if it
    doesn't look like a company or title/role phrase.
    """
    # (1) Verb-based, covering base/plural/past-tense forms — checked first since
    # it's the most reliable signal when present (avoids appositive commas
    # elsewhere in the sentence, e.g. "Manager, Credit Risk," being mistaken
    # for a name).
    match = re.search(
        r"\b(?:appoints?|appointed|names?|named|promotes?|promoted|elevates?|elevated|"
        r"welcomes?|welcomed|hires?|hired|recruits?|recruited)\s+"
        r"([A-Z][a-zA-Z.\'-]+(?:\s+[A-Z][a-zA-Z.\'-]+){1,3})",
        line,
    )
    if match:
        candidate = match.group(1).strip()
        if _looks_like_person_name(candidate):
            return candidate

    # (2) Appositive: ", Name Name," or ", Name Name Name,"
    for match in re.finditer(r",\s*([A-Z][a-zA-Z.\'-]+(?:\s+[A-Z][a-zA-Z.\'-]+){1,2})\s*,", line):
        candidate = match.group(1).strip()
        if _looks_like_person_name(candidate):
            return candidate

    # (3) Leading capitalized sequence at the start of the sentence.
    match = re.match(r"^([A-Z][a-zA-Z.\'-]+(?:\s+[A-Z][a-zA-Z.\'-]+){1,3})", line.strip())
    if match:
        candidate = match.group(1).strip()
        # Trim trailing company names accidentally captured (e.g. "Jane Smith RBC").
        for company in ALL_KNOWN_COMPANIES:
            candidate = re.sub(r"\s+" + re.escape(company) + r"$", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip()
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
    """Best-effort extraction of (prior_title, new_title) from a line.

    Looks for patterns like "from X to Y", "X to Y", "as <title>", or a single
    title following an action verb (treated as the new title, prior left blank).
    """
    # Pattern: "from <prior title> to <new title>"
    m = re.search(
        r"from\s+([A-Z][\w,&/\s-]{2,60}?)\s+to\s+([A-Z][\w,&/\s-]{2,60}?)(?:[,.;]|\s+(?:at|effective|after|since|starting)\b|$)",
        line,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Pattern: "previously <title> at/of <company>, ... (now) <title>" is hard to
    # generalize; fall back to "as <title>" for the new title and "previously <title>"
    # for prior.
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
        # "X, <Title> of/at <Company>," near the start — treat as prior title if
        # verb suggests departure, else new title.
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
    return 0  # unknown title text


def classify_title_delta(prior_title: str, new_title: str) -> str:
    """Infer Promotion / Lateral / Expanded Remit from prior vs new title text."""
    if not new_title:
        return ""
    if not prior_title:
        # No prior title to compare — can't confidently call it a promotion;
        # default to Lateral unless the new title itself signals broadened scope.
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

    # new_rank < prior_rank: could still be an expanded remit at a nominally
    # lower-ranked label (rare) — otherwise call it Lateral rather than a
    # (likely mis-parsed) demotion, since we don't have a Demotion category.
    if scope_expanded:
        return "Expanded Remit"
    return "Lateral"


def classify_function(title: str, full_line: str = "") -> str:
    """Infer a role-family tag from the title text (falls back to full line)."""
    search_text = f"{title} {full_line}".lower()
    scores = {}
    for func, keywords in FUNCTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in search_text)
        if count:
            scores[func] = count
    if not scores:
        return "Other"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Move dataclass + top-level parse
# ---------------------------------------------------------------------------


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


def parse_moves(text: str, peer_set_name: str = "Canada") -> pd.DataFrame:
    lines = split_into_candidate_lines(text)
    moves: list[Move] = []

    peer_names_lower = {c.lower() for c in PEER_SETS.get(peer_set_name, [])}

    for line in lines:
        direction, _ = classify_direction(line)
        prior_title, new_title = _extract_titles(line)
        companies = extract_companies(line)
        person = extract_person(line)
        move_date = extract_date(line)
        source = extract_source(line)

        # Assign prior/new company by direction + order of mention when possible.
        prior_company, new_company = "", ""
        if companies:
            if direction == "Outbound":
                prior_company = companies[0]
            elif direction == "Inbound":
                new_company = companies[0]
            else:  # Internal or Unclassified — same company both sides if only one found
                if len(companies) >= 2:
                    prior_company, new_company = companies[0], companies[1]
                else:
                    prior_company = new_company = companies[0]

        title_delta = classify_title_delta(prior_title, new_title)
        function = classify_function(new_title or prior_title, line)

        moves.append(
            Move(
                person=person,
                direction=direction,
                prior_title=prior_title,
                new_title=new_title,
                prior_company=prior_company,
                new_company=new_company,
                date=move_date,
                source=source,
                title_delta=title_delta,
                function=function,
                raw_text=line,
            )
        )

    if not moves:
        return pd.DataFrame(
            columns=[
                "Person", "Direction", "Prior Title", "New Title", "Prior Company",
                "New Company", "Date", "Source", "Title Delta", "Function", "Raw Text",
            ]
        )

    df = pd.DataFrame(
        [
            {
                "Person": m.person,
                "Direction": m.direction,
                "Prior Title": m.prior_title,
                "New Title": m.new_title,
                "Prior Company": m.prior_company,
                "New Company": m.new_company,
                "Date": m.date,
                "Source": m.source,
                "Title Delta": m.title_delta,
                "Function": m.function,
                "Raw Text": m.raw_text,
            }
            for m in moves
        ]
    )
    return df


# ---------------------------------------------------------------------------
# "So What" analysis (rule-based counts/proportions)
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
            "share (≥60% with 2+ tracked moves) — could indicate deliberate succession planning, "
            "but also a smaller external talent pool being tapped for that function. Worth benchmarking "
            "whether that's a strength (deep bench) or a risk (limited external validation of leadership)."
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
        lines.append(
            f"⚠️ **Disproportionate external hiring:** {', '.join(flagged)} — 70%+ of tracked moves in "
            "this function are external hires rather than internal promotion, which may signal a talent "
            "gap internally or an aggressive external build-out strategy by peers."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

METHODOLOGY_TEXT = """
### What counts as a valid move
A row is treated as a trackable executive/leadership move only when the pasted
text provides, at minimum:
- **Person** — a named individual (not "a spokesperson" or an anonymous reference)
- **Prior or New role context** — at least one side of the move (what they did
  before and/or what they are moving into)
- **Explicit source attribution** — the text names or clearly implies where the
  information came from (a company press release, a named news outlet, a
  quoted internal memo, etc.)

### What is excluded
- **LinkedIn scraping or bulk profile monitoring.** This tool only processes
  text a user has manually pasted from a specific, identifiable source — it
  does not crawl, scrape, or bulk-ingest social profiles.
- **Unconfirmed rumor or anonymous "sources say" content**, unless the pasted
  text itself is from a reputable, named outlet reporting on the record. Vague
  attribution (no outlet, no memo, no named source) should be treated with
  caution and reviewed manually before inclusion in any external-facing report.
- **Non-leadership/non-executive personnel changes.** This tool is scoped to
  publicly disclosed executive and senior leadership moves only.

### Scope statement
Only **publicly disclosed** executive/leadership moves — i.e., moves a company
or credible news outlet has already chosen to announce publicly — are tracked
here. This tool does not perform individual-level surveillance of employees,
does not track rank-and-file personnel changes, and is not a substitute for
consent-based talent intelligence processes. All extracted rows are surfaced
in an editable review table specifically so a human can correct, remove, or
flag any row before it is used in reporting.
"""


def build_markdown_export(
    df: pd.DataFrame,
    peer_set_name: str,
    sourcing_md: str,
    retention_md: str,
    benchmarking_md: str,
) -> str:
    lines = ["# Talent Moves Tracker — Report", "", f"_Peer set: {peer_set_name}_", ""]

    lines.append("## Tracked Moves")
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
    lines.append("## So What")
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
    lines.append("## Methodology")
    lines.append(METHODOLOGY_TEXT)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

DIRECTION_OPTIONS = ["Inbound", "Internal", "Outbound", "Unclassified"]
TITLE_DELTA_OPTIONS = ["Promotion", "Lateral", "Expanded Remit"]
FUNCTION_OPTIONS = ["Risk", "Finance", "HR/People", "Technology", "Operations", "Legal", "Marketing", "Other"]


def render_sidebar() -> str:
    with st.sidebar:
        st.header("Settings")
        peer_set_name = st.radio("Peer set", list(PEER_SETS.keys()), index=0)
        st.caption(
            "Direction classification and company extraction match against this peer set's "
            "known company names."
        )
        st.divider()
        if st.button("Load sample text"):
            st.session_state["source_text"] = SAMPLE_TEXT.strip()
            st.session_state.pop("parsed_df", None)
        if st.button("Clear all"):
            st.session_state["source_text"] = ""
            st.session_state.pop("parsed_df", None)
    return peer_set_name


def render_input_and_parse(peer_set_name: str) -> None:
    st.subheader("1. Paste source text")
    st.caption(
        "Paste press releases, news snippets, or announcement text. One move per line/sentence "
        "works best, but multi-clause sentences are also handled."
    )
    text = st.text_area(
        "Source text",
        value=st.session_state.get("source_text", ""),
        height=220,
        key="source_text",
    )

    if st.button("Parse moves", type="primary", disabled=not text.strip()):
        st.session_state["parsed_df"] = parse_moves(text, peer_set_name)


def render_review_table() -> pd.DataFrame | None:
    df = st.session_state.get("parsed_df")
    if df is None:
        st.info("Paste text above and click **Parse moves** to get started (or load the sample text from the sidebar).")
        return None

    st.subheader("2. Review & correct")
    st.caption("Edit any misclassified cells directly in the table below before exporting.")

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Direction": st.column_config.SelectboxColumn("Direction", options=DIRECTION_OPTIONS, required=False),
            "Title Delta": st.column_config.SelectboxColumn("Title Delta", options=TITLE_DELTA_OPTIONS, required=False),
            "Function": st.column_config.SelectboxColumn("Function", options=FUNCTION_OPTIONS, required=False),
        },
        key="move_editor",
    )
    st.session_state["parsed_df"] = edited_df
    return edited_df


def render_so_what(df: pd.DataFrame) -> tuple[str, str, str]:
    st.subheader("3. So What")

    sourcing_tab, retention_tab, benchmarking_tab = st.tabs(["Sourcing", "Retention", "Benchmarking"])
    sourcing_md = build_sourcing_section(df)
    retention_md = build_retention_section(df)
    benchmarking_md = build_benchmarking_section(df)

    with sourcing_tab:
        st.markdown(sourcing_md)
    with retention_tab:
        st.markdown(retention_md)
    with benchmarking_tab:
        st.markdown(benchmarking_md)

    return sourcing_md, retention_md, benchmarking_md


def render_methodology_tab() -> None:
    st.subheader("Methodology")
    with st.expander("What counts as a move, what's excluded, and scope", expanded=True):
        st.markdown(METHODOLOGY_TEXT)


def render_export(df: pd.DataFrame, peer_set_name: str, sourcing_md: str, retention_md: str, benchmarking_md: str) -> None:
    st.subheader("4. Export")
    markdown_report = build_markdown_export(df, peer_set_name, sourcing_md, retention_md, benchmarking_md)
    st.download_button(
        "Download report (.md)",
        data=markdown_report,
        file_name="talent_moves_report.md",
        mime="text/markdown",
    )
    with st.expander("Preview Markdown"):
        st.code(markdown_report, language="markdown")


def main() -> None:
    st.title("🧭 Talent Moves Tracker")
    st.caption(
        "Rule-based extraction of executive/leadership moves from pasted text — no API key, "
        "no external model calls."
    )

    peer_set_name = render_sidebar()

    tracker_tab, methodology_tab = st.tabs(["Tracker", "📖 Methodology"])

    with tracker_tab:
        render_input_and_parse(peer_set_name)
        df = render_review_table()
        if df is not None and not df.empty:
            sourcing_md, retention_md, benchmarking_md = render_so_what(df)
            render_export(df, peer_set_name, sourcing_md, retention_md, benchmarking_md)
        elif df is not None:
            st.warning("No moves parsed from the text — try adjusting the source text or check the Methodology tab.")

    with methodology_tab:
        render_methodology_tab()


if __name__ == "__main__":
    main()
