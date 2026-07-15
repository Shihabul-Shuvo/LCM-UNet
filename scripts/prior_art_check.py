"""Prior-art re-check (methodology section 16, "mandatory at Week 7").

Prints the six required search strings for the user to run by hand (the
primary, always-available path -- no setup, no dependency, no network
access needed from this script itself). If the OPTIONAL `duckduckgo-search`
package happens to be installed, it also runs a best-effort live query for
each string and applies a LOW-CONFIDENCE keyword heuristic over the result
titles/snippets to flag anything that might show local-conv -> Delta/dt
injection -- printing a bold PIVOT-BEFORE-SUBMISSION warning on any hit.

THIS SCRIPT NEVER INSTALLS duckduckgo-search ITSELF and never requires it --
keeping this prompt minimal and not silently adding a new dependency
(pip install duckduckgo-search yourself if you want the automatic path).

IMPORTANT, READ BEFORE TRUSTING A CLEAN RUN: the keyword heuristic below
only scans SEARCH-RESULT SNIPPETS (a sentence or two per hit), never full
papers -- it is a low-confidence PRE-FILTER, not a verification. A flagged
hit requires manual reading before treating it as a real prior-art conflict;
an UNFLAGGED run does NOT mean no conflict exists. Per methodology section
16: read the top ~10 results for each string by hand at Week 7 regardless of
what this script prints.

Usage:
    python scripts/prior_art_check.py                # print strings (+ best-effort query)
    python scripts/prior_art_check.py --no-query      # print strings only, skip even the optional query
    python scripts/prior_art_check.py --max-results 5 # results per string for the optional query path (default 8)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python scripts/prior_art_check.py` works from any cwd

# Methodology section 16, "Prior-art re-check (mandatory at Week 7)" -- verbatim, in order.
SEARCH_STRINGS: List[str] = [
    "neighbourhood conditioned delta Mamba",
    "local convolution modulate Mamba step size",
    "convolution conditioned selective parameters SSM",
    "spatially varying step-size state space model",
    "depthwise conv modulate dt visual Mamba",
    "scan dynamics conditioned convolution segmentation",
]

# A hit is flagged only if it contains at least one term from EACH of these
# three sets -- a crude but stated, auditable heuristic, not a claim of
# semantic understanding (see module docstring).
CONV_TERMS = ("convolution", "conv", "depthwise", "local")
DELTA_TERMS = ("delta", "step-size", "step size", "dt ", " dt,", "discretization step", "discretisation step")
SSM_TERMS = ("mamba", "state space", "ssm", "selective scan")


def _try_import_ddgs():
    try:
        from duckduckgo_search import DDGS  # type: ignore

        return DDGS
    except ImportError:
        return None


def is_suspicious_hit(title: str, snippet: str) -> bool:
    """Low-confidence keyword heuristic -- see module docstring. Case-insensitive
    substring match over title+snippet for at least one term from each of
    CONV_TERMS/DELTA_TERMS/SSM_TERMS."""
    text = f" {title} {snippet} ".lower()
    return any(t in text for t in CONV_TERMS) and any(t in text for t in DELTA_TERMS) and any(t in text for t in SSM_TERMS)


def query_string(ddgs_cls, query: str, max_results: int) -> List[Dict[str, str]]:
    """Runs one live DuckDuckGo text query via the optional duckduckgo-search
    package. Returns a list of {"title","href","body"} dicts (possibly
    empty). Any exception (network error, rate limit, API change) is
    surfaced to the caller as an empty result list with a printed warning,
    never silently treated as "no prior art found" -- see print_report."""
    with ddgs_cls() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def check_one_string(ddgs_cls, query: str, max_results: int) -> Dict[str, Any]:
    """Returns {"query", "results", "flagged", "error"}. "flagged" is the
    list of results (title/href/body) that tripped is_suspicious_hit."""
    try:
        results = query_string(ddgs_cls, query, max_results)
    except Exception as exc:  # noqa: BLE001 -- best-effort network call, never fatal to the script
        return {"query": query, "results": [], "flagged": [], "error": repr(exc)}

    flagged = [r for r in results if is_suspicious_hit(r.get("title", ""), r.get("body", ""))]
    return {"query": query, "results": results, "flagged": flagged, "error": None}


def print_report(do_query: bool, max_results: int = 8) -> List[Dict[str, Any]]:
    print("=== Prior-art re-check (methodology section 16, mandatory at Week 7) ===\n")
    print("Search each of these strings by hand (Google Scholar / arXiv / a general web search),")
    print("read the top ~10 results per string, and look for any paper doing local/depthwise")
    print("convolution -> Mamba's selective step-size (Delta/dt) injection:\n")
    for i, q in enumerate(SEARCH_STRINGS, 1):
        print(f'  {i}. "{q}"')
    print()

    if not do_query:
        print("(--no-query: skipping the optional automatic query path.)")
        return []

    ddgs_cls = _try_import_ddgs()
    if ddgs_cls is None:
        print(
            "Optional automatic path not available: `duckduckgo-search` is not installed.\n"
            "This is expected -- it is NOT a dependency of this project (kept minimal).\n"
            "Install it yourself for a best-effort automatic pre-filter: pip install duckduckgo-search\n"
            "Manual search (above) is the primary, required path either way."
        )
        return []

    print("`duckduckgo-search` found -- running a best-effort automatic pre-filter")
    print("(LOW CONFIDENCE: snippet-only keyword matching, see this script's module docstring).\n")

    all_results: List[Dict[str, Any]] = []
    any_flagged = False
    for q in SEARCH_STRINGS:
        outcome = check_one_string(ddgs_cls, q, max_results)
        all_results.append(outcome)
        if outcome["error"]:
            print(f'  "{q}": query FAILED ({outcome["error"]}) -- search this one manually.')
            continue
        print(f'  "{q}": {len(outcome["results"])} result(s), {len(outcome["flagged"])} flagged')
        for hit in outcome["flagged"]:
            any_flagged = True
            print(f"      FLAGGED: {hit.get('title')}  <{hit.get('href')}>")

    print()
    if any_flagged:
        print("#" * 78)
        print("### PIVOT-BEFORE-SUBMISSION WARNING ###")
        print("At least one flagged hit above suggests local-conv -> Delta/dt injection may")
        print("already exist in prior work. Per methodology section 16: read every flagged")
        print("hit IN FULL before treating this as a real conflict -- this heuristic works")
        print("off search snippets only and is NOT a substitute for reading the paper. If")
        print("confirmed, PIVOT before submission.")
        print("#" * 78)
    else:
        print(
            "No flagged hits from the automatic pre-filter. This does NOT mean no conflict\n"
            "exists -- it only means none of the returned snippets matched the keyword\n"
            "heuristic. Still read the top ~10 results per string by hand (section 16)."
        )
    return all_results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-query", action="store_true", help="print the search strings only; skip the optional automatic query path")
    parser.add_argument("--max-results", type=int, default=8, help="results per string for the optional query path (default 8)")
    args = parser.parse_args()

    print_report(do_query=not args.no_query, max_results=args.max_results)


if __name__ == "__main__":
    main()
